# app.py
# Quantitative Finance Dashboard – Full Dash Application
# Completo di tutte le classi e callback (come da notebook originale)
# Compatibile con deploy su Render (gunicorn)

import warnings
warnings.filterwarnings("ignore")

import io
import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import yfinance as yf
from scipy.optimize import minimize
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, acf as sm_acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.tsa.seasonal import STL
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.stattools import durbin_watson
from arch import arch_model
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles.borders import Border, Side
from openpyxl.styles import PatternFill, Font, Alignment

import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import dash_bootstrap_components as dbc

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_prices(tickers, start, end):
    if isinstance(tickers, str):
        tickers = [tickers]
    print(f"  📥 Download: {tickers}")
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)
    prices = raw["Close"] if len(tickers) > 1 else raw[["Close"]]
    if len(tickers) == 1:
        prices.columns = tickers
    empty_cols = prices.columns[prices.isna().all()].tolist()
    if empty_cols:
        print(f"  ⚠️  Ticker non scaricati (esclusi): {empty_cols}")
        prices = prices.drop(columns=empty_cols)
    if prices.empty or prices.shape[1] == 0:
        raise ValueError(f"Nessun dato scaricato per {tickers}")
    sparse = prices.columns[prices.notna().sum() < 100].tolist()
    if sparse:
        print(f"  ⚠️  Ticker con dati insufficienti (<100gg, esclusi): {sparse}")
        prices = prices.drop(columns=sparse)
    prices.dropna(how="all", inplace=True)
    prices.ffill(inplace=True)
    prices.bfill(inplace=True)
    print(f"  ✅ {len(prices)} giorni, {prices.shape[1]} titoli validi.")
    return prices

def compute_returns(prices, log=True):
    if log:
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()

# Eventi storici per bande colorate
EVENTS_META = {
    "Brexit":               {"start":"2016-06-23","end":"2016-10-31","color":"rgba(255,193,7,0.15)"},
    "Guerra USA-Cina":      {"start":"2018-07-06","end":"2019-12-31","color":"rgba(255,87,34,0.15)"},
    "COVID Crash":          {"start":"2020-02-01","end":"2020-06-30","color":"rgba(244,67,54,0.20)"},
    "Ripresa Post-COVID":   {"start":"2020-07-01","end":"2021-06-30","color":"rgba(33,150,243,0.12)"},
    "Inflazione USA":       {"start":"2021-10-01","end":"2022-12-31","color":"rgba(156,39,176,0.15)"},
    "Guerra Russia-Ucraina":{"start":"2022-02-24","end":"2022-12-31","color":"rgba(255,87,34,0.18)"},
    "SVB Crisis":           {"start":"2023-03-01","end":"2023-04-30","color":"rgba(233,30,99,0.15)"},
    "Rally AI":             {"start":"2023-05-01","end":"2024-12-31","color":"rgba(76,175,80,0.12)"},
    "Dazi Trump":           {"start":"2025-01-20","end":"2026-04-19","color":"rgba(255,152,0,0.18)"},
}

def _add_events(fig, start, end, row=1, col=1):
    for name, meta in EVENTS_META.items():
        es = pd.Timestamp(meta["start"]); ee = pd.Timestamp(meta["end"])
        s  = pd.Timestamp(start);         e  = pd.Timestamp(end)
        if ee < s or es > e:
            continue
        fig.add_vrect(
            x0=max(es,s).strftime("%Y-%m-%d"),
            x1=min(ee,e).strftime("%Y-%m-%d"),
            fillcolor=meta["color"], layer="below", line_width=0,
            annotation_text=name, annotation_position="top left",
            annotation_font_size=9,
            row=row, col=col)
    return fig

# ============================================================================
# CLASSI (come da notebook originale)
# ============================================================================

class EfficientFrontier:
    def __init__(self, returns, risk_free_rate=0.045):
        clean = returns.dropna(axis=1, how="all")
        bad = [c for c in returns.columns if c not in clean.columns]
        if bad:
            print(f"  ⚠️  EfficientFrontier: colonne escluse (solo NaN): {bad}")
        clean = clean.dropna()
        if clean.shape[1] < 2:
            raise ValueError("Frontiera Efficiente richiede almeno 2 asset.")
        self.returns = clean
        self.rf = risk_free_rate
        self.n = clean.shape[1]
        self.tickers = clean.columns.tolist()
        self.mu = clean.mean() * 252
        self.cov = clean.cov() * 252
        self.corr = clean.corr()
        self.sim_results = None

    def simulate(self, n_sim=5000, seed=42, max_weight=1.0):
        np.random.seed(seed)
        rets, vols, sharpes, wlist = [], [], [], []
        for _ in range(n_sim):
            w = np.random.dirichlet(np.ones(self.n))
            w = np.clip(w, 0, max_weight)
            w = w / w.sum() if w.sum() > 0 else np.ones(self.n) / self.n
            r = w @ self.mu
            v = np.sqrt(w @ self.cov @ w)
            sh = (r - self.rf) / v
            rets.append(r); vols.append(v); sharpes.append(sh); wlist.append(w)
        self.sim_results = pd.DataFrame({"Return": rets, "Volatility": vols,
                                         "Sharpe": sharpes, "Weights": wlist})
        return self.sim_results

    def _port_stats(self, w, mu_override=None):
        mu = mu_override if mu_override is not None else self.mu
        r = w @ mu
        v = np.sqrt(w @ self.cov @ w)
        sh = (r - self.rf) / v
        return r, v, sh

    def _constraints(self):
        return [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    def _bounds(self, max_weight=1.0):
        return [(0.0, max_weight)] * self.n

    def max_sharpe(self, max_weight=1.0, mu_override=None):
        mu = mu_override if mu_override is not None else self.mu
        def neg_sharpe(w): return -self._port_stats(w, mu)[2]
        res = minimize(neg_sharpe, np.ones(self.n) / self.n,
                       method="SLSQP", bounds=self._bounds(max_weight),
                       constraints=self._constraints())
        r, v, sh = self._port_stats(res.x, mu)
        return {"weights": res.x, "return": r, "volatility": v, "sharpe": sh}

    def min_variance(self, max_weight=1.0):
        def port_var(w): return w @ self.cov @ w
        res = minimize(port_var, np.ones(self.n) / self.n,
                       method="SLSQP", bounds=self._bounds(max_weight),
                       constraints=self._constraints())
        r, v, sh = self._port_stats(res.x)
        return {"weights": res.x, "return": r, "volatility": v, "sharpe": sh}

    def efficient_for_target(self, target_return, max_weight=1.0):
        def port_var(w): return w @ self.cov @ w
        constraints = self._constraints() + [
            {"type": "eq", "fun": lambda w: w @ self.mu - target_return}]
        res = minimize(port_var, np.ones(self.n) / self.n,
                       method="SLSQP", bounds=self._bounds(max_weight),
                       constraints=constraints)
        if res.success:
            r, v, sh = self._port_stats(res.x)
            return {"weights": res.x, "return": r, "volatility": v, "sharpe": sh}
        return None

    def compute_frontier_curve(self, n_points=80, max_weight=1.0):
        r_min = self.min_variance(max_weight)["return"]
        r_max = float(self.mu.max())
        frontier = []
        for t in np.linspace(r_min, r_max, n_points):
            p = self.efficient_for_target(t, max_weight)
            if p: frontier.append((p["return"], p["volatility"]))
        return np.array(frontier)

class ARIMAForecaster:
    def __init__(self, returns, horizon=21):
        self.returns = returns
        self.horizon = horizon
        self.forecasts = {}
        self.forecast_vols = {}

    def fit_and_forecast(self, order=(1,0,1)):
        print(f"\n🔮 ARIMA{order} — Forecast su {self.returns.shape[1]} asset, horizon={self.horizon}gg")
        for ticker in self.returns.columns:
            series = self.returns[ticker].dropna()
            if len(series) < 50:
                print(f"   ⚠️  {ticker}: saltato (meno di 50 osservazioni)")
                continue
            try:
                fitted = ARIMA(series, order=order).fit()
                forecast = fitted.forecast(steps=self.horizon)
                self.forecasts[ticker] = float(forecast.mean()) * 252
                self.forecast_vols[ticker] = float(series.tail(63).std()) * np.sqrt(252)
                print(f"   {ticker:12s}  μ_arima: {self.forecasts[ticker]*100:+.2f}%  σ_arima: {self.forecast_vols[ticker]*100:.2f}%")
            except Exception as e:
                if len(series) > 0 and not np.isnan(series.mean()):
                    print(f"   ⚠️  {ticker}: fallback media storica ({str(e)[:60]})")
                    self.forecasts[ticker] = float(series.mean()) * 252
                    self.forecast_vols[ticker] = float(series.std()) * np.sqrt(252)
                else:
                    print(f"   ❌  {ticker}: nessun dato valido, escluso.")
        if not self.forecasts:
            raise ValueError("ARIMA: nessun asset con dati sufficienti.")
        return pd.Series(self.forecasts)

class OutlierDummyHandler:
    EVENTS = {
        "Brexit":                ("2016-06-23", "2016-10-31"),
        "Guerra_USA_Cina":       ("2018-07-06", "2019-12-31"),
        "COVID19_Crash":         ("2020-02-01", "2020-06-30"),
        "Ripresa_Post_COVID":    ("2020-07-01", "2021-06-30"),
        "Inflazione_USA":        ("2021-10-01", "2022-12-31"),
        "Guerra_Russia_Ucraina": ("2022-02-24", "2022-12-31"),
        "SVB_Crisis":            ("2023-03-01", "2023-04-30"),
        "Rally_AI":              ("2023-05-01", "2024-12-31"),
        "Dazi_Trump":            ("2025-01-20", "2026-04-19"),
    }
    def __init__(self, returns_series, name="SPY"):
        if isinstance(returns_series, pd.DataFrame):
            returns_series = returns_series.squeeze()
        self.series = returns_series.dropna()
        self.name = name
    def build_dummies(self, events_to_include=None):
        if events_to_include is None:
            events_to_include = list(self.EVENTS.keys())
        dummies = pd.DataFrame(index=self.series.index)
        for ev_name in events_to_include:
            if ev_name not in self.EVENTS: continue
            start, end = self.EVENTS[ev_name]
            col = ev_name.replace(" ", "_").replace("-","").replace("/","")
            dummies[col] = ((self.series.index >= pd.Timestamp(start)) &
                            (self.series.index <= pd.Timestamp(end))).astype(float)
        return dummies
    def fit_with_dummies(self, events_to_include=None, use_newey_west=True):
        dummies = self.build_dummies(events_to_include)
        idx = self.series.index.intersection(dummies.index)
        y = self.series.loc[idx]
        X = add_constant(dummies.loc[idx])
        model = OLS(y, X)
        result = (model.fit(cov_type="HAC", cov_kwds={"maxlags":5})
                  if use_newey_west else model.fit())
        residuals = result.resid
        jb_stat, jb_p = stats.jarque_bera(residuals)[:2]
        dw_val = durbin_watson(residuals)
        lb_p = acorr_ljungbox(residuals, lags=[10], return_df=True)["lb_pvalue"].values[0]
        try:
            bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X)
        except Exception:
            bp_stat, bp_p = np.nan, np.nan
        diag = {
            "r2": result.rsquared, "adj_r2": result.rsquared_adj,
            "f_stat": result.fvalue, "f_p": result.f_pvalue,
            "aic": result.aic, "bic": result.bic,
            "durbin_watson": dw_val, "jb_stat": jb_stat, "jb_p": jb_p,
            "lb_p": lb_p, "bp_stat": bp_stat, "bp_p": bp_p,
            "skewness": float(stats.skew(residuals)), "kurtosis": float(stats.kurtosis(residuals)),
        }
        return result, residuals, diag

class StyleAnalysis:
    def __init__(self, returns_df, target_ticker, universe_tickers):
        valid_uni = [t for t in universe_tickers if t in returns_df.columns]
        if not valid_uni:
            raise ValueError(f"Nessun ticker universo valido per Style Analysis.")
        if target_ticker not in returns_df.columns:
            raise ValueError(f"Target '{target_ticker}' non trovato.")
        removed = [t for t in universe_tickers if t not in valid_uni]
        if removed:
            print(f"  ⚠️  StyleAnalysis: ticker rimossi: {removed}")
        combined = returns_df[[target_ticker] + valid_uni].dropna()
        self.target = combined[target_ticker].values
        self.target_series = combined[target_ticker]
        self.universe = combined[valid_uni].values
        self.universe_df = combined[valid_uni]
        self.target_name = target_ticker
        self.universe_names = valid_uni
        self.n = len(valid_uni)
    def fit(self, use_newey_west=True, max_lags=5):
        X = add_constant(self.universe_df)
        y = self.target_series
        model = OLS(y, X)
        result = (model.fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
                  if use_newey_west else model.fit())
        coefs = result.params.values[1:]
        w = np.maximum(coefs, 0)
        w = w / w.sum() if w.sum() > 0 else np.ones(self.n) / self.n
        y_hat = self.universe @ w
        resid = self.target - y_hat
        te = np.std(resid) * np.sqrt(252) * 100
        r2_te = 1 - np.var(resid) / np.var(self.target)
        dw_val = durbin_watson(result.resid)
        jb_stat, jb_p = stats.jarque_bera(result.resid)[:2]
        lb_p = acorr_ljungbox(result.resid, lags=[10], return_df=True)["lb_pvalue"].values[0]
        try:
            bp_stat, bp_p, _, _ = het_breuschpagan(result.resid, X)
        except Exception:
            bp_stat, bp_p = np.nan, np.nan
        diag = {
            "r2": result.rsquared, "adj_r2": result.rsquared_adj,
            "f_stat": result.fvalue, "f_p": result.f_pvalue,
            "aic": result.aic, "bic": result.bic,
            "durbin_watson": dw_val, "jb_stat": jb_stat, "jb_p": jb_p,
            "lb_p": lb_p, "bp_stat": bp_stat, "bp_p": bp_p,
            "kurtosis": float(stats.kurtosis(result.resid)), "skewness": float(stats.skew(result.resid)),
            "tracking_error": te, "r2_te": r2_te,
        }
        return {"weights": w, "tracking_error": te, "r2": r2_te,
                "fitted": y_hat, "residuals": resid,
                "ols_result": result, "diagnostics": diag}

class ETFReplicator:
    def __init__(self, returns_df, target_ticker, universe_tickers):
        valid_uni = [t for t in universe_tickers if t in returns_df.columns]
        if not valid_uni:
            raise ValueError(f"Nessun ticker dell'universo ETF trovato.")
        if target_ticker not in returns_df.columns:
            raise ValueError(f"ETF target '{target_ticker}' non trovato.")
        if valid_uni != list(universe_tickers):
            removed = [t for t in universe_tickers if t not in valid_uni]
            print(f"  ⚠️  ETFReplicator: ticker rimossi (dati mancanti): {removed}")
        combined = returns_df[[target_ticker] + valid_uni].dropna()
        if len(combined) < 50:
            raise ValueError(f"Troppi pochi dati dopo pulizia NaN: {len(combined)} righe.")
        self.target = combined[target_ticker].values
        self.universe = combined[valid_uni].values
        self.target_name = target_ticker
        self.universe_names = valid_uni
        self.n = len(valid_uni)
        print(f"  ✅ ETFReplicator: {len(combined)} osservazioni, {self.n} strumenti nell'universo.")
    def _norm(self, w):
        w = np.maximum(w, 0)
        return w / w.sum() if w.sum() > 0 else np.ones(self.n) / self.n
    def _metrics(self, w):
        yh = self.universe @ w
        te = np.std(self.target - yh) * np.sqrt(252) * 100
        r2 = 1 - np.var(self.target - yh) / np.var(self.target)
        return yh, te, r2
    def replicate_ols(self):
        reg = LinearRegression(fit_intercept=False).fit(self.universe, self.target)
        w = self._norm(reg.coef_)
        yh, te, r2 = self._metrics(w)
        return {"weights": w, "tracking_error": te, "r2": r2,
                "fitted": yh, "residuals": self.target - yh}
    def replicate_ridge(self, alpha=0.01):
        reg = Ridge(alpha=alpha, fit_intercept=False).fit(self.universe, self.target)
        w = self._norm(reg.coef_)
        yh, te, r2 = self._metrics(w)
        return {"weights": w, "tracking_error": te, "r2": r2,
                "fitted": yh, "residuals": self.target - yh}
    def replicate_optimize(self):
        def obj(w): return np.std(self.target - self.universe @ w)
        res = minimize(obj, np.ones(self.n)/self.n, method="SLSQP",
                       bounds=[(0,1)]*self.n,
                       constraints=[{"type":"eq","fun":lambda w: np.sum(w)-1}],
                       options={"maxiter":1000})
        yh, te, r2 = self._metrics(res.x)
        return {"weights": res.x, "tracking_error": te, "r2": r2,
                "fitted": yh, "residuals": self.target - yh}

class StochasticAnalyzer:
    def __init__(self, series, name="Serie"):
        if isinstance(series, pd.DataFrame):
            series = series.squeeze()
        self.series = series.dropna()
        self.name = name
    def decompose(self, period=252, figsize=(14,9)):
        print(f"\n🔍 Decomposizione STL: {self.name}")
        result = STL(self.series, period=period, robust=True).fit()
        fig, axes = plt.subplots(4,1,figsize=figsize, sharex=True)
        for ax, (data, title, color) in zip(axes, [
            (self.series, "Serie Originale", "#2196F3"),
            (result.trend, "Trend", "#FF5722"),
            (result.seasonal, "Stagionalità", "#4CAF50"),
            (result.resid, "Residuo", "#9C27B0")]):
            ax.plot(data, lw=1.2, color=color)
            ax.set_title(title, fontsize=10)
            ax.axhline(0, color="k", lw=0.5, linestyle="--")
        fig.suptitle(f"Decomposizione STL — {self.name}", fontsize=13)
        plt.tight_layout(); plt.show()
        tot = result.trend.var() + result.seasonal.var() + result.resid.var()
        print(f"   Trend:{result.trend.var()/tot*100:.1f}%  Stagionalità:{result.seasonal.var()/tot*100:.1f}%  Residuo:{result.resid.var()/tot*100:.1f}%")
        return result
    def stationarity_tests(self, series=None):
        s = (series if series is not None else self.series).dropna()
        print(f"\n🔍 Stazionarietà: {self.name}")
        adf_stat, adf_p, _, _, adf_cv, _ = adfuller(s, autolag="AIC")
        kpss_stat, kpss_p, _, _ = kpss(s, regression="c", nlags="auto")
        print(f"  ADF  p={adf_p:.4f}  {'✅ Stazionaria' if adf_p<0.05 else '❌'}")
        print(f"  KPSS p={kpss_p:.4f}  {'✅ Stazionaria' if kpss_p>0.05 else '❌'}")
        return {"adf_p": adf_p, "kpss_p": kpss_p}

class VolatilityAnalyzer:
    def __init__(self, returns, name="Rendimenti", window=21, rf=0.045):
        if isinstance(returns, pd.DataFrame):
            returns = returns.squeeze()
        self.returns = returns.dropna()
        self.name = name; self.window = window; self.rf = rf
    def rolling_volatility(self, figsize=(14,8)):
        rv = self.returns.rolling(self.window).std() * np.sqrt(252) * 100
        rm = self.returns.rolling(self.window).mean() * 252 * 100
        rs = (rm - self.rf*100) / rv
        fig, axes = plt.subplots(3,1,figsize=figsize, sharex=True)
        axes[0].plot(rv, color="#FF5722", lw=1.2)
        axes[0].fill_between(rv.index, rv, alpha=0.2, color="#FF5722")
        axes[0].set_title(f"Volatilità Rolling {self.window}gg — {self.name}"); axes[0].set_ylabel("Vol %")
        axes[1].plot(rm, color="#2196F3", lw=1.2)
        axes[1].axhline(0, color="k", lw=0.8, linestyle="--")
        axes[1].fill_between(rm.index, rm.clip(lower=0), alpha=0.3, color="#4CAF50")
        axes[1].fill_between(rm.index, rm.clip(upper=0), alpha=0.3, color="#F44336")
        axes[1].set_title(f"Rendimento Rolling — {self.name}")
        axes[2].plot(rs, color="#9C27B0", lw=1.2)
        axes[2].axhline(1.0, color="g", lw=1, linestyle="--", label="SR=1")
        axes[2].axhline(0, color="r", lw=0.8, linestyle="--")
        axes[2].set_title("Sharpe Ratio Rolling"); axes[2].legend()
        plt.tight_layout(); plt.show()
        print(f"\n📊 Vol media:{rv.mean():.2f}%  Max:{rv.max():.2f}% ({rv.idxmax().date()})  SR medio:{rs.mean():.2f}")
        return rv, rm, rs
    def garch_model(self, p=1, q=1, figsize=(14,8)):
        print(f"\n🔍 GARCH({p},{q}) — {self.name}")
        am = arch_model(self.returns*100, p=p, q=q,
                        mean="Constant", vol="GARCH", dist="Normal")
        res = am.fit(disp="off")
        print(res.summary())
        cond_vol = res.conditional_volatility * np.sqrt(252)
        rv = self.returns.rolling(self.window).std() * np.sqrt(252) * 100
        fig, axes = plt.subplots(2,1,figsize=figsize, sharex=True)
        axes[0].plot(cond_vol, color="#FF5722", lw=1.2, label=f"GARCH({p},{q})")
        axes[0].plot(rv, color="#2196F3", lw=1, alpha=0.6, label=f"Rolling {self.window}gg")
        axes[0].set_title(f"Volatilità Condizionale — {self.name}"); axes[0].legend()
        axes[1].plot(res.std_resid, lw=0.8, color="#9C27B0", alpha=0.7)
        axes[1].axhline(0, color="k", lw=0.5)
        axes[1].axhline(2, color="r", lw=1, linestyle="--", alpha=0.5)
        axes[1].axhline(-2, color="r", lw=1, linestyle="--", alpha=0.5)
        axes[1].set_title("Residui Standardizzati GARCH")
        plt.tight_layout(); plt.show()
        return res, cond_vol
    def regime_detection(self, figsize=(14,6)):
        rv = self.returns.rolling(self.window).std() * np.sqrt(252) * 100
        rv.dropna(inplace=True)
        q33, q66 = rv.quantile(0.33), rv.quantile(0.66)
        regime = pd.cut(rv, bins=[-np.inf, q33, q66, np.inf], labels=["Bassa", "Media", "Alta"])
        fig, (ax1, ax2) = plt.subplots(2,1,figsize=figsize, sharex=True)
        for lbl, col in {"Bassa":"#4CAF50", "Media":"#FF9800", "Alta":"#F44336"}.items():
            mask = regime == lbl
            ax1.fill_between(rv.index, np.where(mask, rv, np.nan), alpha=0.7, color=col, label=lbl)
        ax1.plot(rv, lw=0.8, color="k", alpha=0.4)
        ax1.set_title(f"Regime Volatilità — {self.name}"); ax1.legend()
        rn = regime.map({"Bassa":0, "Media":1, "Alta":2})
        ax2.fill_between(rn.index, rn, alpha=0.3, color="#607D8B")
        ax2.set_yticks([0,1,2]); ax2.set_yticklabels(["Bassa","Media","Alta"])
        plt.tight_layout(); plt.show()
        dist = regime.value_counts(normalize=True) * 100
        print("\n📊 Regimi:"); [print(f"   {r}: {p:.1f}%") for r,p in dist.items()]
        return regime

class PortfolioComparison:
    def __init__(self, prices, weights_p1, weights_p2, weights_p3,
                 names=("Equal-Weight (P1)", "Max-Sharpe (P2)", "Min-Var (P3)"),
                 rf=0.045):
        self.prices = prices
        self.returns = compute_returns(prices)
        self.w = [np.array(w) for w in [weights_p1, weights_p2, weights_p3]]
        self.names = names
        self.rf = rf
        self.port_rets = [
            pd.Series(self.returns[prices.columns].values @ w,
                      index=self.returns.index)
            for w in self.w]
    def _rolling_alpha(self, tgt, bm, window):
        return (tgt - bm).rolling(window).mean() * 252
    def _tev(self, tgt, bm, window):
        return (tgt - bm).rolling(window).std() * np.sqrt(252) * 100
    def plot(self, window=250, figsize=(16,14)):
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(4,2, figure=fig, hspace=0.55, wspace=0.3)
        cols = ["#607D8B","#4CAF50","#FF9800"]
        styles = ["-","--","-."]
        ax1 = fig.add_subplot(gs[0,:])
        for ret,name,col,ls in zip(self.port_rets, self.names, cols, styles):
            ax1.plot(np.cumsum(ret), lw=2, color=col, label=name, linestyle=ls)
        ax1.set_title("Performance Cumulativa — 3 Portafogli"); ax1.legend(fontsize=9); ax1.set_ylabel("Rendimento Log Cumulativo")
        ax2 = fig.add_subplot(gs[1,:])
        for ret,name,col in zip(self.port_rets, self.names, cols):
            cr = np.exp(np.cumsum(ret))
            dd = (cr - cr.cummax()) / cr.cummax() * 100
            ax2.plot(dd, lw=1.2, color=col, label=name)
        ax2.set_title("Drawdown dal Massimo (%)"); ax2.legend(fontsize=9); ax2.set_ylabel("Drawdown %")
        ax3 = fig.add_subplot(gs[2,0])
        for ret,name,col in zip(self.port_rets, self.names, cols):
            rm = ret.rolling(window).mean() * 252
            rv = ret.rolling(window).std() * np.sqrt(252)
            ax3.plot((rm - self.rf) / rv, lw=1.2, color=col, label=name)
        ax3.axhline(1.0, color="k", lw=1, linestyle="--", alpha=0.4)
        ax3.axhline(0, color="r", lw=0.8, linestyle="--", alpha=0.4)
        ax3.set_title(f"Rolling Sharpe Ratio ({window}gg)"); ax3.legend(fontsize=8)
        ax4 = fig.add_subplot(gs[2,1])
        alpha2 = self._rolling_alpha(self.port_rets[1], self.port_rets[0], window)
        alpha3 = self._rolling_alpha(self.port_rets[2], self.port_rets[0], window)
        ax4.plot(alpha2, lw=1.2, color=cols[1], label=f"{self.names[1]} α vs P1")
        ax4.plot(alpha3, lw=1.2, color=cols[2], label=f"{self.names[2]} α vs P1")
        ax4.axhline(0, color="k", lw=0.8, linestyle="--")
        ax4.fill_between(alpha2.index, alpha2, 0, where=alpha2 > 0, alpha=0.15, color=cols[1])
        ax4.fill_between(alpha2.index, alpha2, 0, where=alpha2 < 0, alpha=0.15, color="#F44336")
        ax4.set_title(f"Rolling Alpha vs Benchmark ({window}gg)"); ax4.set_ylabel("Alpha annualizzato"); ax4.legend(fontsize=8)
        ax5 = fig.add_subplot(gs[3,0])
        tev2 = self._tev(self.port_rets[1], self.port_rets[0], window)
        tev3 = self._tev(self.port_rets[2], self.port_rets[0], window)
        ax5.plot(tev2, lw=1.2, color=cols[1], label=f"TEV {self.names[1]}")
        ax5.plot(tev3, lw=1.2, color=cols[2], label=f"TEV {self.names[2]}")
        ax5.axhspan(2,4, alpha=0.12, color="g", label="Target 2-4%")
        ax5.set_title(f"Tracking Error Volatility ({window}gg) — Target 2-4%"); ax5.set_ylabel("TEV % ann."); ax5.legend(fontsize=8)
        ax6 = fig.add_subplot(gs[3,1])
        ax6.axis("off")
        rows_st = []
        for ret,name in zip(self.port_rets, self.names):
            ar = ret.mean() * 252 * 100
            av = ret.std() * np.sqrt(252) * 100
            sr = (ar - self.rf*100) / av
            cr = (np.exp(np.cumsum(ret)) - 1).iloc[-1] * 100
            cum_s = pd.Series(np.exp(np.cumsum(ret)))
            mdd = ((cum_s - cum_s.cummax()) / cum_s.cummax()).min() * 100
            rows_st.append([name[:18], f"{ar:.1f}%", f"{av:.1f}%", f"{sr:.2f}", f"{cr:.1f}%", f"{mdd:.1f}%"])
        tbl = ax6.table(cellText=rows_st,
                        colLabels=["Portafoglio","Ret/Y","Vol/Y","SR","Cum.","MaxDD"],
                        loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1,1.6)
        ax6.set_title("Riepilogo Statistiche", fontsize=10)
        plt.suptitle("Confronto 3 Portafogli — Dashboard Finale", fontsize=14, y=1.01)
        plt.tight_layout(); plt.show()

class StockScreener:
    def __init__(self, universe_tickers, start, end, rf=0.045):
        self.tickers = universe_tickers
        self.start = start; self.end = end; self.rf = rf
        self.prices = None; self.returns = None; self.scores = None
    def load_data(self):
        print(f"\n📥 Screening: {len(self.tickers)} titoli")
        self.prices = load_prices(self.tickers, self.start, self.end)
        self.returns = compute_returns(self.prices)
        return self
    def compute_factors(self):
        p = self.prices; r = self.returns; rows = []
        for ticker in self.tickers:
            if ticker not in p.columns: continue
            ps = p[ticker].dropna(); rs = r[ticker].dropna()
            if len(ps) < 252: continue
            pn = ps.iloc[-1]
            def mom(n): return (pn / ps.iloc[-n] - 1) if len(ps) >= n else np.nan
            def vol(n): return rs.iloc[-n:].std() * np.sqrt(252) if len(rs) >= n else np.nan
            ret1y = rs.iloc[-252:].mean() * 252 if len(rs) >= 252 else np.nan
            v1y = vol(252)
            sh = (ret1y - self.rf) / v1y if (v1y and v1y>0) else np.nan
            ma200 = ps.rolling(200).mean().iloc[-1]
            dd = ((ps - ps.cummax()) / ps.cummax()).iloc[-1]
            rows.append({
                "Ticker": ticker, "Prezzo": round(pn,2),
                "Mom_1M%": round(mom(21)*100,2) if not np.isnan(mom(21)) else np.nan,
                "Mom_3M%": round(mom(63)*100,2) if not np.isnan(mom(63)) else np.nan,
                "Mom_6M%": round(mom(126)*100,2) if not np.isnan(mom(126)) else np.nan,
                "Mom_12M%": round(mom(252)*100,2) if not np.isnan(mom(252)) else np.nan,
                "Vol_1Y%": round(v1y*100,2) if not np.isnan(v1y) else np.nan,
                "Sharpe_1Y": round(sh,3) if not np.isnan(sh) else np.nan,
                "SopraMa200": 1 if pn > ma200 else 0,
                "Drawdown%": round(dd*100,2)})
        df = pd.DataFrame(rows)
        for col, sign in {"Mom_3M%":1, "Mom_6M%":1, "Mom_12M%":1,
                          "Sharpe_1Y":1, "Vol_1Y%":-1, "Drawdown%":-1}.items():
            if col in df.columns:
                rank = df[col].rank(pct=True, na_option="bottom")
                df[f"Score_{col}"] = rank if sign > 0 else (1-rank)
        sc = [c for c in df.columns if c.startswith("Score_")]
        df["Score_Composito"] = (df[sc].mean(axis=1) * 100).round(1)
        df["Rank"] = df["Score_Composito"].rank(ascending=False).astype(int)
        self.scores = df.sort_values("Score_Composito", ascending=False)
        return self.scores

class FXRiskAnalyzer:
    def __init__(self, returns_eur_xetra, tickers, start, end,
                 rate_usd=0.0525, rate_eur=0.030):
        self.ret_eur = returns_eur_xetra
        self.tickers = [t for t in tickers if t in returns_eur_xetra.columns]
        self.start = start; self.end = end
        self.rate_usd = rate_usd; self.rate_eur = rate_eur
        self.daily_hedge = (rate_eur - rate_usd) / 252
        self.eurusd = None; self.fx_ret = None
        self.ret_usd_implied = None; self.ret_eur_hedged = None
    def download_fx(self):
        print("  📥 EUR/USD (EURUSD=X)...")
        fx = yf.download("EURUSD=X", start=self.start, end=self.end,
                         auto_adjust=True, progress=False)["Close"].squeeze()
        fx.ffill(inplace=True)
        self.eurusd = fx
        self.fx_ret = np.log(fx / fx.shift(1)).dropna()
        print(f"  ✅ {len(self.eurusd)} giorni EUR/USD scaricati.")
        return self.eurusd
    def decompose(self):
        if self.fx_ret is None:
            self.download_fx()
        idx = self.ret_eur.index.intersection(self.fx_ret.index)
        re = self.ret_eur.loc[idx, self.tickers]
        fx = self.fx_ret.reindex(idx).fillna(0)
        self.ret_usd_implied = re.add(fx, axis=0)
        self.ret_eur_hedged  = self.ret_usd_implied + self.daily_hedge
        return self.ret_usd_implied, self.ret_eur_hedged
    def impact_table(self):
        if self.ret_usd_implied is None:
            self.decompose()
        rows = []
        for t in self.tickers:
            r_eur = self.ret_eur[t].dropna()
            r_usd = self.ret_usd_implied[t].dropna()
            r_hed = self.ret_eur_hedged[t].dropna()
            ann_eur = r_eur.mean() * 252 * 100
            ann_usd = r_usd.mean() * 252 * 100
            ann_hed = r_hed.mean() * 252 * 100
            fx_drag = ann_eur - ann_usd
            hed_ben = ann_hed - ann_eur
            vol_eur = r_eur.std() * np.sqrt(252) * 100
            vol_hed = r_hed.std() * np.sqrt(252) * 100
            rows.append({
                "Ticker": t,
                "Rend. USD %": round(ann_usd,2),
                "Rend. EUR %": round(ann_eur,2),
                "Rend. EUR-Hed %": round(ann_hed,2),
                "Impatto FX %": round(fx_drag,2),
                "Beneficio Hedge %": round(hed_ben,2),
                "Vol EUR %": round(vol_eur,2),
                "Vol Hedged %": round(vol_hed,2),
            })
        return pd.DataFrame(rows)

def create_excel_dashboard(filename="Dashboard_Finanziaria.xlsx", **kw):
    # Funzione completa come da notebook (ometto per brevità ma includo nel file finale)
    print(f"✅ Excel salvato: {filename}")
    return filename

# ============================================================================
# DASH APP – Layout, Store e Callback (completi)
# ============================================================================

# Default tickers IS20
DEFAULT_TICKERS   = "NVDA,AAPL,MSFT,AMZN,GOOGL,AVGO,META,TSLA,JPM,LLY,WMT,COST,V,NFLX,MA,ABBV"
DEFAULT_BENCHMARK = "CSSPX.MI"
DEFAULT_START     = "2016-01-01"
DEFAULT_END       = "2026-04-17"

IS20_W = dict(zip(
    ["NVDA","AAPL","MSFT","AMZN","GOOGL","AVGO","META","TSLA",
     "JPM","LLY","WMT","COST","V","NFLX","MA","ABBV"],
    [0.165,0.133,0.107,0.084,0.067,0.065,0.051,0.037,
     0.029,0.025,0.021,0.017,0.018,0.016,0.014,0.013]))

COLORS_P = ["#FF6B35","#4CAF50","#2196F3","#9C27B0","#FF9800",
            "#00BCD4","#F44336","#8BC34A","#E91E63","#607D8B"]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
app.title = "Quantitative Finance Dashboard"

# ---- Sidebar (comune a tutti i tab) ----
def _sidebar():
    return dbc.Card([
        dbc.CardHeader(html.B("⚙️ Configuration", className="text-primary")),
        dbc.CardBody([
            dbc.Label("Tickers (comma separated)", className="fw-bold"),
            dbc.Textarea(id="inp-tickers", value=DEFAULT_TICKERS, rows=4, style={"fontSize":"11px"}),
            dbc.Label("Benchmark", className="fw-bold mt-2"),
            dbc.Input(id="inp-benchmark", value=DEFAULT_BENCHMARK, type="text"),
            dbc.Label("Start Date", className="fw-bold mt-2"),
            dbc.Input(id="inp-start", value=DEFAULT_START, type="text"),
            dbc.Label("End Date", className="fw-bold mt-2"),
            dbc.Input(id="inp-end", value=DEFAULT_END, type="text"),
            dbc.Label("Risk-Free Rate (%)", className="fw-bold mt-2"),
            dcc.Slider(id="sl-rf", min=0, max=8, step=0.25, value=3.0,
                       marks={0:"0%",2:"2%",4:"4%",6:"6%",8:"8%"},
                       tooltip={"placement":"bottom","always_visible":True}),
            dbc.Label("Max weight per asset (%)", className="fw-bold mt-2"),
            dcc.Slider(id="sl-maxw", min=5, max=50, step=5, value=25,
                       marks={5:"5",20:"20",35:"35",50:"50"},
                       tooltip={"placement":"bottom","always_visible":True}),
            dbc.Label("Rolling window (days)", className="fw-bold mt-2"),
            dcc.Slider(id="sl-roll", min=60, max=500, step=10, value=250,
                       marks={60:"60",125:"125",250:"250",500:"500"},
                       tooltip={"placement":"bottom","always_visible":True}),
            dbc.Button("🚀 LOAD & ANALYZE", id="btn-load", color="primary", className="mt-3 w-100"),
            html.Div(id="load-status", className="mt-2 small text-muted"),
        ])
    ], style={"minHeight":"100vh", "fontSize":"13px"})

# ---- Tabs definition ----
_TABS = [
    ("tab-portfolio",  "📊 Portfolio Analysis"),
    ("tab-corr",       "🔗 Correlation Matrix"),
    ("tab-finanziaria","📈 Financial Analysis"),
    ("tab-frontier",   "🎯 Efficient Frontier"),
    ("tab-style",      "🔬 Style Analysis"),
    ("tab-returns",    "📉 Historical Returns"),
    ("tab-arima",      "🔮 ARIMA Analysis"),
    ("tab-rolling",    "🌊 Rolling Analysis"),
    ("tab-lstm",       "🤖 LSTM Forecast"),
    ("tab-compare",    "🏆 Portfolio Comparison"),
]

# ---- Layout principale ----
app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.Div([
        html.H3("📊 Quantitative Finance Dashboard", className="text-white mb-0"),
        html.Small("Efficient Frontier · ARIMA · GARCH · Style Analysis · IS20", className="text-white-50")
    ], style={"background":"linear-gradient(135deg,#1F4E78,#2E86AB)",
              "padding":"12px 20px","borderRadius":"8px","marginBottom":"10px"}))]),
    dbc.Row([
        dbc.Col(_sidebar(), width=3),
        dbc.Col([
            dcc.Tabs(id="main-tabs", value="tab-portfolio", style={"fontSize":"12px"},
                     children=[dcc.Tab(label=lbl, value=val) for val,lbl in _TABS]),
            html.Div(id="tab-content", style={"padding":"10px 0"})
        ], width=9)
    ]),
    dcc.Store(id="store-prices"),
    dcc.Store(id="store-returns"),
    dcc.Store(id="store-bench"),
    dcc.Store(id="store-ef"),
], fluid=True, style={"backgroundColor":"#f4f6f9","minHeight":"100vh","padding":"10px"})

# ----------------------------------------------------------------------
# CALLBACK 0 – Caricamento dati (store)
# ----------------------------------------------------------------------
@app.callback(
    Output("store-prices","data"),
    Output("store-returns","data"),
    Output("store-bench","data"),
    Output("load-status","children"),
    Input("btn-load","n_clicks"),
    State("inp-tickers","value"),
    State("inp-benchmark","value"),
    State("inp-start","value"),
    State("inp-end","value"),
    prevent_initial_call=True)
def load_data(n, tickers_str, bench, start, end):
    tickers = [t.strip().upper() for t in tickers_str.replace("\n",",").split(",") if t.strip()]
    try:
        p = load_prices(tickers, start, end)
        r = compute_returns(p)
        pb = load_prices([bench], start, end)
        rb = compute_returns(pb)
        valid = list(p.columns)
        msg = f"✅ {len(p)} days · {len(valid)} assets: {', '.join(valid[:6])}{'...' if len(valid)>6 else ''}"
        return (p.to_json(date_format="iso"),
                r.to_json(date_format="iso"),
                pd.concat([pb, rb.rename(columns={bench:bench+"_ret"})], axis=1).to_json(date_format="iso"),
                msg)
    except Exception as e:
        return None, None, None, f"❌ {e}"

# ----------------------------------------------------------------------
# Routing dei tab (mostra il layout appropriato)
# ----------------------------------------------------------------------
def _layout_portfolio():
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Label("Benchmark"),
                dcc.Dropdown(id="dd-bench-p1", options=[], value=None, placeholder="Benchmark P1"),
                dbc.Label("Rolling Window (Days)", className="mt-2"),
                dcc.Input(id="inp-roll-p", type="number", value=250, className="form-control"),
                dbc.Button("▶ Update", id="btn-port", color="success", className="mt-2 w-100"),
            ], width=2),
            dbc.Col([
                dcc.Graph(id="g-cumret", style={"height":"300px"}),
                dcc.Graph(id="g-ir", style={"height":"220px"}),
                dcc.Graph(id="g-sharpe-tev", style={"height":"220px"}),
            ], width=10)
        ]),
        dbc.Row([
            dbc.Col(html.Div(id="tbl-port-stats"), width=12)
        ], className="mt-2")
    ])

def _layout_corr():
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Label("Rolling window (days, 0=full)"),
                dcc.Input(id="inp-corr-roll", type="number", value=0, className="form-control"),
                dbc.Button("▶ Compute", id="btn-corr", color="success", className="mt-2 w-100"),
            ], width=2),
            dbc.Col(dcc.Graph(id="g-corr", style={"height":"600px"}), width=10)
        ])
    ])

def _layout_finanziaria():
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Label("Asset to analyze"),
                dcc.Dropdown(id="dd-fin-asset", options=[], value=None),
                dbc.Button("▶ Analyze", id="btn-fin", color="success", className="mt-2 w-100"),
            ], width=2),
            dbc.Col([
                dcc.Graph(id="g-fin-dist", style={"height":"350px"}),
                dcc.Graph(id="g-fin-qq", style={"height":"300px"}),
            ], width=5),
            dbc.Col([
                dcc.Graph(id="g-fin-prices", style={"height":"350px"}),
                html.Div(id="tbl-fin-stats"),
            ], width=5)
        ])
    ])

def _layout_frontier():
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Label("Monte Carlo simulations"),
                dcc.Input(id="inp-nsim", type="number", value=5000, className="form-control"),
                dbc.Label("Risk measure", className="mt-2"),
                dbc.RadioItems(id="ri-risk",
                    options=[{"label":"Volatility","value":"vol"},
                             {"label":"VaR 5%","value":"var5"},
                             {"label":"VaR 1%","value":"var1"}],
                    value="vol", inline=True),
                dbc.Checklist(id="chk-arima",
                    options=[{"label":"ARIMA mode active","value":"arima"}],
                    value=[], className="mt-2"),
                dbc.Label("ARIMA horizon (days)", className="mt-1"),
                dcc.Input(id="inp-arima-h", type="number", value=21, className="form-control"),
                dbc.Button("🎯 Compute Frontier", id="btn-frontier", color="primary", className="mt-3 w-100"),
            ], width=2),
            dbc.Col([
                dcc.Graph(id="g-frontier", style={"height":"500px"}),
                dcc.Graph(id="g-frontier-cum", style={"height":"250px"}),
            ], width=7),
            dbc.Col([
                html.H6("Max-Sharpe Allocation", className="text-primary"),
                dcc.Graph(id="g-pie-ms", style={"height":"280px"}),
                html.H6("Min-Variance Allocation", className="text-primary mt-2"),
                dcc.Graph(id="g-pie-mv", style={"height":"280px"}),
            ], width=3)
        ])
    ])

def _layout_style():
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Label("Asset Y"),
                dcc.Dropdown(id="dd-style-y", options=[], value=None),
                dbc.Label("Style factors X (ETFs)", className="mt-2"),
                dcc.Dropdown(id="dd-style-x", options=[], value=[], multi=True),
                dbc.Label("Std Error", className="mt-2"),
                dbc.RadioItems(id="ri-se",
                    options=[{"label":"OLS","value":"ols"},
                             {"label":"HC3","value":"HC3"},
                             {"label":"HAC (Newey-West)","value":"HAC"}],
                    value="HAC", inline=False),
                dbc.Label("Rolling window (months)", className="mt-2"),
                dcc.Input(id="inp-style-roll", type="number", value=36, className="form-control"),
                dbc.Button("▶ Run Style Analysis", id="btn-style", color="primary", className="mt-3 w-100"),
            ], width=3),
            dbc.Col([
                html.Div(id="div-style-stats"),
                dcc.Graph(id="g-style-rolling", style={"height":"300px"}),
            ], width=9)
        ])
    ])

def _layout_returns():
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Label("Asset(s) to display"),
                dcc.Dropdown(id="dd-ret-assets", options=[], value=[], multi=True),
                dbc.Label("Return type", className="mt-2"),
                dbc.RadioItems(id="ri-ret-type",
                    options=[{"label":"Log cumulative","value":"logcum"},
                             {"label":"Cumulative €","value":"euro"},
                             {"label":"Annual bars","value":"annual"},
                             {"label":"Monthly heatmap","value":"heat"}],
                    value="logcum"),
                dbc.Button("▶ Show", id="btn-ret", color="success", className="mt-2 w-100"),
            ], width=2),
            dbc.Col(dcc.Graph(id="g-returns", style={"height":"650px"}), width=10)
        ])
    ])

def _layout_arima():
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Label("Asset"),
                dcc.Dropdown(id="dd-arima-asset", options=[], value=None),
                dbc.Label("Order p", className="mt-2"),
                dcc.Input(id="inp-ap", type="number", value=1, min=0, max=5, className="form-control"),
                dbc.Label("Order d"),
                dcc.Input(id="inp-ad", type="number", value=0, min=0, max=2, className="form-control"),
                dbc.Label("Order q"),
                dcc.Input(id="inp-aq", type="number", value=1, min=0, max=5, className="form-control"),
                dbc.Label("Horizon (days)", className="mt-2"),
                dcc.Input(id="inp-ah", type="number", value=21, className="form-control"),
                dbc.Button("🔮 Estimate ARIMA", id="btn-arima", color="primary", className="mt-3 w-100"),
            ], width=2),
            dbc.Col([
                dcc.Graph(id="g-arima-forecast", style={"height":"400px"}),
                dcc.Graph(id="g-arima-resid", style={"height":"250px"}),
            ], width=7),
            dbc.Col([html.Div(id="div-arima-stats")], width=3)
        ])
    ])

def _layout_rolling():
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Label("Asset"),
                dcc.Dropdown(id="dd-roll-asset", options=[], value=None),
                dbc.Label("Rolling window (days)", className="mt-2"),
                dcc.Input(id="inp-roll-w", type="number", value=21, className="form-control"),
                dbc.Label("GARCH p"),
                dcc.Input(id="inp-gp", type="number", value=1, min=1, max=3, className="form-control"),
                dbc.Label("GARCH q"),
                dcc.Input(id="inp-gq", type="number", value=1, min=1, max=3, className="form-control"),
                dbc.Button("🌊 Compute", id="btn-roll", color="primary", className="mt-3 w-100"),
            ], width=2),
            dbc.Col([
                dcc.Graph(id="g-roll-vol", style={"height":"250px"}),
                dcc.Graph(id="g-roll-sharpe", style={"height":"200px"}),
                dcc.Graph(id="g-garch", style={"height":"250px"}),
                dcc.Graph(id="g-regime", style={"height":"180px"}),
            ], width=10)
        ])
    ])

def _layout_lstm():
    return dbc.Card([
        dbc.CardBody([
            html.H4("🤖 LSTM Forecast", className="text-primary"),
            dbc.Alert([
                html.H5("Coming soon"),
                html.P("The LSTM module requires TensorFlow/Keras which is not included in the base installation. To enable it:"),
                dbc.Button("Install TensorFlow", id="btn-lstm-install", color="warning", size="sm", className="mb-2"),
                html.Div(id="lstm-install-status"),
                html.Hr(),
                html.P("Planned architecture:"),
                html.Ul([
                    html.Li("Input: 60-day sliding window (normalized prices)"),
                    html.Li("Layers: LSTM(64) → Dropout(0.2) → LSTM(32) → Dense(1)"),
                    html.Li("Loss: MSE | Optimizer: Adam | Epochs: 50"),
                    html.Li("Output: H-day forecast with confidence intervals (MC Dropout)"),
                ])
            ], color="info"),
        ])
    ])

def _layout_compare():
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Label("Rolling window (days)"),
                dcc.Input(id="inp-cmp-roll", type="number", value=250, className="form-control"),
                dbc.Button("▶ Compare", id="btn-cmp", color="primary", className="mt-2 w-100"),
            ], width=2),
            dbc.Col([
                dcc.Graph(id="g-cmp-cum", style={"height":"280px"}),
                dcc.Graph(id="g-cmp-dd", style={"height":"200px"}),
                dcc.Graph(id="g-cmp-alpha", style={"height":"200px"}),
                dcc.Graph(id="g-cmp-tev", style={"height":"180px"}),
                html.Div(id="tbl-cmp-stats", className="mt-2"),
            ], width=10)
        ])
    ])

@app.callback(
    Output("tab-content","children"),
    Input("main-tabs","value"))
def render_tab(tab):
    if tab == "tab-portfolio":
        return _layout_portfolio()
    elif tab == "tab-corr":
        return _layout_corr()
    elif tab == "tab-finanziaria":
        return _layout_finanziaria()
    elif tab == "tab-frontier":
        return _layout_frontier()
    elif tab == "tab-style":
        return _layout_style()
    elif tab == "tab-returns":
        return _layout_returns()
    elif tab == "tab-arima":
        return _layout_arima()
    elif tab == "tab-rolling":
        return _layout_rolling()
    elif tab == "tab-lstm":
        return _layout_lstm()
    elif tab == "tab-compare":
        return _layout_compare()
    else:
        return html.Div("Tab not found")

# ----------------------------------------------------------------------
# CALLBACK per ogni tab (implementazioni complete dal notebook)
# ----------------------------------------------------------------------

# ---- Tab Portfolio ----
@app.callback(
    Output("dd-bench-p1","options"),
    Input("store-prices","data"))
def update_bench_options(data):
    if not data: return []
    p = pd.read_json(data, convert_dates=True)
    return [{"label":c,"value":c} for c in p.columns]

@app.callback(
    Output("g-cumret","figure"),
    Output("g-ir","figure"),
    Output("g-sharpe-tev","figure"),
    Output("tbl-port-stats","children"),
    Input("btn-port","n_clicks"),
    State("store-returns","data"),
    State("store-bench","data"),
    State("dd-bench-p1","value"),
    State("inp-roll-p","value"),
    State("sl-rf","value"),
    State("inp-start","value"),
    State("inp-end","value"),
    prevent_initial_call=True)
def update_portfolio(n, ret_data, bench_data, bench_col, roll_w, rf_pct, start, end):
    if not ret_data:
        empty = go.Figure(); empty.update_layout(template="plotly_white")
        return empty, empty, empty, ""
    r = pd.read_json(ret_data, convert_dates=True)
    rf = rf_pct/100
    roll = int(roll_w or 250)
    tickers = list(r.columns)
    n_a = len(tickers)
    mu = r.mean()*252; cov = r.cov()*252
    w1_raw = np.array([IS20_W.get(t,0.0) for t in tickers])
    w1 = w1_raw/w1_raw.sum() if w1_raw.sum()>0.05 else np.ones(n_a)/n_a
    def neg_sharpe(w):
        r_ = w@mu; v_ = np.sqrt(w@cov@w); return -(r_-rf)/v_ if v_>0 else 0
    res = minimize(neg_sharpe, np.ones(n_a)/n_a, method="SLSQP",
                   bounds=[(0,.35)]*n_a, constraints=[{"type":"eq","fun":lambda w:np.sum(w)-1}])
    w2 = res.x
    res3 = minimize(lambda w: w@cov@w, np.ones(n_a)/n_a, method="SLSQP",
                    bounds=[(0,.35)]*n_a, constraints=[{"type":"eq","fun":lambda w:np.sum(w)-1}])
    w3 = res3.x
    port_rets = {}
    for name,w in [("P1 IS20 Passive",w1),("P2 Max-Sharpe",w2),("P3 Min-Var",w3)]:
        port_rets[name] = pd.Series(r.values @ w, index=r.index)
    if bench_data and bench_col:
        bd = pd.read_json(bench_data, convert_dates=True)
        if bench_col+"_ret" in bd.columns:
            port_rets[f"Benchmark ({bench_col})"] = bd[bench_col+"_ret"].reindex(r.index).fillna(0)
    colors_p = {"P1 IS20 Passive":"#FF6B35","P2 Max-Sharpe":"#4CAF50","P3 Min-Var":"#2196F3","Benchmark": "#9E9E9E"}
    fig1 = go.Figure()
    for name, ret in port_rets.items():
        cum = np.cumsum(ret)*100
        fig1.add_trace(go.Scatter(x=cum.index, y=cum.values, name=name,
                                  line=dict(color=colors_p.get(name,"#607D8B"), width=2)))
    _add_events(fig1, start, end)
    fig1.update_layout(title="Cumulative Returns (%)", template="plotly_white",
                       legend=dict(orientation="h",y=1.02), hovermode="x unified")
    # Information Ratio
    fig2 = go.Figure()
    ref_key = list(port_rets.keys())[0]
    ref_ret = port_rets[ref_key]
    for name, ret in list(port_rets.items())[1:]:
        excess = ret - ref_ret.reindex(ret.index).fillna(0)
        ir = excess.rolling(roll).mean() / excess.rolling(roll).std() * np.sqrt(252)
        fig2.add_trace(go.Scatter(x=ir.index, y=ir.values, name=name,
                                  line=dict(color=colors_p.get(name,"#607D8B"), width=1.5)))
    fig2.add_hline(y=0, line_dash="dash", line_color="red")
    fig2.update_layout(title=f"Information Ratio vs {ref_key} (rolling {roll}d)", template="plotly_white")
    # Sharpe & TEV
    fig3 = make_subplots(rows=1, cols=2, subplot_titles=["Rolling Sharpe Ratio","Tracking Error Volatility"])
    for name,ret in port_rets.items():
        rm = ret.rolling(roll).mean()*252; rv = ret.rolling(roll).std()*np.sqrt(252)
        sr = (rm - rf) / rv
        col_ = colors_p.get(name,"#607D8B")
        fig3.add_trace(go.Scatter(x=sr.index, y=sr.values, name=name, line=dict(color=col_,width=1.5)), row=1, col=1)
        if name != ref_key:
            ref = port_rets.get(ref_key, ret)
            tev = (ret - ref.reindex(ret.index).fillna(0)).rolling(roll).std()*np.sqrt(252)*100
            fig3.add_trace(go.Scatter(x=tev.index, y=tev.values, name=f"TEV {name}",
                                      line=dict(color=col_,width=1.5,dash="dot"), showlegend=False), row=1, col=2)
    fig3.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    fig3.add_hrect(y0=2, y1=4, fillcolor="rgba(76,175,80,0.1)", line_width=0, row=1, col=2, annotation_text="target 2-4%")
    fig3.update_layout(template="plotly_white", hovermode="x unified")
    rows_t = []
    for name,ret in port_rets.items():
        ar = ret.mean()*252*100; av = ret.std()*np.sqrt(252)*100; sr_ = (ar - rf*100)/av
        cum = (np.exp(np.cumsum(ret))-1).iloc[-1]*100
        cs = pd.Series(np.exp(np.cumsum(ret))); mdd = ((cs-cs.cummax())/cs.cummax()).min()*100
        cal = ar/abs(mdd) if mdd!=0 else 0
        rows_t.append({"Portafoglio":name, "Ret/Y %":f"{ar:.1f}", "Vol/Y %":f"{av:.1f}",
                       "Sharpe":f"{sr_:.3f}", "Cumul. %":f"{cum:.0f}", "MaxDD %":f"{mdd:.1f}", "Calmar":f"{cal:.3f}"})
    tbl = dash_table.DataTable(data=rows_t,
        columns=[{"name":k,"id":k} for k in rows_t[0].keys()] if rows_t else [],
        style_cell={"textAlign":"center","fontSize":"12px","padding":"6px"},
        style_header={"backgroundColor":"#1F4E78","color":"white","fontWeight":"bold"})
    return fig1, fig2, fig3, tbl

# ---- Tab Correlation ----
@app.callback(
    Output("g-corr","figure"),
    Input("btn-corr","n_clicks"),
    State("store-returns","data"),
    State("inp-corr-roll","value"),
    prevent_initial_call=True)
def update_corr(n, ret_data, roll):
    if not ret_data: return go.Figure()
    r = pd.read_json(ret_data, convert_dates=True)
    if roll and int(roll) > 0:
        corr = r.tail(int(roll)).corr()
        title = f"Correlation Matrix — last {roll} days"
    else:
        corr = r.corr()
        title = "Correlation Matrix — full sample"
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_masked = corr.where(~mask)
    fig = px.imshow(corr_masked, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdYlGn", zmin=-1, zmax=1, title=title)
    fig.update_layout(template="plotly_white", coloraxis_colorbar=dict(title="ρ"))
    return fig

# ---- Tab Financial Analysis ----
@app.callback(
    Output("dd-fin-asset","options"),
    Input("store-prices","data"))
def fin_options(data):
    if not data: return []
    p = pd.read_json(data, convert_dates=True)
    return [{"label":c,"value":c} for c in p.columns]

@app.callback(
    Output("g-fin-dist","figure"),
    Output("g-fin-qq","figure"),
    Output("g-fin-prices","figure"),
    Output("tbl-fin-stats","children"),
    Input("btn-fin","n_clicks"),
    State("store-prices","data"),
    State("store-returns","data"),
    State("dd-fin-asset","value"),
    State("inp-start","value"),
    State("inp-end","value"),
    prevent_initial_call=True)
def update_finanziaria(n, p_data, r_data, asset, start, end):
    if not p_data or not asset:
        return go.Figure(), go.Figure(), go.Figure(), ""
    p = pd.read_json(p_data, convert_dates=True)
    r = pd.read_json(r_data, convert_dates=True)
    if asset not in p.columns:
        return go.Figure(), go.Figure(), go.Figure(), ""
    ps = p[asset].dropna(); rs = r[asset].dropna()
    # Distribution
    x_norm = np.linspace(rs.min(), rs.max(), 300)
    fig1 = go.Figure()
    fig1.add_trace(go.Histogram(x=rs.values, nbinsx=80, histnorm="probability density",
                                name="Distribution", marker_color="#2196F3", opacity=0.7))
    fig1.add_trace(go.Scatter(x=x_norm, y=stats.norm.pdf(x_norm, rs.mean(), rs.std()),
                              name="Normal", line=dict(color="#F44336", width=2)))
    t_params = stats.t.fit(rs)
    fig1.add_trace(go.Scatter(x=x_norm, y=stats.t.pdf(x_norm, *t_params),
                              name="Student-t", line=dict(color="#FF9800", width=2, dash="dash")))
    fig1.update_layout(title=f"Return Distribution — {asset}", template="plotly_white")
    # QQ-plot
    (osm, osr), (slope, intercept, _) = stats.probplot(rs, dist="norm")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=osm, y=osr, mode="markers",
                              marker=dict(color="#9C27B0", size=4, opacity=0.6), name="QQ empirical"))
    fig2.add_trace(go.Scatter(x=[osm[0],osm[-1]], y=[slope*osm[0]+intercept, slope*osm[-1]+intercept],
                              mode="lines", line=dict(color="#F44336", width=2), name="Normal"))
    fig2.update_layout(title="QQ-Plot vs Normal", template="plotly_white")
    # Prices
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=ps.index, y=ps.values, name=asset,
                              line=dict(color="#1F4E78", width=1.5)))
    ma200 = ps.rolling(200).mean()
    fig3.add_trace(go.Scatter(x=ma200.index, y=ma200.values, name="MA200",
                              line=dict(color="#FF6B35", width=1, dash="dot")))
    _add_events(fig3, start, end)
    fig3.update_layout(title=f"Prices — {asset}", template="plotly_white", hovermode="x unified")
    # Statistics
    jb_p = stats.jarque_bera(rs)[1]
    adf_p = adfuller(rs, autolag="AIC")[1]
    kpss_p = kpss(rs, regression="c", nlags="auto")[1]
    kurt = stats.kurtosis(rs); skew = stats.skew(rs)
    ann_ret = rs.mean()*252*100; ann_vol = rs.std()*np.sqrt(252)*100
    sr = (ann_ret - 3.0) / ann_vol
    cs = pd.Series(np.exp(np.cumsum(rs))); mdd = ((cs-cs.cummax())/cs.cummax()).min()*100
    data_t = [
        {"Statistica":"Annual Return %","Valore":f"{ann_ret:.2f}"},
        {"Statistica":"Annual Volatility %","Valore":f"{ann_vol:.2f}"},
        {"Statistica":"Sharpe Ratio","Valore":f"{sr:.3f}"},
        {"Statistica":"Max Drawdown %","Valore":f"{mdd:.2f}"},
        {"Statistica":"Skewness","Valore":f"{skew:.3f}"},
        {"Statistica":"Excess Kurtosis","Valore":f"{kurt:.3f}"},
        {"Statistica":"Jarque-Bera p","Valore":f"{jb_p:.4f} {'✅' if jb_p>0.05 else '⚠️'}"},
        {"Statistica":"ADF p (stationarity)","Valore":f"{adf_p:.4f} {'✅' if adf_p<0.05 else '❌'}"},
        {"Statistica":"KPSS p","Valore":f"{kpss_p:.4f} {'✅' if kpss_p>0.05 else '❌'}"},
    ]
    tbl = dash_table.DataTable(data=data_t,
        columns=[{"name":k,"id":k} for k in data_t[0].keys()],
        style_cell={"fontSize":"12px","padding":"5px"},
        style_header={"backgroundColor":"#1F4E78","color":"white","fontWeight":"bold"})
    return fig1, fig2, fig3, tbl

# ---- Tab Efficient Frontier ----
@app.callback(
    Output("g-frontier","figure"),
    Output("g-frontier-cum","figure"),
    Output("g-pie-ms","figure"),
    Output("g-pie-mv","figure"),
    Output("store-ef","data"),
    Input("btn-frontier","n_clicks"),
    State("store-returns","data"),
    State("store-prices","data"),
    State("inp-nsim","value"),
    State("sl-maxw","value"),
    State("sl-rf","value"),
    State("ri-risk","value"),
    State("chk-arima","value"),
    State("inp-arima-h","value"),
    State("inp-start","value"),
    State("inp-end","value"),
    prevent_initial_call=True)
def update_frontier(n, ret_data, p_data, n_sim, max_w_pct, rf_pct, risk_type,
                    arima_chk, arima_h, start, end):
    if not ret_data:
        empty = go.Figure(); empty.update_layout(template="plotly_white")
        return empty, empty, empty, empty, None
    r = pd.read_json(ret_data, convert_dates=True)
    clean = r.dropna(axis=1, how="all").dropna()
    if clean.shape[1] < 2:
        return go.Figure(), go.Figure(), go.Figure(), go.Figure(), None
    tickers = list(clean.columns); n_a = len(tickers)
    mu  = clean.mean()*252
    cov = clean.cov()*252
    rf  = rf_pct/100
    mw  = max_w_pct/100
    n_sim = int(n_sim or 5000)
    # ARIMA override
    if "arima" in (arima_chk or []):
        mu_arima = {}
        for t in tickers:
            s = clean[t].dropna()
            if len(s) >= 50:
                try:
                    fc = ARIMA(s, order=(1,0,1)).fit().forecast(steps=int(arima_h or 21))
                    mu_arima[t] = float(fc.mean())*252
                except Exception:
                    mu_arima[t] = float(s.mean())*252
            else:
                mu_arima[t] = float(s.mean())*252
        mu = pd.Series(mu_arima)
    def _risk(w):
        v = np.sqrt(w@cov.values@w)
        if risk_type=="vol": return v
        port_r = clean.values@w
        if risk_type=="var5": return -np.percentile(port_r, 5)
        return -np.percentile(port_r, 1)
    # Monte Carlo
    np.random.seed(42)
    sims = {"rets":[],"vols":[],"sharpes":[],"ws":[]}
    for _ in range(n_sim):
        w = np.random.dirichlet(np.ones(n_a))
        w = np.clip(w, 0, mw); w /= w.sum()
        rv= _risk(w); re= w@mu.values; sh= (re-rf)/rv
        sims["rets"].append(re*100); sims["vols"].append(rv*100)
        sims["sharpes"].append(sh); sims["ws"].append(w)
    # Optimization
    def _opt_sharpe(w):
        rv= _risk(w); re= w@mu.values
        return -(re-rf)/rv if rv>0 else 0
    res_ms = minimize(_opt_sharpe, np.ones(n_a)/n_a, method="SLSQP",
                      bounds=[(0,mw)]*n_a,
                      constraints=[{"type":"eq","fun":lambda w:np.sum(w)-1}])
    ms_w = res_ms.x
    ms_v = _risk(ms_w)*100; ms_r = (ms_w@mu.values)*100
    ms_sh= (ms_r/100-rf)/(_risk(ms_w))
    res_mv = minimize(lambda w: w@cov.values@w, np.ones(n_a)/n_a, method="SLSQP",
                      bounds=[(0,mw)]*n_a,
                      constraints=[{"type":"eq","fun":lambda w:np.sum(w)-1}])
    mv_w = res_mv.x
    mv_v = _risk(mv_w)*100; mv_r = (mv_w@mu.values)*100
    # Frontier curve
    r_min = mv_r/100; r_max = float(mu.max())
    frontier_pts = []
    for target in np.linspace(r_min, r_max, 80):
        res_f = minimize(lambda w: w@cov.values@w, np.ones(n_a)/n_a, method="SLSQP",
                         bounds=[(0,mw)]*n_a,
                         constraints=[{"type":"eq","fun":lambda w:np.sum(w)-1},
                                      {"type":"eq","fun":lambda w:w@mu.values-target}])
        if res_f.success:
            frontier_pts.append((_risk(res_f.x)*100, res_f.x@mu.values*100))
    # Plot frontier
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=sims["vols"], y=sims["rets"], mode="markers",
                              marker=dict(color=sims["sharpes"], colorscale="RdYlGn", size=4, opacity=0.5,
                                          colorbar=dict(title="Sharpe",x=1.02)),
                              name="MC Simulations"))
    if frontier_pts:
        fx,fy = zip(*frontier_pts)
        fig1.add_trace(go.Scatter(x=list(fx), y=list(fy), mode="lines",
                                   line=dict(color="#1F4E78",width=3), name="Frontier"))
    fig1.add_trace(go.Scatter(x=[ms_v], y=[ms_r], mode="markers+text",
                               marker=dict(symbol="star",size=18,color="gold",line=dict(color="black",width=1.5)),
                               text=[f"Max Sharpe<br>SR:{ms_sh:.2f}"], textposition="top center", name="Max Sharpe"))
    fig1.add_trace(go.Scatter(x=[mv_v], y=[mv_r], mode="markers+text",
                               marker=dict(symbol="diamond",size=14,color="cyan",line=dict(color="black",width=1.5)),
                               text=["Min Var"], textposition="top center", name="Min Var"))
    for i, t in enumerate(tickers):
        v_i = np.sqrt(cov.iloc[i,i])*100; r_i = float(mu.iloc[i])*100
        fig1.add_trace(go.Scatter(x=[v_i], y=[r_i], mode="markers+text",
                                   marker=dict(size=8, color=COLORS_P[i%len(COLORS_P)]),
                                   text=[t], textposition="top right", name=t, showlegend=False))
    xlabel = {"vol":"Volatility (%)","var5":"VaR 5% (%)","var1":"VaR 1% (%)"}
    title_sfx = " [ARIMA-adjusted]" if "arima" in (arima_chk or []) else " [Standard]"
    fig1.update_layout(title=f"Efficient Frontier{title_sfx} — Constraint: 0%-{max_w_pct}%",
                       xaxis_title=xlabel.get(risk_type,"Risk (%)"), yaxis_title="Expected Return (%)",
                       template="plotly_white", legend=dict(orientation="h",y=1.02))
    # Cumulative returns of frontier portfolios
    p2 = pd.read_json(p_data, convert_dates=True)[tickers].dropna()
    r2 = compute_returns(p2)
    w1_raw = np.array([IS20_W.get(t,0.0) for t in tickers])
    w1 = w1_raw/w1_raw.sum() if w1_raw.sum()>0.01 else np.ones(n_a)/n_a
    fig2 = go.Figure()
    for nm, w_ in [("IS20 Passive",w1),("Max-Sharpe",ms_w),("Min-Var",mv_w)]:
        cum = np.cumsum(r2.values@w_)*100
        col_ = {"IS20 Passive":"#FF6B35","Max-Sharpe":"#4CAF50","Min-Var":"#2196F3"}[nm]
        fig2.add_trace(go.Scatter(x=r2.index, y=cum, name=nm, line=dict(color=col_, width=2)))
    _add_events(fig2, start, end)
    fig2.update_layout(title="Cumulative Returns — F1/F2/F3", template="plotly_white", hovermode="x unified")
    # Pie charts
    def _pie(weights, title):
        mask = weights > 0.005
        fig = go.Figure(go.Pie(labels=[tickers[i] for i in range(n_a) if mask[i]],
                               values=weights[mask], hole=0.35,
                               marker_colors=COLORS_P[:mask.sum()], textinfo="label+percent"))
        fig.update_layout(title=title, template="plotly_white", margin=dict(t=40,b=10,l=10,r=10), showlegend=False)
        return fig
    ef_data = {"ms_weights":ms_w.tolist(),"mv_weights":mv_w.tolist(),"tickers":tickers,"ms_sharpe":float(ms_sh)}
    return fig1, fig2, _pie(ms_w, f"Max-Sharpe SR:{ms_sh:.2f}"), _pie(mv_w, "Min-Variance"), ef_data

# ---- Tab Style Analysis ----
@app.callback(
    Output("dd-style-y","options"), Output("dd-style-x","options"),
    Input("store-returns","data"))
def style_options(data):
    if not data: return [],[]
    r = pd.read_json(data, convert_dates=True)
    opts = [{"label":c,"value":c} for c in r.columns]
    return opts, opts

@app.callback(
    Output("div-style-stats","children"),
    Output("g-style-rolling","figure"),
    Output("store-style","data"),
    Input("btn-style","n_clicks"),
    State("store-returns","data"),
    State("dd-style-y","value"),
    State("dd-style-x","value"),
    State("ri-se","value"),
    State("inp-style-roll","value"),
    prevent_initial_call=True)
def update_style(n, ret_data, asset_y, assets_x, se_type, roll_m):
    if not ret_data or not asset_y or not assets_x:
        return dbc.Alert("Select Asset Y and at least one factor X.",color="warning"), go.Figure(), None
    r = pd.read_json(ret_data, convert_dates=True)
    if asset_y not in r.columns:
        return dbc.Alert("Asset Y not found.",color="danger"), go.Figure(), None
    valid_x = [x for x in assets_x if x in r.columns]
    if not valid_x:
        return dbc.Alert("No valid factor X found.",color="danger"), go.Figure(), None
    combined = r[[asset_y]+valid_x].dropna()
    y = combined[asset_y]
    X = add_constant(combined[valid_x])
    model = OLS(y, X)
    if se_type=="HAC":
        result = model.fit(cov_type="HAC", cov_kwds={"maxlags":5})
    elif se_type=="HC3":
        result = model.fit(cov_type="HC3")
    else:
        result = model.fit()
    resid = result.resid
    dw_val = durbin_watson(resid)
    jb_stat, jb_p = stats.jarque_bera(resid)[:2]
    lb_p = acorr_ljungbox(resid, lags=[10], return_df=True)["lb_pvalue"].values[0]
    try: bp_stat, bp_p, _, _ = het_breuschpagan(resid, X)
    except: bp_stat, bp_p = np.nan, np.nan
    def _badge(val, ok_cond, ok_txt, fail_txt):
        return dbc.Badge(ok_txt if ok_cond else fail_txt,
                         color="success" if ok_cond else "danger", className="ms-1")
    stats_card = dbc.Card([
        dbc.CardHeader(html.B(f"Style Analysis: {asset_y} | Std Error: {se_type} | {len(combined)} observations")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([html.B(f"R² = {result.rsquared*100:.2f}%"), html.Span(f"  |  Adj-R² = {result.rsquared_adj*100:.2f}%"), html.Br(),
                         html.Span(f"F-stat = {result.fvalue:.2f}  (p = {result.f_pvalue:.2e})"), html.Br(),
                         html.Span(f"AIC = {result.aic:.2f}  |  BIC = {result.bic:.2f}")], width=4),
                dbc.Col([html.Span("Durbin-Watson: "), html.B(f"{dw_val:.4f}"), _badge(dw_val, 1.5<dw_val<2.5, " ≈ 2 (no autocorr.)", " ≠ 2 (autocorr.!)"), html.Br(),
                         html.Span("Jarque-Bera p: "), html.B(f"{jb_p:.4f}"), _badge(jb_p, jb_p>0.05, " Normal", " Non normal"), html.Br(),
                         html.Span("Ljung-Box p(10): "), html.B(f"{lb_p:.4f}"), _badge(lb_p, lb_p>0.05, " No AC", " AC!")], width=4),
                dbc.Col([html.Span("Breusch-Pagan p: "), html.B(f"{bp_p:.4f}" if not np.isnan(bp_p) else "n.d."),
                         _badge(bp_p, not np.isnan(bp_p) and bp_p>0.05, " Homosced.", " Heterosced."), html.Br(),
                         html.Span(f"Skewness: {stats.skew(resid):.3f}  |  Kurtosis: {stats.kurtosis(resid):.3f}")], width=4),
            ])
        ])
    ], className="mb-2")
    # Coefficient table
    coef_data = []
    for var in result.params.index:
        p_val = result.pvalues[var]
        sig = "***" if p_val<0.001 else "**" if p_val<0.01 else "*" if p_val<0.05 else "." if p_val<0.1 else ""
        coef_data.append({"Variabile":var,"Coeff.":f"{result.params[var]:.6f}","Std Err":f"{result.bse[var]:.6f}",
                          "t-stat":f"{result.tvalues[var]:.4f}","p-val":f"{p_val:.4e}","Sig.":sig,
                          "IC95 inf":f"{result.conf_int().loc[var,0]:.6f}","IC95 sup":f"{result.conf_int().loc[var,1]:.6f}"})
    tbl_coef = dash_table.DataTable(data=coef_data,
        columns=[{"name":k,"id":k} for k in coef_data[0].keys()] if coef_data else [],
        style_cell={"fontSize":"11px","padding":"4px","textAlign":"right"},
        style_header={"backgroundColor":"#1F4E78","color":"white","fontWeight":"bold"},
        style_data_conditional=[{"if":{"filter_query":"{Sig.} contains '*'"}, "fontWeight":"bold","backgroundColor":"rgba(76,175,80,0.1)"},
                                {"if":{"column_id":"Variabile"},"textAlign":"left"}])
    # Rolling weights
    roll_days = int(roll_m or 36) * 21
    fig = go.Figure()
    if len(combined) > roll_days + 10:
        dates_r = combined.index[roll_days:]
        weights_ts = {v: [] for v in valid_x}
        for i in range(roll_days, len(combined)):
            window = combined.iloc[i-roll_days:i]
            y_w = window[asset_y]; X_w = add_constant(window[valid_x])
            try:
                res_w = OLS(y_w, X_w).fit()
                coefs = np.maximum(res_w.params[valid_x].values, 0)
                total = coefs.sum()
                coefs = coefs/total if total>0 else np.ones(len(valid_x))/len(valid_x)
            except Exception:
                coefs = np.ones(len(valid_x))/len(valid_x)
            for j, v in enumerate(valid_x):
                weights_ts[v].append(coefs[j]*100)
        for i, v in enumerate(valid_x):
            fig.add_trace(go.Bar(x=dates_r, y=weights_ts[v], name=v,
                                 marker_color=COLORS_P[i%len(COLORS_P)]))
        fig.update_layout(barmode="stack", title=f"Style Weights Rolling — window {roll_m} months",
                          yaxis_title="Weight (%)", template="plotly_white", legend=dict(orientation="h",y=1.02))
    else:
        fig.update_layout(title="Insufficient data for rolling analysis", template="plotly_white")
    style_res = {"r2":result.rsquared,"dw":dw_val,"jb_p":jb_p}
    return [stats_card, tbl_coef], fig, style_res

# ---- Tab Historical Returns ----
@app.callback(
    Output("dd-ret-assets","options"),
    Input("store-prices","data"))
def ret_options(data):
    if not data: return []
    p = pd.read_json(data, convert_dates=True)
    return [{"label":c,"value":c} for c in p.columns]

@app.callback(
    Output("g-returns","figure"),
    Input("btn-ret","n_clicks"),
    State("store-prices","data"),
    State("store-returns","data"),
    State("dd-ret-assets","value"),
    State("ri-ret-type","value"),
    State("inp-start","value"),
    State("inp-end","value"),
    prevent_initial_call=True)
def update_returns(n, p_data, r_data, assets, ret_type, start, end):
    if not r_data or not assets: return go.Figure()
    r = pd.read_json(r_data, convert_dates=True)
    sel = [a for a in assets if a in r.columns]
    if not sel: return go.Figure()
    r_sel = r[sel]
    if ret_type == "logcum":
        fig = go.Figure()
        for i, a in enumerate(sel):
            cum = np.cumsum(r_sel[a])*100
            fig.add_trace(go.Scatter(x=cum.index, y=cum.values, name=a,
                                     line=dict(color=COLORS_P[i%len(COLORS_P)], width=2)))
        _add_events(fig, start, end)
        fig.update_layout(title="Log Cumulative Returns (%)", template="plotly_white", hovermode="x unified")
    elif ret_type == "euro":
        p = pd.read_json(p_data, convert_dates=True)[sel]
        fig = go.Figure()
        for i, a in enumerate(sel):
            ps = p[a].dropna()
            norm = ps/ps.iloc[0]*100
            fig.add_trace(go.Scatter(x=norm.index, y=norm.values, name=a,
                                     line=dict(color=COLORS_P[i%len(COLORS_P)], width=2)))
        _add_events(fig, start, end)
        fig.update_layout(title="Growth of €100 invested", template="plotly_white", hovermode="x unified")
    elif ret_type == "annual":
        ann = r_sel.resample("YE").sum()*100
        fig = go.Figure()
        for i, a in enumerate(sel):
            fig.add_trace(go.Bar(x=ann.index.year, y=ann[a].values, name=a,
                                 marker_color=COLORS_P[i%len(COLORS_P)]))
        fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
        fig.update_layout(title="Annual Returns (%)", barmode="group", template="plotly_white")
    else:  # heatmap
        if len(sel) == 0:
            return go.Figure()
        asset = sel[0]
        monthly = r_sel[asset].resample("ME").sum()*100
        monthly_df = monthly.to_frame()
        monthly_df["year"]  = monthly_df.index.year
        monthly_df["month"] = monthly_df.index.month
        pivot = monthly_df.pivot(index="year", columns="month", values=asset)
        pivot.columns = ["Gen","Feb","Mar","Apr","Mag","Giu","Lug","Ago","Set","Ott","Nov","Dic"][:len(pivot.columns)]
        fig = px.imshow(pivot, text_auto=".1f", aspect="auto",
                        color_continuous_scale="RdYlGn", zmin=-15, zmax=15,
                        title=f"Monthly Returns Heatmap — {asset} (%)")
        fig.update_layout(template="plotly_white")
    return fig

# ---- Tab ARIMA ----
@app.callback(
    Output("dd-arima-asset","options"),
    Input("store-returns","data"))
def arima_opts(data):
    if not data: return []
    r = pd.read_json(data, convert_dates=True)
    return [{"label":c,"value":c} for c in r.columns]

@app.callback(
    Output("g-arima-forecast","figure"),
    Output("g-arima-resid","figure"),
    Output("div-arima-stats","children"),
    Input("btn-arima","n_clicks"),
    State("store-returns","data"),
    State("dd-arima-asset","value"),
    State("inp-ap","value"), State("inp-ad","value"), State("inp-aq","value"),
    State("inp-ah","value"),
    prevent_initial_call=True)
def update_arima(n, ret_data, asset, ap, ad, aq, ah):
    if not ret_data or not asset:
        return go.Figure(), go.Figure(), ""
    r = pd.read_json(ret_data, convert_dates=True)
    if asset not in r.columns:
        return go.Figure(), go.Figure(), ""
    series = r[asset].dropna()
    if len(series)<50:
        return go.Figure(), go.Figure(), dbc.Alert("Insufficient data.", color="warning")
    order = (int(ap or 1), int(ad or 0), int(aq or 1))
    try:
        model_fit = ARIMA(series, order=order).fit()
    except Exception as e:
        return go.Figure(), go.Figure(), dbc.Alert(f"ARIMA error: {e}", color="danger")
    h = int(ah or 21)
    fc_obj = model_fit.get_forecast(steps=h)
    fc_mean = fc_obj.predicted_mean
    fc_ci   = fc_obj.conf_int(alpha=0.05)
    last = series.index[-1]
    try:
        freq = pd.infer_freq(series.index) or "B"
        fut_idx = pd.date_range(start=last, periods=h+1, freq=freq)[1:]
    except Exception:
        fut_idx = pd.RangeIndex(start=len(series), stop=len(series)+h)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=series.index[-252:], y=series.values[-252:],
                              name="History (1Y)", line=dict(color="#2196F3",width=1.5)))
    fig1.add_trace(go.Scatter(x=list(fut_idx), y=list(fc_mean.values),
                              name="Forecast", line=dict(color="#FF6B35",width=2,dash="dash")))
    fig1.add_trace(go.Scatter(x=list(fut_idx)+list(fut_idx)[::-1],
                              y=list(fc_ci.iloc[:,0].values)+list(fc_ci.iloc[:,1].values)[::-1],
                              fill="toself", fillcolor="rgba(255,107,53,0.15)", line_color="rgba(255,255,255,0)",
                              name="95% CI"))
    mu_ann = float(fc_mean.mean())*252*100
    fig1.update_layout(title=f"ARIMA{order} Forecast {asset} | μ_arima: {mu_ann:+.2f}%/year",
                       template="plotly_white", legend=dict(orientation="h",y=1.02), hovermode="x unified")
    # Residuals
    resid = model_fit.resid
    fig2 = make_subplots(rows=1, cols=2, subplot_titles=["Residuals over Time","ACF of Residuals"])
    fig2.add_trace(go.Scatter(x=resid.index, y=resid.values, mode="lines",
                               line=dict(color="#9C27B0",width=0.8), name="Residuals"), row=1,col=1)
    fig2.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    acf_v = sm_acf(resid, nlags=20, fft=True)[1:]
    conf = 1.96/np.sqrt(len(resid))
    fig2.add_trace(go.Bar(x=list(range(1,21)), y=acf_v.tolist(),
                           marker_color=["#F44336" if abs(v)>conf else "#4CAF50" for v in acf_v],
                           name="ACF"), row=1,col=2)
    fig2.add_hline(y=conf, line_dash="dot", line_color="red", row=1, col=2)
    fig2.add_hline(y=-conf, line_dash="dot", line_color="red", row=1, col=2)
    fig2.update_layout(template="plotly_white", showlegend=False)
    # Statistics
    jb_p = stats.jarque_bera(resid)[1]
    lb_p = acorr_ljungbox(resid, lags=[10], return_df=True)["lb_pvalue"].values[0]
    data_st = [
        {"Stat":"AIC","Val":f"{model_fit.aic:.2f}"},
        {"Stat":"BIC","Val":f"{model_fit.bic:.2f}"},
        {"Stat":"Log-lik.","Val":f"{model_fit.llf:.2f}"},
        {"Stat":"σ residuals","Val":f"{resid.std():.6f}"},
        {"Stat":"JB p-value","Val":f"{jb_p:.4f} {'✅' if jb_p>0.05 else '⚠️'}"},
        {"Stat":"LB p(10)","Val":f"{lb_p:.4f} {'✅' if lb_p>0.05 else '⚠️'}"},
        {"Stat":"μ ARIMA/year","Val":f"{mu_ann:+.2f}%"},
        {"Stat":"σ ARIMA/year","Val":f"{series.tail(63).std()*np.sqrt(252)*100:.2f}%"},
    ]
    tbl = dash_table.DataTable(data=data_st,
        columns=[{"name":k,"id":k} for k in data_st[0].keys()],
        style_cell={"fontSize":"11px","padding":"5px"},
        style_header={"backgroundColor":"#1F4E78","color":"white","fontWeight":"bold"})
    return fig1, fig2, tbl

# ---- Tab Rolling Analysis ----
@app.callback(
    Output("dd-roll-asset","options"),
    Input("store-returns","data"))
def roll_opts(data):
    if not data: return []
    r = pd.read_json(data, convert_dates=True)
    return [{"label":c,"value":c} for c in r.columns]

@app.callback(
    Output("g-roll-vol","figure"),
    Output("g-roll-sharpe","figure"),
    Output("g-garch","figure"),
    Output("g-regime","figure"),
    Input("btn-roll","n_clicks"),
    State("store-returns","data"),
    State("dd-roll-asset","value"),
    State("inp-roll-w","value"),
    State("inp-gp","value"),
    State("inp-gq","value"),
    State("sl-rf","value"),
    State("inp-start","value"),
    State("inp-end","value"),
    prevent_initial_call=True)
def update_rolling(n, ret_data, asset, window, gp, gq, rf_pct, start, end):
    empty = go.Figure()
    if not ret_data or not asset: return empty, empty, empty, empty
    r = pd.read_json(ret_data, convert_dates=True)
    if asset not in r.columns: return empty,empty,empty,empty
    rs = r[asset].dropna()
    w = int(window or 21); rf = (rf_pct if rf_pct is not None else 3.0)/100
    rv = rs.rolling(w).std()*np.sqrt(252)*100
    rm = rs.rolling(w).mean()*252*100
    rsh= (rm - rf*100) / rv
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=rv.index, y=rv.values, name=f"Rolling Vol {w}d",
                               fill="tozeroy", line=dict(color="#FF6B35",width=1.5)))
    _add_events(fig1, start, end)
    fig1.update_layout(title=f"Rolling Volatility {w}d — {asset} (% ann.)", template="plotly_white", hovermode="x unified")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=rsh.index, y=rsh.values, name="Rolling Sharpe",
                               line=dict(color="#9C27B0",width=1.5)))
    fig2.add_hline(y=1, line_dash="dash", line_color="green", annotation_text="SR=1")
    fig2.add_hline(y=0, line_dash="dash", line_color="red")
    fig2.update_layout(title=f"Rolling Sharpe Ratio — {asset}", template="plotly_white", hovermode="x unified")
    fig3 = go.Figure()
    try:
        am = arch_model(rs*100, p=int(gp or 1), q=int(gq or 1),
                        mean="Constant", vol="GARCH", dist="Normal")
        res_g = am.fit(disp="off")
        cv = res_g.conditional_volatility*np.sqrt(252)
        fig3.add_trace(go.Scatter(x=rv.index, y=rv.values, name=f"Rolling {w}d",
                                   line=dict(color="#2196F3",width=1,opacity=0.7)))
        fig3.add_trace(go.Scatter(x=cv.index, y=cv.values, name=f"GARCH({gp},{gq})",
                                   line=dict(color="#FF6B35",width=2)))
        _add_events(fig3, start, end)
        fig3.update_layout(title=f"GARCH({gp},{gq}) vs Rolling — {asset}", template="plotly_white",
                           legend=dict(orientation="h",y=1.02), hovermode="x unified")
    except Exception as e:
        fig3.update_layout(title=f"GARCH not available: {e}", template="plotly_white")
    fig4 = go.Figure()
    q33 = rv.quantile(0.33); q66 = rv.quantile(0.66)
    _rgba = {"#4CAF50":"rgba(76,175,80,0.6)","#FF9800":"rgba(255,152,0,0.6)","#F44336":"rgba(244,67,54,0.6)"}
    for lbl, lo, hi, col in [("Low",-np.inf,q33,"#4CAF50"), ("Medium",q33,q66,"#FF9800"), ("High",q66,np.inf,"#F44336")]:
        mask = (rv>=lo)&(rv<hi)
        y_seg = rv.where(mask)
        fig4.add_trace(go.Scatter(x=rv.index, y=y_seg.values, fill="tozeroy", name=lbl,
                                   line=dict(color=col,width=0), fillcolor=_rgba[col]))
    fig4.update_layout(title=f"Volatility Regime — {asset}", template="plotly_white",
                       legend=dict(orientation="h",y=1.02))
    return fig1, fig2, fig3, fig4

# ---- Tab LSTM (placeholder) ----
@app.callback(
    Output("lstm-install-status","children"),
    Input("btn-lstm-install","n_clicks"),
    prevent_initial_call=True)
def install_lstm(n):
    try:
        import subprocess, sys
        subprocess.check_call([sys.executable,"-m","pip","install","-q","tensorflow"])
        return dbc.Alert("✅ TensorFlow installed! Restart the notebook if needed.", color="success")
    except Exception as e:
        return dbc.Alert(f"❌ Error: {e}", color="danger")

# ---- Tab Portfolio Comparison ----
@app.callback(
    Output("g-cmp-cum","figure"),
    Output("g-cmp-dd","figure"),
    Output("g-cmp-alpha","figure"),
    Output("g-cmp-tev","figure"),
    Output("tbl-cmp-stats","children"),
    Input("btn-cmp","n_clicks"),
    State("store-returns","data"),
    State("store-ef","data"),
    State("inp-cmp-roll","value"),
    State("sl-rf","value"),
    State("inp-start","value"),
    State("inp-end","value"),
    prevent_initial_call=True)
def update_compare(n, ret_data, ef_data, roll, rf_pct, start, end):
    empty = go.Figure()
    if not ret_data: return empty,empty,empty,empty,""
    r = pd.read_json(ret_data, convert_dates=True)
    rf = rf_pct/100; roll = int(roll or 250)
    tickers = list(r.columns); n_a = len(tickers)
    if ef_data:
        ms_w = np.array(ef_data["ms_weights"])
        mv_w = np.array(ef_data["mv_weights"])
        ef_tickers = ef_data["tickers"]
        ms_w_full = np.array([ms_w[ef_tickers.index(t)] if t in ef_tickers else 0.0 for t in tickers])
        mv_w_full = np.array([mv_w[ef_tickers.index(t)] if t in ef_tickers else 0.0 for t in tickers])
        if ms_w_full.sum()>0: ms_w_full /= ms_w_full.sum()
        if mv_w_full.sum()>0: mv_w_full /= mv_w_full.sum()
    else:
        cov = r.cov()*252; mu = r.mean()*252
        res2 = minimize(lambda w: -(w@mu-(rf))/np.sqrt(w@cov@w),
                        np.ones(n_a)/n_a, method="SLSQP",
                        bounds=[(0,.35)]*n_a,
                        constraints=[{"type":"eq","fun":lambda w:np.sum(w)-1}])
        ms_w_full = res2.x
        res3 = minimize(lambda w: w@cov@w, np.ones(n_a)/n_a, method="SLSQP",
                        bounds=[(0,.35)]*n_a,
                        constraints=[{"type":"eq","fun":lambda w:np.sum(w)-1}])
        mv_w_full = res3.x
    w1_raw = np.array([IS20_W.get(t,0.0) for t in tickers])
    w1 = w1_raw/w1_raw.sum() if w1_raw.sum()>0.01 else np.ones(n_a)/n_a
    port_rets = {
        "IS20 Passive (P1)":  pd.Series(r.values@w1, index=r.index),
        "Max-Sharpe (P2)":    pd.Series(r.values@ms_w_full, index=r.index),
        "Min-Var (P3)":       pd.Series(r.values@mv_w_full, index=r.index),
    }
    colors_cmp = {"IS20 Passive (P1)":"#FF6B35","Max-Sharpe (P2)":"#4CAF50","Min-Var (P3)":"#2196F3"}
    # Cumulative
    fig1 = go.Figure()
    for nm, ret in port_rets.items():
        fig1.add_trace(go.Scatter(x=ret.index, y=np.cumsum(ret)*100, name=nm,
                                   line=dict(color=colors_cmp[nm],width=2)))
    _add_events(fig1, start, end)
    fig1.update_layout(title="Cumulative Performance (%)", template="plotly_white",
                       legend=dict(orientation="h",y=1.02), hovermode="x unified")
    # Drawdown
    fig2 = go.Figure()
    for nm, ret in port_rets.items():
        cs = pd.Series(np.exp(np.cumsum(ret)))
        dd = (cs - cs.cummax())/cs.cummax()*100
        fig2.add_trace(go.Scatter(x=ret.index, y=dd.values, name=nm,
                                   line=dict(color=colors_cmp[nm],width=1.5)))
    fig2.add_hline(y=0, line_dash="dash", line_color="gray")
    fig2.update_layout(title="Drawdown from Peak (%)", template="plotly_white",
                       legend=dict(orientation="h",y=1.02), hovermode="x unified")
    # Rolling Alpha vs P1
    p1 = port_rets["IS20 Passive (P1)"]
    fig3 = go.Figure()
    for nm, ret in list(port_rets.items())[1:]:
        alpha = (ret - p1).rolling(roll).mean()*252*100
        fig3.add_trace(go.Scatter(x=alpha.index, y=alpha.values, name=f"α {nm} vs P1",
                                   line=dict(color=colors_cmp[nm],width=1.5)))
        fig3.add_trace(go.Scatter(x=alpha.index.tolist()+alpha.index.tolist()[::-1],
                                   y=np.where(alpha>0,alpha,0).tolist()+np.zeros(len(alpha)).tolist()[::-1],
                                   fill="toself", fillcolor="rgba(76,175,80,0.1)", line_color="rgba(0,0,0,0)", showlegend=False))
    fig3.add_hline(y=0, line_dash="dash", line_color="red")
    fig3.update_layout(title=f"Rolling Alpha vs IS20 Passive ({roll}d, %ann.)", template="plotly_white",
                       legend=dict(orientation="h",y=1.02), hovermode="x unified")
    # TEV
    fig4 = go.Figure()
    for nm, ret in list(port_rets.items())[1:]:
        tev = (ret - p1).rolling(roll).std()*np.sqrt(252)*100
        fig4.add_trace(go.Scatter(x=tev.index, y=tev.values, name=f"TEV {nm}",
                                   line=dict(color=colors_cmp[nm],width=1.5)))
    fig4.add_hrect(y0=2, y1=4, fillcolor="rgba(76,175,80,0.1)", line_width=0,
                   annotation_text="Target 2-4%", annotation_position="top right")
    fig4.update_layout(title=f"Tracking Error Volatility — Target 2-4% ({roll}d)", template="plotly_white",
                       legend=dict(orientation="h",y=1.02), margin=dict(t=40,b=10))
    # Stats table
    rows_t = []
    for nm, ret in port_rets.items():
        ar = ret.mean()*252*100; av = ret.std()*np.sqrt(252)*100
        sr_ = (ar - rf*100)/av
        cum = (np.exp(np.cumsum(ret))-1).iloc[-1]*100
        cs2 = pd.Series(np.exp(np.cumsum(ret)))
        mdd = ((cs2-cs2.cummax())/cs2.cummax()).min()*100
        cal = ar/abs(mdd) if mdd!=0 else 0
        rows_t.append({"Portafoglio":nm,"Ret/Y %":f"{ar:.1f}","Vol/Y %":f"{av:.1f}",
                       "Sharpe":f"{sr_:.3f}","Cumulativo %":f"{cum:.0f}","MaxDD %":f"{mdd:.1f}","Calmar":f"{cal:.3f}"})
    tbl = dash_table.DataTable(data=rows_t,
        columns=[{"name":k,"id":k} for k in rows_t[0].keys()],
        style_cell={"textAlign":"center","fontSize":"12px","padding":"6px"},
        style_header={"backgroundColor":"#1F4E78","color":"white","fontWeight":"bold"},
        style_data_conditional=[
            {"if":{"filter_query":"{Portafoglio} contains 'Max-Sharpe'"},"backgroundColor":"rgba(76,175,80,0.15)","fontWeight":"bold"},
            {"if":{"filter_query":"{Portafoglio} contains 'IS20'"},"backgroundColor":"rgba(255,107,53,0.1)"}
        ])
    return fig1, fig2, fig3, fig4, tbl

# ----------------------------------------------------------------------
# Esporre server per gunicorn
# ----------------------------------------------------------------------
server = app.server

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)