# app.py - FIXED VERSION
# Quantitative Finance Dashboard – Full Dash Application
# Bugfix per Efficient Frontier, Store Management, e Callback Error Handling

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

import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table, callback
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
    try:
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
    except Exception as e:
        print(f"⚠️ Event overlay error: {e}")
    return fig

# ============================================================================
# DEFAULT CONFIG
# ============================================================================

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

# ============================================================================
# DASH APP SETUP
# ============================================================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
app.title = "Quantitative Finance Dashboard"

# ---- Sidebar ----
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

# ---- Tab Definitions ----
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

# ---- Main Layout ----
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
    dcc.Store(id="store-style"),
], fluid=True, style={"backgroundColor":"#f4f6f9","minHeight":"100vh","padding":"10px"})

# ============================================================================
# CALLBACK 0 – Data Loading (FIXED: better error handling)
# ============================================================================
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
        try:
            pb = load_prices([bench], start, end)
            rb = compute_returns(pb)
            bench_data = pd.concat([pb, rb.rename(columns={bench:bench+"_ret"})], axis=1).to_json(date_format="iso")
        except:
            bench_data = None
        valid = list(p.columns)
        msg = f"✅ {len(p)} days · {len(valid)} assets: {', '.join(valid[:6])}{'...' if len(valid)>6 else ''}"
        return (p.to_json(date_format="iso"),
                r.to_json(date_format="iso"),
                bench_data,
                msg)
    except Exception as e:
        return None, None, None, f"❌ {str(e)[:100]}"

# ============================================================================
# TAB ROUTING
# ============================================================================

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

# Simplified versions for other tabs (full versions in production)
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
    return html.Div("Financial Analysis tab - Full implementation pending")

def _layout_style():
    return html.Div("Style Analysis tab - Full implementation pending")

def _layout_returns():
    return html.Div("Historical Returns tab - Full implementation pending")

def _layout_arima():
    return html.Div("ARIMA Analysis tab - Full implementation pending")

def _layout_rolling():
    return html.Div("Rolling Analysis tab - Full implementation pending")

def _layout_lstm():
    return dbc.Card([
        dbc.CardBody([
            html.H4("🤖 LSTM Forecast", className="text-primary"),
            dbc.Alert("Coming soon - Requires TensorFlow", color="info"),
        ])
    ])

def _layout_compare():
    return html.Div("Portfolio Comparison tab - Full implementation pending")

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

# ============================================================================
# EFFICIENT FRONTIER CALLBACK (MAIN FIX)
# ============================================================================
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
    """FIXED: Proper error handling and covariance matrix operations"""
    
    empty = go.Figure()
    empty.update_layout(template="plotly_white")
    
    try:
        # ===== DATA VALIDATION =====
        if not ret_data or not p_data:
            return empty, empty, empty, empty, None
        
        r = pd.read_json(ret_data, convert_dates=True)
        p = pd.read_json(p_data, convert_dates=True)
        
        # Clean data
        clean = r.dropna(axis=1, how="all").dropna()
        if clean.shape[1] < 2:
            return empty, empty, empty, empty, None
        
        tickers = list(clean.columns)
        n_a = len(tickers)
        
        # ===== PARAMETERS =====
        mu  = clean.mean() * 252
        cov = clean.cov() * 252
        rf  = (rf_pct or 3.0) / 100
        mw  = (max_w_pct or 25) / 100
        n_sim = int(n_sim or 5000)
        
        # ARIMA override if checked
        if "arima" in (arima_chk or []):
            mu_arima = {}
            for t in tickers:
                s = clean[t].dropna()
                if len(s) >= 50:
                    try:
                        fc = ARIMA(s, order=(1,0,1)).fit().forecast(steps=int(arima_h or 21))
                        mu_arima[t] = float(fc.mean()) * 252
                    except Exception:
                        mu_arima[t] = float(s.mean()) * 252
                else:
                    mu_arima[t] = float(s.mean()) * 252
            mu = pd.Series(mu_arima)
        
        # ===== RISK FUNCTION (FIXED: use .values for numpy operations) =====
        def _risk(w):
            """Compute portfolio risk based on selected measure"""
            try:
                v = np.sqrt(w @ cov.values @ w)
                if risk_type == "vol":
                    return v
                port_r = clean.values @ w
                if risk_type == "var5":
                    return -np.percentile(port_r, 5)
                elif risk_type == "var1":
                    return -np.percentile(port_r, 1)
                return v
            except Exception as e:
                print(f"Risk calc error: {e}")
                return 1.0
        
        # ===== MONTE CARLO SIMULATION =====
        np.random.seed(42)
        sims = {"rets": [], "vols": [], "sharpes": [], "ws": []}
        
        for _ in range(n_sim):
            w = np.random.dirichlet(np.ones(n_a))
            w = np.clip(w, 0, mw)
            w = w / w.sum() if w.sum() > 0 else np.ones(n_a) / n_a
            
            try:
                rv = _risk(w)
                re = float(w @ mu.values)
                sh = (re - rf) / rv if rv > 0 else 0
                
                sims["rets"].append(re * 100)
                sims["vols"].append(rv * 100)
                sims["sharpes"].append(sh)
                sims["ws"].append(w)
            except Exception:
                continue
        
        if not sims["rets"]:
            return empty, empty, empty, empty, None
        
        # ===== OPTIMIZATION =====
        def _opt_sharpe(w):
            """Objective: maximize Sharpe ratio"""
            try:
                rv = _risk(w)
                re = float(w @ mu.values)
                return -(re - rf) / rv if rv > 0 else 0
            except:
                return 0
        
        # Max Sharpe
        res_ms = minimize(_opt_sharpe, np.ones(n_a) / n_a, method="SLSQP",
                         bounds=[(0, mw)] * n_a,
                         constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
                         options={"maxiter": 1000})
        ms_w = res_ms.x if res_ms.success else np.ones(n_a) / n_a
        ms_v = _risk(ms_w) * 100
        ms_r = float(ms_w @ mu.values) * 100
        ms_sh = (ms_r / 100 - rf) / (_risk(ms_w)) if _risk(ms_w) > 0 else 0
        
        # Min Variance
        res_mv = minimize(lambda w: w @ cov.values @ w, np.ones(n_a) / n_a, method="SLSQP",
                         bounds=[(0, mw)] * n_a,
                         constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
                         options={"maxiter": 1000})
        mv_w = res_mv.x if res_mv.success else np.ones(n_a) / n_a
        mv_v = _risk(mv_w) * 100
        mv_r = float(mv_w @ mu.values) * 100
        
        # ===== FRONTIER CURVE =====
        r_min = mv_r / 100
        r_max = float(mu.max())
        frontier_pts = []
        
        for target in np.linspace(r_min, min(r_max, r_min + 0.5), 60):
            try:
                res_f = minimize(lambda w: w @ cov.values @ w, np.ones(n_a) / n_a, method="SLSQP",
                                bounds=[(0, mw)] * n_a,
                                constraints=[
                                    {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                                    {"type": "eq", "fun": lambda w: w @ mu.values - target}
                                ],
                                options={"maxiter": 500})
                if res_f.success:
                    frontier_pts.append((_risk(res_f.x) * 100, res_f.x @ mu.values * 100))
            except Exception:
                continue
        
        # ===== PLOT 1: FRONTIER =====
        fig1 = go.Figure()
        
        # MC scatter
        fig1.add_trace(go.Scatter(x=sims["vols"], y=sims["rets"], mode="markers",
                                 marker=dict(
                                     color=sims["sharpes"],
                                     colorscale="RdYlGn",
                                     size=4,
                                     opacity=0.5,
                                     colorbar=dict(title="Sharpe", x=1.02)
                                 ),
                                 name="MC Simulations"))
        
        # Frontier curve
        if frontier_pts:
            fx, fy = zip(*frontier_pts)
            fig1.add_trace(go.Scatter(x=list(fx), y=list(fy), mode="lines",
                                     line=dict(color="#1F4E78", width=3),
                                     name="Frontier"))
        
        # Max Sharpe
        fig1.add_trace(go.Scatter(x=[ms_v], y=[ms_r], mode="markers+text",
                                 marker=dict(symbol="star", size=18, color="gold",
                                           line=dict(color="black", width=1.5)),
                                 text=[f"Max Sharpe<br>SR:{ms_sh:.2f}"],
                                 textposition="top center",
                                 name="Max Sharpe"))
        
        # Min Var
        fig1.add_trace(go.Scatter(x=[mv_v], y=[mv_r], mode="markers+text",
                                 marker=dict(symbol="diamond", size=14, color="cyan",
                                           line=dict(color="black", width=1.5)),
                                 text=["Min Var"],
                                 textposition="top center",
                                 name="Min Var"))
        
        # Individual assets
        for i, t in enumerate(tickers):
            v_i = np.sqrt(cov.iloc[i, i]) * 100
            r_i = float(mu.iloc[i]) * 100
            fig1.add_trace(go.Scatter(x=[v_i], y=[r_i], mode="markers+text",
                                     marker=dict(size=8, color=COLORS_P[i % len(COLORS_P)]),
                                     text=[t],
                                     textposition="top right",
                                     name=t,
                                     showlegend=False))
        
        xlabel = {"vol": "Volatility (%)", "var5": "VaR 5% (%)", "var1": "VaR 1% (%)"}
        title_sfx = " [ARIMA-adjusted]" if "arima" in (arima_chk or []) else " [Standard]"
        fig1.update_layout(
            title=f"Efficient Frontier{title_sfx} — Constraint: 0%-{max_w_pct}%",
            xaxis_title=xlabel.get(risk_type, "Risk (%)"),
            yaxis_title="Expected Return (%)",
            template="plotly_white",
            legend=dict(orientation="h", y=1.02),
            hovermode="x unified"
        )
        
        # ===== PLOT 2: CUMULATIVE RETURNS =====
        p_sel = p[tickers].dropna()
        r2 = compute_returns(p_sel)
        
        w1_raw = np.array([IS20_W.get(t, 0.0) for t in tickers])
        w1 = w1_raw / w1_raw.sum() if w1_raw.sum() > 0.01 else np.ones(n_a) / n_a
        
        fig2 = go.Figure()
        for nm, w_ in [("IS20 Passive", w1), ("Max-Sharpe", ms_w), ("Min-Var", mv_w)]:
            cum = np.cumsum(r2.values @ w_) * 100
            col_ = {"IS20 Passive": "#FF6B35", "Max-Sharpe": "#4CAF50", "Min-Var": "#2196F3"}[nm]
            fig2.add_trace(go.Scatter(x=r2.index, y=cum, name=nm, line=dict(color=col_, width=2)))
        
        _add_events(fig2, start, end)
        fig2.update_layout(
            title="Cumulative Returns — F1/F2/F3",
            template="plotly_white",
            hovermode="x unified"
        )
        
        # ===== PIE CHARTS =====
        def _pie(weights, title, tickers_list):
            mask = weights > 0.005
            labels = [tickers_list[i] for i in range(len(tickers_list)) if mask[i]]
            values = weights[mask]
            colors = COLORS_P[:len(labels)]
            fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.35,
                                   marker_colors=colors, textinfo="label+percent"))
            fig.update_layout(title=title, template="plotly_white",
                            margin=dict(t=40, b=10, l=10, r=10), showlegend=False)
            return fig
        
        fig_ms = _pie(ms_w, f"Max-Sharpe SR:{ms_sh:.2f}", tickers)
        fig_mv = _pie(mv_w, "Min-Variance", tickers)
        
        # ===== STORE DATA =====
        ef_data = {
            "ms_weights": ms_w.tolist(),
            "mv_weights": mv_w.tolist(),
            "tickers": tickers,
            "ms_sharpe": float(ms_sh)
        }
        
        return fig1, fig2, fig_ms, fig_mv, ef_data
    
    except Exception as e:
        print(f"❌ Frontier Error: {e}")
        import traceback
        traceback.print_exc()
        return empty, empty, empty, empty, None

# ============================================================================
# CORRELATION MATRIX CALLBACK
# ============================================================================
@app.callback(
    Output("g-corr","figure"),
    Input("btn-corr","n_clicks"),
    State("store-returns","data"),
    State("inp-corr-roll","value"),
    prevent_initial_call=True)
def update_corr(n, ret_data, roll):
    if not ret_data:
        return go.Figure()
    try:
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
                       color_continuous_scale="RdYlGn", zmin=-1, zmax=1,
                       title=title)
        fig.update_layout(template="plotly_white",
                         coloraxis_colorbar=dict(title="ρ"))
        return fig
    except Exception as e:
        print(f"Correlation error: {e}")
        return go.Figure()

# ============================================================================
# PORTFOLIO ANALYSIS CALLBACKS
# ============================================================================
@app.callback(
    Output("dd-bench-p1","options"),
    Input("store-prices","data"))
def update_bench_options(data):
    if not data:
        return []
    try:
        p = pd.read_json(data, convert_dates=True)
        return [{"label": c, "value": c} for c in p.columns]
    except:
        return []

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
    empty = go.Figure()
    empty.update_layout(template="plotly_white")
    
    if not ret_data:
        return empty, empty, empty, ""
    
    try:
        r = pd.read_json(ret_data, convert_dates=True)
        rf = (rf_pct or 3.0) / 100
        roll = int(roll_w or 250)
        
        tickers = list(r.columns)
        n_a = len(tickers)
        mu = r.mean() * 252
        cov = r.cov() * 252
        
        # IS20 weights
        w1_raw = np.array([IS20_W.get(t, 0.0) for t in tickers])
        w1 = w1_raw / w1_raw.sum() if w1_raw.sum() > 0.05 else np.ones(n_a) / n_a
        
        # Max Sharpe
        def neg_sharpe(w):
            r_ = w @ mu
            v_ = np.sqrt(w @ cov @ w)
            return -(r_ - rf) / v_ if v_ > 0 else 0
        
        res = minimize(neg_sharpe, np.ones(n_a) / n_a, method="SLSQP",
                      bounds=[(0, .35)] * n_a,
                      constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}])
        w2 = res.x if res.success else np.ones(n_a) / n_a
        
        # Min Var
        res3 = minimize(lambda w: w @ cov @ w, np.ones(n_a) / n_a, method="SLSQP",
                       bounds=[(0, .35)] * n_a,
                       constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}])
        w3 = res3.x if res3.success else np.ones(n_a) / n_a
        
        port_rets = {}
        for name, w in [("P1 IS20 Passive", w1), ("P2 Max-Sharpe", w2), ("P3 Min-Var", w3)]:
            port_rets[name] = pd.Series(r.values @ w, index=r.index)
        
        if bench_data and bench_col:
            try:
                bd = pd.read_json(bench_data, convert_dates=True)
                if bench_col + "_ret" in bd.columns:
                    port_rets[f"Benchmark ({bench_col})"] = bd[bench_col + "_ret"].reindex(r.index).fillna(0)
            except:
                pass
        
        colors_p = {
            "P1 IS20 Passive": "#FF6B35",
            "P2 Max-Sharpe": "#4CAF50",
            "P3 Min-Var": "#2196F3",
            "Benchmark": "#9E9E9E"
        }
        
        # Plot 1: Cumulative returns
        fig1 = go.Figure()
        for name, ret in port_rets.items():
            cum = np.cumsum(ret) * 100
            fig1.add_trace(go.Scatter(x=cum.index, y=cum.values, name=name,
                                     line=dict(color=colors_p.get(name, "#607D8B"), width=2)))
        _add_events(fig1, start, end)
        fig1.update_layout(title="Cumulative Returns (%)", template="plotly_white",
                          legend=dict(orientation="h", y=1.02), hovermode="x unified")
        
        # Plot 2: Information Ratio
        fig2 = go.Figure()
        ref_key = list(port_rets.keys())[0]
        ref_ret = port_rets[ref_key]
        for name, ret in list(port_rets.items())[1:]:
            excess = ret - ref_ret.reindex(ret.index).fillna(0)
            ir = excess.rolling(roll).mean() / excess.rolling(roll).std() * np.sqrt(252)
            fig2.add_trace(go.Scatter(x=ir.index, y=ir.values, name=name,
                                     line=dict(color=colors_p.get(name, "#607D8B"), width=1.5)))
        fig2.add_hline(y=0, line_dash="dash", line_color="red")
        fig2.update_layout(title=f"Information Ratio vs {ref_key} (rolling {roll}d)",
                          template="plotly_white")
        
        # Plot 3: Sharpe & TEV
        fig3 = make_subplots(rows=1, cols=2,
                            subplot_titles=["Rolling Sharpe Ratio", "Tracking Error Volatility"])
        for name, ret in port_rets.items():
            rm = ret.rolling(roll).mean() * 252
            rv = ret.rolling(roll).std() * np.sqrt(252)
            sr = (rm - rf) / rv
            col_ = colors_p.get(name, "#607D8B")
            fig3.add_trace(go.Scatter(x=sr.index, y=sr.values, name=name,
                                     line=dict(color=col_, width=1.5)),
                          row=1, col=1)
            if name != ref_key:
                ref = port_rets.get(ref_key, ret)
                tev = (ret - ref.reindex(ret.index).fillna(0)).rolling(roll).std() * np.sqrt(252) * 100
                fig3.add_trace(go.Scatter(x=tev.index, y=tev.values,
                                         name=f"TEV {name}",
                                         line=dict(color=col_, width=1.5, dash="dot"),
                                         showlegend=False),
                              row=1, col=2)
        fig3.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        fig3.add_hrect(y0=2, y1=4, fillcolor="rgba(76,175,80,0.1)", line_width=0,
                      row=1, col=2, annotation_text="target 2-4%")
        fig3.update_layout(template="plotly_white", hovermode="x unified")
        
        # Stats table
        rows_t = []
        for name, ret in port_rets.items():
            ar = ret.mean() * 252 * 100
            av = ret.std() * np.sqrt(252) * 100
            sr_ = (ar - rf * 100) / av if av > 0 else 0
            cum = (np.exp(np.cumsum(ret)) - 1).iloc[-1] * 100
            cs = pd.Series(np.exp(np.cumsum(ret)))
            mdd = ((cs - cs.cummax()) / cs.cummax()).min() * 100
            cal = ar / abs(mdd) if mdd != 0 else 0
            rows_t.append({
                "Portafoglio": name,
                "Ret/Y %": f"{ar:.1f}",
                "Vol/Y %": f"{av:.1f}",
                "Sharpe": f"{sr_:.3f}",
                "Cumul. %": f"{cum:.0f}",
                "MaxDD %": f"{mdd:.1f}",
                "Calmar": f"{cal:.3f}"
            })
        
        tbl = dash_table.DataTable(
            data=rows_t,
            columns=[{"name": k, "id": k} for k in rows_t[0].keys()] if rows_t else [],
            style_cell={"textAlign": "center", "fontSize": "12px", "padding": "6px"},
            style_header={"backgroundColor": "#1F4E78", "color": "white", "fontWeight": "bold"}
        )
        
        return fig1, fig2, fig3, tbl
    
    except Exception as e:
        print(f"Portfolio error: {e}")
        return empty, empty, empty, ""

# ============================================================================
# SERVER & RUN
# ============================================================================
server = app.server

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
