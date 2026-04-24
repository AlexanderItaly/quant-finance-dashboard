# 📊 Dashboard Finanziaria Quantitativa — IS20

Dash web app con 10 tab di analisi finanziaria quantitativa.

## 🚀 Deploy su Render.com

1. **Fork / push** su GitHub
2. Vai su [render.com](https://render.com) → **New Web Service**
3. Collega il repo GitHub
4. Impostazioni:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:server`
   - **Environment:** Python 3
5. Click **Create Web Service**

## 💻 Esecuzione locale

```bash
pip install -r requirements.txt
python app.py
# oppure con gunicorn:
gunicorn app:server
```
Apri `http://localhost:8050`

## 📋 Tab disponibili

| # | Tab | Descrizione |
|---|-----|-------------|
| 1 | Analisi Portafoglio | Pesi IS20, metriche rischio/rendimento |
| 2 | Matrice Correlazioni | Heatmap correlazioni |
| 3 | Analisi Finanziaria | Beta, alfa, drawdown, Sharpe |
| 4 | Frontiera Efficiente | Ottimizzazione Markowitz |
| 5 | Style Analysis | Regressione Newey-West Fama-French |
| 6 | Rendimenti Storici | Cumulative returns e volatilità |
| 7 | Analisi ARIMA | Previsioni serie temporali |
| 8 | Analisi Rolling | Volatilità rolling + GARCH + regime |
| 9 | Previsione LSTM | Rete neurale ricorrente |
| 10 | Confronto Portafogli | Benchmark IS20 vs ottimizzati |
