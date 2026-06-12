# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Primary app
streamlit run app.py           # Opens at http://localhost:8501

# Train / retrain the ML model (takes several minutes — downloads 10y of data for 44 tickers)
python train_model.py

# Standalone backtester (hardcoded to RELIANCE.NS, writes TrailingStopStrategy.html)
python trading_bot.py

# Install dependencies
pip install yfinance pandas pandas-ta backtesting streamlit plotly vaderSentiment scikit-learn joblib
```

## Architecture

### File roles

| File | Role |
|------|------|
| `app.py` | **Primary app** — 6-mode Streamlit dashboard (1955 lines) |
| `dashboard.py` | Legacy prototype — 2-mode US-only version, superseded by `app.py` |
| `trading_bot.py` | Standalone `backtesting` script for RELIANCE.NS |
| `train_model.py` | Trains `GradientBoostingClassifier`, writes `.pkl` artifacts |
| `trade_signal_model.pkl` / `feature_scaler.pkl` | ML artifacts loaded at runtime |
| `trade_journal.json` | Paper-trade state (persisted across sessions) |
| `watchlist.json` | User watchlist (persisted across sessions) |

### `app.py` — six operating modes

Selected via the sidebar `dashboard_mode` selectbox. All modes share a common sidebar that sets `PROFIT_TARGET_PCT`, `STOP_LOSS_PCT`, `AI_SENTIMENT_FLOOR`, `news_provider`, `api_key`, currency toggle (`IS_INR` / `GLOBAL_SYM`), email alert credentials, and Gemini API key.

1. **Sector Scanner** — scans a sector universe (or user watchlist) ticker-by-ticker: technical pre-filter → sentiment → `generate_signal()` → sorted results table with top pick.
2. **Deep-Dive Analysis** — single ticker: 3-panel Plotly chart (candlestick + EMA/SMA overlay, RSI sub-panel, volume sub-panel), technical snapshot table, confidence gauge, optional Gemini AI narrative.
3. **Backtest Lab** — embeds `TrailingStopStrategy` inline (mirrors `trading_bot.py`), runs via the `backtesting` library against user-chosen ticker/date range, renders equity curve + drawdown chart + trades table.
4. **ML Insights** — feature importance bar chart, training config table, live quick-scan scorer.
5. **Trade Journal** — paper trading with live price monitoring, target/stop email alerts, P&L analytics (cumulative curve, win/loss donut, expectancy).
6. **Position Sizer** — fixed-fractional risk calculator with scenarios table and price-level visualiser.

### Shared data pipeline

```
yfinance.download() → compute_indicators() → generate_signal()
                                ↓
                    get_sentiment() [VADER + news provider]
```

- **`get_data(ticker, days=400)`** — cached 300 s; flattens multi-level yfinance columns.
- **`compute_indicators(df)`** — adds EMA_10, SMA_50, SMA_200, RSI, MACD/Signal/Hist, ADX, ATR and the 8 ML feature columns (ATR_Pct, Price_vs_EMA10, Volume_Ratio, MA_Trend, MACD_Cross).
- **`generate_signal(last, profit_pct, stop_pct, sentiment, floor, ml_model, ml_scaler)`** — returns a dict with `verdict`, `color`, `regime`, `target_price`, `stop_price`, `ml_prob`, `confidence`, `rr_ratio`. Logic:
  - ADX > 22 → **Trending**: entry when `price > EMA_10` and `volume > 1.2 × Vol_SMA`; target = `price × (1 + profit_pct)`, stop = `price × (1 − stop_pct)`.
  - ADX ≤ 22 → **Choppy**: entry when RSI < 35; target = `price + 2 × ATR`, stop = `price − ATR`.
  - BUY verdict is downgraded to `"HOLD (ML Caution)"` when ML probability < 0.45.
  - Confidence = `rule_score × 0.5 + ml_prob × 0.5` (or `rule_score` alone if no model).

### ML model

- **Label**: 1 if price hits +4% before −2% within 5 trading days; 0 otherwise (tie-break on final close).
- **Features** (`ML_FEATURE_COLS`): RSI, MACD_Hist, ADX, ATR_Pct, Price_vs_EMA10, Volume_Ratio, MA_Trend, MACD_Cross.
- **Algorithm**: `GradientBoostingClassifier(n_estimators=300, max_depth=4, lr=0.05, subsample=0.8, min_samples_leaf=20)`.
- **Universe**: 25 US tickers + 20 Nifty-50 NSE tickers; trained 2015–2024, held-out test 2025.
- `load_ml_model()` is `@st.cache_resource` — model is shared across all sessions.

### TrailingStopStrategy (backtesting)

Defined identically in both `trading_bot.py` (standalone) and inline in `app.py` (Backtest Lab). Entry regimes use ADX > 25 (note: signal engine uses ADX > 22). Exits: 2× ATR trailing stop, 4× ATR take-profit, breakeven stop at 50% of distance to target, RSI > `rsi_high` (65).

### Key hardcoded values

- `app.py:57–63` — `SECTOR_UNIVERSES` dict (add new sectors here)
- `app.py:65–68` — `ML_FEATURE_COLS` (must match `train_model.py:FEATURE_COLS`)
- `app.py:571` — `JOURNAL_INITIAL`: ₹50,000 (INR) / $600 (USD)
- `app.py:675–676` — Backtest Lab `bt_cash` number_input (min 100, max 1,000,000)
- `trading_bot.py:99` — standalone backtest ticker (`"RELIANCE.NS"`)
- `train_model.py:31–33` — label parameters (PROFIT_PCT, STOP_PCT, HORIZON) — must stay in sync with `generate_signal()` defaults
