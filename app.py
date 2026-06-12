import streamlit as st
import yfinance as yf
import pandas as pd
import indicators as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import timedelta, date
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
import re
import time
import os
import uuid
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit.components.v1 as components
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
try:
    from config import GEMINI_MODEL
except ImportError:
    GEMINI_MODEL = "gemini-3.5-flash"

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)
except ImportError:
    pass

try:
    from ai_reasoning import get_ai_explanation as _get_ai_explanation
    _AI_REASONING_OK = True
except ImportError:
    _AI_REASONING_OK = False

try:
    import joblib
    _JOBLIB_OK = True
except ImportError:
    _JOBLIB_OK = False

try:
    from backtesting import Backtest, Strategy
    _BT_OK = True
except ImportError:
    _BT_OK = False

# ════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Trading Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .main .block-container { padding-top: 1rem; }
  div[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 700; }
  div[data-testid="stMetricDelta"] { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
SECTOR_UNIVERSES = {
    "Semiconductors (AI Infra)": ["NVDA", "AMD", "AVGO", "TSM", "QCOM", "MU", "ARM"],
    "Clean Energy & EV":         ["TSLA", "ENPH", "FSLR", "NEE", "RIVN", "PLUG"],
    "Aerospace & Defense":       ["LMT", "RTX", "NOC", "GD", "BA", "PLTR"],
    "Big Tech (FAANG+)":         ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NFLX"],
    "Indian Large-Caps (NSE)":   ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "WIPRO.NS"],
}

ML_FEATURE_COLS = [
    "RSI", "MACD_Hist", "ADX", "ATR_Pct",
    "Price_vs_EMA10", "Volume_Ratio", "MA_Trend", "MACD_Cross",
]

NEWS_PROVIDERS = [
    "Hybrid Auto-Router (Free)",
    "Yahoo Finance (Free)",
    "Google News RSS (Free)",
    "Finviz Web Scraper (Free)",
    "NewsAPI.org (Key Required)",
    "Alpha Vantage (Key Required)",
]

_BASE_DIR             = os.path.dirname(os.path.abspath(__file__))
WATCHLIST_FILE        = "watchlist.json"
PORTFOLIO_FILE        = os.path.join(_BASE_DIR, "portfolio.json")
CLOSED_POSITIONS_FILE = os.path.join(_BASE_DIR, "closed_positions.json")
UNIVERSE_FILE         = os.path.join(_BASE_DIR, "universe.json")

_DEFAULT_UNIVERSE = [
    "NVDA", "AMD", "AVGO", "TSM", "QCOM", "MU", "ARM",
    "TSLA", "ENPH", "FSLR", "NEE", "RIVN", "PLUG",
    "LMT", "RTX", "NOC", "GD", "BA", "PLTR",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NFLX",
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "WIPRO.NS", "SBIN.NS", "BHARTIARTL.NS", "HINDUNILVR.NS", "BAJFINANCE.NS",
    "MARUTI.NS", "TATAMOTORS.NS", "HCLTECH.NS", "ITC.NS", "AXISBANK.NS",
    "KOTAKBANK.NS", "LT.NS", "NESTLEIND.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS",
]


def load_universe() -> list:
    if os.path.exists(UNIVERSE_FILE):
        try:
            with open(UNIVERSE_FILE) as f:
                data = json.load(f)
                if isinstance(data, list) and data:
                    return data
        except Exception:
            pass
    return list(_DEFAULT_UNIVERSE)


def save_universe(tickers: list) -> None:
    with open(UNIVERSE_FILE, "w") as f:
        json.dump(tickers, f, indent=2)


# Loaded fresh on every Streamlit rerun so universe.json additions are reflected immediately
AI_PICKS_UNIVERSE = load_universe()

# ════════════════════════════════════════════════════════════════════════════
#  ML MODEL LOADER
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_ml_model():
    if not _JOBLIB_OK:
        return None, None
    m_path = "trade_signal_model.pkl"
    s_path = "feature_scaler.pkl"
    if os.path.exists(m_path) and os.path.exists(s_path):
        return joblib.load(m_path), joblib.load(s_path)
    return None, None

# ════════════════════════════════════════════════════════════════════════════
#  DATA ENGINE
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def get_data(ticker: str, days: int = 400) -> pd.DataFrame:
    try:
        end   = date.today()
        start = end - timedelta(days=days)
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df.dropna()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_corr_returns(tickers: tuple, lookback_days: int = 90) -> pd.DataFrame:
    """Fetch daily close returns for each ticker; return last-60-row slice."""
    frames = {}
    for t in tickers:
        df_c = get_data(t, days=lookback_days)
        if not df_c.empty and len(df_c) > 5:
            frames[t] = df_c["Close"].pct_change().dropna()
    if len(frames) < 2:
        return pd.DataFrame()
    combined = pd.DataFrame(frames).dropna(how="any")
    return combined.iloc[-60:] if len(combined) >= 60 else combined


@st.cache_data(ttl=60)
def get_current_price(ticker: str) -> float | None:
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA_10"]  = ta.ema(df["Close"], length=10)
    df["SMA_50"]  = ta.sma(df["Close"], length=50)
    df["SMA_200"] = ta.sma(df["Close"], length=200)
    df["RSI"]     = ta.rsi(df["Close"], length=14)
    df["Vol_SMA"] = ta.sma(df["Volume"], length=20)

    macd_df         = ta.macd(df["Close"])
    df["MACD"]      = macd_df.iloc[:, 0]
    df["MACD_Sig"]  = macd_df.iloc[:, 2]
    df["MACD_Hist"] = macd_df.iloc[:, 1]

    adx_df   = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    df["ADX"] = adx_df.iloc[:, 0]
    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    df["ATR_Pct"]        = df["ATR"] / df["Close"]
    df["Price_vs_EMA10"] = (df["Close"] - df["EMA_10"]) / df["EMA_10"]
    df["Volume_Ratio"]   = df["Volume"] / df["Vol_SMA"]
    df["MA_Trend"]       = (df["SMA_50"] > df["SMA_200"]).astype(int)
    df["MACD_Cross"]     = (df["MACD"] > df["MACD_Sig"]).astype(int)

    return df.dropna()

# ════════════════════════════════════════════════════════════════════════════
#  SENTIMENT ENGINE
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def get_sentiment(ticker: str, provider: str, key: str = ""):
    analyzer  = SentimentIntensityAnalyzer()
    headlines = []
    headers   = {"User-Agent": "Mozilla/5.0"}

    def score_title(title, source):
        s = analyzer.polarity_scores(title)["compound"]
        headlines.append({"title": f"[{source}] {title}", "score": s})
        return s

    total = 0.0
    try:
        if provider == "Alpha Vantage (Key Required)" and key:
            base = ticker.split(".")[0]
            url  = (f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT"
                    f"&tickers={base}&limit=6&apikey={key}")
            req  = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as r:
                data = json.loads(r.read())
            for item in data.get("feed", [])[:6]:
                s = float(item.get("overall_sentiment_score", 0))
                headlines.append({"title": f"[AlphaV] {item.get('title', '')}", "score": s})
                total += s

        elif provider == "NewsAPI.org (Key Required)" and key:
            q   = urllib.parse.quote(f"{ticker} stock")
            url = (f"https://newsapi.org/v2/everything?q={q}&language=en"
                   f"&sortBy=publishedAt&pageSize=6&apiKey={key}")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as r:
                data = json.loads(r.read())
            for item in data.get("articles", []):
                t = item.get("title", "")
                if t and t != "[Removed]":
                    total += score_title(t, "NewsAPI")

        if provider in ("Yahoo Finance (Free)", "Hybrid Auto-Router (Free)"):
            try:
                for article in (yf.Ticker(ticker).news or [])[:6]:
                    t = article.get("title", "")
                    if t:
                        total += score_title(t, "Yahoo")
            except Exception:
                pass

        if provider == "Google News RSS (Free)" or (
                provider == "Hybrid Auto-Router (Free)" and not headlines):
            try:
                q   = urllib.parse.quote(f"{ticker} stock")
                url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=8) as r:
                    root = ET.fromstring(r.read())
                for item in root.findall(".//item")[:6]:
                    node = item.find("title")
                    if node is not None and node.text:
                        total += score_title(node.text.split(" - ")[0], "Google")
            except Exception:
                pass

        if provider == "Finviz Web Scraper (Free)" or (
                provider == "Hybrid Auto-Router (Free)" and not headlines):
            try:
                url = f"https://finviz.com/quote.ashx?t={ticker.split('.')[0]}"
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=8) as r:
                    html = r.read().decode("utf-8")
                for t in re.findall(r'class="tab-link-news"[^>]*>(.*?)</a>', html)[:6]:
                    total += score_title(re.sub(r"<[^>]+>", "", t), "Finviz")
            except Exception:
                pass

        if not headlines:
            return 0.0, [{"title": "No news data accessible.", "score": 0}]
        return total / len(headlines), headlines

    except Exception:
        return 0.0, [{"title": "Provider connection failed.", "score": 0}]

# ════════════════════════════════════════════════════════════════════════════
#  SIGNAL ENGINE
# ════════════════════════════════════════════════════════════════════════════
def generate_signal(last: pd.Series, profit_pct: float, stop_pct: float,
                    sentiment: float, sentiment_floor: float,
                    ml_model=None, ml_scaler=None) -> dict:
    price   = float(last["Close"])
    ema     = float(last["EMA_10"])
    rsi     = float(last["RSI"])
    adx     = float(last["ADX"])
    atr     = float(last["ATR"])
    vol     = float(last["Volume"])
    vol_avg = float(last["Vol_SMA"])

    is_trending = adx > 22
    news_ok     = sentiment >= sentiment_floor

    if is_trending:
        regime       = "Trending"
        target_price = price * (1.0 + profit_pct)
        stop_price   = price * (1.0 - stop_pct)
        tech_buy     = price > ema and vol > vol_avg * 1.2
        tech_sell    = rsi > 70
    else:
        regime       = "Choppy"
        target_price = price + 2.0 * atr
        stop_price   = price - 1.0 * atr
        tech_buy     = rsi < 35
        tech_sell    = rsi > 65

    # ML probability
    ml_prob = None
    if ml_model is not None and ml_scaler is not None:
        try:
            feats = np.array([[
                float(last.get("RSI",            rsi)),
                float(last.get("MACD_Hist",      0)),
                float(last.get("ADX",            adx)),
                float(last.get("ATR_Pct",        atr / price if price else 0)),
                float(last.get("Price_vs_EMA10", (price - ema) / ema if ema else 0)),
                float(last.get("Volume_Ratio",   vol / vol_avg if vol_avg else 1)),
                float(last.get("MA_Trend",       0)),
                float(last.get("MACD_Cross",     0)),
            ]])
            ml_prob = float(ml_model.predict_proba(ml_scaler.transform(feats))[0][1])
        except Exception:
            ml_prob = None

    if tech_buy and news_ok:
        verdict = "BUY"
        color   = "#00ff88"
    elif tech_sell:
        verdict = "SELL"
        color   = "#ff4444"
    else:
        verdict = "HOLD"
        color   = "#ffaa00"

    if verdict == "BUY" and ml_prob is not None and ml_prob < 0.45:
        verdict = "HOLD (ML Caution)"
        color   = "#ffaa00"

    rule_score = 1.0 if tech_buy else 0.5 if verdict.startswith("HOLD") else 0.0
    if ml_prob is not None:
        confidence = (rule_score * 0.5 + ml_prob * 0.5) * 100
    else:
        confidence = rule_score * 100

    rr = ((target_price - price) / (price - stop_price)
          if price - stop_price > 0 else 0.0)

    return {
        "verdict":      verdict,
        "color":        color,
        "regime":       regime,
        "target_price": target_price,
        "stop_price":   stop_price,
        "ml_prob":      ml_prob,
        "confidence":   confidence,
        "rr_ratio":     round(rr, 2),
        "news_ok":      news_ok,
        "price":        price,
    }

# ════════════════════════════════════════════════════════════════════════════
#  INLINE BACKTEST STRATEGY  (mirrors trading_bot.py)
# ════════════════════════════════════════════════════════════════════════════
if _BT_OK:
    class TrailingStopStrategy(Strategy):
        rsi_period = 14
        rsi_low    = 35
        rsi_high   = 65
        adx_period = 14
        atr_period = 14

        def init(self):
            close = pd.Series(self.data.Close)
            high  = pd.Series(self.data.High)
            low   = pd.Series(self.data.Low)

            self.rsi_ind    = self.I(ta.rsi, close, length=self.rsi_period)
            macd_df         = ta.macd(close)
            self.macd_line  = self.I(lambda: macd_df.iloc[:, 0].values)
            self.macd_sig   = self.I(lambda: macd_df.iloc[:, 2].values)
            self.fast_ma    = self.I(ta.sma, close, length=50)
            self.slow_ma    = self.I(ta.sma, close, length=200)
            adx_df          = ta.adx(high, low, close, length=self.adx_period)
            self.adx_ind    = self.I(lambda: adx_df.iloc[:, 0].values)
            self.atr_ind    = self.I(ta.atr, high, low, close, length=self.atr_period)
            self._entry     = 0.0
            self._tp        = 0.0
            self._sl        = 0.0

        def next(self):
            if len(self.adx_ind) < 200:
                return
            adx   = self.adx_ind[-1]
            rsi   = self.rsi_ind[-1]
            price = self.data.Close[-1]
            atr   = self.atr_ind[-1]
            uptrend  = self.fast_ma[-1] > self.slow_ma[-1]
            macd_up  = self.macd_line[-1] > self.macd_sig[-1]

            if self.position:
                new_sl = price - 2 * atr
                if new_sl > self._sl:
                    self._sl = new_sl
                    self.position.sl = self._sl
                halfway = self._entry + (self._tp - self._entry) * 0.5
                if price >= halfway and self.position.sl < self._entry:
                    self.position.sl = self._entry
                if price >= self._tp or rsi > self.rsi_high:
                    self.position.close()
                return

            if (adx > 25 and uptrend and macd_up) or (adx <= 25 and rsi < self.rsi_low):
                self._entry = price
                self._tp    = price + 4 * atr
                self._sl    = price - 2 * atr
                self.buy(sl=self._sl, tp=self._tp)

# ════════════════════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ════════════════════════════════════════════════════════════════════════════
def currency_sym(ticker: str) -> str:
    return "₹" if ".NS" in ticker else "$"


def render_headlines(headlines: list) -> None:
    for h in headlines:
        c = "green" if h["score"] >= 0.05 else "red" if h["score"] <= -0.05 else "gray"
        st.markdown(
            f'- <span style="color:{c}">**[{h["score"]:+.2f}]**</span> {h["title"]}',
            unsafe_allow_html=True,
        )


def signal_banner(sig: dict, extra: str = "") -> None:
    color = sig["color"]
    st.markdown(
        f'<div style="background:{color}22; border-left:4px solid {color}; '
        f'padding:10px 18px; border-radius:6px; margin:8px 0;">'
        f'<span style="color:{color}; font-size:1.3rem; font-weight:700;">'
        f'{sig["verdict"]}</span>'
        f'<span style="color:#bbb; margin-left:18px; font-size:0.9rem;">'
        f'Confidence: {sig["confidence"]:.0f}%  ·  Regime: {sig["regime"]}{extra}'
        f'</span></div>',
        unsafe_allow_html=True,
    )

# ════════════════════════════════════════════════════════════════════════════
#  WATCHLIST · EMAIL · ALERT HELPERS
# ════════════════════════════════════════════════════════════════════════════
def load_watchlist() -> list:
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE) as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            pass
    return ["NVDA", "AAPL", "TSLA", "RELIANCE.NS"]


def save_watchlist(tickers: list) -> None:
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(tickers, f, indent=2)


def load_portfolio() -> list:
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE) as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            pass
    return []


def save_portfolio(positions: list) -> None:
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(positions, f, indent=2)


def load_closed_positions() -> list:
    if os.path.exists(CLOSED_POSITIONS_FILE):
        try:
            with open(CLOSED_POSITIONS_FILE) as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            pass
    return []


def save_closed_positions(positions: list) -> None:
    with open(CLOSED_POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2)


def send_email_alert(recipient: str, subject: str, body: str,
                     sender: str, password: str) -> bool:
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = sender
        msg["To"]      = recipient
        msg.attach(MIMEText(body, "html"))
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as srv:
            srv.login(sender, password)
            srv.sendmail(sender, recipient, msg.as_string())
        return True
    except Exception:
        return False


def _build_alert_email(ticker: str, color: str, regime: str,
                       entry: float, target: float, stop: float,
                       sentiment: float, ml_prob: str, sym: str) -> str:
    rr = (target - entry) / (entry - stop) if entry - stop > 0 else 0
    return f"""
<html><body style="font-family:Arial,sans-serif;background:#111;color:#eee;padding:24px;max-width:560px;">
<h2 style="color:{color};margin-bottom:4px;">⚡ BUY Signal: {ticker}</h2>
<p style="color:#888;margin-top:0;">Regime: {regime}  ·  Generated by AI Trading Suite</p>
<table style="border-collapse:collapse;width:100%;margin-top:16px;">
  <tr style="background:#1e1e2e;">
    <td style="padding:10px 14px;border:1px solid #333;color:#aaa;">Entry Price</td>
    <td style="padding:10px 14px;border:1px solid #333;font-weight:700;">{sym}{entry:.2f}</td>
  </tr>
  <tr>
    <td style="padding:10px 14px;border:1px solid #333;color:#aaa;">Take Profit</td>
    <td style="padding:10px 14px;border:1px solid #333;color:#00ff88;font-weight:700;">{sym}{target:.2f}
    &nbsp;<span style="color:#666;font-size:0.85em;">(+{(target-entry)/entry*100:.1f}%)</span></td>
  </tr>
  <tr style="background:#1e1e2e;">
    <td style="padding:10px 14px;border:1px solid #333;color:#aaa;">Stop Loss</td>
    <td style="padding:10px 14px;border:1px solid #333;color:#ff4444;font-weight:700;">{sym}{stop:.2f}
    &nbsp;<span style="color:#666;font-size:0.85em;">(-{(entry-stop)/entry*100:.1f}%)</span></td>
  </tr>
  <tr>
    <td style="padding:10px 14px;border:1px solid #333;color:#aaa;">R:R Ratio</td>
    <td style="padding:10px 14px;border:1px solid #333;">{rr:.2f}</td>
  </tr>
  <tr style="background:#1e1e2e;">
    <td style="padding:10px 14px;border:1px solid #333;color:#aaa;">Sentiment Score</td>
    <td style="padding:10px 14px;border:1px solid #333;">{sentiment:+.2f}</td>
  </tr>
  <tr>
    <td style="padding:10px 14px;border:1px solid #333;color:#aaa;">ML Probability</td>
    <td style="padding:10px 14px;border:1px solid #333;">{ml_prob}</td>
  </tr>
</table>
<p style="color:#555;font-size:0.75rem;margin-top:20px;">
  This is an automated alert. Not financial advice.
</p>
</body></html>
"""


@st.cache_data(ttl=300)
def get_gemini_analysis(ticker: str, prompt: str, gemini_key: str) -> str:
    """Send a prompt to Gemini 2.5 Flash and return the text response."""
    if not gemini_key:
        return ""
    try:
        from google import genai as _genai
        _client   = _genai.Client(api_key=gemini_key)
        _response = _client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return _response.text.strip()
    except Exception as e:
        return f"_(Gemini error: {str(e)})_"


@st.cache_data(ttl=86400)
def cached_ai_explanation(ticker: str, indicators_json: str,
                          gemini_key: str, date_str: str) -> dict:
    """Daily-cached Gemini reasoning. date_str causes cache bust at midnight."""
    if not _AI_REASONING_OK:
        return {
            "verdict": "hold", "confidence": "low",
            "reasoning": "`ai_reasoning.py` not found — check installation.",
        }
    return _get_ai_explanation(ticker, json.loads(indicators_json), gemini_key)


def _build_price_alert_email(trade: dict, current_price: float,
                              alert_type: str, sym: str) -> str:
    is_target = alert_type == "target"
    color     = "#00ff88" if is_target else "#ff4444"
    heading   = "🎯 TARGET HIT" if is_target else "🛑 STOP HIT"
    pnl       = (current_price - trade["entry_price"]) * trade["quantity"]
    pnl_pct   = (current_price - trade["entry_price"]) / trade["entry_price"] * 100
    level_lbl = "Target" if is_target else "Stop"
    level_val = trade["target_price"] if is_target else trade["stop_price"]
    return f"""
<html><body style="font-family:Arial,sans-serif;background:#111;color:#eee;padding:24px;max-width:560px;">
<h2 style="color:{color};margin-bottom:4px;">{heading}: {trade['ticker']}</h2>
<p style="color:#888;margin-top:0;">Paper trade alert — AI Trading Suite</p>
<table style="border-collapse:collapse;width:100%;margin-top:16px;">
  <tr style="background:#1e1e2e;">
    <td style="padding:10px 14px;border:1px solid #333;color:#aaa;">Current Price</td>
    <td style="padding:10px 14px;border:1px solid #333;font-weight:700;color:{color};">{sym}{current_price:.2f}</td>
  </tr>
  <tr>
    <td style="padding:10px 14px;border:1px solid #333;color:#aaa;">{level_lbl} Level</td>
    <td style="padding:10px 14px;border:1px solid #333;">{sym}{level_val:.2f}</td>
  </tr>
  <tr style="background:#1e1e2e;">
    <td style="padding:10px 14px;border:1px solid #333;color:#aaa;">Entry Price</td>
    <td style="padding:10px 14px;border:1px solid #333;">{sym}{trade['entry_price']:.2f}</td>
  </tr>
  <tr>
    <td style="padding:10px 14px;border:1px solid #333;color:#aaa;">Quantity</td>
    <td style="padding:10px 14px;border:1px solid #333;">{trade['quantity']} shares</td>
  </tr>
  <tr style="background:#1e1e2e;">
    <td style="padding:10px 14px;border:1px solid #333;color:#aaa;">Unrealised P&L</td>
    <td style="padding:10px 14px;border:1px solid #333;font-weight:700;color:{color};">
      {sym}{pnl:+,.0f} ({pnl_pct:+.1f}%)</td>
  </tr>
  <tr>
    <td style="padding:10px 14px;border:1px solid #333;color:#aaa;">Entry Date</td>
    <td style="padding:10px 14px;border:1px solid #333;">{trade.get('entry_date','—')}</td>
  </tr>
</table>
<p style="margin-top:18px;color:#aaa;">Consider {"closing for a profit" if is_target else "cutting the loss"} now.</p>
<p style="color:#555;font-size:0.75rem;">Automated alert · Not financial advice.</p>
</body></html>
"""


# ════════════════════════════════════════════════════════════════════════════
#  SETTINGS PERSISTENCE
# ════════════════════════════════════════════════════════════════════════════
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def _save_env(env_key: str, value: str) -> None:
    """Write one key=value to .env immediately (no quotes around value)."""
    try:
        from dotenv import set_key
        set_key(_ENV_PATH, env_key, value, quote_mode="never")
    except Exception:
        pass


# ── on_change callbacks — each fires when its widget changes ──────────────
def _cb_currency():        _save_env("CURRENCY",          st.session_state.get("cfg_currency",        "USD  ($)"))
def _cb_dashboard_mode():  _save_env("DASHBOARD_MODE",    st.session_state.get("cfg_dashboard_mode",  "🔍 Sector Scanner"))
def _cb_profit_pct():      _save_env("PROFIT_TARGET_PCT", str(st.session_state.get("cfg_profit_pct",  4.0)))
def _cb_stop_pct():        _save_env("STOP_LOSS_PCT",     str(st.session_state.get("cfg_stop_pct",    2.0)))
def _cb_sent_floor():      _save_env("SENTIMENT_FLOOR",   str(st.session_state.get("cfg_sent_floor",  0.1)))
def _cb_news_provider():   _save_env("NEWS_PROVIDER",     st.session_state.get("cfg_news_provider",   NEWS_PROVIDERS[0]))
def _cb_news_api_key():    _save_env("NEWS_API_KEY",      st.session_state.get("cfg_news_api_key",    ""))
def _cb_alert_enabled():   _save_env("ALERT_ENABLED",     str(st.session_state.get("cfg_alert_enabled", False)))
def _cb_alert_rcpt():      _save_env("EMAIL_RECIPIENT",   st.session_state.get("alert_rcpt",          ""))
def _cb_alert_from():      _save_env("EMAIL_SENDER",      st.session_state.get("alert_from",          ""))
def _cb_alert_pw():        _save_env("EMAIL_PASSWORD",    st.session_state.get("alert_pw",            ""))
def _cb_gemini_key():      _save_env("GEMINI_API_KEY",    st.session_state.get("gemini_key_input",    ""))


# ── one-time session init: load .env → session_state on fresh load/refresh ─
if "cfg_loaded" not in st.session_state:
    _np = os.getenv("NEWS_PROVIDER", NEWS_PROVIDERS[0])
    st.session_state["cfg_currency"]      = os.getenv("CURRENCY",          "USD  ($)")
    st.session_state["cfg_profit_pct"]    = float(os.getenv("PROFIT_TARGET_PCT", "4.0"))
    st.session_state["cfg_stop_pct"]      = float(os.getenv("STOP_LOSS_PCT",     "2.0"))
    st.session_state["cfg_sent_floor"]    = float(os.getenv("SENTIMENT_FLOOR",   "0.1"))
    st.session_state["cfg_news_provider"]   = _np if _np in NEWS_PROVIDERS else NEWS_PROVIDERS[0]
    st.session_state["cfg_news_api_key"]   = os.getenv("NEWS_API_KEY",         "")
    _MODES = ["🔍 Sector Scanner", "📊 Deep-Dive Analysis", "🧪 Backtest Lab",
              "🤖 ML Insights", "📓 Trade Journal", "📐 Position Sizer", "🎯 AI Picks",
              "🔎 Quick Analyze", "📁 My Portfolio"]
    _dm = os.getenv("DASHBOARD_MODE", "🔍 Sector Scanner")
    st.session_state["cfg_dashboard_mode"] = _dm if _dm in _MODES else "🔍 Sector Scanner"
    st.session_state["cfg_alert_enabled"]  = os.getenv("ALERT_ENABLED",       "False") == "True"
    st.session_state["alert_rcpt"]        = os.getenv("EMAIL_RECIPIENT",    "")
    st.session_state["alert_from"]        = os.getenv("EMAIL_SENDER",       "")
    st.session_state["alert_pw"]          = os.getenv("EMAIL_PASSWORD",     "")
    st.session_state["gemini_key_input"]  = os.getenv("GEMINI_API_KEY",     "")
    st.session_state["cfg_loaded"]        = True


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
if "watchlist" not in st.session_state:
    st.session_state.watchlist = load_watchlist()

ml_model, ml_scaler = load_ml_model()

# ════════════════════════════════════════════════════════════════════════════
#  TOP BAR  — mode, sector universe, watchlist
# ════════════════════════════════════════════════════════════════════════════

_tb_logo, _tb_mode, _tb_sector, _tb_watch = st.columns([1.2, 2.2, 2.4, 2.2])

with _tb_logo:
    st.markdown("### ⚡ AI Trading Suite")

with _tb_mode:
    dashboard_mode = st.selectbox(
        "Operating Mode",
        ["🔍 Sector Scanner", "📊 Deep-Dive Analysis", "🧪 Backtest Lab",
         "🤖 ML Insights", "📓 Trade Journal", "📐 Position Sizer", "🎯 AI Picks",
         "🔎 Quick Analyze", "📁 My Portfolio"],
        key="cfg_dashboard_mode", on_change=_cb_dashboard_mode,
    )

with _tb_sector:
    _sector_opts = (
        ["⭐ My Watchlist"] + list(SECTOR_UNIVERSES.keys())
        if st.session_state.watchlist else list(SECTOR_UNIVERSES.keys())
    )
    if "cfg_sector" not in st.session_state:
        st.session_state["cfg_sector"] = _sector_opts[0]
    selected_sector = st.selectbox(
        "Sector Universe", _sector_opts, key="cfg_sector",
    )

with _tb_watch:
    _wl_label = f"⭐ Watchlist  ({len(st.session_state.watchlist)})"
    with st.expander(_wl_label, expanded=False):
        wl = st.session_state.watchlist
        _wc1, _wc2 = st.columns([3, 1])
        _new_tick = _wc1.text_input(
            "wl_add_input", placeholder="e.g. INFY.NS",
            label_visibility="collapsed", key="wl_new_input",
        )
        if _wc2.button("Add", key="wl_add_btn", use_container_width=True):
            _cleaned = _new_tick.upper().strip()
            if _cleaned and _cleaned not in wl:
                wl = wl + [_cleaned]
                st.session_state.watchlist = wl
                save_watchlist(wl)
                st.rerun()
        if wl:
            _to_keep = st.multiselect(
                "Tickers (deselect to remove)",
                options=wl, default=wl, key="wl_keep",
            )
            if set(_to_keep) != set(wl):
                st.session_state.watchlist = _to_keep
                save_watchlist(_to_keep)
                st.rerun()
            st.caption(f"{len(wl)} ticker(s) · saved to watchlist.json")
        else:
            st.caption("Watchlist is empty.")

st.divider()

with st.sidebar:
    st.divider()

    # ── Currency toggle ───────────────────────────────────────────────────
    st.markdown("**Account Currency**")
    CURRENCY      = st.radio(
        "currency", ["USD  ($)", "INR  (₹)"],
        horizontal=True, label_visibility="collapsed",
        key="cfg_currency", on_change=_cb_currency,
    )
    IS_INR        = "INR" in CURRENCY
    GLOBAL_SYM    = "₹" if IS_INR else "$"
    JOURNAL_INITIAL = 50_000.0 if IS_INR else 600.0
    st.divider()

    # ── Strategy parameters ───────────────────────────────────────────────
    st.header("Strategy Parameters")
    PROFIT_TARGET_PCT  = st.slider("Profit Target (%)",  2.0, 10.0, 4.0, 0.5, key="cfg_profit_pct",  on_change=_cb_profit_pct)  / 100.0
    STOP_LOSS_PCT      = st.slider("Stop Loss (%)",       1.0,  5.0, 2.0, 0.5, key="cfg_stop_pct",   on_change=_cb_stop_pct)    / 100.0
    AI_SENTIMENT_FLOOR = st.slider("Sentiment Floor",    -0.5,  0.5, 0.1, 0.05, key="cfg_sent_floor", on_change=_cb_sent_floor)
    st.divider()

    # ── News source ───────────────────────────────────────────────────────
    st.header("News Source")
    news_provider = st.selectbox("Provider", NEWS_PROVIDERS, key="cfg_news_provider", on_change=_cb_news_provider)
    api_key = ""
    if "Key Required" in news_provider:
        api_key = st.text_input("API Key", type="password", key="cfg_news_api_key", on_change=_cb_news_api_key)
    st.divider()

    # ── Email alerts ──────────────────────────────────────────────────────
    with st.expander("📧 Email Alerts", expanded=False):
        alert_enabled = st.toggle("Send BUY signal alerts", key="cfg_alert_enabled", on_change=_cb_alert_enabled)
        alert_email   = st.text_input("Recipient email", placeholder="you@gmail.com",
                                      key="alert_rcpt", on_change=_cb_alert_rcpt)
        alert_sender  = st.text_input("Sender Gmail",    placeholder="bot@gmail.com",
                                      key="alert_from", on_change=_cb_alert_from)
        alert_pass    = st.text_input("Gmail App Password", type="password",
                                      key="alert_pw",   on_change=_cb_alert_pw)
        st.caption(
            "Requires a Gmail **App Password** (not your login password).  \n"
            "Generate at: myaccount.google.com → Security → App Passwords."
        )
    st.divider()

    # ── Gemini AI ─────────────────────────────────────────────────────────
    with st.expander("✨ Gemini AI", expanded=False):
        gemini_key = st.text_input(
            "Gemini API Key", type="password", key="gemini_key_input",
            placeholder="AIza…", on_change=_cb_gemini_key,
        )
        st.caption(
            "Powers AI analysis in Deep-Dive and AI Picks modes.  \n"
            "Free key at **aistudio.google.com** → Get API key."
        )
    st.divider()

    # ── Universe Manager ──────────────────────────────────────────────────
    with st.expander(f"🧬 Universe Manager  ({len(AI_PICKS_UNIVERSE)} stocks)", expanded=False):
        st.caption(
            "Add a new stock to the trained universe and retrain the ML model.  \n"
            "Takes ~30–60 seconds. Current model stays active until retrain completes."
        )
        _um_ticker = st.text_input(
            "New ticker (yfinance format)",
            placeholder="e.g. ADANIENT.NS · NVDL · SMCI",
            key="um_ticker_input",
        ).upper().strip()

        if st.button("Add & Retrain", key="um_retrain_btn", type="primary",
                     use_container_width=True):
            if not _um_ticker:
                st.error("Enter a ticker symbol.")
            elif _um_ticker in AI_PICKS_UNIVERSE:
                st.warning(f"**{_um_ticker}** is already in the universe.")
            else:
                # Step 1: validate via yfinance
                with st.spinner(f"Validating {_um_ticker}…"):
                    _um_test = get_data(_um_ticker, days=60)

                if _um_test.empty:
                    st.error(
                        f"**{_um_ticker}** not found or returned no data.  "
                        "Check the symbol (yfinance format) and try again."
                    )
                else:
                    _um_new_universe = AI_PICKS_UNIVERSE + [_um_ticker]
                    save_universe(_um_new_universe)

                    # Step 2: retrain — current model is backed up inside train_on_universe()
                    with st.spinner(
                        f"Retraining model on {len(_um_new_universe)} stocks…  "
                        "(30–60 s, please wait)"
                    ):
                        try:
                            from train_model import train_on_universe as _retrain
                            _ok, _msg = _retrain(_um_new_universe)
                        except Exception as _exc:
                            _ok, _msg = False, str(_exc)

                    if _ok:
                        load_ml_model.clear()   # bust @st.cache_resource
                        st.success(
                            f"✅ **{_um_ticker}** added.  "
                            f"Model retrained on **{len(_um_new_universe)} stocks**.  \n"
                            f"_{_msg}_"
                        )
                        st.rerun()
                    else:
                        # universe.json already updated — keep the new ticker there
                        # but log retrain failure so user can investigate
                        st.error(
                            f"Retrain failed — previous model preserved.  \n"
                            f"**Error:** `{_msg}`  \n"
                            f"Ticker saved to universe; retry with **Add & Retrain** later."
                        )

        # Show current universe list
        with st.expander("📋 Current universe", expanded=False):
            for _ut in AI_PICKS_UNIVERSE:
                st.caption(_ut)
    st.divider()

    # ── Mode-specific controls ────────────────────────────────────────────
    if dashboard_mode == "🔍 Sector Scanner":
        st.header("Scanner Settings")
        st.caption(f"Universe: **{selected_sector}**")

    elif dashboard_mode == "🧪 Backtest Lab":
        st.header("Backtest Settings")
        _bt_default = "RELIANCE.NS" if IS_INR else "AAPL"
        bt_ticker     = st.text_input("Ticker", _bt_default).upper().strip()
        bt_start      = st.date_input("Start Date", value=date(2020, 1, 1))
        bt_end        = st.date_input("End Date",   value=date(2025, 1, 1))
        bt_cash       = st.number_input(f"Starting Cash ({GLOBAL_SYM})",
                                         100, 1_000_000, int(JOURNAL_INITIAL), 1_000)
        bt_commission = st.number_input("Commission (%)", 0.0, 1.0, 0.10, 0.01) / 100.0
        run_bt        = st.button("▶  Run Backtest", type="primary", use_container_width=True)

    st.divider()
    if ml_model is not None:
        st.success("ML Model  ✓  Loaded")
    else:
        st.warning("ML Model: not found\n`python train_model.py`")

    # ── Persist all sidebar values to .env on every rerun ────────────────
    _save_env("GEMINI_API_KEY",    gemini_key)
    _save_env("EMAIL_RECIPIENT",   alert_email)
    _save_env("EMAIL_SENDER",      alert_sender)
    _save_env("EMAIL_PASSWORD",    alert_pass)
    _save_env("ALERT_ENABLED",     str(alert_enabled))
    _save_env("CURRENCY",          CURRENCY)
    _save_env("PROFIT_TARGET_PCT", str(st.session_state.get("cfg_profit_pct", 4.0)))
    _save_env("STOP_LOSS_PCT",     str(st.session_state.get("cfg_stop_pct",   2.0)))
    _save_env("SENTIMENT_FLOOR",   str(st.session_state.get("cfg_sent_floor", 0.1)))
    _save_env("NEWS_PROVIDER",     news_provider)
    _save_env("NEWS_API_KEY",      api_key)
    _save_env("DASHBOARD_MODE",    dashboard_mode)

    # ── Auto-refresh countdown (JS — reloads page every 5 min) ───────────
    st.divider()
    components.html("""
<div style="font-family:sans-serif;color:#777;font-size:0.70rem;
            text-align:center;padding:2px 0;line-height:1.6;">
  ⏱ Next refresh: <span id="cd" style="color:#aaa;font-weight:600;">5:00</span>
</div>
<script>
  var rem = 300;
  var el  = document.getElementById('cd');
  var iv  = setInterval(function(){
    rem--;
    var m = Math.floor(rem/60), s = rem%60;
    el.textContent = m+':'+(s<10?'0':'')+s;
    if(rem <= 0){ clearInterval(iv); window.top.location.reload(); }
  }, 1000);
</script>
""", height=28)

# ════════════════════════════════════════════════════════════════════════════
#  MODE 1 — SECTOR SCANNER
# ════════════════════════════════════════════════════════════════════════════
if dashboard_mode == "🔍 Sector Scanner":
    if selected_sector == "⭐ My Watchlist":
        tickers = list(st.session_state.watchlist)
        if not tickers:
            st.warning("Your watchlist is empty. Add tickers via the ⭐ **My Watchlist** expander in the sidebar.")
            st.stop()
    else:
        tickers = SECTOR_UNIVERSES[selected_sector]
    # Auto-pick currency by majority of tickers
    ns_count = sum(1 for t in tickers if ".NS" in t)
    sym = "₹" if ns_count > len(tickers) / 2 else "$"

    st.title(f"🔍 Sector Scanner — {selected_sector}")
    st.caption(
        f"Scanning **{len(tickers)}** tickers via **{news_provider}**  ·  "
        f"Target +{PROFIT_TARGET_PCT*100:.1f}%  ·  Stop -{STOP_LOSS_PCT*100:.1f}%  ·  "
        f"Sentiment ≥ {AI_SENTIMENT_FLOOR:+.2f}"
    )

    scan_results = []
    rejected     = []
    prog         = st.progress(0, text="Starting scan…")

    for idx, t in enumerate(tickers):
        prog.progress((idx + 1) / len(tickers), text=f"Scanning {t}…")

        df = get_data(t)
        if df.empty or len(df) < 50:
            rejected.append({"Ticker": t, "Reason": "Insufficient data"})
            time.sleep(0.3)
            continue

        df = compute_indicators(df)
        if df.empty:
            rejected.append({"Ticker": t, "Reason": "Indicator error"})
            continue

        last = df.iloc[-1]
        is_trending = float(last["ADX"]) > 22
        tech_pass   = (
            (is_trending and float(last["Close"]) > float(last["EMA_10"])
             and float(last["Volume"]) > float(last["Vol_SMA"]) * 1.2)
            or (not is_trending and float(last["RSI"]) < 35)
        )

        if not tech_pass:
            rejected.append({
                "Ticker": t,
                "Reason": f"Tech filter: ADX={float(last['ADX']):.1f}, RSI={float(last['RSI']):.1f}",
            })
            time.sleep(0.3)
            continue

        sent_score, _ = get_sentiment(t, news_provider, api_key)
        if sent_score < AI_SENTIMENT_FLOOR:
            rejected.append({"Ticker": t, "Reason": f"Sentiment {sent_score:+.2f} < floor"})
            continue

        sig = generate_signal(last, PROFIT_TARGET_PCT, STOP_LOSS_PCT,
                              sent_score, AI_SENTIMENT_FLOOR, ml_model, ml_scaler)

        ml_str = f"{sig['ml_prob']:.1%}" if sig["ml_prob"] is not None else "—"
        scan_results.append({
            "Ticker":      t,
            "Setup":       sig["regime"],
            "Signal":      sig["verdict"],
            "Entry":       sig["price"],
            "Target":      sig["target_price"],
            "Stop":        sig["stop_price"],
            "R:R":         sig["rr_ratio"],
            "Sentiment":   sent_score,
            "ML Prob":     sig["ml_prob"],
            "_ml_str":     ml_str,
            "Confidence":  sig["confidence"],
            "_color":      sig["color"],
        })
        time.sleep(0.8)

    prog.empty()

    if scan_results:
        scan_results.sort(key=lambda x: x["Confidence"], reverse=True)
        top = scan_results[0]

        # ── TOP PICK ──────────────────────────────────────────────────────
        reward = top["Target"] - top["Entry"]
        risk   = top["Entry"]  - top["Stop"]
        signal_banner(
            {"verdict": f"🏆 Top Pick: {top['Ticker']}  —  {top['Setup']}",
             "color": top["_color"], "confidence": top["Confidence"], "regime": ""},
        )

        # Email alert — once per ticker per calendar day
        if alert_enabled and alert_email and alert_sender and alert_pass:
            _ak = f"alerted_{top['Ticker']}_{date.today()}"
            if _ak not in st.session_state:
                _body = _build_alert_email(
                    top["Ticker"], top["_color"], top["Setup"],
                    top["Entry"], top["Target"], top["Stop"],
                    top["Sentiment"], top["_ml_str"], sym,
                )
                if send_email_alert(alert_email, f"⚡ BUY Signal: {top['Ticker']}", _body,
                                    alert_sender, alert_pass):
                    st.toast(f"📧 Alert sent for {top['Ticker']}", icon="📧")
                    st.session_state[_ak] = True
                else:
                    st.warning("Email alert failed — check your Gmail App Password in the sidebar.")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Entry",     f"{sym}{top['Entry']:.2f}")
        c2.metric("Target",    f"{sym}{top['Target']:.2f}", f"+{reward/top['Entry']*100:.2f}%")
        c3.metric("Stop",      f"{sym}{top['Stop']:.2f}",   f"-{risk/top['Entry']*100:.2f}%")
        c4.metric("R:R Ratio", f"{top['R:R']:.2f}")
        c5.metric("ML Prob",   top["_ml_str"])

        st.divider()

        # ── RESULTS TABLE ─────────────────────────────────────────────────
        st.subheader(f"All Qualifying Candidates ({len(scan_results)} of {len(tickers)})")
        display_rows = []
        for d in scan_results:
            display_rows.append({
                "Ticker":     d["Ticker"],
                "Setup":      d["Setup"],
                "Signal":     d["Signal"],
                "Entry":      f"{sym}{d['Entry']:.2f}",
                "Target":     f"{sym}{d['Target']:.2f}",
                "Stop":       f"{sym}{d['Stop']:.2f}",
                "R:R":        d["R:R"],
                "Sentiment":  f"{d['Sentiment']:+.2f}",
                "ML Prob":    d["_ml_str"],
                "Confidence": f"{d['Confidence']:.0f}%",
            })
        st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

        health = len(scan_results) / len(tickers) * 100
        st.metric("Sector Health Score",
                  f"{health:.0f}%",
                  f"{len(scan_results)}/{len(tickers)} passed all filters")

    else:
        st.warning(
            f"No tickers in **{selected_sector}** passed all filters. "
            "Try lowering the Sentiment Floor or checking market hours."
        )

    if rejected:
        with st.expander(f"Filtered out — {len(rejected)} tickers"):
            st.dataframe(pd.DataFrame(rejected), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
#  MODE 2 — DEEP-DIVE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif dashboard_mode == "📊 Deep-Dive Analysis":
    # ── Ticker + window selectors (universe-restricted; full ML available) ─
    _dd_c1, _dd_c2 = st.columns([3, 1])
    try:
        _dd_default_idx = AI_PICKS_UNIVERSE.index("RELIANCE.NS" if IS_INR else "NVDA")
    except ValueError:
        _dd_default_idx = 0
    ticker_input = _dd_c1.selectbox(
        "Ticker", AI_PICKS_UNIVERSE, index=_dd_default_idx, key="dd_ticker",
    )
    chart_window = _dd_c2.selectbox(
        "Chart Window", ["3 months", "6 months", "1 year"], index=2, key="dd_window",
    )
    st.caption(
        f"Full ML analysis — {len(AI_PICKS_UNIVERSE)}-stock trained universe.  "
        "For any other ticker use **🔎 Quick Analyze**."
    )

    days_map = {"3 months": 90, "6 months": 180, "1 year": 365}
    chart_days = days_map[chart_window]
    sym = currency_sym(ticker_input)

    st.title(f"📊 Deep-Dive: {ticker_input}")

    df = get_data(ticker_input, days=max(chart_days + 120, 400))
    if df.empty:
        st.error(f"No data for **{ticker_input}**. Check the ticker symbol and try again.")
        st.stop()

    df = compute_indicators(df)
    if df.empty:
        st.error("Not enough historical data to compute indicators (need ≥ 30 bars).")
        st.stop()

    last    = df.iloc[-1]
    price   = float(last["Close"])
    prev    = float(df["Close"].iloc[-2]) if len(df) >= 2 else price
    pct_chg = (price - prev) / prev * 100

    sent_score, headlines = get_sentiment(ticker_input, news_provider, api_key)

    sig = generate_signal(last, PROFIT_TARGET_PCT, STOP_LOSS_PCT,
                          sent_score, AI_SENTIMENT_FLOOR, ml_model, ml_scaler)

    # ── METRICS ROW ───────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Price",       f"{sym}{price:.2f}",               f"{pct_chg:+.2f}%")
    c2.metric("Take Profit", f"{sym}{sig['target_price']:.2f}",
              f"+{(sig['target_price']-price)/price*100:.2f}%")
    c3.metric("Stop Loss",   f"{sym}{sig['stop_price']:.2f}",
              f"-{(price-sig['stop_price'])/price*100:.2f}%")
    c4.metric("R:R Ratio",   f"{sig['rr_ratio']:.2f}")
    ml_disp = f"{sig['ml_prob']:.1%}" if sig["ml_prob"] is not None else "N/A"
    c5.metric("ML Probability", ml_disp)

    rsi_v = float(last["RSI"])
    adx_v = float(last["ADX"])
    signal_banner(sig, f"  ·  RSI: {rsi_v:.1f}  ·  ADX: {adx_v:.1f}  ·  Sentiment: {sent_score:+.2f}")

    # Email alert — fires once per ticker per calendar day on BUY
    if sig["verdict"].startswith("BUY") and alert_enabled and alert_email and alert_sender and alert_pass:
        _ak = f"alerted_{ticker_input}_{date.today()}"
        if _ak not in st.session_state:
            _body = _build_alert_email(
                ticker_input, sig["color"], sig["regime"],
                sig["price"], sig["target_price"], sig["stop_price"],
                sent_score, ml_disp, sym,
            )
            if send_email_alert(alert_email, f"⚡ BUY Signal: {ticker_input}", _body,
                                alert_sender, alert_pass):
                st.toast(f"📧 Alert sent for {ticker_input}", icon="📧")
                st.session_state[_ak] = True
            else:
                st.warning("Email alert failed — check your Gmail App Password in the sidebar.")

    st.divider()

    # ── 3-PANEL CHART ─────────────────────────────────────────────────────
    display_df = df.tail(chart_days)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.60, 0.20, 0.20],
        vertical_spacing=0.025,
        subplot_titles=("Price & Indicators", "RSI (14)", "Volume"),
    )

    fig.add_trace(go.Candlestick(
        x=display_df.index,
        open=display_df["Open"], high=display_df["High"],
        low=display_df["Low"],   close=display_df["Close"],
        name="Price",
        increasing_line_color="#00ff88", decreasing_line_color="#ff4444",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=display_df.index, y=display_df["EMA_10"],
        line=dict(color="#f7c948", width=1.5), name="EMA 10",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=display_df.index, y=display_df["SMA_50"],
        line=dict(color="#60b4ff", width=1.0, dash="dot"), name="SMA 50",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=display_df.index, y=display_df["SMA_200"],
        line=dict(color="#bf94e4", width=1.0, dash="dash"), name="SMA 200",
    ), row=1, col=1)

    fig.add_hline(y=sig["target_price"], line_dash="dash", line_color="#00ff88",
                  annotation_text=f"Target {sym}{sig['target_price']:.2f}",
                  annotation_font_color="#00ff88", row=1, col=1)
    fig.add_hline(y=sig["stop_price"], line_dash="dash", line_color="#ff4444",
                  annotation_text=f"Stop {sym}{sig['stop_price']:.2f}",
                  annotation_font_color="#ff4444", row=1, col=1)

    fig.add_trace(go.Scatter(
        x=display_df.index, y=display_df["RSI"],
        line=dict(color="#bf94e4", width=1.5), name="RSI",
    ), row=2, col=1)
    fig.add_hline(y=70, line_color="#ff4444", line_dash="dot", row=2, col=1)
    fig.add_hline(y=30, line_color="#00ff88", line_dash="dot", row=2, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.03)",
                  line_width=0, row=2, col=1)

    bar_colors = [
        "#00ff88" if c >= o else "#ff4444"
        for c, o in zip(display_df["Close"], display_df["Open"])
    ]
    fig.add_trace(go.Bar(
        x=display_df.index, y=display_df["Volume"],
        marker_color=bar_colors, name="Volume", showlegend=False,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=display_df.index, y=display_df["Vol_SMA"],
        line=dict(color="white", width=1, dash="dot"), name="Vol SMA 20",
    ), row=3, col=1)

    fig.update_layout(
        height=700, template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(t=50, l=0, r=0, b=0),
    )
    fig.update_yaxes(title_text="Price",  row=1, col=1)
    fig.update_yaxes(title_text="RSI",    row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="Volume", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # ── TECHNICAL BREAKDOWN + CONFIDENCE GAUGE ────────────────────────────
    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        st.subheader("Technical Snapshot")
        vol_ratio = float(last.get("Volume_Ratio", 0))
        ema_diff  = float(last.get("Price_vs_EMA10", 0)) * 100
        snapshot  = pd.DataFrame([
            {"Indicator": "RSI (14)",       "Value": f"{rsi_v:.1f}",
             "Status": "Overbought" if rsi_v > 70 else "Oversold" if rsi_v < 30 else "Neutral"},
            {"Indicator": "ADX (14)",       "Value": f"{adx_v:.1f}",
             "Status": "Strong Trend" if adx_v > 25 else "No Clear Trend"},
            {"Indicator": "ATR (14)",       "Value": f"{sym}{float(last['ATR']):.2f}",
             "Status": "Volatility measure"},
            {"Indicator": "Volume Ratio",   "Value": f"{vol_ratio:.2f}×",
             "Status": "Above Avg" if vol_ratio > 1.2 else "Below Avg"},
            {"Indicator": "Price vs EMA10", "Value": f"{ema_diff:+.2f}%",
             "Status": "Above EMA" if ema_diff > 0 else "Below EMA"},
            {"Indicator": "MA Trend",       "Value": "Uptrend" if float(last.get("MA_Trend", 0)) else "Downtrend",
             "Status": "SMA50 vs SMA200"},
            {"Indicator": "MACD Cross",     "Value": "Bullish" if float(last.get("MACD_Cross", 0)) else "Bearish",
             "Status": "MACD vs Signal"},
            {"Indicator": "Sentiment",      "Value": f"{sent_score:+.2f}",
             "Status": "Bullish" if sent_score >= 0.05 else "Bearish" if sent_score <= -0.05 else "Neutral"},
        ])
        st.dataframe(snapshot, use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("Signal Confidence")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sig["confidence"],
            title={"text": f"{sig['verdict']}"},
            gauge={
                "axis":  {"range": [0, 100]},
                "bar":   {"color": sig["color"]},
                "steps": [
                    {"range": [0,  40], "color": "rgba(255,68,68,0.15)"},
                    {"range": [40, 60], "color": "rgba(255,170,0,0.15)"},
                    {"range": [60,100], "color": "rgba(0,255,136,0.15)"},
                ],
                "threshold": {
                    "line":      {"color": "white", "width": 2},
                    "thickness": 0.75,
                    "value":     60,
                },
            },
            number={"suffix": "%"},
        ))
        fig_gauge.update_layout(
            height=280, template="plotly_dark",
            margin=dict(t=50, b=10, l=20, r=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── NEWS HEADLINES ────────────────────────────────────────────────────
    st.divider()
    with st.expander(f"📰 {news_provider} Headlines — {ticker_input}"):
        render_headlines(headlines)

    # ── GEMINI AI ANALYSIS ────────────────────────────────────────────────
    if gemini_key:
        st.divider()
        with st.expander("✨ Gemini AI Analysis", expanded=True):
            _prompt = f"""You are a short-term trading analyst. Analyze this setup concisely:

Ticker: {ticker_input}
Price: {sym}{price:.2f} ({pct_chg:+.2f}% today)
Signal: {sig['verdict']} | Regime: {sig['regime']}
RSI: {rsi_v:.1f} | ADX: {adx_v:.1f} | Confidence: {sig['confidence']:.0f}%
Target: {sym}{sig['target_price']:.2f} | Stop: {sym}{sig['stop_price']:.2f} | R:R: {sig['rr_ratio']:.2f}
News Sentiment: {sent_score:+.2f} | ML Probability: {ml_disp}

Write 4-5 focused sentences covering:
1. What the technicals are saying right now
2. Whether the entry makes sense at this level
3. Key risk factors specific to this setup
4. Your overall take — worth taking or wait?
Be direct and specific. Skip boilerplate disclaimers."""
            with st.spinner("Asking Gemini…"):
                _analysis = get_gemini_analysis(ticker_input, _prompt, gemini_key)
            st.markdown(_analysis)
    elif not gemini_key:
        st.caption("_Add a Gemini API key in the sidebar to enable AI analysis._")


# ════════════════════════════════════════════════════════════════════════════
#  MODE 3 — BACKTEST LAB
# ════════════════════════════════════════════════════════════════════════════
elif dashboard_mode == "🧪 Backtest Lab":
    st.title("🧪 Backtest Lab")

    if not _BT_OK:
        st.error("The `backtesting` library is not installed.  Run: `pip install backtesting`")
        st.stop()

    st.info(
        f"**Strategy:** TrailingStopStrategy  ·  **Ticker:** {bt_ticker}  ·  "
        f"**Period:** {bt_start} → {bt_end}  ·  "
        f"**Cash:** ${bt_cash:,}  ·  **Commission:** {bt_commission*100:.2f}%"
    )

    if not run_bt:
        st.markdown("""
        #### How the strategy works
        | Regime | Entry trigger | Stop | Target |
        |--------|--------------|------|--------|
        | **Trending** (ADX > 25) | SMA50 > SMA200 + MACD bullish cross | 2× ATR trailing | 4× ATR |
        | **Choppy** (ADX ≤ 25)  | RSI < 35 oversold bounce | 2× ATR trailing | 4× ATR |

        - **Breakeven stop** activates when price reaches 50% of the distance to target
        - RSI > 65 triggers early exit in choppy regime
        """)
        st.info("Configure the ticker and date range in the sidebar, then click **▶ Run Backtest**.")
        st.stop()

    # Run
    with st.spinner(f"Downloading {bt_ticker} and running backtest…"):
        raw = yf.download(bt_ticker, start=str(bt_start), end=str(bt_end), progress=False)
        if raw.empty or len(raw) < 220:
            st.error(
                "Not enough data (need ≥ 220 bars). "
                "Try a longer date range or a different ticker."
            )
            st.stop()
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
        raw = raw.dropna()

        try:
            bt     = Backtest(raw, TrailingStopStrategy, cash=bt_cash, commission=bt_commission)
            stats  = bt.run()
        except Exception as e:
            st.error(f"Backtest error: {e}")
            st.stop()

    st.success(f"Backtest complete — {len(raw)} bars processed.")

    # ── SUMMARY METRICS ───────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return",    f"{stats['Return [%]']:.2f}%")
    c2.metric("Buy & Hold",      f"{stats['Buy & Hold Return [%]']:.2f}%")
    c3.metric("Max Drawdown",    f"{stats['Max. Drawdown [%]']:.2f}%")
    c4.metric("Sharpe Ratio",    f"{stats.get('Sharpe Ratio', 0):.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("# Trades",        stats["# Trades"])
    c6.metric("Win Rate",        f"{stats['Win Rate [%]']:.1f}%")
    c7.metric("Best Trade",      f"{stats.get('Best Trade [%]', 0):.2f}%")
    c8.metric("Worst Trade",     f"{stats.get('Worst Trade [%]', 0):.2f}%")

    st.divider()

    # ── EQUITY CURVE + DRAWDOWN ───────────────────────────────────────────
    eq_curve = stats["_equity_curve"]
    equity   = eq_curve["Equity"]
    drawdown = (equity / equity.cummax() - 1) * 100  # % drawdown, negative

    fig_eq = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.70, 0.30],
        vertical_spacing=0.04,
        subplot_titles=("Portfolio Equity", "Drawdown (%)"),
    )
    fig_eq.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        fill="tozeroy", fillcolor="rgba(96,180,255,0.15)",
        line=dict(color="#60b4ff", width=2),
        name="Equity",
    ), row=1, col=1)
    fig_eq.add_hline(y=bt_cash, line_dash="dash", line_color="gray",
                     annotation_text="Starting Capital", row=1, col=1)

    fig_eq.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        fill="tozeroy", fillcolor="rgba(255,68,68,0.2)",
        line=dict(color="#ff4444", width=1),
        name="Drawdown %",
    ), row=2, col=1)

    fig_eq.update_layout(
        height=520, template="plotly_dark",
        showlegend=False,
        margin=dict(t=50, l=0, r=0, b=0),
    )
    fig_eq.update_yaxes(title_text=f"Portfolio ({currency_sym(bt_ticker)})", row=1, col=1)
    fig_eq.update_yaxes(title_text="Drawdown %", row=2, col=1)
    st.plotly_chart(fig_eq, use_container_width=True)

    # ── TRADES TABLE ──────────────────────────────────────────────────────
    trades = stats["_trades"]
    if not trades.empty:
        with st.expander(f"📋 All {len(trades)} Trades"):
            keep = [c for c in
                    ["EntryTime", "ExitTime", "EntryPrice", "ExitPrice",
                     "PnL", "ReturnPct", "Duration"]
                    if c in trades.columns]
            st.dataframe(trades[keep].round(3), use_container_width=True)

    # ── FULL STATS ────────────────────────────────────────────────────────
    with st.expander("📑 Full Statistics"):
        rows = [(k, v) for k, v in stats.items() if not str(k).startswith("_")]
        st.dataframe(
            pd.DataFrame(rows, columns=["Metric", "Value"]),
            use_container_width=True, hide_index=True,
        )


# ════════════════════════════════════════════════════════════════════════════
#  MODE 4 — ML INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
elif dashboard_mode == "🤖 ML Insights":
    st.title("🤖 ML Model Insights")

    if ml_model is None:
        st.warning(
            "No trained model found. Run `python train_model.py` in this directory "
            "to generate `trade_signal_model.pkl` and `feature_scaler.pkl`."
        )
        st.markdown("""
        **What the model learns:**
        - 8 hand-crafted features from price, volume, and momentum indicators
        - Label: did price hit **+4%** before **−2%** within **5 trading days**?
        - Algorithm: `GradientBoostingClassifier` (300 trees, max depth 4)
        - Training universe: 44 tickers — 24 US equities + 20 Nifty-50 NSE large-caps
        - Training window: 2015–2024  ·  Held-out test: 2025
        """)
    else:
        importances = sorted(
            zip(ML_FEATURE_COLS, ml_model.feature_importances_),
            key=lambda x: x[1],
        )
        names  = [i[0] for i in importances]
        values = [i[1] for i in importances]

        fig_imp = go.Figure(go.Bar(
            x=values, y=names,
            orientation="h",
            marker=dict(
                color=values,
                colorscale="Blues",
                showscale=False,
            ),
        ))
        fig_imp.update_layout(
            title="Feature Importances — GradientBoostingClassifier",
            xaxis_title="Importance",
            height=380, template="plotly_dark",
            margin=dict(t=50, l=0, r=0, b=0),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Feature Descriptions")
            desc_df = pd.DataFrame([
                ("RSI",             "14-period Relative Strength Index — momentum oscillator"),
                ("MACD_Hist",       "MACD histogram — direction and strength of momentum"),
                ("ADX",             "Average Directional Index — trend strength (>25 = strong)"),
                ("ATR_Pct",         "ATR / Close price — normalised relative volatility"),
                ("Price_vs_EMA10",  "% deviation of Close from 10-EMA — short-term positioning"),
                ("Volume_Ratio",    "Volume ÷ 20-day SMA volume — participation level"),
                ("MA_Trend",        "Binary: SMA50 > SMA200 (long-term uptrend flag)"),
                ("MACD_Cross",      "Binary: MACD line > Signal line (bullish momentum flag)"),
            ], columns=["Feature", "Description"])
            st.dataframe(desc_df, use_container_width=True, hide_index=True)

        with col_b:
            st.subheader("Training Configuration")
            cfg_df = pd.DataFrame([
                ("Algorithm",      "GradientBoostingClassifier"),
                ("n_estimators",   "300"),
                ("max_depth",      "4"),
                ("learning_rate",  "0.05"),
                ("subsample",      "0.8"),
                ("min_samples_leaf","20"),
                ("Training period","2015-01-01 → 2025-01-01"),
                ("Test period",    "2025-01-01 → 2026-01-01"),
                ("Label horizon",  "5 trading days"),
                ("Profit target",  "+4%  (matches dashboard default)"),
                ("Stop level",     "−2%  (matches dashboard default)"),
                ("Universe",       "44 tickers: 24 US + 20 NSE"),
            ], columns=["Parameter", "Value"])
            st.dataframe(cfg_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Live Score — Quick Scan")

        presets = (
            ["NVDA", "AAPL", "MSFT", "GOOGL", "TSLA", "AMD", "META", "AMZN"]
            + ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
        )
        quick_tickers = st.multiselect(
            "Tickers to score",
            presets,
            default=["NVDA", "AAPL", "TSLA"],
        )

        if quick_tickers and st.button("▶  Score Tickers", type="primary"):
            results = []
            for t in quick_tickers:
                df_q = get_data(t)
                if df_q.empty or len(df_q) < 50:
                    continue
                df_q = compute_indicators(df_q)
                if df_q.empty:
                    continue
                last_q = df_q.iloc[-1]
                feats  = np.array([[
                    float(last_q.get("RSI",            50)),
                    float(last_q.get("MACD_Hist",      0)),
                    float(last_q.get("ADX",            20)),
                    float(last_q.get("ATR_Pct",        0.01)),
                    float(last_q.get("Price_vs_EMA10", 0)),
                    float(last_q.get("Volume_Ratio",   1)),
                    float(last_q.get("MA_Trend",       0)),
                    float(last_q.get("MACD_Cross",     0)),
                ]])
                prob   = float(ml_model.predict_proba(ml_scaler.transform(feats))[0][1])
                s      = currency_sym(t)
                results.append({
                    "Ticker":  t,
                    "Price":   f"{s}{float(last_q['Close']):.2f}",
                    "RSI":     f"{float(last_q.get('RSI', 50)):.1f}",
                    "ADX":     f"{float(last_q.get('ADX', 20)):.1f}",
                    "ML Prob": f"{prob:.1%}",
                    "Signal":  "BUY" if prob >= 0.55 else "HOLD" if prob >= 0.45 else "SKIP",
                })
            if results:
                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
            else:
                st.warning("Could not retrieve data for selected tickers.")

    st.divider()
    st.caption(
        "Model files: `trade_signal_model.pkl` + `feature_scaler.pkl`  ·  "
        "Generate with: `python train_model.py`"
    )


# ════════════════════════════════════════════════════════════════════════════
#  MODE 5 — TRADE JOURNAL
# ════════════════════════════════════════════════════════════════════════════
elif dashboard_mode == "📓 Trade Journal":
    JOURNAL_FILE     = "trade_journal.json"
    INITIAL_BALANCE  = JOURNAL_INITIAL   # follows the sidebar currency toggle

    def _load_journal() -> dict:
        if os.path.exists(JOURNAL_FILE):
            try:
                with open(JOURNAL_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"initial_balance": INITIAL_BALANCE, "cash": INITIAL_BALANCE, "trades": []}

    def _save_journal(data: dict) -> None:
        with open(JOURNAL_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    if "journal" not in st.session_state:
        st.session_state.journal = _load_journal()

    j            = st.session_state.journal
    all_trades   = j["trades"]
    open_trades  = [t for t in all_trades if t["status"] == "OPEN"]
    closed_trades= [t for t in all_trades if t["status"] != "OPEN"]

    # Live prices for all open positions (cached 60 s)
    current_prices: dict[str, float] = {}
    for ot in open_trades:
        tick = ot["ticker"]
        if tick not in current_prices:
            cp = get_current_price(tick)
            current_prices[tick] = cp if cp is not None else ot["entry_price"]

    # Portfolio mathematics
    open_cost    = sum(t["investment"] for t in open_trades)
    live_value   = sum(current_prices[t["ticker"]] * t["quantity"] for t in open_trades)
    unrealized   = live_value - open_cost
    realized     = sum(t.get("pnl_inr") or 0 for t in closed_trades)
    portfolio_val= j["cash"] + live_value
    total_pnl    = portfolio_val - INITIAL_BALANCE

    # ── PRICE ALERTS — check every open trade against target / stop ───────
    _price_alerts: list[tuple] = []
    for _t in open_trades:
        _cp  = current_prices[_t["ticker"]]
        _tid = _t["id"]
        if _cp >= _t["target_price"]:
            _ak = f"price_alerted_tp_{_tid}"
            if _ak not in st.session_state:
                _price_alerts.append(("target", _t, _cp))
                st.session_state[_ak] = True
                if alert_enabled and alert_email and alert_sender and alert_pass:
                    _subj = f"🎯 Target Hit: {_t['ticker']} @ {GLOBAL_SYM}{_cp:.2f}"
                    send_email_alert(
                        alert_email, _subj,
                        _build_price_alert_email(_t, _cp, "target", GLOBAL_SYM),
                        alert_sender, alert_pass,
                    )
        elif _cp <= _t["stop_price"]:
            _ak = f"price_alerted_sl_{_tid}"
            if _ak not in st.session_state:
                _price_alerts.append(("stop", _t, _cp))
                st.session_state[_ak] = True
                if alert_enabled and alert_email and alert_sender and alert_pass:
                    _subj = f"🛑 Stop Hit: {_t['ticker']} @ {GLOBAL_SYM}{_cp:.2f}"
                    send_email_alert(
                        alert_email, _subj,
                        _build_price_alert_email(_t, _cp, "stop", GLOBAL_SYM),
                        alert_sender, alert_pass,
                    )

    # ── HEADER KPIs ───────────────────────────────────────────────────────
    st.title("📓 Trade Journal")
    st.caption(f"Paper trading — simulated {GLOBAL_SYM}{INITIAL_BALANCE:,.0f} starting balance  ·  Currency: {CURRENCY}")

    # Render price-alert banners directly under the title
    for _alert_type, _trade, _cp in _price_alerts:
        _pnl     = (_cp - _trade["entry_price"]) * _trade["quantity"]
        _pnl_pct = (_cp - _trade["entry_price"]) / _trade["entry_price"] * 100
        if _alert_type == "target":
            st.success(
                f"🎯 **TARGET HIT — {_trade['ticker']}** reached "
                f"**{GLOBAL_SYM}{_cp:.2f}** (target was {GLOBAL_SYM}{_trade['target_price']:.2f})  ·  "
                f"Unrealised P&L: **+{GLOBAL_SYM}{_pnl:,.0f}** ({_pnl_pct:+.1f}%)  ·  "
                f"Go to Open Positions → Close Position."
            )
        else:
            st.error(
                f"🛑 **STOP HIT — {_trade['ticker']}** dropped to "
                f"**{GLOBAL_SYM}{_cp:.2f}** (stop was {GLOBAL_SYM}{_trade['stop_price']:.2f})  ·  "
                f"Unrealised loss: **{GLOBAL_SYM}{_pnl:,.0f}** ({_pnl_pct:.1f}%)  ·  "
                f"Go to Open Positions → Close Position."
            )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Starting Capital",   f"{GLOBAL_SYM}{INITIAL_BALANCE:,.0f}")
    k2.metric("Cash Available",     f"{GLOBAL_SYM}{j['cash']:,.0f}")
    k3.metric("Open Positions",     f"{GLOBAL_SYM}{live_value:,.0f}",
              f"Invested {GLOBAL_SYM}{open_cost:,.0f}")
    k4.metric("Unrealized P&L",     f"{GLOBAL_SYM}{unrealized:+,.0f}",
              f"{unrealized/open_cost*100:+.1f}%" if open_cost else "—")
    k5.metric("Total Portfolio",    f"{GLOBAL_SYM}{portfolio_val:,.0f}",
              f"{GLOBAL_SYM}{total_pnl:+,.0f}  ({total_pnl/INITIAL_BALANCE*100:+.1f}%)")

    st.divider()

    tab_open, tab_add, tab_hist, tab_stats = st.tabs(
        ["📈 Open Positions", "➕ Log Trade", "📋 History", "📊 Analytics"]
    )

    # ── TAB 1: OPEN POSITIONS ─────────────────────────────────────────────
    with tab_open:
        if not open_trades:
            st.info("No open positions. Go to **Log Trade** to add one.")
        else:
            rows = []
            for t in open_trades:
                cp     = current_prices[t["ticker"]]
                upnl   = (cp - t["entry_price"]) * t["quantity"]
                upnl_p = (cp - t["entry_price"]) / t["entry_price"] * 100
                to_tgt = (t["target_price"] - cp) / cp * 100
                to_stp = (cp - t["stop_price"]) / cp * 100
                rows.append({
                    "ID":          t["id"],
                    "Ticker":      t["ticker"],
                    "Date":        t["entry_date"],
                    "Qty":         t["quantity"],
                    "Entry ₹":     round(t["entry_price"], 2),
                    "Current ₹":   round(cp, 2),
                    "Target ₹":    round(t["target_price"], 2),
                    "Stop ₹":      round(t["stop_price"], 2),
                    "Unreal. P&L": f"₹{upnl:+,.0f} ({upnl_p:+.1f}%)",
                    "→ Target":    f"{to_tgt:+.1f}%",
                    "→ Stop":      f"-{to_stp:.1f}%",
                    "Notes":       t.get("notes", ""),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Close position form
        if open_trades:
            st.subheader("Close a Position")
            with st.form("close_form"):
                options = {
                    f"{t['ticker']}  ·  Entry ₹{t['entry_price']:.2f}  ×  {t['quantity']} units  [{t['id']}]": t["id"]
                    for t in open_trades
                }
                chosen_label = st.selectbox("Select position to close", list(options.keys()))
                chosen_id    = options[chosen_label]
                chosen_trade = next(t for t in open_trades if t["id"] == chosen_id)
                live_cp      = current_prices.get(chosen_trade["ticker"], chosen_trade["entry_price"])

                col_ep, col_er = st.columns(2)
                with col_ep:
                    exit_px = st.number_input(
                        "Exit Price (₹)",
                        min_value=0.01,
                        value=round(live_cp, 2),
                        step=0.50,
                        format="%.2f",
                    )
                with col_er:
                    exit_reason = st.selectbox(
                        "Exit Reason",
                        ["Take Profit Hit", "Stop Loss Hit", "Manual Exit", "Strategy Signal"],
                    )
                close_btn = st.form_submit_button("✅ Close Position", type="primary",
                                                   use_container_width=True)

            if close_btn:
                pnl     = (exit_px - chosen_trade["entry_price"]) * chosen_trade["quantity"]
                pnl_pct = (exit_px - chosen_trade["entry_price"]) / chosen_trade["entry_price"] * 100
                status  = "CLOSED" if pnl >= 0 else "STOPPED"
                for t in j["trades"]:
                    if t["id"] == chosen_id:
                        t.update({
                            "status":      status,
                            "exit_price":  round(exit_px, 4),
                            "exit_date":   str(date.today()),
                            "pnl_inr":     round(pnl, 2),
                            "pnl_pct":     round(pnl_pct, 2),
                            "exit_reason": exit_reason,
                        })
                        break
                j["cash"] = round(j["cash"] + exit_px * chosen_trade["quantity"], 2)
                _save_journal(j)
                icon = "✅" if pnl >= 0 else "🛑"
                st.toast(
                    f"{icon} {chosen_trade['ticker']} closed  ·  "
                    f"P&L: ₹{pnl:+,.0f} ({pnl_pct:+.1f}%)",
                    icon=icon,
                )
                st.rerun()

    # ── TAB 2: LOG NEW TRADE ──────────────────────────────────────────────
    with tab_add:
        st.subheader("Log a New Trade")
        if j["cash"] < 1:
            st.error("No cash available. Close an open position to free up capital.")
        else:
            with st.form("add_trade_form"):
                col1, col2 = st.columns(2)
                with col1:
                    f_ticker = st.text_input(
                        "Ticker Symbol",
                        placeholder="RELIANCE.NS  or  NVDA",
                    )
                    f_entry  = st.number_input(
                        "Entry Price (₹ or $)",
                        min_value=0.01, value=100.0, step=0.5, format="%.2f",
                    )
                    f_target = st.number_input(
                        "Target Price (₹ or $)",
                        min_value=0.01, value=104.0, step=0.5, format="%.2f",
                    )
                with col2:
                    f_stop   = st.number_input(
                        "Stop Loss (₹ or $)",
                        min_value=0.01, value=98.0, step=0.5, format="%.2f",
                    )
                    f_invest = st.number_input(
                        "Investment Amount (₹)",
                        min_value=1.0,
                        max_value=float(j["cash"]),
                        value=min(5_000.0, float(j["cash"])),
                        step=100.0,
                    )
                    f_notes  = st.text_input(
                        "Notes (optional)",
                        placeholder="Signal source, setup reason…",
                    )

                # Live preview row (computed from current widget values)
                qty_prev  = max(1, int(f_invest // f_entry))
                cost_prev = round(qty_prev * f_entry, 2)
                rr_prev   = ((f_target - f_entry) / (f_entry - f_stop)
                             if f_entry > f_stop else 0)
                st.markdown(
                    f"**Qty:** {qty_prev} units  ·  "
                    f"**Cost:** ₹{cost_prev:,.0f}  ·  "
                    f"**R:R:** {rr_prev:.2f}  ·  "
                    f"**Cash after:** ₹{j['cash'] - cost_prev:,.0f}"
                )

                add_btn = st.form_submit_button(
                    "➕ Add Trade", type="primary", use_container_width=True
                )

            if add_btn:
                ticker_clean = f_ticker.upper().strip()
                err = None
                if not ticker_clean:
                    err = "Ticker symbol is required."
                elif f_target <= f_entry:
                    err = "Target must be above entry price."
                elif f_stop >= f_entry:
                    err = "Stop must be below entry price."
                elif cost_prev > j["cash"]:
                    err = f"Insufficient cash — available ₹{j['cash']:,.0f}."

                if err:
                    st.error(err)
                else:
                    trade_rec = {
                        "id":           uuid.uuid4().hex[:8],
                        "ticker":       ticker_clean,
                        "entry_price":  round(f_entry, 4),
                        "target_price": round(f_target, 4),
                        "stop_price":   round(f_stop, 4),
                        "quantity":     qty_prev,
                        "investment":   cost_prev,
                        "entry_date":   str(date.today()),
                        "status":       "OPEN",
                        "exit_price":   None,
                        "exit_date":    None,
                        "pnl_inr":      None,
                        "pnl_pct":      None,
                        "exit_reason":  None,
                        "notes":        f_notes,
                    }
                    j["trades"].append(trade_rec)
                    j["cash"] = round(j["cash"] - cost_prev, 2)
                    _save_journal(j)
                    st.toast(
                        f"✅ {qty_prev} × {ticker_clean} @ ₹{f_entry:.2f}  "
                        f"(Cost ₹{cost_prev:,.0f}  ·  R:R {rr_prev:.2f})",
                        icon="✅",
                    )
                    st.rerun()

    # ── TAB 3: HISTORY ────────────────────────────────────────────────────
    with tab_hist:
        if not closed_trades:
            st.info("No closed trades yet.")
        else:
            hist_rows = []
            for t in sorted(closed_trades, key=lambda x: x.get("exit_date", ""), reverse=True):
                hist_rows.append({
                    "Ticker":     t["ticker"],
                    "Entry Date": t["entry_date"],
                    "Exit Date":  t.get("exit_date", "—"),
                    "Qty":        t["quantity"],
                    "Entry ₹":    f"₹{t['entry_price']:.2f}",
                    "Exit ₹":     f"₹{t.get('exit_price', 0):.2f}",
                    "Target ₹":   f"₹{t['target_price']:.2f}",
                    "Stop ₹":     f"₹{t['stop_price']:.2f}",
                    "P&L ₹":      t.get("pnl_inr", 0),
                    "P&L %":      f"{t.get('pnl_pct', 0):+.2f}%",
                    "Result":     t["status"],
                    "Reason":     t.get("exit_reason", "—"),
                    "Notes":      t.get("notes", ""),
                })
            st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)

            fig_bar = go.Figure(go.Bar(
                x=[f"{t['ticker']} ({t.get('exit_date','')[:10]})" for t in closed_trades],
                y=[t.get("pnl_inr", 0) for t in closed_trades],
                marker_color=[
                    "#00ff88" if (t.get("pnl_inr") or 0) >= 0 else "#ff4444"
                    for t in closed_trades
                ],
                text=[f"₹{t.get('pnl_inr', 0):+,.0f}" for t in closed_trades],
                textposition="outside",
            ))
            fig_bar.update_layout(
                title="P&L per Closed Trade",
                yaxis_title="P&L (₹)",
                height=360, template="plotly_dark",
                margin=dict(t=50, l=0, r=0, b=80),
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # ── TAB 4: ANALYTICS ─────────────────────────────────────────────────
    with tab_stats:
        finished = [t for t in all_trades
                    if t["status"] != "OPEN" and t.get("pnl_inr") is not None]
        if not finished:
            st.info("Close some trades to see analytics.")
        else:
            winners   = [t for t in finished if t["pnl_inr"] >= 0]
            losers    = [t for t in finished if t["pnl_inr"] < 0]
            win_rate  = len(winners) / len(finished) * 100
            avg_win   = float(np.mean([t["pnl_inr"] for t in winners])) if winners else 0.0
            avg_loss  = float(np.mean([t["pnl_inr"] for t in losers]))  if losers  else 0.0
            expectancy= (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)
            best_t    = max(finished, key=lambda x: x["pnl_inr"])
            worst_t   = min(finished, key=lambda x: x["pnl_inr"])

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Win Rate",    f"{win_rate:.1f}%",
                      f"{len(winners)}W / {len(losers)}L")
            s2.metric("Avg Win",     f"₹{avg_win:+,.0f}")
            s3.metric("Avg Loss",    f"₹{avg_loss:+,.0f}")
            s4.metric("Expectancy",  f"₹{expectancy:+,.0f}", "per trade")

            s5, s6 = st.columns(2)
            s5.metric("Realized P&L", f"₹{realized:+,.0f}",
                      f"{realized/INITIAL_BALANCE*100:+.2f}% on starting capital")
            s6.metric("Trades Closed", len(finished))

            st.divider()

            # Cumulative P&L line chart
            sorted_fin   = sorted(finished, key=lambda x: x.get("exit_date", ""))
            cum_pnl      = np.cumsum([t["pnl_inr"] for t in sorted_fin])
            trade_labels = [f"#{i+1} {t['ticker']}" for i, t in enumerate(sorted_fin)]

            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(
                x=list(range(1, len(cum_pnl) + 1)),
                y=cum_pnl,
                mode="lines+markers",
                fill="tozeroy",
                fillcolor="rgba(96,180,255,0.12)",
                line=dict(color="#60b4ff", width=2),
                marker=dict(
                    size=8,
                    color=["#00ff88" if v >= 0 else "#ff4444" for v in cum_pnl],
                    line=dict(color="white", width=1),
                ),
                text=trade_labels,
                hovertemplate="%{text}<br>Cum P&L: ₹%{y:+,.0f}<extra></extra>",
            ))
            fig_cum.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
            fig_cum.update_layout(
                title="Cumulative P&L Curve",
                xaxis_title="Trade #",
                yaxis_title="Cumulative P&L (₹)",
                height=360, template="plotly_dark",
                showlegend=False,
                margin=dict(t=50, l=0, r=0, b=0),
            )
            st.plotly_chart(fig_cum, use_container_width=True)

            # Win/Loss donut + best/worst cards
            col_pie, col_cards = st.columns([1, 1])
            with col_pie:
                fig_pie = go.Figure(go.Pie(
                    labels=["Winners", "Losers"],
                    values=[len(winners), max(len(losers), 0)],
                    marker=dict(colors=["#00ff88", "#ff4444"]),
                    hole=0.55,
                    textinfo="label+percent",
                ))
                fig_pie.update_layout(
                    title="Win / Loss Split",
                    height=300, template="plotly_dark",
                    showlegend=False,
                    margin=dict(t=50, b=0, l=0, r=0),
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_cards:
                st.markdown("#### Best Trade")
                st.success(
                    f"**{best_t['ticker']}**  ·  "
                    f"₹{best_t['pnl_inr']:+,.0f} ({best_t.get('pnl_pct', 0):+.1f}%)\n\n"
                    f"Entry ₹{best_t['entry_price']:.2f} → "
                    f"Exit ₹{best_t.get('exit_price', 0):.2f}  ·  "
                    f"{best_t.get('exit_date', '—')}"
                )
                st.markdown("#### Worst Trade")
                st.error(
                    f"**{worst_t['ticker']}**  ·  "
                    f"₹{worst_t['pnl_inr']:+,.0f} ({worst_t.get('pnl_pct', 0):+.1f}%)\n\n"
                    f"Entry ₹{worst_t['entry_price']:.2f} → "
                    f"Exit ₹{worst_t.get('exit_price', 0):.2f}  ·  "
                    f"{worst_t.get('exit_date', '—')}"
                )

    # ── DANGER ZONE ───────────────────────────────────────────────────────
    st.divider()
    with st.expander("⚠️ Danger Zone"):
        st.warning(
            f"Resetting permanently deletes all trades and restores the "
            f"{GLOBAL_SYM}{INITIAL_BALANCE:,.0f} balance. This cannot be undone."
        )
        if st.checkbox("I understand this is irreversible"):
            if st.button("🔄 Reset Journal", type="secondary"):
                st.session_state.journal = {
                    "initial_balance": INITIAL_BALANCE,
                    "cash":            INITIAL_BALANCE,
                    "trades":          [],
                }
                _save_journal(st.session_state.journal)
                st.toast(f"Journal reset to {GLOBAL_SYM}{INITIAL_BALANCE:,.0f}", icon="🔄")
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════
#  MODE 6 — POSITION SIZER
# ════════════════════════════════════════════════════════════════════════════
elif dashboard_mode == "📐 Position Sizer":
    sym = GLOBAL_SYM

    st.title("📐 Position Sizing Calculator")
    st.caption(
        f"Risk a fixed % of your capital per trade.  "
        f"Currency: **{CURRENCY}**  ·  Capital default: **{sym}{JOURNAL_INITIAL:,.0f}**"
    )

    col1, col2 = st.columns(2)
    with col1:
        ps_capital = st.number_input(
            f"Total Account Capital ({sym})",
            min_value=1.0,
            value=float(JOURNAL_INITIAL),
            step=100.0,
            format="%.2f",
        )
        ps_risk_pct = st.slider(
            "Risk per Trade (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.1,
            help="% of account capital you are willing to lose on this trade.",
        )
        ps_entry = st.number_input(
            f"Entry Price ({sym})", min_value=0.01, value=100.0, step=0.5, format="%.2f",
        )

    with col2:
        ps_stop = st.number_input(
            f"Stop Loss Price ({sym})", min_value=0.01, value=97.0, step=0.5, format="%.2f",
        )
        ps_target = st.number_input(
            f"Target Price ({sym})", min_value=0.01, value=106.0, step=0.5, format="%.2f",
        )
        ps_commission = st.number_input(
            f"Commission per side ({sym})", min_value=0.0, value=0.0, step=1.0, format="%.2f",
            help="Brokerage cost per leg (buy + sell counted separately).",
        )

    st.divider()

    risk_per_share = ps_entry - ps_stop

    if risk_per_share <= 0:
        st.error("Stop Loss must be **below** Entry Price.")
    elif ps_target <= ps_entry:
        st.error("Target Price must be **above** Entry Price.")
    else:
        risk_amount      = ps_capital * ps_risk_pct / 100
        shares           = max(1, int(risk_amount / risk_per_share))
        position_cost    = shares * ps_entry
        total_commission = ps_commission * 2                       # buy + sell
        actual_risk      = shares * risk_per_share + total_commission
        actual_risk_pct  = actual_risk / ps_capital * 100
        reward_per_share = ps_target - ps_entry
        potential_profit = shares * reward_per_share - total_commission
        rr_ratio         = potential_profit / actual_risk if actual_risk > 0 else 0
        position_pct     = position_cost / ps_capital * 100
        remaining_cash   = ps_capital - position_cost

        # ── KEY RESULTS ───────────────────────────────────────────────────
        st.subheader("Sizing Result")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Shares to Buy",    f"{shares:,}",
                  f"{sym}{position_cost:,.2f} position")
        r2.metric("Capital Used",     f"{sym}{position_cost:,.2f}",
                  f"{position_pct:.1f}% of account")
        r3.metric("Max Risk",         f"{sym}{actual_risk:,.2f}",
                  f"{actual_risk_pct:.2f}% of capital")
        r4.metric("R:R Ratio",        f"{rr_ratio:.2f}",
                  f"{'Good' if rr_ratio >= 2 else 'Low'}")

        r5, r6, r7, r8 = st.columns(4)
        r5.metric("Potential Profit", f"{sym}{potential_profit:,.2f}")
        r6.metric("Risk per Share",   f"{sym}{risk_per_share:.2f}")
        r7.metric("Stop Distance",    f"{(ps_entry-ps_stop)/ps_entry*100:.2f}%")
        r8.metric("Remaining Cash",   f"{sym}{remaining_cash:,.2f}")

        # ── TRADE SUMMARY BANNER ──────────────────────────────────────────
        banner_col = "#00ff88" if rr_ratio >= 2 else "#ffaa00"
        st.markdown(
            f'<div style="background:{banner_col}18; border-left:4px solid {banner_col}; '
            f'padding:12px 18px; border-radius:6px; margin:12px 0; font-size:0.95rem;">'
            f'Buy <b>{shares:,} shares</b> at {sym}{ps_entry:.2f}  ·  '
            f'Stop: {sym}{ps_stop:.2f}  ·  Target: {sym}{ps_target:.2f}  ·  '
            f'Max loss: <b>{sym}{actual_risk:,.2f}</b> ({actual_risk_pct:.2f}%)  ·  '
            f'Potential gain: <b>{sym}{potential_profit:,.2f}</b>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.divider()

        # ── CAPITAL ALLOCATION CHART ──────────────────────────────────────
        col_chart, col_scenarios = st.columns([1, 1])

        with col_chart:
            fig_alloc = go.Figure(go.Bar(
                x=["Position Cost", "Max Risk", "Potential Profit", "Remaining Cash"],
                y=[position_cost, actual_risk, potential_profit, max(remaining_cash, 0)],
                marker_color=["#60b4ff", "#ff4444", "#00ff88", "#555"],
                text=[f"{sym}{v:,.0f}" for v in
                      [position_cost, actual_risk, potential_profit, max(remaining_cash, 0)]],
                textposition="outside",
            ))
            fig_alloc.update_layout(
                title="Capital Breakdown",
                yaxis_title=f"Amount ({sym})",
                height=360, template="plotly_dark",
                showlegend=False,
                margin=dict(t=50, l=0, r=0, b=40),
            )
            st.plotly_chart(fig_alloc, use_container_width=True)

        # ── RISK SCENARIOS TABLE ──────────────────────────────────────────
        with col_scenarios:
            st.subheader("Risk % Scenarios")
            scenarios = []
            for r_pct in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
                r_amt  = ps_capital * r_pct / 100
                sh     = max(1, int(r_amt / risk_per_share))
                cost   = sh * ps_entry
                profit = sh * reward_per_share - total_commission
                rr     = profit / (sh * risk_per_share + total_commission) if risk_per_share > 0 else 0
                scenarios.append({
                    "Risk %":    f"{r_pct:.1f}%",
                    "Risk Amt":  f"{sym}{r_amt:,.0f}",
                    "Shares":    sh,
                    "Cost":      f"{sym}{cost:,.0f}",
                    "% Capital": f"{cost/ps_capital*100:.1f}%",
                    "Profit":    f"{sym}{profit:,.0f}",
                    "R:R":       f"{rr:.2f}",
                })
            # Highlight the selected risk row
            df_sc = pd.DataFrame(scenarios)
            st.dataframe(df_sc, use_container_width=True, hide_index=True, height=260)
            st.caption(f"Highlighted row = your current {ps_risk_pct:.1f}% selection")

        # ── RISK / REWARD VISUALISER ──────────────────────────────────────
        st.subheader("Price Level Visualiser")
        fig_rr = go.Figure()
        price_range = [ps_stop * 0.97, ps_target * 1.03]

        fig_rr.add_hrect(
            y0=ps_stop, y1=ps_entry, fillcolor="rgba(255,68,68,0.12)",
            line_width=0, annotation_text="Risk zone", annotation_position="right",
        )
        fig_rr.add_hrect(
            y0=ps_entry, y1=ps_target, fillcolor="rgba(0,255,136,0.10)",
            line_width=0, annotation_text="Reward zone", annotation_position="right",
        )
        for y_val, label, colour in [
            (ps_entry,  f"Entry  {sym}{ps_entry:.2f}",  "#f7c948"),
            (ps_target, f"Target {sym}{ps_target:.2f}", "#00ff88"),
            (ps_stop,   f"Stop   {sym}{ps_stop:.2f}",   "#ff4444"),
        ]:
            fig_rr.add_hline(y=y_val, line_color=colour, line_dash="solid", line_width=1.5,
                             annotation_text=label, annotation_font_color=colour)

        fig_rr.update_layout(
            height=280, template="plotly_dark",
            yaxis=dict(range=price_range, title=f"Price ({sym})"),
            xaxis=dict(visible=False),
            margin=dict(t=20, l=0, r=120, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_rr, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  MODE 7 — AI PICKS
# ════════════════════════════════════════════════════════════════════════════
elif dashboard_mode == "🎯 AI Picks":
    st.title("🎯 AI Picks")
    st.caption(
        f"GradientBoosting ML model scored across all {len(AI_PICKS_UNIVERSE)} tickers  ·  "
        f"Top 5 long + short candidates  ·  "
        f"Gemini reasoning loaded on demand (1 API call per ticker per day)"
    )

    if ml_model is None:
        st.error(
            "No trained model found. Run `python train_model.py` to generate "
            "`trade_signal_model.pkl` and `feature_scaler.pkl`."
        )
        st.stop()

    # ── SCAN TRIGGER ──────────────────────────────────────────────────────
    col_btn, col_info = st.columns([1, 4])
    run_scan = col_btn.button("▶  Run Full Scan", type="primary", use_container_width=True)
    col_info.caption(
        f"Scores all {len(AI_PICKS_UNIVERSE)} tickers using the ML model only (no news fetch).  "
        f"Takes ~30–60 s on first run; results cached until you re-scan."
    )

    if run_scan:
        st.session_state.pop("ai_picks_results", None)

    if "ai_picks_results" not in st.session_state:
        if not run_scan:
            st.info("Click **▶ Run Full Scan** to score all tickers and see today's AI picks.")
            st.stop()

        scan_prog = st.progress(0, text="Starting scan…")
        pick_results = []
        for idx, t in enumerate(AI_PICKS_UNIVERSE):
            scan_prog.progress((idx + 1) / len(AI_PICKS_UNIVERSE), text=f"Scoring {t}…")
            df_t = get_data(t)
            if df_t.empty or len(df_t) < 50:
                continue
            df_t = compute_indicators(df_t)
            if df_t.empty:
                continue
            last_t = df_t.iloc[-1]
            feats = np.array([[
                float(last_t.get("RSI",            50)),
                float(last_t.get("MACD_Hist",       0)),
                float(last_t.get("ADX",            20)),
                float(last_t.get("ATR_Pct",      0.01)),
                float(last_t.get("Price_vs_EMA10",  0)),
                float(last_t.get("Volume_Ratio",    1)),
                float(last_t.get("MA_Trend",        0)),
                float(last_t.get("MACD_Cross",      0)),
            ]])
            ml_prob_t = float(ml_model.predict_proba(ml_scaler.transform(feats))[0][1])
            sig_t = generate_signal(
                last_t, PROFIT_TARGET_PCT, STOP_LOSS_PCT,
                0.0, AI_SENTIMENT_FLOOR, ml_model, ml_scaler,
            )
            pick_results.append({
                "ticker":  t,
                "price":   float(last_t["Close"]),
                "ml_prob": ml_prob_t,
                "sig":     sig_t,
                "indicators": {
                    "RSI":            float(last_t.get("RSI",            50)),
                    "MACD_Hist":      float(last_t.get("MACD_Hist",       0)),
                    "ADX":            float(last_t.get("ADX",            20)),
                    "ATR_Pct":        float(last_t.get("ATR_Pct",      0.01)),
                    "Price_vs_EMA10": float(last_t.get("Price_vs_EMA10",  0)),
                    "Volume_Ratio":   float(last_t.get("Volume_Ratio",    1)),
                    "MA_Trend":       float(last_t.get("MA_Trend",        0)),
                    "MACD_Cross":     float(last_t.get("MACD_Cross",      0)),
                    "price":          float(last_t["Close"]),
                    "ml_prob":        ml_prob_t,
                    "verdict":        sig_t["verdict"],
                },
            })
        scan_prog.empty()
        st.session_state["ai_picks_results"] = pick_results

    pick_results = st.session_state["ai_picks_results"]

    if not pick_results:
        st.warning("No tickers returned data. Check your internet connection and try again.")
        st.stop()

    sorted_long  = sorted(pick_results, key=lambda x: x["ml_prob"], reverse=True)[:5]
    sorted_short = sorted(pick_results, key=lambda x: x["ml_prob"])[:5]

    # ── CARD RENDERER ──────────────────────────────────────────────────────
    def _pick_card(item: dict, card_key: str) -> None:
        t     = item["ticker"]
        price = item["price"]
        prob  = item["ml_prob"]
        sig   = item["sig"]
        sym   = currency_sym(t)
        color = "#00ff88" if prob >= 0.6 else "#ff4444" if prob < 0.4 else "#ffaa00"

        st.markdown(
            f'<div style="border:1px solid {color}55; border-radius:8px; '
            f'padding:14px 16px; margin:6px 0; background:{color}0d;">'
            f'<div style="display:flex; justify-content:space-between; align-items:center;">'
            f'<span style="font-size:1.1rem; font-weight:700; color:{color};">{t}</span>'
            f'<span style="color:#aaa; font-size:0.9rem;">{sym}{price:.2f}</span>'
            f'</div>'
            f'<div style="margin-top:8px;">'
            f'<span style="background:{color}33; color:{color}; padding:3px 10px; '
            f'border-radius:12px; font-size:0.82rem; font-weight:600;">'
            f'{prob:.1%} ML confidence</span>'
            f'<span style="color:#888; margin-left:10px; font-size:0.82rem;">'
            f'{sig["verdict"]} · {sig["regime"]}</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        cache_key = f"ai_pick_{t}_{date.today()}"
        if cache_key in st.session_state:
            ai = st.session_state[cache_key]
            verd_color = (
                "#00ff88" if ai["verdict"] == "buy"
                else "#ff4444" if ai["verdict"] == "sell"
                else "#ffaa00"
            )
            conf_color = (
                "#00ff88" if ai["confidence"] == "high"
                else "#ffaa00" if ai["confidence"] == "medium"
                else "#888"
            )
            st.markdown(
                f'<span style="color:{verd_color}; font-size:0.82rem; font-weight:600;">'
                f'● {ai["verdict"].upper()}</span>'
                f'<span style="color:{conf_color}; font-size:0.78rem; margin-left:8px;">'
                f'({ai["confidence"]} confidence)</span>',
                unsafe_allow_html=True,
            )
            st.caption(ai["reasoning"])
        else:
            if st.button("✨ Get AI Reasoning", key=f"ai_btn_{card_key}"):
                if not gemini_key:
                    st.warning("Add a Gemini API key in the sidebar (✨ Gemini AI expander).")
                else:
                    with st.spinner(f"Asking Gemini about {t}…"):
                        ai_result = cached_ai_explanation(
                            t,
                            json.dumps(item["indicators"]),
                            gemini_key,
                            str(date.today()),
                        )
                    st.session_state[cache_key] = ai_result
                    st.rerun()

    # ── TWO-COLUMN PICK CARDS ─────────────────────────────────────────────
    st.divider()
    col_long, col_short = st.columns(2)

    with col_long:
        st.subheader("🟢 Top 5 Long Candidates")
        st.caption("Highest ML buy-signal probability")
        for i, item in enumerate(sorted_long):
            _pick_card(item, f"long_{i}")

    with col_short:
        st.subheader("🔴 Top 5 Short Candidates")
        st.caption("Lowest ML buy-signal probability (avoid / short)")
        for i, item in enumerate(sorted_short):
            _pick_card(item, f"short_{i}")

    # ── FULL SCORES TABLE (collapsed) ────────────────────────────────────
    st.divider()
    with st.expander(f"📋 All {len(pick_results)} ticker scores"):
        table_rows = []
        for r in sorted(pick_results, key=lambda x: x["ml_prob"], reverse=True):
            sym_r = currency_sym(r["ticker"])
            table_rows.append({
                "Ticker":     r["ticker"],
                "Price":      f"{sym_r}{r['price']:.2f}",
                "ML Prob":    f"{r['ml_prob']:.1%}",
                "Signal":     r["sig"]["verdict"],
                "Regime":     r["sig"]["regime"],
                "Confidence": f"{r['sig']['confidence']:.0f}%",
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # ── CORRELATION HEATMAP ───────────────────────────────────────────────
    st.divider()
    st.subheader("📉 60-Day Return Correlation")
    st.caption(
        "Pairwise Pearson correlation of daily returns across all 44 tickers "
        "(last 60 trading days). Pairs with |r| > 0.8 are flagged below."
    )

    with st.spinner("Loading 60-day price history for correlation…"):
        ret_df = get_corr_returns(tuple(AI_PICKS_UNIVERSE))

    if ret_df.empty or len(ret_df.columns) < 2:
        st.warning("Not enough overlapping data to compute the correlation matrix.")
    else:
        corr = ret_df.corr()

        fig_corr = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdYlGn",
            zmid=0, zmin=-1, zmax=1,
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
        ))
        fig_corr.update_layout(
            title=f"Return Correlation — last 60 days  ({len(corr.columns)} tickers)",
            height=680, template="plotly_dark",
            margin=dict(t=50, l=0, r=0, b=80),
            xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
            yaxis=dict(tickfont=dict(size=8)),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # High-correlation warning pairs
        high_pairs = []
        cols_c = corr.columns.tolist()
        for i in range(len(cols_c)):
            for j in range(i + 1, len(cols_c)):
                val = corr.iloc[i, j]
                if abs(val) > 0.8:
                    high_pairs.append({
                        "Ticker A":    cols_c[i],
                        "Ticker B":    cols_c[j],
                        "Correlation": round(val, 3),
                        "Type":        "⚠️ High positive" if val > 0 else "⚠️ High negative",
                    })

        if high_pairs:
            high_pairs.sort(key=lambda x: abs(x["Correlation"]), reverse=True)
            with st.expander(f"⚠️ {len(high_pairs)} highly correlated pairs  (|r| > 0.8)"):
                st.caption(
                    "Holding both sides of a high-correlation pair reduces diversification. "
                    "Consider treating them as a single position."
                )
                st.dataframe(
                    pd.DataFrame(high_pairs),
                    use_container_width=True, hide_index=True,
                )
        else:
            st.success("No pairs with |r| > 0.8 in the current 60-day window.")

# ════════════════════════════════════════════════════════════════════════════
#  MODE 8 — QUICK ANALYZE  (any ticker, ML only for trained universe)
# ════════════════════════════════════════════════════════════════════════════
elif dashboard_mode == "🔎 Quick Analyze":
    st.title("🔎 Quick Analyze")
    st.caption(
        "Analyze any stock worldwide. "
        "ML signal available only for the 44-stock trained universe."
    )

    _qa_c1, _qa_c2 = st.columns([3, 1])
    _qa_ticker = (
        _qa_c1.text_input(
            "Ticker symbol",
            placeholder="e.g. RELIANCE.NS · TSLA · AAPL · BTC-USD · ^NSEI",
            key="qa_ticker_input",
        )
        .upper()
        .strip()
    )
    _qa_window = _qa_c2.selectbox(
        "Chart window", ["3 months", "6 months", "1 year"], index=2, key="qa_window"
    )

    if not _qa_ticker:
        st.info("Type any valid ticker above (yfinance format) and press Enter.")
        st.stop()

    _qa_days_map = {"3 months": 90, "6 months": 180, "1 year": 365}
    _qa_chart_days = _qa_days_map[_qa_window]

    with st.spinner(f"Loading {_qa_ticker}…"):
        _qa_df = get_data(_qa_ticker, days=_qa_chart_days + 300)

    if _qa_df.empty:
        st.error(
            f"Ticker **{_qa_ticker}** not found or returned no data.  "
            "Check the symbol (use yfinance format, e.g. `RELIANCE.NS`, `BTC-USD`) and try again."
        )
        st.stop()

    _qa_df = compute_indicators(_qa_df)
    if _qa_df.empty:
        st.error("Not enough historical data to compute indicators (need ≥ 30 bars).")
        st.stop()

    _qa_in_universe = _qa_ticker in AI_PICKS_UNIVERSE
    _qa_ml_m  = ml_model  if _qa_in_universe else None
    _qa_ml_s  = ml_scaler if _qa_in_universe else None
    _qa_sym   = currency_sym(_qa_ticker)

    _qa_last  = _qa_df.iloc[-1]
    _qa_price = float(_qa_last["Close"])
    _qa_prev  = float(_qa_df["Close"].iloc[-2]) if len(_qa_df) >= 2 else _qa_price
    _qa_pct   = (_qa_price - _qa_prev) / _qa_prev * 100

    _qa_sent, _qa_headlines = get_sentiment(_qa_ticker, news_provider, api_key)

    _qa_sig = generate_signal(
        _qa_last, PROFIT_TARGET_PCT, STOP_LOSS_PCT,
        _qa_sent, AI_SENTIMENT_FLOOR, _qa_ml_m, _qa_ml_s,
    )

    # Universe status banner
    if _qa_in_universe:
        st.success(
            f"✅ **{_qa_ticker}** is in the trained universe — full ML signal included.  "
            "This is identical to Deep-Dive Analysis."
        )
    else:
        st.info(
            f"ℹ️ **{_qa_ticker}** is outside the 44-stock trained universe.  "
            "ML signal is not available; analysis is based on technical indicators only."
        )

    # ── Metrics row ───────────────────────────────────────────────────────
    _qa_rsi_v = float(_qa_last["RSI"])
    _qa_adx_v = float(_qa_last["ADX"])
    _qa_atr_v = float(_qa_last["ATR"])

    _qc1, _qc2, _qc3, _qc4, _qc5 = st.columns(5)
    _qc1.metric("Price",       f"{_qa_sym}{_qa_price:.2f}", f"{_qa_pct:+.2f}%")
    _qc2.metric("Take Profit", f"{_qa_sym}{_qa_sig['target_price']:.2f}",
                f"+{(_qa_sig['target_price'] - _qa_price) / _qa_price * 100:.2f}%")
    _qc3.metric("Stop Loss",   f"{_qa_sym}{_qa_sig['stop_price']:.2f}",
                f"-{(_qa_price - _qa_sig['stop_price']) / _qa_price * 100:.2f}%")
    _qc4.metric("R:R Ratio",   f"{_qa_sig['rr_ratio']:.2f}")
    if _qa_in_universe and _qa_sig["ml_prob"] is not None:
        _qc5.metric("ML Probability", f"{_qa_sig['ml_prob']:.1%}")
    else:
        _qc5.metric("ML Probability", "N/A  (not trained)")

    signal_banner(
        _qa_sig,
        f"  ·  RSI: {_qa_rsi_v:.1f}  ·  ADX: {_qa_adx_v:.1f}  ·  "
        f"Sentiment: {_qa_sent:+.2f}"
        + ("" if _qa_in_universe else "  ·  ⚠️ ML excluded"),
    )

    st.divider()

    # ── Price chart (same style as Deep-Dive) ─────────────────────────────
    _qa_disp = _qa_df.tail(_qa_chart_days)

    _qa_fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.60, 0.20, 0.20],
        vertical_spacing=0.025,
        subplot_titles=("Price & Indicators", "RSI (14)", "Volume"),
    )
    _qa_fig.add_trace(go.Candlestick(
        x=_qa_disp.index,
        open=_qa_disp["Open"], high=_qa_disp["High"],
        low=_qa_disp["Low"],   close=_qa_disp["Close"],
        name="Price",
        increasing_line_color="#00ff88", decreasing_line_color="#ff4444",
    ), row=1, col=1)
    _qa_fig.add_trace(go.Scatter(
        x=_qa_disp.index, y=_qa_disp["EMA_10"],
        line=dict(color="#f7c948", width=1.5), name="EMA 10",
    ), row=1, col=1)
    _qa_fig.add_trace(go.Scatter(
        x=_qa_disp.index, y=_qa_disp["SMA_50"],
        line=dict(color="#60b4ff", width=1.0, dash="dot"), name="SMA 50",
    ), row=1, col=1)
    _qa_fig.add_trace(go.Scatter(
        x=_qa_disp.index, y=_qa_disp["SMA_200"],
        line=dict(color="#bf94e4", width=1.0, dash="dash"), name="SMA 200",
    ), row=1, col=1)
    _qa_fig.add_hline(
        y=_qa_sig["target_price"], line_dash="dash", line_color="#00ff88",
        annotation_text=f"Target {_qa_sym}{_qa_sig['target_price']:.2f}",
        annotation_font_color="#00ff88", row=1, col=1,
    )
    _qa_fig.add_hline(
        y=_qa_sig["stop_price"], line_dash="dash", line_color="#ff4444",
        annotation_text=f"Stop {_qa_sym}{_qa_sig['stop_price']:.2f}",
        annotation_font_color="#ff4444", row=1, col=1,
    )
    _qa_fig.add_trace(go.Scatter(
        x=_qa_disp.index, y=_qa_disp["RSI"],
        line=dict(color="#bf94e4", width=1.5), name="RSI",
    ), row=2, col=1)
    _qa_fig.add_hline(y=70, line_color="#ff4444", line_dash="dot", row=2, col=1)
    _qa_fig.add_hline(y=30, line_color="#00ff88", line_dash="dot", row=2, col=1)
    _qa_bar_colors = [
        "#00ff88" if c >= o else "#ff4444"
        for c, o in zip(_qa_disp["Close"], _qa_disp["Open"])
    ]
    _qa_fig.add_trace(go.Bar(
        x=_qa_disp.index, y=_qa_disp["Volume"],
        marker_color=_qa_bar_colors, name="Volume", showlegend=False,
    ), row=3, col=1)
    _qa_fig.add_trace(go.Scatter(
        x=_qa_disp.index, y=_qa_disp["Vol_SMA"],
        line=dict(color="white", width=1, dash="dot"), name="Vol SMA 20",
    ), row=3, col=1)
    _qa_fig.update_layout(
        height=700, template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(t=50, l=0, r=0, b=0),
    )
    _qa_fig.update_yaxes(title_text="Price",  row=1, col=1)
    _qa_fig.update_yaxes(title_text="RSI",    row=2, col=1, range=[0, 100])
    _qa_fig.update_yaxes(title_text="Volume", row=3, col=1)
    st.plotly_chart(_qa_fig, use_container_width=True)

    # ── Technical Snapshot + Confidence gauge ─────────────────────────────
    _qa_col_l, _qa_col_r = st.columns([1.4, 1])

    with _qa_col_l:
        st.subheader("Technical Snapshot")
        _qa_vol_ratio = float(_qa_last.get("Volume_Ratio", 0))
        _qa_ema_diff  = float(_qa_last.get("Price_vs_EMA10", 0)) * 100
        _qa_snapshot  = pd.DataFrame([
            {"Indicator": "RSI (14)",       "Value": f"{_qa_rsi_v:.1f}",
             "Status": "Overbought" if _qa_rsi_v > 70 else "Oversold" if _qa_rsi_v < 30 else "Neutral"},
            {"Indicator": "ADX (14)",       "Value": f"{_qa_adx_v:.1f}",
             "Status": "Strong Trend" if _qa_adx_v > 25 else "No Clear Trend"},
            {"Indicator": "ATR (14)",       "Value": f"{_qa_sym}{_qa_atr_v:.2f}",
             "Status": "Volatility measure"},
            {"Indicator": "Volume Ratio",   "Value": f"{_qa_vol_ratio:.2f}×",
             "Status": "Above Avg" if _qa_vol_ratio > 1.2 else "Below Avg"},
            {"Indicator": "Price vs EMA10", "Value": f"{_qa_ema_diff:+.2f}%",
             "Status": "Above EMA" if _qa_ema_diff > 0 else "Below EMA"},
            {"Indicator": "MA Trend",
             "Value": "Uptrend" if float(_qa_last.get("MA_Trend", 0)) else "Downtrend",
             "Status": "SMA50 vs SMA200"},
            {"Indicator": "MACD Cross",
             "Value": "Bullish" if float(_qa_last.get("MACD_Cross", 0)) else "Bearish",
             "Status": "MACD vs Signal"},
            {"Indicator": "Sentiment",      "Value": f"{_qa_sent:+.2f}",
             "Status": ("Bullish" if _qa_sent >= 0.05
                        else "Bearish" if _qa_sent <= -0.05 else "Neutral")},
            {"Indicator": "ML Signal",
             "Value": (f"{_qa_sig['ml_prob']:.1%}" if _qa_in_universe and _qa_sig['ml_prob'] is not None
                       else "Not available"),
             "Status": ("Trained universe" if _qa_in_universe
                        else "Outside trained universe")},
        ])
        st.dataframe(_qa_snapshot, use_container_width=True, hide_index=True)

    with _qa_col_r:
        st.subheader("Signal Confidence")
        _qa_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=_qa_sig["confidence"],
            title={"text": _qa_sig["verdict"]},
            gauge={
                "axis":  {"range": [0, 100]},
                "bar":   {"color": _qa_sig["color"]},
                "steps": [
                    {"range": [0,  40], "color": "rgba(255,68,68,0.15)"},
                    {"range": [40, 60], "color": "rgba(255,170,0,0.15)"},
                    {"range": [60,100], "color": "rgba(0,255,136,0.15)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.75, "value": 60,
                },
            },
            number={"suffix": "%"},
        ))
        _qa_gauge.update_layout(
            height=280, template="plotly_dark",
            margin=dict(t=50, b=10, l=20, r=20),
        )
        st.plotly_chart(_qa_gauge, use_container_width=True)
        if not _qa_in_universe:
            st.caption(
                "Confidence score is rule-based only (RSI + volume + EMA).  "
                "ML component excluded — ticker outside trained universe."
            )

    # ── News headlines ────────────────────────────────────────────────────
    st.divider()
    with st.expander(f"📰 {news_provider} Headlines — {_qa_ticker}"):
        render_headlines(_qa_headlines)

    # ── Gemini AI analysis ────────────────────────────────────────────────
    if gemini_key:
        st.divider()
        with st.expander("✨ Gemini AI Analysis", expanded=True):
            if _qa_in_universe and _qa_sig["ml_prob"] is not None:
                _qa_prompt = f"""You are a short-term trading analyst. Analyze this setup concisely:

Ticker: {_qa_ticker}
Price: {_qa_sym}{_qa_price:.2f} ({_qa_pct:+.2f}% today)
Signal: {_qa_sig['verdict']} | Regime: {_qa_sig['regime']}
RSI: {_qa_rsi_v:.1f} | ADX: {_qa_adx_v:.1f} | Confidence: {_qa_sig['confidence']:.0f}%
Target: {_qa_sym}{_qa_sig['target_price']:.2f} | Stop: {_qa_sym}{_qa_sig['stop_price']:.2f} | R:R: {_qa_sig['rr_ratio']:.2f}
Sentiment: {_qa_sent:+.2f} | ML Probability: {_qa_sig['ml_prob']:.1%}

Write 4-5 focused sentences covering the technicals, entry quality, key risks, and overall take.
Be direct. Skip boilerplate disclaimers."""
            else:
                _qa_prompt = f"""You are a short-term trading analyst. No ML model signal is available for this ticker — base your analysis solely on the technical indicators below.

Ticker: {_qa_ticker}
Price: {_qa_sym}{_qa_price:.2f} ({_qa_pct:+.2f}% today)
Rule-based Signal: {_qa_sig['verdict']} | Regime: {_qa_sig['regime']}
RSI: {_qa_rsi_v:.1f} | ADX: {_qa_adx_v:.1f} | ATR: {_qa_sym}{_qa_atr_v:.2f}
Price vs EMA10: {float(_qa_last.get('Price_vs_EMA10', 0)) * 100:+.2f}% | Volume Ratio: {float(_qa_last.get('Volume_Ratio', 1)):.2f}×
MA Trend: {"Uptrend (SMA50>SMA200)" if float(_qa_last.get('MA_Trend', 0)) else "Downtrend (SMA50<SMA200)"}
MACD Cross: {"Bullish" if float(_qa_last.get('MACD_Cross', 0)) else "Bearish"}
Target: {_qa_sym}{_qa_sig['target_price']:.2f} | Stop: {_qa_sym}{_qa_sig['stop_price']:.2f} | R:R: {_qa_sig['rr_ratio']:.2f}
Sentiment: {_qa_sent:+.2f}

Write 4-5 focused sentences covering what the technicals show, whether the entry makes sense, key risks, and your overall take. No ML data is available — rely only on the indicators above.
Be direct. Skip boilerplate disclaimers."""

            _qa_key = f"qa_gemini_{_qa_ticker}_{date.today()}"
            if _qa_key not in st.session_state:
                if st.button("Get AI Analysis", key="qa_gemini_btn", type="primary"):
                    with st.spinner("Asking Gemini…"):
                        st.session_state[_qa_key] = get_gemini_analysis(
                            _qa_ticker, _qa_prompt, gemini_key
                        )
            if _qa_key in st.session_state:
                st.markdown(st.session_state[_qa_key])
    else:
        st.caption("_Add a Gemini API key in the sidebar to enable AI analysis._")


# ════════════════════════════════════════════════════════════════════════════
#  MODE 9 — MY PORTFOLIO  (paper trading tracker)
# ════════════════════════════════════════════════════════════════════════════
elif dashboard_mode == "📁 My Portfolio":
    st.title("📁 My Portfolio")
    st.caption("Paper trading tracker — live prices, ML signals, ATR-based stops.")

    # ── Add position form ─────────────────────────────────────────────────
    with st.expander("➕ Add New Position", expanded=not bool(load_portfolio())):
        with st.form("pf_add_form", clear_on_submit=True):
            _fa, _fb, _fc, _fd = st.columns([2, 2, 1, 2])
            _pf_ticker    = _fa.selectbox("Ticker", AI_PICKS_UNIVERSE)
            _pf_buy_price = _fb.number_input("Buy Price", min_value=0.01, value=100.0,
                                              step=0.01, format="%.2f")
            _pf_qty       = _fc.number_input("Quantity", min_value=1, value=1, step=1)
            _pf_buy_date  = _fd.date_input("Buy Date", value=date.today())
            if st.form_submit_button("Add Position", type="primary", use_container_width=True):
                _pf_list = load_portfolio()
                _pf_list.append({
                    "ticker":    _pf_ticker,
                    "buy_price": float(_pf_buy_price),
                    "quantity":  int(_pf_qty),
                    "buy_date":  str(_pf_buy_date),
                })
                save_portfolio(_pf_list)
                st.success(f"Added {int(_pf_qty)} × {_pf_ticker} @ {_pf_buy_price:.2f}")
                st.rerun()

    st.divider()

    # ── Open positions ────────────────────────────────────────────────────
    _pf_positions = load_portfolio()
    if not _pf_positions:
        st.info("No open positions. Use ➕ Add New Position above.")
    else:
        st.subheader(f"Open Positions  ({len(_pf_positions)})")
        _pf_rows   = []
        _pf_prog   = st.progress(0, text="Loading portfolio data…")

        for _pi, _pos in enumerate(_pf_positions):
            _pf_prog.progress((_pi + 1) / len(_pf_positions),
                               text=f"Fetching {_pos['ticker']}…")
            _tk  = _pos["ticker"]
            _bpx = float(_pos["buy_price"])
            _qty = int(_pos["quantity"])
            _sym = "₹" if ".NS" in _tk else "$"

            _df_p = get_data(_tk)
            if _df_p.empty or len(_df_p) < 50:
                _pf_rows.append({
                    "_idx": _pi, "_sym": _sym,
                    "Ticker": _tk, "Buy": _bpx, "Current": None, "Qty": _qty,
                    "P&L": None, "P&L%": None, "ATR": None, "ATR Stop": None,
                    "Signal": "N/A", "Conf%": None, "_conf_level": "low",
                    "_pnl_pct": 0.0, "Rec": "⚠️ No data",
                })
                continue

            _df_p = compute_indicators(_df_p)
            if _df_p.empty:
                continue

            _last_p  = _df_p.iloc[-1]
            _cur_p   = float(_last_p["Close"])
            _atr_p   = float(_last_p["ATR"])
            _atr_stop_p = _bpx - 2.0 * _atr_p
            _pnl_abs = (_cur_p - _bpx) * _qty
            _pnl_pct = (_cur_p - _bpx) / _bpx * 100 if _bpx else 0.0

            _sig_p  = generate_signal(_last_p, PROFIT_TARGET_PCT, STOP_LOSS_PCT,
                                       0.0, -999.0, ml_model, ml_scaler)
            _v_p    = _sig_p["verdict"]
            _cp_p   = _sig_p["confidence"]
            _cl_p   = "high" if _cp_p >= 65 else "medium" if _cp_p >= 35 else "low"

            _pf_rows.append({
                "_idx": _pi, "_sym": _sym,
                "Ticker": _tk, "Buy": _bpx, "Current": _cur_p, "Qty": _qty,
                "P&L": _pnl_abs, "P&L%": _pnl_pct, "ATR": _atr_p,
                "ATR Stop": _atr_stop_p, "Signal": _v_p,
                "Conf%": _cp_p, "_conf_level": _cl_p,
                "_pnl_pct": _pnl_pct, "Rec": "_PENDING_",
            })

        _pf_prog.empty()

        # Total portfolio value needed for ADD-size check
        _total_pf_val = sum(
            r["Current"] * r["Qty"] for r in _pf_rows if r["Current"] is not None
        )

        # Resolve recommendations
        for _r in _pf_rows:
            if _r["Rec"] != "_PENDING_":
                continue
            _cur2, _qty2  = _r["Current"], _r["Qty"]
            _v2, _cl2     = _r["Signal"], _r["_conf_level"]
            _atr_stop2    = _r["ATR Stop"]
            _in_profit    = _r["_pnl_pct"] > 0

            if _cur2 <= _atr_stop2:
                _r["Rec"] = "🔴 SELL (stop-loss hit)"
            elif "SELL" in _v2 and _cl2 in ("high", "medium") and _in_profit:
                _r["Rec"] = "🟡 Consider SELL (lock profit)"
            elif "SELL" in _v2 and _cl2 == "low":
                _r["Rec"] = "🟡 HOLD (weak sell signal)"
            elif "BUY" in _v2 and _cl2 == "high":
                _pos_val = _cur2 * _qty2
                if _total_pf_val > 0 and _pos_val / _total_pf_val < 0.05:
                    _r["Rec"] = "🟢 Consider ADD"
                else:
                    _r["Rec"] = "⚪ HOLD"
            else:
                _r["Rec"] = "⚪ HOLD"

        # Build display dataframe
        _disp = []
        for _r in _pf_rows:
            _s = _r["_sym"]
            _disp.append({
                "Ticker":   _r["Ticker"],
                "Buy":      f"{_s}{_r['Buy']:.2f}",
                "Current":  f"{_s}{_r['Current']:.2f}"  if _r["Current"]  is not None else "N/A",
                "Qty":      _r["Qty"],
                "P&L":      f"{_s}{_r['P&L']:+.2f}"     if _r["P&L"]      is not None else "N/A",
                "P&L %":    f"{_r['P&L%']:+.1f}%"       if _r["P&L%"]     is not None else "N/A",
                "ATR":      f"{_s}{_r['ATR']:.2f}"       if _r["ATR"]      is not None else "N/A",
                "ATR Stop": f"{_s}{_r['ATR Stop']:.2f}"  if _r["ATR Stop"] is not None else "N/A",
                "Signal":   _r["Signal"],
                "Conf":     f"{_r['Conf%']:.0f}%"        if _r["Conf%"]    is not None else "N/A",
                "Rec":      _r["Rec"],
            })

        _pf_disp_df = pd.DataFrame(_disp)

        def _style_rec(val):
            if "stop-loss" in val:     return "background-color:#3d0000;color:#ff6666"
            if "Consider SELL" in val: return "background-color:#3d2600;color:#ffaa33"
            if "Consider ADD"  in val: return "background-color:#003d00;color:#66ff66"
            return "color:#cccccc"

        def _style_pnl(val):
            if isinstance(val, str) and val.lstrip("$₹").startswith("-"):
                return "color:#ff6666"
            if isinstance(val, str) and "+" in val:
                return "color:#66ff66"
            return ""

        try:
            _styled_pf = (
                _pf_disp_df.style
                .map(_style_rec, subset=["Rec"])
                .map(_style_pnl, subset=["P&L", "P&L %"])
            )
        except AttributeError:
            _styled_pf = (
                _pf_disp_df.style
                .applymap(_style_rec, subset=["Rec"])
                .applymap(_style_pnl, subset=["P&L", "P&L %"])
            )

        st.dataframe(_styled_pf, use_container_width=True, hide_index=True)

        # ── Portfolio summary ─────────────────────────────────────────────
        st.divider()
        _total_cost_pf = sum(r["Buy"] * r["Qty"] for r in _pf_rows)
        _total_pnl_pf  = sum(r["P&L"] for r in _pf_rows if r["P&L"] is not None)
        _ms1, _ms2, _ms3 = st.columns(3)
        _ms1.metric(
            "Total Portfolio Value",
            f"${_total_pf_val:,.2f}" if _total_pf_val else "N/A",
        )
        _ms2.metric(
            "Total P&L",
            f"${_total_pnl_pf:+,.2f}",
            delta=f"{_total_pnl_pf / _total_cost_pf * 100:+.1f}%" if _total_cost_pf else None,
        )
        _ms3.metric("Open Positions", len(_pf_positions))

        if _total_pf_val > 0:
            _wt_rows = [
                {
                    "Ticker": r["Ticker"],
                    "Value":  f"{r['_sym']}{r['Current'] * r['Qty']:,.2f}",
                    "Weight": f"{r['Current'] * r['Qty'] / _total_pf_val * 100:.1f}%",
                }
                for r in _pf_rows if r["Current"] is not None
            ]
            with st.expander("📊 Position Weights"):
                st.dataframe(pd.DataFrame(_wt_rows),
                              use_container_width=True, hide_index=True)

        # ── Close a position ──────────────────────────────────────────────
        st.divider()
        st.subheader("Close a Position")
        _open_labels = [
            f"{p['ticker']}  ·  bought @ {p['buy_price']:.2f}  ×{p['quantity']}"
            for p in _pf_positions
        ]
        _cl1, _cl2, _cl3 = st.columns([3, 2, 1])
        _close_sel   = _cl1.selectbox("Position", _open_labels, key="pf_close_sel",
                                       label_visibility="collapsed")
        _close_price = _cl2.number_input("Sell price", min_value=0.01, value=100.0,
                                          step=0.01, key="pf_close_price",
                                          label_visibility="collapsed")
        if _cl3.button("Close", type="primary", key="pf_close_btn",
                        use_container_width=True):
            _ci    = _open_labels.index(_close_sel)
            _cp_pos = _pf_positions[_ci]
            _pnl_c  = (_close_price - _cp_pos["buy_price"]) * _cp_pos["quantity"]
            _sym_c  = "₹" if ".NS" in _cp_pos["ticker"] else "$"
            _closed_list = load_closed_positions()
            _closed_list.append({
                "ticker":     _cp_pos["ticker"],
                "buy_price":  _cp_pos["buy_price"],
                "quantity":   _cp_pos["quantity"],
                "buy_date":   _cp_pos["buy_date"],
                "sell_price": float(_close_price),
                "close_date": str(date.today()),
                "pnl":        round(_pnl_c, 2),
                "pnl_pct":    round(
                    (_close_price - _cp_pos["buy_price"]) / _cp_pos["buy_price"] * 100, 2
                ),
            })
            save_closed_positions(_closed_list)
            _pf_positions.pop(_ci)
            save_portfolio(_pf_positions)
            st.success(
                f"Closed {_cp_pos['ticker']} @ {_sym_c}{_close_price:.2f}  —  "
                f"P&L: {_sym_c}{_pnl_c:+.2f}"
            )
            st.rerun()

    # ── Closed positions log ──────────────────────────────────────────────
    _closed_log = load_closed_positions()
    if _closed_log:
        st.divider()
        with st.expander(f"📋 Closed Positions  ({len(_closed_log)})", expanded=False):
            _cl_df = pd.DataFrame(_closed_log)
            _cl_df.rename(columns={
                "ticker": "Ticker", "buy_price": "Buy Price", "quantity": "Qty",
                "buy_date": "Buy Date", "sell_price": "Sell Price",
                "close_date": "Close Date", "pnl": "P&L", "pnl_pct": "P&L %",
            }, inplace=True)
            st.dataframe(_cl_df, use_container_width=True, hide_index=True)
            _realized = sum(p["pnl"] for p in _closed_log)
            st.metric("Total Realized P&L", f"{_realized:+.2f}")
