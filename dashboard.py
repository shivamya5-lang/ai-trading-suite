import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import timedelta, date
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
import re
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Short-Term Trading Engine", layout="wide")
st.title("⚡ 1-Week Automated Trading Suite")

# --- 2. SIDEBAR CONFIGURATION ---
st.sidebar.header("Navigation Control")
dashboard_mode = st.sidebar.selectbox(
    "Select Operating Mode",
    ["Sector Watchlist Scanner", "Single Stock Deep-Dive"]
)

st.sidebar.divider()

st.sidebar.header("⚙️ 1-Week Strategy Rules")
PROFIT_TARGET_PCT = st.sidebar.slider("Profit Target (%)", 2.0, 10.0, 4.0) / 100.0
STOP_LOSS_PCT = st.sidebar.slider("Stop Loss (%)", 1.0, 5.0, 2.0) / 100.0
AI_SENTIMENT_THRESHOLD = st.sidebar.slider("AI Sentiment Floor", -0.5, 0.5, 0.1)

st.sidebar.divider()

# --- THE MASTER NEWS ROUTER RETURNS ---
st.sidebar.header("News & Sentiment Source")
news_provider = st.sidebar.selectbox(
    "Select Data Provider", 
    [
        "Hybrid Auto-Router (Free)", 
        "Google News RSS (Free)", 
        "Yahoo Finance (Free)", 
        "Finviz Web Scraper (Free)", 
        "NewsAPI.org (Key Required)", 
        "Alpha Vantage (Key Required)"
    ]
)

api_key = ""
if "Key Required" in news_provider:
    api_key = st.sidebar.text_input("Enter API Key", type="password")

SECTOR_UNIVERSES = {
    "Semiconductors (AI Infra)": ["NVDA", "AMD", "AVGO", "TSM", "QCOM", "MU", "ARM"],
    "Clean Energy & EV": ["TSLA", "ENPH", "FSLR", "NEE", "RIVN", "PLUG"],
    "Aerospace & Defense": ["LMT", "RTX", "NOC", "GD", "BA", "PLTR"],
    "Big Tech (FAANG)": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NFLX"]
}

if dashboard_mode == "Single Stock Deep-Dive":
    st.sidebar.header("Deep-Dive Target")
    ticker_input = st.sidebar.text_input("US Stock Ticker", "NVDA").upper().strip()
else:
    st.sidebar.header("Scanner Target")
    selected_sector = st.sidebar.selectbox("Select Target Sector Universe", list(SECTOR_UNIVERSES.keys()))

# --- 3. DATA ENGINE ---
@st.cache_data
def get_short_term_data(ticker):
    try:
        end = date.today()
        start = end - timedelta(days=365)
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty: return pd.DataFrame()
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        return df.dropna()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_sentiment(ticker, provider, key=""):
    """Scrapes breaking headlines using the selected provider and scores them"""
    analyzer = SentimentIntensityAnalyzer()
    headlines = []
    total_score = 0
    headers = {'User-Agent': 'Mozilla/5.0'}

    def process_headline(title, source):
        score = analyzer.polarity_scores(title)['compound']
        headlines.append({"title": f"[{source}] {title}", "score": score})
        return score

    try:
        if provider == "Alpha Vantage (Key Required)" and key:
            base_ticker = ticker.split('.')[0]
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={base_ticker}&limit=6&apikey={key}"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode('utf-8'))
            if 'feed' in data:
                for item in data['feed'][:6]:
                    score = float(item.get('overall_sentiment_score', 0))
                    headlines.append({"title": f"[AlphaV] {item.get('title', '')}", "score": score})
                    total_score += score

        elif provider == "NewsAPI.org (Key Required)" and key:
            query = urllib.parse.quote(f"{ticker} stock")
            url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=6&apiKey={key}"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode('utf-8'))
            if 'articles' in data:
                for item in data['articles']:
                    title = item.get('title', '')
                    if title and title != "[Removed]": total_score += process_headline(title, "NewsAPI")

        elif provider == "Yahoo Finance (Free)" or provider == "Hybrid Auto-Router (Free)":
            try:
                y_news = yf.Ticker(ticker).news
                if y_news:
                    for article in y_news[:6]:
                        title = article.get('title', '')
                        if title: total_score += process_headline(title, "Yahoo")
            except Exception: pass

        if provider == "Google News RSS (Free)" or (provider == "Hybrid Auto-Router (Free)" and not headlines):
            try:
                query = urllib.parse.quote(f"{ticker} stock")
                url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req) as response:
                    root = ET.fromstring(response.read())
                for item in root.findall('.//item')[:6]:
                    title = item.find('title').text
                    if title: total_score += process_headline(title.split(" - ")[0], "Google")
            except Exception: pass

        if provider == "Finviz Web Scraper (Free)" or (provider == "Hybrid Auto-Router (Free)" and not headlines):
            try:
                url = f"https://finviz.com/quote.ashx?t={ticker.split('.')[0]}"
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req) as response:
                    html = response.read().decode('utf-8')
                matches = re.findall(r'class="tab-link-news"[^>]*>(.*?)</a>', html)
                for title in matches[:6]:
                    total_score += process_headline(re.sub(r'<[^>]+>', '', title), "Finviz")
            except Exception: pass

        if not headlines: return 0, [{"title": "No news data accessible.", "score": 0}]
        return total_score / len(headlines), headlines
    except Exception:
        return 0, [{"title": "Provider Connection Failed.", "score": 0}]

# --- 4. EXECUTION ROADMAPS ---

if dashboard_mode == "Sector Watchlist Scanner":
    tickers = SECTOR_UNIVERSES[selected_sector]
    st.markdown(f"## 🤖 AI Market Scanner: {selected_sector}")
    st.write(f"Scanning the market using **{news_provider}** to find the absolute best 1-week setup...")

    scan_data = []
    p_bar = st.progress(0)

    for i, t in enumerate(tickers):
        p_bar.progress((i + 1) / len(tickers))
        df = get_short_term_data(t)
        if df.empty or len(df) < 30: continue

        # Technical Indicators
        df['EMA_10'] = ta.ema(df['Close'], length=10)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['Vol_SMA'] = ta.sma(df['Volume'], length=20)
        adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        df['ADX'] = adx_df.iloc[:, 0]
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df = df.dropna()
        if df.empty: continue

        p = float(df['Close'].iloc[-1])
        ema_v = float(df['EMA_10'].iloc[-1])
        rsi_v = float(df['RSI'].iloc[-1])
        adx_v = float(df['ADX'].iloc[-1])
        v_v = float(df['Volume'].iloc[-1])
        v_a = float(df['Vol_SMA'].iloc[-1])
        atr_v = float(df['ATR'].iloc[-1])

        # Mathematical Triggers
        is_trending = adx_v > 22
        above_ema = bool(p > ema_v)
        vol_ok = bool(v_v > (v_a * 1.2))
        oversold = bool(rsi_v < 35)

        # Quick pre-filter before requesting expensive news parsing
        tech_pass = (is_trending and above_ema and vol_ok) or (not is_trending and oversold)

        if tech_pass:
            score, _ = get_sentiment(t, news_provider, api_key)
            if score >= AI_SENTIMENT_THRESHOLD:
                
                # Dynamic targets based on market regime (Trending vs Choppy)
                if is_trending:
                    t_price = p * (1.0 + PROFIT_TARGET_PCT)
                    s_price = p * (1.0 - STOP_LOSS_PCT)
                    regime_label = "Trend Breakout"
                else:
                    t_price = p + (2.0 * atr_v)
                    s_price = p - (1.0 * atr_v)
                    regime_label = "Range Bottom"

                scan_data.append({
                    "Ticker": t,
                    "Setup": regime_label,
                    "Entry Price": p,
                    "Target Price": t_price,
                    "Stop Loss": s_price,
                    "AI Score": score
                })
        time.sleep(1.0) # Protect against IP Bans

    # --- AI SUGGESTION ENGINE ---
    if scan_data:
        # Sort the passed stocks by their AI Sentiment Score (Highest first)
        scan_data = sorted(scan_data, key=lambda x: x['AI Score'], reverse=True)
        top_pick = scan_data[0]

        st.success(f"### 🏆 Top AI Suggestion: Buy **{top_pick['Ticker']}**")
        st.markdown(f"**Why?** It passed all technical momentum filters and has the most bullish news sentiment in the sector (**+{top_pick['AI Score']:.2f}**).")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Recommended Entry", f"${top_pick['Entry Price']:.2f}")
        c2.metric("Take Profit Target", f"${top_pick['Target Price']:.2f}")
        c3.metric("Protective Stop Loss", f"${top_pick['Stop Loss']:.2f}")
        
        st.divider()
        st.markdown("#### 📋 Other Qualifying Candidates")
        
        # Format for clean display
        display_list = []
        for d in scan_data:
            display_list.append({
                "Ticker": d['Ticker'],
                "Market Setup": d['Setup'],
                "Entry": f"${d['Entry Price']:.2f}",
                "Target": f"${d['Target Price']:.2f}",
                "Stop": f"${d['Stop Loss']:.2f}",
                "News Sentiment": f"{d['AI Score']:+.2f}"
            })
        st.dataframe(pd.DataFrame(display_list), use_container_width=True, hide_index=True)
    else:
        st.warning(f"No stocks in {selected_sector} are safe to buy right now. The AI rejected them based on your strict momentum and sentiment rules.")

else:
    # --- SINGLE STOCK DEEP-DIVE MODE ---
    st.markdown(f"## 📊 Short-Term Deep Dive: {ticker_input}")
    df = get_short_term_data(ticker_input)
    
    if df.empty:
        st.error(f"Unable to pull data for '{ticker_input}'.")
    else:
        df['EMA_10'] = ta.ema(df['Close'], length=10)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['Vol_SMA'] = ta.sma(df['Volume'], length=20)
        adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        df['ADX'] = adx_df.iloc[:, 0]
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df = df.dropna()

        price = float(df['Close'].iloc[-1])
        ema_val = float(df['EMA_10'].iloc[-1])
        rsi_val = float(df['RSI'].iloc[-1])
        adx_val = float(df['ADX'].iloc[-1])
        vol_val = float(df['Volume'].iloc[-1])
        vol_avg = float(df['Vol_SMA'].iloc[-1])
        atr_val = float(df['ATR'].iloc[-1])

        score, headlines_list = get_sentiment(ticker_input, news_provider, api_key)
        news_passed = bool(score >= AI_SENTIMENT_THRESHOLD)

        is_trending = adx_val > 22
        verdict = "HOLD 🟡"
        
        if is_trending:
            regime = "TRENDING 📈"
            target_price = price * (1.0 + PROFIT_TARGET_PCT)
            stop_price = price * (1.0 - STOP_LOSS_PCT)
            if price > ema_val and vol_val > (vol_avg * 1.2):
                verdict = "FAST BUY 🟢" if news_passed else "HOLD (AI News Block) 🛑"
            elif rsi_val > 70: verdict = "SELL 🔴"
        else:
            regime = "SIDEWAYS / CHOPPY ↔️"
            target_price = price + (2.0 * atr_val)
            stop_price = price - (1.0 * atr_val)
            if rsi_val < 35:
                verdict = "BUY (Range Bottom) 🟢" if news_passed else "HOLD (AI News Block) 🛑"
            elif rsi_val > 65: verdict = "SELL 🔴"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", f"${price:.2f}", regime)
        c2.metric("Take Profit Target", f"${target_price:.2f}")
        c3.metric("Protective Stop Loss", f"${stop_price:.2f}")
        c4.metric("AI Recommendation", verdict, f"News Score: {score:+.2f}")

        st.divider()
        with st.expander(f"📰 View {news_provider} Headlines"):
            for h in headlines_list:
                c = "green" if h['score'] >= 0.05 else "red" if h['score'] <= -0.05 else "gray"
                st.markdown(f"- <span style='color:{c}'>**[{h['score']:.2f}]**</span> {h['title']}", unsafe_allow_html=True)

        st.divider()
        display_df = df.tail(60)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=display_df.index, open=display_df['Open'], high=display_df['High'], low=display_df['Low'], close=display_df['Close'], name='Price'))
        fig.add_trace(go.Scatter(x=display_df.index, y=display_df['EMA_10'], line=dict(color='yellow', width=1.5), name='10 EMA'))
        fig.add_hline(y=target_price, line_dash="dash", line_color="green", annotation_text="Take Profit")
        fig.add_hline(y=stop_price, line_dash="dash", line_color="red", annotation_text="Stop Loss")
        fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)