import json

from google import genai
from config import GEMINI_MODEL


def get_ai_explanation(ticker: str, indicators: dict, gemini_key: str) -> dict:
    """Call Gemini to produce a structured trade-signal explanation.

    indicators keys expected: RSI, MACD_Hist, ADX, ATR_Pct, Price_vs_EMA10,
    Volume_Ratio, MA_Trend, MACD_Cross, price, ml_prob, verdict.

    Returns {"verdict": "buy|sell|hold",
             "confidence": "high|medium|low",
             "reasoning": "2-3 sentences"}
    """
    if not gemini_key:
        return {
            "verdict": "hold",
            "confidence": "low",
            "reasoning": "No Gemini API key configured. Add one in the sidebar.",
        }

    rsi       = indicators.get("RSI", 50)
    macd_hist = indicators.get("MACD_Hist", 0)
    adx       = indicators.get("ADX", 20)
    atr_pct   = indicators.get("ATR_Pct", 0.01)
    pve       = indicators.get("Price_vs_EMA10", 0)
    vol_ratio = indicators.get("Volume_Ratio", 1)
    ma_trend  = indicators.get("MA_Trend", 0)
    macd_x    = indicators.get("MACD_Cross", 0)
    price     = indicators.get("price", 0)
    ml_prob   = indicators.get("ml_prob", 0.5)
    verdict   = indicators.get("verdict", "HOLD")

    prompt = f"""You are a concise quantitative trading analyst. Analyze this stock and respond ONLY with valid JSON — no markdown, no extra text.

Ticker: {ticker}
Price: {price:.2f}
ML Signal Probability: {ml_prob:.1%}
Current Signal: {verdict}

Indicators:
- RSI(14): {rsi:.1f}
- MACD Histogram: {macd_hist:.4f}
- ADX(14): {adx:.1f}
- Volatility (ATR%): {atr_pct * 100:.2f}%
- Price vs EMA10: {pve * 100:+.2f}%
- Volume vs 20-day avg: {vol_ratio:.2f}x
- Long-term MA trend: {"Uptrend (SMA50>SMA200)" if ma_trend else "Downtrend (SMA50<SMA200)"}
- MACD bullish cross: {"Yes" if macd_x else "No"}

Respond ONLY with this JSON (no other text):
{{"verdict": "buy", "confidence": "high", "reasoning": "Your 2-3 sentence explanation here."}}

verdict must be exactly one of: buy, sell, hold
confidence must be exactly one of: high, medium, low"""

    try:
        client   = genai.Client(api_key=gemini_key)
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = response.text.strip()

        # Strip markdown fences if Gemini wraps the JSON
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else parts[0]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        result = json.loads(text)
        if all(k in result for k in ("verdict", "confidence", "reasoning")):
            return result
        return {"verdict": "hold", "confidence": "low", "reasoning": str(result)}

    except json.JSONDecodeError as e:
        return {
            "verdict": "hold",
            "confidence": "low",
            "reasoning": f"_(Could not parse Gemini response: {str(e)[:80]})_",
        }
    except Exception as e:
        return {
            "verdict": "hold",
            "confidence": "low",
            "reasoning": f"_(Gemini error: {str(e)})_",
        }
