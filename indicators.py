"""
Pure pandas/numpy implementations of the technical indicators used across this project.
Drop-in replacement for pandas_ta — same function names and return shapes.

RSI / ATR / ADX  use Wilder's smoothing  (com = length-1, alpha = 1/length)
EMA / MACD       use standard EMA        (span = length,  alpha = 2/(length+1))
"""

import pandas as pd
import numpy as np


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length, min_periods=length).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0.0)
    loss     = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(com=length - 1, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(com=length - 1, adjust=False, min_periods=length).mean()
    rs       = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=length - 1, adjust=False, min_periods=length).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    ema_fast    = series.ewm(span=fast,   adjust=False).mean()
    ema_slow    = series.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    # Column order matches pandas_ta: iloc[:,0]=MACD  iloc[:,1]=Hist  iloc[:,2]=Signal
    return pd.DataFrame({
        "MACD":  macd_line,
        "MACDh": histogram,
        "MACDs": signal_line,
    }, index=series.index)


def adx(high: pd.Series, low: pd.Series, close: pd.Series,
        length: int = 14) -> pd.DataFrame:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    up   = high.diff()
    down = (-low).diff()
    plus_dm  = up.where((up > down) & (up > 0.0),   0.0)
    minus_dm = down.where((down > up) & (down > 0.0), 0.0)

    smooth_tr    = tr.ewm(com=length - 1, adjust=False, min_periods=length).mean()
    plus_di      = 100.0 * plus_dm.ewm( com=length - 1, adjust=False, min_periods=length).mean() / smooth_tr
    minus_di     = 100.0 * minus_dm.ewm(com=length - 1, adjust=False, min_periods=length).mean() / smooth_tr

    di_sum  = (plus_di + minus_di).replace(0.0, np.nan)
    dx      = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx_val = dx.ewm(com=length - 1, adjust=False, min_periods=length).mean()

    # Column order: iloc[:,0]=ADX  (same as pandas_ta ADX_14_6_6)
    return pd.DataFrame({
        "ADX": adx_val,
        "DMP": plus_di,
        "DMN": minus_di,
    }, index=close.index)
