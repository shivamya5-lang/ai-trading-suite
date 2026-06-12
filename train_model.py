import os
import json
import shutil
import yfinance as yf
import pandas as pd
import indicators as ta
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings("ignore")

_DIR = os.path.dirname(os.path.abspath(__file__))

# --- CONFIG ---
_DEFAULT_TICKERS = [
    "NVDA", "AMD", "AVGO", "TSM", "QCOM", "MU", "ARM",
    "TSLA", "ENPH", "FSLR", "NEE", "RIVN", "PLUG",
    "LMT", "RTX", "NOC", "GD", "BA", "PLTR",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NFLX",
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "WIPRO.NS", "SBIN.NS", "BHARTIARTL.NS", "HINDUNILVR.NS", "BAJFINANCE.NS",
    "MARUTI.NS", "TATAMOTORS.NS", "HCLTECH.NS", "ITC.NS", "AXISBANK.NS",
    "KOTAKBANK.NS", "LT.NS", "NESTLEIND.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS",
]

UNIVERSE_FILE = os.path.join(_DIR, "universe.json")
MODEL_PATH    = os.path.join(_DIR, "trade_signal_model.pkl")
SCALER_PATH   = os.path.join(_DIR, "feature_scaler.pkl")
BACKUP_MODEL  = os.path.join(_DIR, "trade_signal_model_backup.pkl")
BACKUP_SCALER = os.path.join(_DIR, "feature_scaler_backup.pkl")

TRAIN_START  = "2015-01-01"
SPLIT_DATE   = "2025-01-01"
TRAIN_END    = "2026-01-01"
PROFIT_PCT   = 0.04
STOP_PCT     = 0.02
HORIZON      = 5

FEATURE_COLS = [
    "RSI",
    "MACD_Hist",
    "ADX",
    "ATR_Pct",
    "Price_vs_EMA10",
    "Volume_Ratio",
    "MA_Trend",
    "MACD_Cross",
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
    return list(_DEFAULT_TICKERS)


def save_universe(tickers: list) -> None:
    with open(UNIVERSE_FILE, "w") as f:
        json.dump(tickers, f, indent=2)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["RSI"] = ta.rsi(df["Close"], length=14)

    macd_df = ta.macd(df["Close"])
    df["MACD"]        = macd_df.iloc[:, 0]
    df["MACD_Signal"] = macd_df.iloc[:, 2]
    df["MACD_Hist"]   = macd_df.iloc[:, 1]

    adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    df["ADX"] = adx_df.iloc[:, 0]

    df["ATR"]     = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["ATR_Pct"] = df["ATR"] / df["Close"]

    df["EMA_10"]  = ta.ema(df["Close"], length=10)
    df["SMA_50"]  = ta.sma(df["Close"], length=50)
    df["SMA_200"] = ta.sma(df["Close"], length=200)
    df["Vol_SMA"] = ta.sma(df["Volume"], length=20)

    df["Price_vs_EMA10"] = (df["Close"] - df["EMA_10"]) / df["EMA_10"]
    df["Volume_Ratio"]   = df["Volume"] / df["Vol_SMA"]
    df["MA_Trend"]       = (df["SMA_50"] > df["SMA_200"]).astype(int)
    df["MACD_Cross"]     = (df["MACD"] > df["MACD_Signal"]).astype(int)

    return df


def label_outcomes(df: pd.DataFrame) -> list:
    closes = df["Close"].values
    highs  = df["High"].values
    lows   = df["Low"].values
    n      = len(df)
    labels = []

    for i in range(n):
        if i + HORIZON >= n:
            labels.append(np.nan)
            continue
        entry = closes[i]
        tp    = entry * (1 + PROFIT_PCT)
        sl    = entry * (1 - STOP_PCT)
        result = np.nan
        for j in range(i + 1, i + 1 + HORIZON):
            if highs[j] >= tp:
                result = 1
                break
            if lows[j] <= sl:
                result = 0
                break
        if np.isnan(result):
            result = 1 if closes[i + HORIZON] >= entry else 0
        labels.append(result)

    return labels


def build_dataset(start: str, end: str, tickers: list = None) -> pd.DataFrame:
    _tickers = tickers if tickers is not None else load_universe()
    frames = []
    for ticker in _tickers:
        print(f"  {ticker}", end=" ", flush=True)
        raw = yf.download(ticker, start=start, end=end, progress=False)
        if raw.empty or len(raw) < 220:
            print("(skipped — insufficient data)")
            continue
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
        raw = raw.dropna()

        df = compute_features(raw)
        df["Label"] = label_outcomes(df)
        df = df.dropna(subset=FEATURE_COLS + ["Label"])
        frames.append(df[FEATURE_COLS + ["Label"]])
        print(f"({len(df)} rows)")

    if not frames:
        raise RuntimeError("No data was downloaded. Check your internet connection.")
    return pd.concat(frames, ignore_index=True)


def _fit_and_save(tickers: list) -> tuple:
    """Core training logic. Returns (success, message). Does NOT save on failure."""
    train_df = build_dataset(TRAIN_START, SPLIT_DATE, tickers)
    test_df  = build_dataset(SPLIT_DATE, TRAIN_END,   tickers)

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["Label"].values.astype(int)
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df["Label"].values.astype(int)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42,
    )
    model.fit(X_train, y_train)

    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    test_auc  = roc_auc_score(y_test,  model.predict_proba(X_test)[:, 1])

    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    msg = (
        f"AUC train {train_auc:.4f} / test {test_auc:.4f}  ·  "
        f"{len(tickers)} tickers  ·  {len(X_train):,} training samples"
    )
    return True, msg


def train_on_universe(tickers: list) -> tuple:
    """
    Retrain on the given tickers list.
    Backs up the current model before overwriting.
    Returns (success: bool, message: str).
    On failure the backup is preserved; the broken partial state is NOT saved.
    """
    # Back up existing model so the dashboard keeps working if retrain fails
    if os.path.exists(MODEL_PATH):
        shutil.copy2(MODEL_PATH,  BACKUP_MODEL)
    if os.path.exists(SCALER_PATH):
        shutil.copy2(SCALER_PATH, BACKUP_SCALER)

    try:
        return _fit_and_save(tickers)
    except Exception as exc:
        # Restore backup if new model files were partially written
        for src, dst in [(BACKUP_MODEL, MODEL_PATH), (BACKUP_SCALER, SCALER_PATH)]:
            if os.path.exists(src):
                shutil.copy2(src, dst)
        return False, str(exc)


def train() -> None:
    tickers = load_universe()
    print(f"=== Universe: {len(tickers)} tickers ===")
    print("=== Building training set ===")
    train_df = build_dataset(TRAIN_START, SPLIT_DATE, tickers)
    print(f"\n=== Building test set ===")
    test_df  = build_dataset(SPLIT_DATE, TRAIN_END,   tickers)

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["Label"].values.astype(int)
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df["Label"].values.astype(int)

    print(f"\nTrain: {len(X_train):,} samples | class balance: {np.bincount(y_train)}")
    print(f"Test : {len(X_test):,}  samples | class balance: {np.bincount(y_test)}")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42,
    )

    print("\nTraining GradientBoostingClassifier...")
    model.fit(X_train, y_train)

    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    test_auc  = roc_auc_score(y_test,  model.predict_proba(X_test)[:, 1])
    print(f"\nAUC — train: {train_auc:.4f}  |  test (out-of-sample): {test_auc:.4f}")

    print("\nClassification report (test set):")
    print(classification_report(y_test, model.predict(X_test),
                                target_names=["No Trade (0)", "Buy (1)"]))

    importances = sorted(zip(FEATURE_COLS, model.feature_importances_),
                         key=lambda x: -x[1])
    print("Feature importances:")
    for name, imp in importances:
        bar = "█" * int(imp * 50)
        print(f"  {name:<20} {imp:.4f}  {bar}")

    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nSaved: {MODEL_PATH}, {SCALER_PATH}")


if __name__ == "__main__":
    train()
