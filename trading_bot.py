import yfinance as yf
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy

class TrailingStopStrategy(Strategy):
    rsi_period = 14
    rsi_low = 35
    rsi_high = 65
    adx_period = 14
    atr_period = 14

    def init(self):
        close_series = pd.Series(self.data.Close)
        high_series = pd.Series(self.data.High)
        low_series = pd.Series(self.data.Low)

        self.rsi = self.I(ta.rsi, close_series, length=self.rsi_period)

        macd_df = ta.macd(close_series)
        self.macd_line = self.I(lambda: macd_df.iloc[:, 0])
        self.macd_signal = self.I(lambda: macd_df.iloc[:, 2])

        self.fast_ma = self.I(ta.sma, close_series, length=50)
        self.slow_ma = self.I(ta.sma, close_series, length=200)

        adx_df = ta.adx(high_series, low_series,
                        close_series, length=self.adx_period)
        self.adx = self.I(lambda: adx_df.iloc[:, 0])
        self.atr = self.I(ta.atr, high_series, low_series,
                         close_series, length=self.atr_period)

        # Track manually
        self.entry_price = 0
        self.take_profit_price = 0
        self.trailing_sl = 0

    def next(self):
        if len(self.adx) < 200:
            return

        current_adx = self.adx[-1]
        current_rsi = self.rsi[-1]
        current_price = self.data.Close[-1]
        current_atr = self.atr[-1]

        is_uptrend = self.fast_ma[-1] > self.slow_ma[-1]
        macd_cross_up = self.macd_line[-1] > self.macd_signal[-1]

        # --- MANAGE OPEN POSITION ---
        if self.position:

            # Manual trailing stop — move up as price rises
            new_sl = current_price - (2 * current_atr)
            if new_sl > self.trailing_sl:
                self.trailing_sl = new_sl
                self.position.sl = self.trailing_sl

            # Breakeven trigger — if 50% to target, move stop to entry
            halfway = self.entry_price + (
                (self.take_profit_price - self.entry_price) * 0.5
            )
            if current_price >= halfway:
                if self.position.sl < self.entry_price:
                    self.position.sl = self.entry_price

            # Take profit hit manually
            if current_price >= self.take_profit_price:
                self.position.close()
                return

            # RSI overbought exit
            if current_rsi > self.rsi_high:
                self.position.close()
                return

            return

        # --- ENTRY LOGIC ---
        entry_triggered = False

        if current_adx > 25 and is_uptrend and macd_cross_up:
            entry_triggered = True
        elif current_adx <= 25 and current_rsi < self.rsi_low:
            entry_triggered = True

        if entry_triggered:
            current_atr_val = 2 * current_atr
            self.entry_price = current_price
            self.take_profit_price = current_price + (4 * current_atr)
            self.trailing_sl = current_price - current_atr_val

            self.buy(
                sl=self.trailing_sl,
                tp=self.take_profit_price
            )

if __name__ == "__main__":
    TICKER = "RELIANCE.NS"
    df = yf.download(TICKER, start="2015-01-01", end="2025-01-01")
    df.columns = [col[0] if isinstance(col, tuple) else col
                 for col in df.columns]
    df = df.dropna()

    bt = Backtest(df, TrailingStopStrategy,
                 cash=10000, commission=0.001)
    stats = bt.run()
    print(stats)
    bt.plot()