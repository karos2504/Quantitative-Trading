"""
Intraday Resistance Breakout Strategy — Advanced Backtest

Uses rolling high/low breakouts with volume confirmation and ATR-based
stop-losses.  Includes advanced validation via vectorbt + optional ML
signal enhancement.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import datetime as dt
from backtesting import Backtest, Strategy

import yfinance as yf
from indicators.atr import calculate_atr
from utils.backtesting import VBTBacktester

# ----------------------------- CONFIG ----------------------------- #
TICKERS = ["NVDA", "AAPL", "GOOGL", "META", "AMZN", "MSFT", "TSLA"]
START_DATE = dt.datetime.today() - dt.timedelta(days=60)
END_DATE = dt.datetime.today()
INTERVAL = '15m'
CASH = 100_000
COMMISSION = 0.001


# ---------------------- INDICATOR HELPERS ---------------------- #
def _precompute_indicators(df, atr_period=20, roll_period=20):
    """Pre-compute ATR and rolling breakout levels."""
    df = calculate_atr(df, atr_period)
    df['roll_max_cp'] = df['High'].rolling(roll_period).max().shift(1)
    df['roll_min_cp'] = df['Low'].rolling(roll_period).min().shift(1)
    df['roll_max_vol'] = df['Volume'].rolling(roll_period).max().shift(1)
    df.dropna(inplace=True)
    return df


# ---------------------- STRATEGY CLASS ---------------------- #
class BreakoutStrategy(Strategy):
    """
    Breakout strategy:
    - Buy when High breaks rolling max AND volume exceeds threshold
    - Sell when Low breaks rolling min AND volume exceeds threshold
    - ATR-based stop-loss management
    """
    vol_factor = 1.5

    def init(self):
        self.atr = self.I(lambda: self.data.ATR, name='ATR', overlay=False)
        self.roll_max = self.I(lambda: self.data.roll_max_cp, name='Resist.', overlay=True)
        self.roll_min = self.I(lambda: self.data.roll_min_cp, name='Support', overlay=True)
        self.roll_max_vol = self.I(lambda: self.data.roll_max_vol, name='Max Vol', overlay=False)

    def next(self):
        high = self.data.High[-1]
        low = self.data.Low[-1]
        close = self.data.Close[-1]
        vol = self.data.Volume[-1]
        atr = self.atr[-1]
        r_max = self.roll_max[-1]
        r_min = self.roll_min[-1]
        r_max_vol = self.roll_max_vol[-1]

        vol_breakout = vol > self.vol_factor * r_max_vol

        if not self.position:
            if high >= r_max and vol_breakout:
                self.buy(sl=close - atr)
            elif low <= r_min and vol_breakout:
                self.sell(sl=close + atr)

        elif self.position.is_long:
            new_stop = close - atr
            if hasattr(self, '_long_stop') and new_stop > self._long_stop:
                self._long_stop = new_stop
            else:
                self._long_stop = new_stop

            if low <= r_min and vol_breakout:
                self.position.close()
                self.sell(sl=close + atr)

        elif self.position.is_short:
            new_stop = close + atr
            if hasattr(self, '_short_stop') and new_stop < self._short_stop:
                self._short_stop = new_stop
            else:
                self._short_stop = new_stop

            if high >= r_max and vol_breakout:
                self.position.close()
                self.buy(sl=close - atr)


def _generate_vbt_signals(df, vol_factor=1.5):
    """Generate vectorbt entry/exit signals from breakout indicator data."""
    vol_breakout = df['Volume'] > vol_factor * df['roll_max_vol']
    entries = (df['High'] >= df['roll_max_cp']) & vol_breakout
    exits = (df['Low'] <= df['roll_min_cp']) & vol_breakout
    return entries, exits


# ----------------------------- MAIN ----------------------------- #
def main():
    print("=" * 60)
    print("  Resistance Breakout Strategy — Advanced Backtest")
    print("=" * 60)

    print("\nFetching intraday OHLCV data...")
    ohlcv = {}
    for ticker in TICKERS:
        try:
            data = yf.download(ticker, start=START_DATE, end=END_DATE,
                               interval=INTERVAL, progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.rename(columns={'Adj Close': 'Close'}, inplace=True, errors='ignore')
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            data = data.between_time('09:35', '16:00')
            if not data.empty:
                ohlcv[ticker] = data
                print(f"  ✅ {ticker}: {len(data)} rows")
        except Exception as e:
            print(f"  ❌ {ticker}: {e}")

    tickers = list(ohlcv.keys())
    if not tickers:
        print("No data fetched. Exiting.")
        return

    # --- backtesting.py pass ---
    all_stats = {}
    for ticker in tickers:
        print(f"\n📊 Backtesting {ticker} (backtesting.py)...")
        try:
            df = _precompute_indicators(ohlcv[ticker])
            if len(df) < 10:
                print(f"  ⚠️ Skipping {ticker}: insufficient data")
                continue

            bt = Backtest(df, BreakoutStrategy,
                          cash=CASH, commission=COMMISSION,
                          exclusive_orders=True, finalize_trades=True)
            stats = bt.run()
            all_stats[ticker] = {
                'Return [%]': stats['Return [%]'],
                'Sharpe Ratio': stats['Sharpe Ratio'],
                'Max Drawdown [%]': stats['Max. Drawdown [%]'],
                '# Trades': stats['# Trades'],
                'Win Rate [%]': stats['Win Rate [%]'],
            }
            print(f"  Return: {stats['Return [%]']:.2f}%  "
                  f"Sharpe: {stats['Sharpe Ratio']:.2f}  "
                  f"Max DD: {stats['Max. Drawdown [%]']:.2f}%  "
                  f"Trades: {stats['# Trades']}")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    if all_stats:
        print("\n" + "=" * 60)
        print("--- 📈 Resistance Breakout — backtesting.py Results ---")
        print("=" * 60)
        print(pd.DataFrame(all_stats).T.to_string(float_format=lambda x: f"{x:.2f}"))

    # --- vectorbt advanced analysis (per-ticker) ---
    for ticker in tickers:
        if ticker not in all_stats:
            continue
        print(f"\n{'=' * 60}")
        print(f"  🔬 Advanced Analysis: {ticker}")
        print(f"{'=' * 60}")

        try:
            df = _precompute_indicators(ohlcv[ticker])
            entries, exits = _generate_vbt_signals(df)

            bt_vbt = VBTBacktester(
                close=df['Close'],
                entries=entries,
                exits=exits,
                freq='15min',
                init_cash=CASH,
                commission=COMMISSION,
            )
            bt_vbt.full_analysis(n_mc=500, n_wf_splits=4, n_trials=len(TICKERS))
        except Exception as e:
            print(f"  ❌ VBT analysis error: {e}")

    # --- ML/DL/RL Signal Enhancement ---
    from utils.ml_signals import run_ml_comparison
    for ticker in tickers:
        if ticker not in all_stats:
            continue
        try:
            df = _precompute_indicators(ohlcv[ticker])
            entries, exits = _generate_vbt_signals(df)
            run_ml_comparison(df, entries, exits, ticker, freq='15min')
        except Exception as e:
            print(f"  ❌ ML error for {ticker}: {e}")


if __name__ == '__main__':
    main()
