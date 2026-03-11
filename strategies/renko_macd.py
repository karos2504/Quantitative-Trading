"""
Renko + MACD Strategy — Advanced Backtest

Uses Renko brick patterns combined with MACD/Signal line slopes for
entry/exit signals.  Backtests on intraday data using backtesting.py,
then runs advanced validation via vectorbt.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy

from indicators.renko import convert_to_renko
from indicators.macd import calculate_macd
from indicators.slope import calculate_slope
from utils.backtesting import VBTBacktester

# ----------------------------- CONFIG ----------------------------- #
TICKERS = ["NVDA", "AAPL", "GOOGL", "META", "AMZN", "MSFT", "TSLA"]
CASH = 100_000
COMMISSION = 0.001


# ---------------------- INDICATOR HELPERS ---------------------- #
def _precompute_indicators(df):
    """Merge Renko + MACD + slopes into a single DataFrame."""
    renko = convert_to_renko(df)
    df = df.copy()
    df['Date'] = df.index
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    if not renko.empty:
        renko['date'] = pd.to_datetime(renko['date']).dt.tz_localize(None)
        renko.rename(columns={'date': 'Date'}, inplace=True)
        merged = df.merge(renko[['Date', 'bar_num']], how='outer', on='Date')
    else:
        merged = df.copy()
        merged['bar_num'] = np.nan

    merged['bar_num'] = merged['bar_num'].ffill()
    merged = calculate_macd(merged, fast=12, slow=26, signal=9)
    merged.dropna(subset=['MACD', 'Signal', 'bar_num'], inplace=True)

    if len(merged) >= 5:
        merged['macd_slope'] = calculate_slope(merged['MACD'], 5)
        merged['signal_slope'] = calculate_slope(merged['Signal'], 5)
    else:
        merged['macd_slope'] = np.nan
        merged['signal_slope'] = np.nan

    merged.dropna(inplace=True)

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col not in merged.columns:
            if col == 'Close' and 'Adj Close' in merged.columns:
                merged['Close'] = merged['Adj Close']
            elif col == 'Open' and 'Close' in merged.columns:
                merged['Open'] = merged['Close']

    merged.set_index('Date', inplace=True)
    merged.index.name = None
    return merged


# ---------------------- STRATEGY CLASS ---------------------- #
class RenkoMACDStrategy(Strategy):
    """Renko + MACD: Buy/sell on combined bar_num + MACD/Signal crossover + slope."""

    def init(self):
        self.bar_num = self.I(lambda: self.data.bar_num, name='bar_num', overlay=False)
        self.macd = self.I(lambda: self.data.MACD, name='MACD', overlay=False)
        self.signal = self.I(lambda: self.data.Signal, name='Signal', overlay=False)
        self.macd_slope = self.I(lambda: self.data.macd_slope, name='MACD Slope', overlay=False)
        self.signal_slope = self.I(lambda: self.data.signal_slope, name='Signal Slope', overlay=False)

    def next(self):
        bar = self.bar_num[-1]
        macd_val = self.macd[-1]
        sig_val = self.signal[-1]
        m_slope = self.macd_slope[-1]
        s_slope = self.signal_slope[-1]

        buy_signal = bar >= 2 and macd_val > sig_val and m_slope > s_slope
        sell_signal = bar <= -2 and macd_val < sig_val and m_slope < s_slope

        if not self.position:
            if buy_signal:
                self.buy()
            elif sell_signal:
                self.sell()
        elif self.position.is_long:
            if sell_signal:
                self.position.close()
                self.sell()
            elif macd_val < sig_val and m_slope < s_slope:
                self.position.close()
        elif self.position.is_short:
            if buy_signal:
                self.position.close()
                self.buy()
            elif macd_val > sig_val and m_slope > s_slope:
                self.position.close()


def _generate_vbt_signals(df):
    """Generate vectorbt entry/exit signals from indicator data."""
    entries = ((df['bar_num'] >= 2) &
               (df['MACD'] > df['Signal']) &
               (df['macd_slope'] > df['signal_slope']))
    exits = ((df['bar_num'] <= -2) &
             (df['MACD'] < df['Signal']) &
             (df['macd_slope'] < df['signal_slope']))
    return entries, exits


# ----------------------------- MAIN ----------------------------- #
def main():
    print("=" * 60)
    print("  Renko + MACD Strategy — Advanced Backtest")
    print("=" * 60)

    print("\n--- Downloading intraday data ---")
    ohlc_intraday = {}
    for ticker in TICKERS:
        try:
            data = yf.download(ticker, interval='15m', period='60d',
                               progress=False, auto_adjust=True)
            data.columns = ["Open", "High", "Low", "Adj Close", "Volume"]
            data['Close'] = data['Adj Close']
            data.dropna(inplace=True)
            ohlc_intraday[ticker] = data
            print(f"  ✅ {ticker}: {len(data)} rows")
        except Exception as e:
            print(f"  ❌ {ticker}: {e}")

    tickers = list(ohlc_intraday.keys())
    if not tickers:
        raise ValueError("No data downloaded.")

    # --- backtesting.py pass ---
    all_stats = {}
    for ticker in tickers:
        print(f"\n📊 Backtesting {ticker} (backtesting.py)...")
        try:
            df = _precompute_indicators(ohlc_intraday[ticker])
            if len(df) < 10:
                print(f"  ⚠️ Skipping {ticker}: insufficient data")
                continue

            bt = Backtest(df, RenkoMACDStrategy,
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
        print("--- 📈 Renko + MACD — backtesting.py Results ---")
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
            df = _precompute_indicators(ohlc_intraday[ticker])
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
            df = _precompute_indicators(ohlc_intraday[ticker])
            entries, exits = _generate_vbt_signals(df)
            run_ml_comparison(df, entries, exits, ticker, freq='15min')
        except Exception as e:
            print(f"  ❌ ML error for {ticker}: {e}")


if __name__ == '__main__':
    main()
