"""
Renko + OBV Strategy — Advanced Backtest

Uses Renko brick patterns combined with OBV slope for entry/exit signals.
Backtests on intraday data using backtesting.py, then runs advanced
validation (Monte Carlo, Walk-Forward, Stress Test, DSR) via vectorbt.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy

from indicators.renko import convert_to_renko
from indicators.obv import calculate_obv
from indicators.slope import calculate_slope
from utils.backtesting import VBTBacktester

# ----------------------------- CONFIG ----------------------------- #
TICKERS = ["NVDA", "AAPL", "GOOGL", "META", "AMZN", "MSFT", "TSLA"]
CASH = 100_000
COMMISSION = 0.001


# ---------------------- INDICATOR HELPERS ---------------------- #
from indicators.renko import convert_to_renko
from indicators.obv import calculate_obv
from indicators.atr import calculate_atr
from utils.strategy_utils import align_indicator_data, standardize_ohlcv

# ---------------------- INDICATOR HELPERS ---------------------- #
def _precompute_indicators(df):
    """Merge Renko + OBV + OBV MA + ATR into a single DataFrame."""
    renko = convert_to_renko(df)
    
    # Use shared utility to align Renko bricks to intraday timeframe
    merged = align_indicator_data(df, renko, merge_col='bar_num')

    # Calculate OBV and moving average of OBV (to define trend)
    merged = calculate_obv(merged)
    merged['obv_ma'] = merged['OBV'].rolling(window=20).mean()

    # Calculate ATR for stops
    merged = calculate_atr(merged, period=14)
    
    # Kaufman's Efficiency Ratio (ER)
    er_period = 14
    change = merged['Close'].diff(er_period).abs()
    volatility = merged['Close'].diff().abs().rolling(er_period).sum()
    merged['ER'] = change / volatility.replace(0, np.nan)
    
    # OBV Rate of Change Z-Score (Momentum Anomaly)
    obv_roc = merged['OBV'].diff(5)
    obv_roc_mean = obv_roc.rolling(20).mean()
    obv_roc_std  = obv_roc.rolling(20).std()
    merged['OBV_Z'] = (obv_roc - obv_roc_mean) / (obv_roc_std + 1e-9)

    merged.dropna(subset=['bar_num', 'OBV', 'obv_ma', 'ATR', 'ER', 'OBV_Z'], inplace=True)
    return standardize_ohlcv(merged)


# ---------------------- STRATEGY CLASS ---------------------- #
class RenkoOBVStrategy(Strategy):
    """Renko + OBV: Buy bar_num>=2 & slope>30, Sell bar_num<=-2 & slope<-30."""

    def init(self):
        self.bar_num = self.I(lambda: self.data.bar_num, name='bar_num', overlay=False)
        self.obv = self.I(lambda: self.data.OBV, name='OBV', overlay=False)
        self.obv_ma = self.I(lambda: self.data.obv_ma, name='OBV_MA', overlay=False)
        self.atr = self.I(lambda: self.data.ATR, name='ATR', overlay=False)
        self.er = self.I(lambda: self.data.ER, name='Eff_Ratio', overlay=False)
        self.obv_z = self.I(lambda: self.data.OBV_Z, name='OBV_Z_Score', overlay=False)

    def next(self):
        bar = self.bar_num[-1]
        obv_val = self.obv[-1]
        obv_ma_val = self.obv_ma[-1]
        atr_val = self.atr[-1]
        er_val = self.er[-1]
        obv_z_val = self.obv_z[-1]
        close = self.data.Close[-1]

        # Require OBV above MA, high Efficiency Ratio (>0.3), and positive OBV momentum anomaly
        buy_signal = bar >= 2 and obv_val > obv_ma_val and er_val > 0.3 and obv_z_val > 1.0
        sell_signal = bar <= -2 and obv_val < obv_ma_val and er_val > 0.3 and obv_z_val < -1.0

        tp_factor = 1.5
        sl_factor = 1.0

        if not self.position:
            if buy_signal:
                self.buy(sl=close - atr_val * sl_factor, tp=close + atr_val * tp_factor)
            elif sell_signal:
                self.sell(sl=close + atr_val * sl_factor, tp=close - atr_val * tp_factor)
        elif self.position.is_long:
            new_stop = close - atr_val * sl_factor
            if hasattr(self, '_long_stop'):
                self._long_stop = max(self._long_stop, new_stop)
            else:
                self._long_stop = new_stop

            # Tighten SL dynamically on the active trade
            for trade in self.trades:
                if trade.is_long:
                    if trade.sl is None or self._long_stop > trade.sl:
                        trade.sl = self._long_stop

            if sell_signal or bar < 2:
                self.position.close()
                if sell_signal:
                    self.sell(sl=close + atr_val * sl_factor, tp=close - atr_val * tp_factor)
        elif self.position.is_short:
            new_stop = close + atr_val * sl_factor
            if hasattr(self, '_short_stop'):
                self._short_stop = min(self._short_stop, new_stop)
            else:
                self._short_stop = new_stop

            # Tighten SL dynamically on the active trade
            for trade in self.trades:
                if trade.is_short:
                    if trade.sl is None or self._short_stop < trade.sl:
                        trade.sl = self._short_stop

            if buy_signal or bar > -2:
                self.position.close()
                if buy_signal:
                    self.buy(sl=close - atr_val * sl_factor, tp=close + atr_val * tp_factor)


def _generate_vbt_signals(df):
    """Generate vectorbt entry/exit signals with ER and Z-score filters."""
    entries = (df['bar_num'] >= 2) & (df['OBV'] > df['obv_ma']) & (df['ER'] > 0.3) & (df['OBV_Z'] > 1.0)
    exits = (df['bar_num'] <= -2) & (df['OBV'] < df['obv_ma']) & (df['ER'] > 0.3) & (df['OBV_Z'] < -1.0)
    return entries, exits


# ----------------------------- MAIN ----------------------------- #
def main():
    print("=" * 60)
    print("  Renko + OBV Strategy — Advanced Backtest")
    print("=" * 60)

    print("\n--- Downloading intraday data ---")
    ohlc_intraday = {}
    for ticker in TICKERS:
        try:
            data = yf.download(ticker, interval='1h', period='730d',
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

            bt = Backtest(df, RenkoOBVStrategy,
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
        print("--- 📈 Renko + OBV — backtesting.py Results ---")
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
    # from utils.ml_signals import run_ml_comparison
    # for ticker in tickers:
    #     if ticker not in all_stats:
    #         continue
    #     try:
    #         df = _precompute_indicators(ohlc_intraday[ticker])
    #         entries, exits = _generate_vbt_signals(df)
    #         run_ml_comparison(df, entries, exits, ticker, freq='15min')
    #     except Exception as e:
    #         print(f"  ❌ ML error for {ticker}: {e}")


if __name__ == '__main__':
    main()
