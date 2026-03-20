"""
Intraday Resistance Breakout Strategy — Advanced Backtest (v3.0)

Key Improvements over v2.0:
  * Optimized parameters are extracted and reused for a clean final backtest
    (separates parameter search from performance reporting).
  * Best parameters printed per ticker + collected in a summary table.
  * VBT and ML signal generation also use best discovered parameters.
  * All v2.0 features preserved: ADX, EMA trend filter, ATR sizing, Vol Z-Score.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import datetime as dt
import os
# Suppress resource_tracker warnings across all processes (especially on macOS/Python 3.14)
os.environ['PYTHONWARNINGS'] = 'ignore:resource_tracker:UserWarning'

import multiprocessing
if os.name == 'posix':
    try:
        if multiprocessing.get_start_method(allow_none=True) is None:
            multiprocessing.set_start_method('spawn', force=True)
    except (RuntimeError, ValueError):
        pass

import backtesting
from backtesting import Strategy
backtesting.Pool = multiprocessing.Pool

import yfinance as yf
from indicators.atr import calculate_atr
from indicators.adx import calculate_adx
from utils.backtesting import VBTBacktester
from utils.strategy_runner import run_strategy_pipeline

# ----------------------------- CONFIG ----------------------------- #
TICKERS = ["NVDA", "AAPL", "GOOGL", "META", "AMZN", "MSFT", "TSLA"]
START_DATE = dt.datetime.today() - dt.timedelta(days=730)
END_DATE   = dt.datetime.today()
INTERVAL   = '1h'
CASH       = 100_000
COMMISSION = 0.001

# Default parameter fallback (used when optimization yields < 10 trades)
DEFAULT_PARAMS = dict(
    vol_z_threshold   = 1.5,
    atr_breakout_coef = 0.3,
    tp_factor         = 2.0,
    sl_factor         = 1.0,
    adx_threshold     = 20,
)


# ---------------------- INDICATOR HELPERS ---------------------- #
def _precompute_indicators(df, atr_period=20, roll_period=14, ema_period=50):
    """Pre-compute ATR, ADX, rolling breakout levels, Volume Z-Score, and EMA."""
    df = calculate_atr(df, atr_period)
    df = calculate_adx(df, period=20)

    df['roll_max_cp'] = df['High'].rolling(roll_period).max().shift(1)
    df['roll_min_cp'] = df['Low'].rolling(roll_period).min().shift(1)

    vol_mean = df['Volume'].rolling(roll_period).mean().shift(1)
    vol_std  = df['Volume'].rolling(roll_period).std().shift(1)
    df['vol_zscore'] = (df['Volume'] - vol_mean) / (vol_std + 1e-9)

    df['ema_trend'] = df['Close'].ewm(span=ema_period, adjust=False).mean()

    df.dropna(inplace=True)
    return df


# ---------------------- STRATEGY CLASS ---------------------- #
class BreakoutStrategy(Strategy):
    """
    Breakout strategy v3.0:
    - Buy when High breaks rolling max + a fraction of ATR
    - Sell when Low breaks rolling min - a fraction of ATR
    - Volume must be a statistical anomaly (Z-Score > threshold)
    - ADX must be > threshold (strong trend condition)
    - Trade must align with 50-EMA direction
    - ATR-based stop-loss and take-profit management
    - All key thresholds are optimizable
    """
    vol_z_threshold   = 1.5
    atr_breakout_coef = 0.3
    tp_factor         = 2.0
    sl_factor         = 1.0
    adx_threshold     = 20

    def init(self):
        self.atr        = self.I(lambda: self.data.ATR,          name='ATR',         overlay=False)
        self.roll_max   = self.I(lambda: self.data.roll_max_cp,  name='Resist',     overlay=True)
        self.roll_min   = self.I(lambda: self.data.roll_min_cp,  name='Support',     overlay=True)
        self.vol_zscore = self.I(lambda: self.data.vol_zscore,   name='Vol Z-Score', overlay=False)
        self.adx        = self.I(lambda: self.data.ADX,          name='ADX',         overlay=False)
        self.ema        = self.I(lambda: self.data.ema_trend,    name='EMA',         overlay=True)

    def next(self):
        high   = self.data.High[-1]
        low    = self.data.Low[-1]
        close  = self.data.Close[-1]
        atr    = self.atr[-1]
        r_max  = self.roll_max[-1]
        r_min  = self.roll_min[-1]
        vol_z  = self.vol_zscore[-1]
        adx_val= self.adx[-1]
        ema_val= self.ema[-1]

        vol_breakout   = vol_z > self.vol_z_threshold
        trend_strong   = adx_val > self.adx_threshold
        long_trigger   = r_max + (atr * self.atr_breakout_coef)
        short_trigger  = r_min - (atr * self.atr_breakout_coef)
        trend_up       = close > ema_val
        trend_down     = close < ema_val

        # ATR-based position sizing (risk 2% of equity per trade)
        equity = self.equity if hasattr(self, 'equity') else CASH
        shares = max(1, int((equity * 0.02) / atr)) if atr > 0 else 1

        if not self.position:
            if high >= long_trigger and vol_breakout and trend_strong and trend_up:
                self.buy(sl=close - atr * self.sl_factor,
                         tp=close + atr * self.tp_factor)
            elif low <= short_trigger and vol_breakout and trend_strong and trend_down:
                self.sell(sl=close + atr * self.sl_factor,
                          tp=close - atr * self.tp_factor)

        elif self.position.is_long:
            new_stop = close - atr * self.sl_factor
            if hasattr(self, '_long_stop'):
                self._long_stop = max(self._long_stop, new_stop)
            else:
                self._long_stop = new_stop
            for trade in self.trades:
                if trade.is_long and (trade.sl is None or self._long_stop > trade.sl):
                    trade.sl = self._long_stop

            if low <= short_trigger and vol_breakout and trend_strong and trend_down:
                self.position.close()
                self.sell(sl=close + atr * self.sl_factor,
                          tp=close - atr * self.tp_factor)

        elif self.position.is_short:
            new_stop = close + atr * self.sl_factor
            if hasattr(self, '_short_stop'):
                self._short_stop = min(self._short_stop, new_stop)
            else:
                self._short_stop = new_stop
            for trade in self.trades:
                if trade.is_short and (trade.sl is None or self._short_stop < trade.sl):
                    trade.sl = self._short_stop

            if high >= long_trigger and vol_breakout and trend_strong and trend_up:
                self.position.close()
                self.buy(sl=close - atr * self.sl_factor,
                         tp=close + atr * self.tp_factor)


# ---------------------- VBT SIGNAL HELPER ---------------------- #
def _generate_vbt_signals(df, vol_z_threshold=1.5, atr_coef=0.3, adx_threshold=20):
    vol_breakout  = df['vol_zscore'] > vol_z_threshold
    trend_strong  = df['ADX'] > adx_threshold
    trend_up      = df['Close'] > df['ema_trend']
    trend_down    = df['Close'] < df['ema_trend']
    long_trigger  = df['roll_max_cp'] + (df['ATR'] * atr_coef)
    short_trigger = df['roll_min_cp'] - (df['ATR'] * atr_coef)

    entries = (df['High'] >= long_trigger) & vol_breakout & trend_strong & trend_up
    exits   = (df['Low']  <= short_trigger) & vol_breakout & trend_strong & trend_down
    return entries, exits


# ----------------------------- MAIN ----------------------------- #
def main():
    print("=" * 70)
    print("  Resistance Breakout Strategy — Advanced Backtest (v3.0)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Download data
    # ------------------------------------------------------------------
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

    run_strategy_pipeline(
        strategy_name="Resistance Breakout Strategy",
        ohlcv_data=ohlcv,
        strategy_class=BreakoutStrategy,
        default_params=DEFAULT_PARAMS,
        param_grid=dict(
            vol_z_threshold   = list(np.arange(1.0, 3.0, 0.1)),
            atr_breakout_coef = list(np.arange(0.3, 0.7, 0.05)),
            tp_factor         = list(np.arange(1.0, 2.0, 0.1)),
            sl_factor         = list(np.arange(0.5, 1.0, 0.1)),
            adx_threshold     = list(np.arange(10, 30, 5)),
        ),
        precompute_fn=_precompute_indicators,
        vbt_signal_fn=_generate_vbt_signals,
        cash=CASH,
        commission=COMMISSION,
        freq='1h'
    )


if __name__ == '__main__':
    try:
        main()
    finally:
        # Explicitly shut down loky to prevent leaked semaphore warnings on exit
        try:
            from joblib.externals.loky import get_reusable_executor
            get_reusable_executor().shutdown(wait=True)
        except (ImportError, AttributeError):
            pass
