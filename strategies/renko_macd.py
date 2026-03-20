"""
Renko + MACD Strategy — ADVANCED & IMPROVED Backtest (v4.0)

Key Improvements over v3.0:
  * Optimized parameters are extracted and reused for a dedicated final backtest
    instead of reporting the optimization run itself (avoids in-sample bias
    reporting — the final bt.run(**best_params) is a clean, single-pass run
    using the best discovered parameters, clearly separated from the search).
  * Best parameters are printed per ticker and collected in a summary table.
  * VBT signal generation also uses the best discovered parameters.
  * All v3.0 features preserved: Stochastic RSI, VWAP proximity, ML hooks.
"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

import yfinance as yf
import backtesting
backtesting.Pool = multiprocessing.Pool

from backtesting import Backtest, Strategy
from utils.strategy_runner import run_strategy_pipeline
from indicators.renko import convert_to_renko
from indicators.macd import calculate_macd
from indicators.slope import calculate_slope
from indicators.rsi import calculate_rsi
from indicators.atr import calculate_atr
from indicators.stochastic import calculate_stochastic
from indicators.vwap import calculate_vwap
from utils.backtesting import VBTBacktester
from utils.strategy_utils import align_indicator_data, standardize_ohlcv

# ----------------------------- CONFIG ----------------------------- #
TICKERS = ["NVDA", "AAPL", "GOOGL", "META", "AMZN", "MSFT", "TSLA"]
CASH = 100_000
COMMISSION = 0.001

# Default parameter fallback (used when optimization yields < 10 trades)
DEFAULT_PARAMS = dict(
    bar_threshold=3,
    rsi_threshold=50,
    er_th=0.3,
    tp_factor=1.5,
    sl_factor=1.0,
    vol_ratio_th=0.8,
    vwap_dist_max=3.0,
)


# ---------------------- INDICATOR HELPERS ---------------------- #
def _precompute_indicators(df):
    """Merge Renko + MACD + RSI + ATR + ER + Volume + Stochastic + VWAP."""
    renko = convert_to_renko(df)
    merged = align_indicator_data(df, renko, merge_col='bar_num')
    merged = calculate_macd(merged, fast=12, slow=26, signal=9)
    merged = calculate_rsi(merged, period=14)
    merged = calculate_atr(merged, period=14)
    merged = calculate_stochastic(merged, k_period=14, d_period=3, smooth_k=3)
    merged = calculate_vwap(merged)

    # Kaufman's Efficiency Ratio
    er_period = 14
    change = merged['Close'].diff(er_period).abs()
    volatility = merged['Close'].diff().abs().rolling(er_period).sum()
    merged['ER'] = change / volatility.replace(0, np.nan)

    # Volume filter
    merged['Vol_SMA'] = merged['Volume'].rolling(window=20).mean()
    merged['Vol_Ratio'] = merged['Volume'] / merged['Vol_SMA'].replace(0, np.nan)

    # VWAP proximity (distance in ATR units)
    merged['VWAP_Dist'] = (merged['Close'] - merged['VWAP']).abs() / merged['ATR'].replace(0, np.nan)

    merged.dropna(subset=['MACD', 'Signal', 'bar_num', 'RSI', 'ATR', 'ER',
                          'Stoch_K', 'Stoch_D', 'VWAP'], inplace=True)

    if len(merged) >= 5:
        merged['macd_slope'] = calculate_slope(merged['MACD'], 5)
        merged['signal_slope'] = calculate_slope(merged['Signal'], 5)
    else:
        merged['macd_slope'] = np.nan
        merged['signal_slope'] = np.nan

    merged.dropna(inplace=True)
    return standardize_ohlcv(merged)


# ---------------------- STRATEGY CLASS (OPTIMIZABLE) ---------------------- #
class RenkoMACDStrategy(Strategy):
    """Renko + MACD with Stochastic + VWAP filters + full optimization."""

    # === OPTIMIZABLE PARAMETERS ===
    bar_threshold: int = 3
    rsi_threshold: int = 50
    er_th: float = 0.3
    tp_factor: float = 1.5
    sl_factor: float = 1.0
    vol_ratio_th: float = 0.8
    vwap_dist_max: float = 3.0

    def init(self):
        self.bar_num     = self.I(lambda: self.data.bar_num,      name='bar_num',    overlay=False)
        self.macd        = self.I(lambda: self.data.MACD,         name='MACD',       overlay=False)
        self.signal      = self.I(lambda: self.data.Signal,       name='Signal',     overlay=False)
        self.macd_slope  = self.I(lambda: self.data.macd_slope,   name='MACD Slope', overlay=False)
        self.signal_slope= self.I(lambda: self.data.signal_slope, name='Sig Slope',  overlay=False)
        self.rsi         = self.I(lambda: self.data.RSI,          name='RSI',        overlay=False)
        self.atr         = self.I(lambda: self.data.ATR,          name='ATR',        overlay=False)
        self.er          = self.I(lambda: self.data.ER,           name='Eff_Ratio',  overlay=False)
        self.vol_ratio   = self.I(lambda: self.data.Vol_Ratio,    name='Vol_Ratio',  overlay=False)
        self.stoch_k     = self.I(lambda: self.data.Stoch_K,      name='Stoch_K',    overlay=False)
        self.stoch_d     = self.I(lambda: self.data.Stoch_D,      name='Stoch_D',    overlay=False)
        self.vwap_dist   = self.I(lambda: self.data.VWAP_Dist,    name='VWAP_Dist',  overlay=False)

    def next(self):
        bar      = self.bar_num[-1]
        macd_val = self.macd[-1]
        sig_val  = self.signal[-1]
        m_slope  = self.macd_slope[-1]
        s_slope  = self.signal_slope[-1]
        rsi_val  = self.rsi[-1]
        atr_val  = self.atr[-1]
        er_val   = self.er[-1]
        vol_val  = self.vol_ratio[-1]
        stoch_k  = self.stoch_k[-1]
        stoch_d  = self.stoch_d[-1]
        vwap_d   = self.vwap_dist[-1]
        close    = self.data.Close[-1]

        stoch_bull = stoch_k > stoch_d
        stoch_bear = stoch_k < stoch_d
        near_vwap  = vwap_d <= self.vwap_dist_max

        buy_signal = (
            bar >= self.bar_threshold and
            macd_val > sig_val and
            m_slope > s_slope and
            rsi_val > self.rsi_threshold and
            er_val > self.er_th and
            vol_val > self.vol_ratio_th and
            stoch_bull and
            near_vwap
        )
        sell_signal = (
            bar <= -self.bar_threshold and
            macd_val < sig_val and
            m_slope < s_slope and
            rsi_val < self.rsi_threshold and
            er_val > self.er_th and
            vol_val > self.vol_ratio_th and
            stoch_bear and
            near_vwap
        )

        if not self.position:
            if buy_signal:
                self.buy(sl=close - atr_val * self.sl_factor,
                         tp=close + atr_val * self.tp_factor)
            elif sell_signal:
                self.sell(sl=close + atr_val * self.sl_factor,
                          tp=close - atr_val * self.tp_factor)

        elif self.position.is_long:
            new_stop = close - atr_val * self.sl_factor
            if not hasattr(self, '_long_stop') or new_stop > self._long_stop:
                self._long_stop = new_stop
            for trade in self.trades:
                if trade.is_long and (trade.sl is None or self._long_stop > trade.sl):
                    trade.sl = self._long_stop

            if sell_signal or (macd_val < sig_val and m_slope < s_slope):
                self.position.close()
                if sell_signal:
                    self.sell(sl=close + atr_val * self.sl_factor,
                              tp=close - atr_val * self.tp_factor)

        elif self.position.is_short:
            new_stop = close + atr_val * self.sl_factor
            if not hasattr(self, '_short_stop') or new_stop < self._short_stop:
                self._short_stop = new_stop
            for trade in self.trades:
                if trade.is_short and (trade.sl is None or self._short_stop < trade.sl):
                    trade.sl = self._short_stop

            if buy_signal or (macd_val > sig_val and m_slope > s_slope):
                self.position.close()
                if buy_signal:
                    self.buy(sl=close - atr_val * self.sl_factor,
                             tp=close + atr_val * self.tp_factor)


# ---------------------- VBT SIGNAL HELPER ---------------------- #
def _generate_vbt_signals(df, bar_threshold=3, rsi_threshold=50,
                           er_th=0.3, vol_ratio_th=0.8, vwap_dist_max=3.0):
    entries = (
        (df['bar_num'] >= bar_threshold) &
        (df['MACD'] > df['Signal']) &
        (df['macd_slope'] > df['signal_slope']) &
        (df['RSI'] > rsi_threshold) &
        (df['ER'] > er_th) &
        (df['Vol_Ratio'] > vol_ratio_th) &
        (df['Stoch_K'] > df['Stoch_D']) &
        (df['VWAP_Dist'] <= vwap_dist_max)
    )
    exits = (
        (df['bar_num'] <= -bar_threshold) &
        (df['MACD'] < df['Signal']) &
        (df['macd_slope'] < df['signal_slope']) &
        (df['RSI'] < rsi_threshold) &
        (df['ER'] > er_th) &
        (df['Vol_Ratio'] > vol_ratio_th) &
        (df['Stoch_K'] < df['Stoch_D']) &
        (df['VWAP_Dist'] <= vwap_dist_max)
    )
    return entries, exits


# ----------------------------- MAIN ----------------------------- #
def main():
    print("=" * 70)
    print("  Renko + MACD Strategy — ADVANCED & IMPROVED Backtest (v4.0)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Download data
    # ------------------------------------------------------------------
    print("\n--- Downloading 1h intraday data (2 years) ---")
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

    run_strategy_pipeline(
        strategy_name="Renko + MACD Strategy",
        ohlcv_data=ohlc_intraday,
        strategy_class=RenkoMACDStrategy,
        default_params=DEFAULT_PARAMS,
        param_grid=dict(
            bar_threshold = list(range(3, 7, 1)),
            rsi_threshold = list(range(30, 70, 5)),
            er_th         = list(np.arange(0.3, 0.7, 0.1)),
            tp_factor     = list(np.arange(1.0, 2.0, 0.1)),
            sl_factor     = list(np.arange(0.5, 1.0, 0.1)),
            vol_ratio_th  = list(np.arange(0.3, 0.7, 0.1)),
            vwap_dist_max = list(np.arange(1.0, 5.0, 0.1)),
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
