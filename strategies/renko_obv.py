"""
Renko + OBV Strategy — Advanced Backtest (v3.0)

Key Improvements over v2.0:
  * Optimized parameters are extracted and reused for a clean final backtest
    (separates the parameter search from performance reporting).
  * Best parameters printed per ticker + collected in a summary table.
  * VBT and ML signal generation also use best discovered parameters.
  * All v2.0 features preserved: ADX, BB squeeze, ER, OBV Z-score.
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
from indicators.obv import calculate_obv
from indicators.atr import calculate_atr
from indicators.adx import calculate_adx
from indicators.bollinger_bands import calculate_bollinger_bands
from utils.backtesting import VBTBacktester
from utils.strategy_utils import align_indicator_data, standardize_ohlcv

# ----------------------------- CONFIG ----------------------------- #
TICKERS = ["NVDA", "AAPL", "GOOGL", "META", "AMZN", "MSFT", "TSLA"]
CASH = 100_000
COMMISSION = 0.001

# Default parameter fallback (used when optimization yields < 10 trades)
DEFAULT_PARAMS = dict(
    bar_threshold  = 6,
    er_threshold   = 0.65,
    obv_z_threshold= 0.8,
    tp_factor      = 1.5,
    sl_factor      = 1.0,
    adx_threshold  = 20,
)


# ---------------------- INDICATOR HELPERS ---------------------- #
def _precompute_indicators(df):
    """Merge Renko + OBV + OBV MA + ATR + ADX + Bollinger into a single DataFrame."""
    renko  = convert_to_renko(df)
    merged = align_indicator_data(df, renko, merge_col='bar_num')

    merged = calculate_obv(merged)
    merged['obv_ma'] = merged['OBV'].rolling(window=20).mean()
    merged = calculate_atr(merged, period=14)
    merged = calculate_adx(merged, period=20)
    merged = calculate_bollinger_bands(merged, window=20, std_dev=2)
    merged['BB_Width_Pct'] = merged['BB_Width'].rolling(100).rank(pct=True)
    merged['BB_Expanding'] = merged['BB_Width'].diff() > 0

    # Kaufman's Efficiency Ratio
    er_period  = 14
    change     = merged['Close'].diff(er_period).abs()
    volatility = merged['Close'].diff().abs().rolling(er_period).sum()
    merged['ER'] = change / volatility.replace(0, np.nan)

    # OBV Rate-of-Change Z-Score
    obv_roc      = merged['OBV'].diff(5)
    obv_roc_mean = obv_roc.rolling(20).mean()
    obv_roc_std  = obv_roc.rolling(20).std()
    merged['OBV_Z'] = (obv_roc - obv_roc_mean) / (obv_roc_std + 1e-9)

    merged.dropna(subset=['bar_num', 'OBV', 'obv_ma', 'ATR', 'ER',
                           'OBV_Z', 'ADX', 'BB_Width_Pct'], inplace=True)
    return standardize_ohlcv(merged)


# ---------------------- STRATEGY CLASS (OPTIMIZABLE) ---------------------- #
class RenkoOBVStrategy(Strategy):
    """Renko + OBV with ADX filter, BB squeeze, and full optimization."""

    # === OPTIMIZABLE PARAMETERS ===
    bar_threshold   = 6
    er_threshold    = 0.65
    obv_z_threshold = 0.8
    tp_factor       = 1.5
    sl_factor       = 1.0
    adx_threshold   = 20

    def init(self):
        self.bar_num     = self.I(lambda: self.data.bar_num,      name='bar_num',    overlay=False)
        self.obv         = self.I(lambda: self.data.OBV,          name='OBV',        overlay=False)
        self.obv_ma      = self.I(lambda: self.data.obv_ma,       name='OBV_MA',     overlay=False)
        self.atr         = self.I(lambda: self.data.ATR,          name='ATR',        overlay=False)
        self.er          = self.I(lambda: self.data.ER,           name='Eff_Ratio',  overlay=False)
        self.obv_z       = self.I(lambda: self.data.OBV_Z,        name='OBV_Z',      overlay=False)
        self.adx         = self.I(lambda: self.data.ADX,          name='ADX',        overlay=False)
        self.bb_expanding= self.I(lambda: self.data.BB_Expanding, name='BB_Expand',  overlay=False)
        self.bb_width_pct= self.I(lambda: self.data.BB_Width_Pct, name='BB_W_Pct',   overlay=False)

    def next(self):
        bar        = self.bar_num[-1]
        obv_val    = self.obv[-1]
        obv_ma_val = self.obv_ma[-1]
        atr_val    = self.atr[-1]
        er_val     = self.er[-1]
        obv_z_val  = self.obv_z[-1]
        adx_val    = self.adx[-1]
        bb_exp     = self.bb_expanding[-1]
        bb_w_pct   = self.bb_width_pct[-1]
        close      = self.data.Close[-1]

        trend_strong  = adx_val > self.adx_threshold
        vol_expanding = bb_exp and bb_w_pct > 0.2

        buy_signal = (
            bar >= self.bar_threshold and
            obv_val > obv_ma_val and
            er_val > self.er_threshold and
            obv_z_val > self.obv_z_threshold and
            trend_strong and
            vol_expanding
        )
        sell_signal = (
            bar <= -self.bar_threshold and
            obv_val < obv_ma_val and
            er_val > self.er_threshold and
            obv_z_val < -self.obv_z_threshold and
            trend_strong and
            vol_expanding
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

            if sell_signal or bar < self.bar_threshold:
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

            if buy_signal or bar > -self.bar_threshold:
                self.position.close()
                if buy_signal:
                    self.buy(sl=close - atr_val * self.sl_factor,
                             tp=close + atr_val * self.tp_factor)


# ---------------------- VBT SIGNAL HELPER ---------------------- #
def _generate_vbt_signals(df, bar_threshold=6, er_threshold=0.65,
                           obv_z_threshold=0.8, adx_threshold=20):
    trend_strong  = df['ADX'] > adx_threshold
    vol_expanding = df['BB_Expanding'] & (df['BB_Width_Pct'] > 0.2)

    entries = (
        (df['bar_num'] >= bar_threshold) &
        (df['OBV'] > df['obv_ma']) &
        (df['ER'] > er_threshold) &
        (df['OBV_Z'] > obv_z_threshold) &
        trend_strong &
        vol_expanding
    )
    exits = (
        (df['bar_num'] <= -bar_threshold) &
        (df['OBV'] < df['obv_ma']) &
        (df['ER'] > er_threshold) &
        (df['OBV_Z'] < -obv_z_threshold) &
        trend_strong &
        vol_expanding
    )
    return entries, exits


# ----------------------------- MAIN ----------------------------- #
def main():
    print("=" * 70)
    print("  Renko + OBV Strategy — Advanced Backtest (v3.0)")
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
        strategy_name="Renko + OBV Strategy",
        ohlcv_data=ohlc_intraday,
        strategy_class=RenkoOBVStrategy,
        default_params=DEFAULT_PARAMS,
        param_grid=dict(
            bar_threshold   = list(range(3, 7, 1)),
            er_threshold    = list(np.arange(0.3, 0.7, 0.1)),
            tp_factor       = list(np.arange(1.0, 2.0, 0.1)),
            sl_factor       = list(np.arange(0.5, 1.0, 0.1)),
            obv_z_threshold = list(np.arange(1.0, 3.0, 0.1)),
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
