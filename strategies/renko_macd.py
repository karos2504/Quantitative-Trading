"""
Renko + MACD Strategy — v6.1 (Institutional Math & Advanced Exits)

Upgrades vs v6.0:
═════════════════
Layer 3 (Exits) completely overhauled:
  • Dynamic Time-Decay: Stop loss systematically tightens (1.0x -> 0.5x -> 0.25x ATR) 
    the longer a trade sits without hitting its target, choking out stalled momentum.
  • True Chandelier Exit: High-watermark now anchors to the absolute High/Low of the 
    bar, rather than the Close, preventing intra-bar spikes from eating profits.
  • Hard Time Stop: Forces a close at 1.5x time limit if unrealized R is < 0.5.
"""

import numpy as np
import sys
import os
import warnings
from pathlib import Path
from datetime import timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["PYTHONWARNINGS"] = "ignore:resource_tracker:UserWarning"
warnings.filterwarnings("ignore")

import multiprocessing
if os.name == "posix":
    try:
        if multiprocessing.get_start_method(allow_none=True) is None:
            multiprocessing.set_start_method("spawn", force=True)
    except (RuntimeError, ValueError):
        pass

import yfinance as yf
import pandas as pd
import backtesting
backtesting.Pool = multiprocessing.Pool

from backtesting import Backtest, Strategy
from backtesting_engine.strategy_runner import run_strategy_pipeline
from indicators.renko import convert_to_renko
from indicators.macd import calculate_macd
from indicators.slope import calculate_slope
from indicators.atr import calculate_atr
from indicators.vwap import calculate_vwap
from backtesting_engine.backtesting import VBTBacktester
from alpha_discovery.strategy_utils import align_indicator_data, standardize_ohlcv

# ─────────────────────────── CONFIG ─────────────────────────── #
from config.settings import (
    TICKERS, CASH, COMMISSION, TARGET_RISK, MIN_TRADES, 
    WF_TRAIN_MONTHS, WF_TEST_MONTHS, INTERVAL
)

DEFAULT_PARAMS = dict(
    score_threshold=4,      # weighted score needed to enter (max is 6)
    er_th=0.3,              # efficiency ratio gate
    tp_atr=3.5,             # take-profit in ATR multiples
    sl_atr=2.0,             # stop-loss in ATR multiples
    time_stop_bars=15,      # time stop in bars
)

# ──────────────────── PREDICTIVE MATH HELPERS ────────────────── #

from core.math_utils import calculate_matrix_ols_slope, calculate_hw_trend

def _renko_momentum(bar_series: pd.Series, halflife: int = 5) -> pd.Series:
    """Continuous Renko momentum: exponentially-weighted sum of bar_num deltas."""
    delta = bar_series.diff().fillna(0)
    alpha = 1 - np.exp(-np.log(2) / halflife)
    return delta.ewm(alpha=alpha, adjust=False).mean()

# ──────────────────── GLOBAL PRECOMPUTATION ──────────────────── #

def _precompute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """All indicators and heavy forecasting math computed strictly once."""
    renko  = convert_to_renko(df)
    merged = align_indicator_data(df, renko, merge_col="bar_num")
    merged = calculate_macd(merged, fast=12, slow=26, signal=9)
    merged = calculate_atr(merged, period=14)
    merged = calculate_vwap(merged)

    # ── Core Momentum ──────────────────────────────────────────
    merged["renko_mom"]  = _renko_momentum(merged["bar_num"], halflife=5)
    merged["macd_hist"]  = merged["MACD"] - merged["Signal"]
    merged["hist_slope"] = merged["macd_hist"].diff(3)

    # ── Efficiency Ratio ───────────────────────────────────────
    er_period = 14
    change     = merged["Close"].diff(er_period).abs()
    volatility = merged["Close"].diff().abs().rolling(er_period).sum()
    merged["ER"] = change / volatility.replace(0, np.nan)

    # ── Realised Volatility ────────────────────────────────────
    log_ret = np.log(merged["Close"] / merged["Close"].shift(1))
    merged["RealVol"] = log_ret.rolling(22).std() * np.sqrt(252 * 6.5)

    # ── Predictive Math Regimes ────────────────────────────────
    merged["OLS_Slope"] = calculate_matrix_ols_slope(merged["Close"], window=40)
    merged["HW_Trend"]  = calculate_hw_trend(merged["Close"], window=120, seasonal_periods=24)

    required = ["MACD", "Signal", "bar_num", "ATR", "ER", "VWAP", 
                "OLS_Slope", "HW_Trend", "renko_mom", "hist_slope", "RealVol"]
    merged.dropna(subset=required, inplace=True)
    return standardize_ohlcv(merged)

# ────────────────────── SCORING HELPER ───────────────────────── #

def _bull_score(row, er_th):
    """
    Returns (bull_score, bear_score). Max score is now 6.
    Weights: renko_mom=2, hist_slope=2, ER=2.
    """
    bull, bear = 0, 0

    if row["renko_mom"] > 0.1: bull += 2
    elif row["renko_mom"] > 0.0: bull += 1
    if row["renko_mom"] < -0.1: bear += 2
    elif row["renko_mom"] < 0.0: bear += 1

    if row["hist_slope"] > 0: bull += 2
    if row["hist_slope"] < 0: bear += 2

    if row["ER"] > er_th:
        bull += 2
        bear += 2

    return bull, bear

# ─────────────────────── STRATEGY CLASS ──────────────────────── #

class RenkoMACDStrategyV6_1(Strategy):
    score_threshold: int   = 4      
    er_th:           float = 0.3    
    tp_atr:          float = 3.5    
    sl_atr:          float = 2.0    
    time_stop_bars:  int   = 15     

    def init(self):
        self.bar_num    = self.I(lambda: self.data.bar_num,     name="bar_num")
        self.renko_mom  = self.I(lambda: self.data.renko_mom,   name="renko_mom")
        self.hist_slope = self.I(lambda: self.data.hist_slope,  name="hist_slope")
        self.atr        = self.I(lambda: self.data.ATR,         name="ATR")
        self.er         = self.I(lambda: self.data.ER,          name="ER")
        self.real_vol   = self.I(lambda: self.data.RealVol,     name="RealVol")
        self.ols_slope  = self.I(lambda: self.data.OLS_Slope,   name="OLS_Slope")
        self.hw_trend   = self.I(lambda: self.data.HW_Trend,    name="HW_Trend")

        self._trade_open_bar:  int   = -1
        self._long_hwm:        float = -np.inf
        self._short_hwm:       float =  np.inf
        self._entry_price:     float = 0.0

    def _reset_trade_state(self):
        self._trade_open_bar = len(self.data) - 1
        self._long_hwm       = -np.inf
        self._short_hwm      =  np.inf
        self._entry_price    = self.data.Close[-1]

    def _vol_size(self, close, atr):
        risk_per_share = self.sl_atr * atr
        if risk_per_share <= 0 or close <= 0:
            return 1                             
        shares = TARGET_RISK / (risk_per_share * close)
        max_shares = (self.equity * 0.95) / close
        return max(1, int(min(shares, max_shares)))

    def next(self):
        close    = self.data.Close[-1]
        atr      = self.atr[-1]
        er       = self.er[-1]
        real_vol = self.real_vol[-1]
        ols      = self.ols_slope[-1]
        hw       = self.hw_trend[-1]
        bar_idx  = len(self.data) - 1

        # ── Compute entry score ────────────────────────────────
        row = {
            "renko_mom": self.renko_mom[-1],
            "hist_slope": self.hist_slope[-1],
            "ER":        er,
        }
        bull_score, bear_score = _bull_score(row, self.er_th)

        # ── Predictive Math Regimes ────────────────────────────
        math_bull = (ols > 0.0) and (hw > 0.0)
        math_bear = (ols < 0.0) and (hw < 0.0)

        bull_entry = (bull_score >= self.score_threshold) and math_bull
        bear_entry = (bear_score >= self.score_threshold) and math_bear

        # ── Vol regime sizing ──────────────────────────────────
        vol_scale = 1.0
        if real_vol > 0.6: vol_scale = 0.5         
        elif real_vol > 0.4: vol_scale = 0.75

        if not self.position:
            if bull_entry:
                sz = max(1, int(self._vol_size(close, atr) * vol_scale))
                self.buy(size=sz, sl=close - atr * self.sl_atr, tp=close + atr * self.tp_atr)
                self._reset_trade_state()

            elif bear_entry:
                sz = max(1, int(self._vol_size(close, atr) * vol_scale))
                self.sell(size=sz, sl=close + atr * self.sl_atr, tp=close - atr * self.tp_atr)
                self._reset_trade_state()

        # ── LAYER 3: V6.1 Advanced Exit Protocol ────────────────
        elif self.position.is_long:
            bars_held = bar_idx - self._trade_open_bar
            current_high = self.data.High[-1]

            # 1. Dynamic Time-Decay Multiplier
            decay_factor = 1.0
            if bars_held > self.time_stop_bars:
                decay_factor = 0.25
            elif bars_held > int(self.time_stop_bars * 0.5):
                decay_factor = 0.50

            dynamic_sl_distance = atr * self.sl_atr * decay_factor

            # 2. True Chandelier Exit (High-Watermark)
            if current_high > self._long_hwm:
                self._long_hwm = current_high
            
            new_trail = self._long_hwm - dynamic_sl_distance

            for trade in self.trades:
                if trade.is_long and (trade.sl is None or new_trail > trade.sl):
                    trade.sl = new_trail

            # 3. Hard Time Stop
            unrealised_r = (close - self._entry_price) / (atr * self.sl_atr + 1e-9)
            if bars_held >= (self.time_stop_bars * 1.5) and unrealised_r < 0.5:
                self.position.close()
                return

            if bear_entry:
                self.position.close()
                sz = max(1, int(self._vol_size(close, atr) * vol_scale))
                self.sell(size=sz, sl=close + atr * self.sl_atr, tp=close - atr * self.tp_atr)
                self._reset_trade_state()

        elif self.position.is_short:
            bars_held = bar_idx - self._trade_open_bar
            current_low = self.data.Low[-1]

            # 1. Dynamic Time-Decay Multiplier
            decay_factor = 1.0
            if bars_held > self.time_stop_bars:
                decay_factor = 0.25
            elif bars_held > int(self.time_stop_bars * 0.5):
                decay_factor = 0.50

            dynamic_sl_distance = atr * self.sl_atr * decay_factor

            # 2. True Chandelier Exit (Low-Watermark)
            if current_low < self._short_hwm:
                self._short_hwm = current_low
            
            new_trail = self._short_hwm + dynamic_sl_distance

            for trade in self.trades:
                if trade.is_short and (trade.sl is None or new_trail < trade.sl):
                    trade.sl = new_trail

            # 3. Hard Time Stop
            unrealised_r = (self._entry_price - close) / (atr * self.sl_atr + 1e-9)
            if bars_held >= (self.time_stop_bars * 1.5) and unrealised_r < 0.5:
                self.position.close()
                return

            if bull_entry:
                self.position.close()
                sz = max(1, int(self._vol_size(close, atr) * vol_scale))
                self.buy(size=sz, sl=close - atr * self.sl_atr, tp=close + atr * self.tp_atr)
                self._reset_trade_state()

# ──────────────── VBT SIGNAL GENERATOR ───────────────────────── #

def _generate_vbt_signals(df, score_threshold=4, er_th=0.3, **_):
    renko_mom_bull = df["renko_mom"] > 0.0
    renko_mom_bear = df["renko_mom"] < 0.0
    hist_bull      = df["hist_slope"] > 0
    hist_bear      = df["hist_slope"] < 0
    er_ok          = df["ER"] > er_th

    math_bull = (df["OLS_Slope"] > 0.0) & (df["HW_Trend"] > 0.0)
    math_bear = (df["OLS_Slope"] < 0.0) & (df["HW_Trend"] < 0.0)

    bull_score = (
        renko_mom_bull.astype(int) * 2 +
        hist_bull.astype(int)      * 2 +
        er_ok.astype(int)          * 2 
    )
    
    bear_score = (
        renko_mom_bear.astype(int) * 2 +
        hist_bear.astype(int)      * 2 +
        er_ok.astype(int)          * 2 
    )

    entries = math_bull & (bull_score >= score_threshold)
    exits   = math_bear & (bear_score >= score_threshold)
    return entries, exits

# ──────────────── WALK-FORWARD STRATEGY VARIANT ─────────────────── #

class RenkoMACDStrategyWF(RenkoMACDStrategyV6_1):
    def _vol_size(self, close, atr):
        if close <= 0: return 1
        return max(1, int(self.equity * 0.90 / close))

from backtesting_engine.walk_forward import run_walk_forward

# ────────────────────────── MAIN ─────────────────────────────── #

def main():
    print("=" * 70)
    print("  Renko + MACD Strategy — v6.1 (Institutional Math & Advanced Exits)")
    print("=" * 70)

    print(f"\n--- Loading {INTERVAL} intraday data from store ---")
    from data_ingestion.data_store import load_universe_data
    ohlc_intraday = load_universe_data(TICKERS, interval=INTERVAL)

    tickers = list(ohlc_intraday.keys())
    if not tickers: raise ValueError("No data loaded.")

    print("\n--- Global Precomputation (OLS Math & Holt-Winters) ---")
    precomputed_data = {}
    for ticker in tickers:
        print(f"  Crunching matrices and HW forecast for {ticker}...")
        precomputed_data[ticker] = _precompute_indicators(ohlc_intraday[ticker])

    PARAM_GRID = dict(
        score_threshold = [3, 4, 5, 6],
        er_th           = list(np.arange(0.2, 0.6, 0.1)),
        tp_atr          = list(np.arange(2.0, 4.0, 0.5)),
        sl_atr          = [1.0, 1.5, 2.0],
        time_stop_bars  = [15, 25],
    )

    def _tp_gt_sl(p): return p.tp_atr > p.sl_atr
    def _maximize(stats): return stats["Sharpe Ratio"] if stats["# Trades"] >= 10 else -9999

    run_strategy_pipeline(
        strategy_name="Renko + MACD Strategy v6.1",
        ohlcv_data=precomputed_data, 
        strategy_class=RenkoMACDStrategyV6_1,
        default_params=DEFAULT_PARAMS,
        param_grid=PARAM_GRID,
        precompute_fn=lambda x: x,  # Dummy function to pass precomputed data through
        vbt_signal_fn=_generate_vbt_signals,
        cash=CASH,
        commission=COMMISSION,
        freq=INTERVAL,
        maximize=_maximize,
        constraint=_tp_gt_sl,
    )

    print("\n" + "=" * 70)
    print("  Extracting per-ticker best params for walk-forward …")
    print("=" * 70)

    best_params_map = {}
    for ticker in tickers:
        proc = precomputed_data[ticker]
        extracted = False
        try:
            bt = Backtest(proc, RenkoMACDStrategyV6_1, cash=CASH, commission=COMMISSION, trade_on_close=True)
            def _tp_gt_sl_local(p): return p.tp_atr > p.sl_atr
            def _maximize_local(stats): return stats["Sharpe Ratio"] if stats["# Trades"] >= 10 else -9999

            opt = bt.optimize(**PARAM_GRID, maximize=_maximize_local, constraint=_tp_gt_sl_local, return_heatmap=False)

            strat = getattr(opt, '_strategy', None)
            bp, errors = {}, []
            for k in PARAM_GRID:
                val = getattr(strat, k, None) if strat else None
                if val is None and isinstance(opt, pd.Series) and k in opt.index: val = opt[k]
                if val is None: errors.append(k)
                else: bp[k] = int(val) if isinstance(val, (np.integer, int)) else float(val) if isinstance(val, (np.floating, float)) else val

            if errors: raise KeyError(f"params not found: {errors}")

            best_params_map[ticker] = bp
            extracted = True
            print(f"  ✅ {ticker}: {bp}")

        except Exception as e:
            best_params_map[ticker] = DEFAULT_PARAMS.copy()
            print(f"  ⚠️  {ticker}: extraction failed — {e}")

    print("\n" + "=" * 70)
    print("  Walk-Forward Out-of-Sample Validation (per-ticker best params)")
    print("=" * 70)

    for ticker in tickers:
        run_walk_forward(ticker, precomputed_data[ticker], RenkoMACDStrategyWF, best_params_map.get(ticker, DEFAULT_PARAMS), precompute_fn=lambda x: x)

if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            from joblib.externals.loky import get_reusable_executor
            get_reusable_executor().shutdown(wait=True)
        except (ImportError, AttributeError):
            pass