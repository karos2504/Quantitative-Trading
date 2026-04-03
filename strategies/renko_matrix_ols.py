"""
Renko + Matrix OLS Strategy
"""

import numpy as np
import sys
import os
import warnings
from pathlib import Path

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

import pandas as pd
import backtesting
backtesting.Pool = multiprocessing.Pool

from backtesting import Backtest, Strategy
from backtesting_engine.strategy_runner import run_strategy_pipeline
from backtesting_engine.walk_forward import run_walk_forward
from indicators.renko import convert_to_renko
from indicators.atr import calculate_atr
from indicators.vwap import calculate_vwap
from alpha_discovery.strategy_utils import align_indicator_data, standardize_ohlcv
from core.math_utils import calculate_matrix_ols_slope, renko_momentum
from config.settings import TICKERS, CASH, COMMISSION, TARGET_RISK, INTERVAL

# ─────────────────────────── CONFIG ─────────────────────────── #

DEFAULT_PARAMS = dict(
    score_threshold=3,      # Max score is now 4 (Renko=2, ER=2)
    er_th=0.3,              # Efficiency ratio gate
    ols_window=40,          # The lookback for our matrix math
    tp_atr=3.5,             # Take-profit in ATR multiples
    sl_atr=2.0,             # Stop-loss in ATR multiples
    time_stop_bars=15,      # Time stop in bars
)

# ──────────────────── GLOBAL PRECOMPUTATION ──────────────────── #

def _precompute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates purely reactive indicators and matrices. No lagging seasonal forecasts."""
    renko  = convert_to_renko(df)
    merged = align_indicator_data(df, renko, merge_col="bar_num")
    merged = calculate_atr(merged, period=14)
    merged = calculate_vwap(merged)

    # ── Core Momentum ──────────────────────────────────────────
    merged["renko_mom"]  = renko_momentum(merged["bar_num"], halflife=5)

    # ── Efficiency Ratio ───────────────────────────────────────
    er_period = 14
    change     = merged["Close"].diff(er_period).abs()
    volatility = merged["Close"].diff().abs().rolling(er_period).sum()
    merged["ER"] = change / volatility.replace(0, np.nan)

    # ── Realised Volatility ────────────────────────────────────
    log_ret = np.log(merged["Close"] / merged["Close"].shift(1))
    merged["RealVol"] = log_ret.rolling(22).std() * np.sqrt(252 * 6.5)

    # ── Predictive Math Regimes (Dynamic OLS Windows) ──────────
    # We precompute multiple lengths so the optimizer can find the best linear fit
    for w in [20, 30, 40, 50, 60]:
        merged[f"OLS_Slope_{w}"] = calculate_matrix_ols_slope(merged["Close"], window=w)

    required = ["bar_num", "ATR", "ER", "VWAP", "renko_mom", "RealVol", "OLS_Slope_40"]
    merged.dropna(subset=required, inplace=True)
    return standardize_ohlcv(merged)

# ────────────────────── SCORING HELPER ───────────────────────── #

def _bull_score(row, er_th):
    """
    Returns (bull_score, bear_score). Max score is 4.
    Weights: renko_mom=2, ER=2.
    """
    bull, bear = 0, 0

    # Renko Momentum
    if row["renko_mom"] > 0.1:   bull += 2
    elif row["renko_mom"] > 0.0: bull += 1
    if row["renko_mom"] < -0.1:  bear += 2
    elif row["renko_mom"] < 0.0: bear += 1

    # Efficiency Ratio Filter
    if row["ER"] > er_th:
        bull += 2
        bear += 2

    return bull, bear

# ─────────────────────── STRATEGY CLASS ──────────────────────── #

class RenkoMatrixStrategy(Strategy):
    score_threshold: int   = 3
    er_th:           float = 0.3
    ols_window:      int   = 40
    tp_atr:          float = 3.5
    sl_atr:          float = 2.0
    time_stop_bars:  int   = 15

    def init(self):
        self.bar_num    = self.I(lambda: self.data.bar_num,   name="bar_num")
        self.renko_mom  = self.I(lambda: self.data.renko_mom, name="renko_mom")
        self.atr        = self.I(lambda: self.data.ATR,       name="ATR")
        self.er         = self.I(lambda: self.data.ER,        name="ER")
        self.real_vol   = self.I(lambda: self.data.RealVol,   name="RealVol")
        
        # Dynamically fetch the correct precomputed OLS column based on the parameter
        col_name = f"OLS_Slope_{self.ols_window}"
        self.ols_slope  = self.I(lambda: getattr(self.data, col_name), name="OLS_Slope")

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
        real_vol = self.real_vol[-1]
        ols      = self.ols_slope[-1]
        bar_idx  = len(self.data) - 1

        row = {
            "renko_mom": self.renko_mom[-1],
            "ER":        self.er[-1],
        }
        bull_score, bear_score = _bull_score(row, self.er_th)

        # Pure Linear Math Filter
        math_bull = (ols > 0.0)
        math_bear = (ols < 0.0)

        bull_entry = (bull_score >= self.score_threshold) and math_bull
        bear_entry = (bear_score >= self.score_threshold) and math_bear

        # Volatility Scaling
        vol_scale = 1.0
        if real_vol > 0.6:   vol_scale = 0.50
        elif real_vol > 0.4: vol_scale = 0.75

        # --- Entry Logic ---
        if not self.position:
            if bull_entry:
                sz = max(1, int(self._vol_size(close, atr) * vol_scale))
                self.buy(size=sz, sl=close - atr * self.sl_atr, tp=close + atr * self.tp_atr)
                self._reset_trade_state()

            elif bear_entry:
                sz = max(1, int(self._vol_size(close, atr) * vol_scale))
                self.sell(size=sz, sl=close + atr * self.sl_atr, tp=close - atr * self.tp_atr)
                self._reset_trade_state()

        # --- Exit & Trailing Logic ---
        elif self.position.is_long:
            bars_held = bar_idx - self._trade_open_bar
            current_high = self.data.High[-1]

            decay_factor = 0.25 if bars_held > self.time_stop_bars else (0.50 if bars_held > int(self.time_stop_bars * 0.5) else 1.0)
            dynamic_sl_distance = atr * self.sl_atr * decay_factor

            if current_high > self._long_hwm:
                self._long_hwm = current_high
            new_trail = self._long_hwm - dynamic_sl_distance

            for trade in self.trades:
                if trade.is_long and (trade.sl is None or new_trail > trade.sl):
                    trade.sl = new_trail

            unrealised_r = (close - self._entry_price) / (atr * self.sl_atr + 1e-9)
            if bars_held >= (self.time_stop_bars * 1.5) and unrealised_r < 0.5:
                self.position.close()
                return

            # Reversal
            if bear_entry:
                self.position.close()
                sz = max(1, int(self._vol_size(close, atr) * vol_scale))
                self.sell(size=sz, sl=close + atr * self.sl_atr, tp=close - atr * self.tp_atr)
                self._reset_trade_state()

        elif self.position.is_short:
            bars_held = bar_idx - self._trade_open_bar
            current_low = self.data.Low[-1]

            decay_factor = 0.25 if bars_held > self.time_stop_bars else (0.50 if bars_held > int(self.time_stop_bars * 0.5) else 1.0)
            dynamic_sl_distance = atr * self.sl_atr * decay_factor

            if current_low < self._short_hwm:
                self._short_hwm = current_low
            new_trail = self._short_hwm + dynamic_sl_distance

            for trade in self.trades:
                if trade.is_short and (trade.sl is None or new_trail < trade.sl):
                    trade.sl = new_trail

            unrealised_r = (self._entry_price - close) / (atr * self.sl_atr + 1e-9)
            if bars_held >= (self.time_stop_bars * 1.5) and unrealised_r < 0.5:
                self.position.close()
                return

            # Reversal
            if bull_entry:
                self.position.close()
                sz = max(1, int(self._vol_size(close, atr) * vol_scale))
                self.buy(size=sz, sl=close - atr * self.sl_atr, tp=close + atr * self.tp_atr)
                self._reset_trade_state()

# ──────────────── VBT SIGNAL GENERATOR ───────────────────────── #

def _generate_vbt_signals(df, score_threshold=3, er_th=0.3, ols_window=40, **_):
    renko_mom_bull_strong = df["renko_mom"] > 0.1
    renko_mom_bull_weak   = (df["renko_mom"] > 0.0) & ~renko_mom_bull_strong
    renko_mom_bear_strong = df["renko_mom"] < -0.1
    renko_mom_bear_weak   = (df["renko_mom"] < 0.0) & ~renko_mom_bear_strong

    er_ok = df["ER"] > er_th

    col_name = f"OLS_Slope_{ols_window}"
    math_bull = df[col_name] > 0.0
    math_bear = df[col_name] < 0.0

    bull_score = (
        renko_mom_bull_strong.astype(int) * 2 +
        renko_mom_bull_weak.astype(int)   * 1 +
        er_ok.astype(int)                 * 2
    )
    
    bear_score = (
        renko_mom_bear_strong.astype(int) * 2 +
        renko_mom_bear_weak.astype(int)   * 1 +
        er_ok.astype(int)                 * 2
    )

    entries = math_bull & (bull_score >= score_threshold)
    exits   = math_bear & (bear_score >= score_threshold)
    return entries, exits

# ──────────────── WALK-FORWARD STRATEGY VARIANT ─────────────────── #

class RenkoMatrixStrategyWF(RenkoMatrixStrategy):
    def _vol_size(self, close, atr):
        if close <= 0: return 1
        return max(1, int(self.equity * 0.90 / close))

# ────────────────────────── MAIN ─────────────────────────────── #

def main():
    print("=" * 70)
    print("  Renko + Matrix Strategy (Lean OLS Optimization)")
    print("=" * 70)

    from data_ingestion.data_store import load_universe_data
    ohlc_intraday = load_universe_data(TICKERS, interval=INTERVAL)

    tickers = list(ohlc_intraday.keys())
    if not tickers: raise ValueError("No data loaded.")

    print("\n--- Global Precomputation ---")
    precomputed_data = {}
    for ticker in tickers:
        print(f"  Crunching matrix grids for {ticker}...")
        precomputed_data[ticker] = _precompute_indicators(ohlc_intraday[ticker])

    # NEW OPTIMIZATION GRID: Focused entirely on OLS geometric fit and ER noise filtering
    PARAM_GRID = dict(
        score_threshold = [2, 3, 4],
        er_th           = [0.2, 0.3, 0.4, 0.5],
        ols_window      = [20, 30, 40, 50, 60],
        tp_atr          = list(np.arange(2.0, 4.0, 0.5)),
        sl_atr          = [1.0, 1.5, 2.0],
        time_stop_bars  = [15, 25],
    )

    def _tp_gt_sl(p): return p.tp_atr > p.sl_atr
    def _maximize(stats): return stats["Sharpe Ratio"] if stats["# Trades"] >= 10 else -9999

    run_strategy_pipeline(
        strategy_name="Renko + Matrix Strategy",
        ohlcv_data=precomputed_data,
        strategy_class=RenkoMatrixStrategy,
        default_params=DEFAULT_PARAMS,
        param_grid=PARAM_GRID,
        precompute_fn=lambda x: x,
        vbt_signal_fn=_generate_vbt_signals,
        cash=CASH,
        commission=COMMISSION,
        freq=INTERVAL,
        maximize=_maximize,
        constraint=_tp_gt_sl,
        verbose=True
    )

    print("\n" + "=" * 70)
    print("  Extracting per-ticker best params for walk-forward …")
    print("=" * 70)

    best_params_map = {}
    for ticker in tickers:
        proc = precomputed_data[ticker]
        try:
            bt = Backtest(proc, RenkoMatrixStrategy, cash=CASH, commission=COMMISSION, trade_on_close=True)
            opt = bt.optimize(**PARAM_GRID, maximize=_maximize, constraint=_tp_gt_sl, return_heatmap=False)

            strat = getattr(opt, '_strategy', None)
            bp, errors = {}, []
            for k in PARAM_GRID:
                val = getattr(strat, k, None) if strat else None
                if val is None and isinstance(opt, pd.Series) and k in opt.index: val = opt[k]
                if val is None: errors.append(k)
                else: bp[k] = int(val) if isinstance(val, (np.integer, int)) else float(val) if isinstance(val, (np.floating, float)) else val

            if errors: raise KeyError(f"params not found: {errors}")

            best_params_map[ticker] = bp
            print(f"  ✅ {ticker}: {bp}")

        except Exception as e:
            best_params_map[ticker] = DEFAULT_PARAMS.copy()
            print(f"  ⚠️  {ticker}: extraction failed — {e}")

    print("\n" + "=" * 70)
    print("  Walk-Forward Out-of-Sample Validation")
    print("=" * 70)

    for ticker in tickers:
        run_walk_forward(
            ticker,
            precomputed_data[ticker],
            RenkoMatrixStrategyWF,
            best_params_map.get(ticker, DEFAULT_PARAMS),
            precompute_fn=lambda x: x
        )

if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            from joblib.externals.loky import get_reusable_executor
            get_reusable_executor().shutdown(wait=True)
        except (ImportError, AttributeError):
            pass
