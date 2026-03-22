"""
Renko + MACD + OBV Hybrid Strategy
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
from indicators.obv import calculate_obv
from indicators.slope import calculate_slope
from indicators.rsi import calculate_rsi
from indicators.atr import calculate_atr
from indicators.stochastic import calculate_stochastic
from indicators.vwap import calculate_vwap
from alpha_discovery.strategy_utils import align_indicator_data, standardize_ohlcv
from core.math_utils import renko_momentum

# ─────────────────────────── CONFIG ─────────────────────────── #
from config.settings import (
    TICKERS, CASH, COMMISSION, TARGET_RISK, MIN_TRADES, 
    WF_TRAIN_MONTHS, WF_TEST_MONTHS, INTERVAL
)

DEFAULT_PARAMS = dict(
    score_threshold=5,      
    er_th=0.3,              
    adx_th=20,              
    tp_atr=3.5,             
    sl_atr=2.0,             
    time_stop_bars=15,      
)

from indicators.adx import calculate_adx

def _precompute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    renko  = convert_to_renko(df)
    merged = align_indicator_data(df, renko, merge_col="bar_num")
    merged = calculate_macd(merged, fast=12, slow=26, signal=9)
    merged = calculate_rsi(merged, period=14)
    merged = calculate_atr(merged, period=14)
    merged = calculate_stochastic(merged, k_period=14, d_period=3, smooth_k=3)
    
    # ── HYBRID ADDITION: OBV Z-Score ───────────────────────────
    merged = calculate_obv(merged)
    obv_roc = merged["OBV"].diff(5)
    obv_roc_mean = obv_roc.rolling(20).mean()
    obv_roc_std = obv_roc.rolling(20).std()
    merged["OBV_Z"] = (obv_roc - obv_roc_mean) / (obv_roc_std + 1e-9)

    merged["renko_mom"] = renko_momentum(merged["bar_num"], halflife=5)
    merged["macd_hist"]  = merged["MACD"] - merged["Signal"]
    merged["hist_slope"] = merged["macd_hist"].diff(3)          

    er_period = 14
    change     = merged["Close"].diff(er_period).abs()
    volatility = merged["Close"].diff().abs().rolling(er_period).sum()
    merged["ER"] = change / volatility.replace(0, np.nan)

    merged = calculate_adx(merged, period=14)
    merged["EMA200"] = merged["Close"].ewm(span=200, adjust=False).mean()
    merged["EMA200_slope"] = merged["EMA200"].diff(5) / 5

    log_ret = np.log(merged["Close"] / merged["Close"].shift(1))
    merged["RealVol"] = log_ret.rolling(22).std() * np.sqrt(252 * 6.5)

    required = ["MACD", "Signal", "bar_num", "RSI", "ATR", "ER",
                "Stoch_K", "Stoch_D", "ADX", "EMA200",
                "EMA200_slope", "renko_mom", "hist_slope", "RealVol", "OBV_Z"]
    
    merged.dropna(subset=required, inplace=True)
    merged.dropna(inplace=True)
    return standardize_ohlcv(merged)


# ────────────────────── SCORING HELPER ───────────────────────── #
def _bull_score(row, er_th):
    """
    Returns (bull_score, bear_score).
    Base weight max: 10. Hybrid volume bonus: +1.
    """
    bull = 0
    bear = 0

    if row["renko_mom"] > 0.1: bull += 2
    elif row["renko_mom"] > 0.0: bull += 1
    if row["renko_mom"] < -0.1: bear += 2
    elif row["renko_mom"] < 0.0: bear += 1

    if row["hist_slope"] > 0: bull += 2
    if row["hist_slope"] < 0: bear += 2

    if row["RSI"] > 55: bull += 2
    elif row["RSI"] > 50: bull += 1
    if row["RSI"] < 45: bear += 2
    elif row["RSI"] < 50: bear += 1

    if row["ER"] > er_th:
        bull += 2
        bear += 2   

    if row["Stoch_K"] > row["Stoch_D"]: bull += 2
    else: bear += 2

    # ── HYBRID ADDITION: Volume Bonus ──────────────────────────
    if row.get("OBV_Z", 0) > 1.0: bull += 1
    elif row.get("OBV_Z", 0) < -1.0: bear += 1

    return bull, bear


# ─────────────────────── STRATEGY CLASS ──────────────────────── #
class RenkoHybridStrategy(Strategy):
    score_threshold: int   = 5      
    er_th:           float = 0.3    
    adx_th:          float = 20.0   
    tp_atr:          float = 3.5    
    sl_atr:          float = 2.0    
    time_stop_bars:  int   = 15     

    def init(self):
        self.renko_mom  = self.I(lambda: self.data.renko_mom,   name="renko_mom")
        self.hist_slope = self.I(lambda: self.data.hist_slope,  name="hist_slope")
        self.rsi        = self.I(lambda: self.data.RSI,         name="RSI")
        self.atr        = self.I(lambda: self.data.ATR,         name="ATR")
        self.er         = self.I(lambda: self.data.ER,          name="ER")
        self.adx        = self.I(lambda: self.data.ADX,         name="ADX")
        self.ema200     = self.I(lambda: self.data.EMA200,      name="EMA200")
        self.ema200_slope = self.I(lambda: self.data.EMA200_slope, name="EMA200_slope")
        self.real_vol   = self.I(lambda: self.data.RealVol,     name="RealVol")
        self.stoch_k    = self.I(lambda: self.data.Stoch_K,     name="Stoch_K")
        self.stoch_d    = self.I(lambda: self.data.Stoch_D,     name="Stoch_D")
        self.obv_z      = self.I(lambda: self.data.OBV_Z,       name="OBV_Z")

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
        if risk_per_share <= 0 or close <= 0: return 1
        shares = TARGET_RISK / (risk_per_share * close)
        max_shares = (self.equity * 0.95) / close
        return max(1, int(min(shares, max_shares)))

    def next(self):
        close    = self.data.Close[-1]
        atr      = self.atr[-1]
        adx      = self.adx[-1]
        ema200   = self.ema200[-1]
        ema_slope = self.ema200_slope[-1]
        er       = self.er[-1]
        real_vol = self.real_vol[-1]
        bar_idx  = len(self.data) - 1

        if adx < self.adx_th:
            return     

        row = {
            "renko_mom": self.renko_mom[-1],
            "hist_slope": self.hist_slope[-1],
            "RSI":       self.rsi[-1],
            "ER":        self.er[-1],
            "Stoch_K":   self.stoch_k[-1],
            "Stoch_D":   self.stoch_d[-1],
            "OBV_Z":     self.obv_z[-1],
        }
        bull_score, bear_score = _bull_score(row, self.er_th)

        ema_bull = (close > ema200) and (ema_slope > 0)
        ema_bear = (close < ema200) and (ema_slope < 0)

        bull_entry = (bull_score >= self.score_threshold) and ema_bull
        bear_entry = (bear_score >= self.score_threshold) and ema_bear

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

        elif self.position.is_long:
            bars_held = bar_idx - self._trade_open_bar
            if close > self._long_hwm:
                self._long_hwm = close
                new_trail = self._long_hwm - atr * self.sl_atr
                for trade in self.trades:
                    if trade.is_long and (trade.sl is None or new_trail > trade.sl):
                        trade.sl = new_trail

            unrealised_r = (close - self._entry_price) / (atr * self.sl_atr + 1e-9)
            if bars_held >= self.time_stop_bars and unrealised_r < 0.5:
                self.position.close()
                return

            if bear_entry:
                self.position.close()
                sz = max(1, int(self._vol_size(close, atr) * vol_scale))
                self.sell(size=sz, sl=close + atr * self.sl_atr, tp=close - atr * self.tp_atr)
                self._reset_trade_state()

        elif self.position.is_short:
            bars_held = bar_idx - self._trade_open_bar
            if close < self._short_hwm:
                self._short_hwm = close
                new_trail = self._short_hwm + atr * self.sl_atr
                for trade in self.trades:
                    if trade.is_short and (trade.sl is None or new_trail < trade.sl):
                        trade.sl = new_trail

            unrealised_r = (self._entry_price - close) / (atr * self.sl_atr + 1e-9)
            if bars_held >= self.time_stop_bars and unrealised_r < 0.5:
                self.position.close()
                return

            if bull_entry:
                self.position.close()
                sz = max(1, int(self._vol_size(close, atr) * vol_scale))
                self.buy(size=sz, sl=close - atr * self.sl_atr, tp=close + atr * self.tp_atr)
                self._reset_trade_state()


# ──────────────── VBT SIGNAL GENERATOR ──────────────────── #
def _generate_vbt_signals(df, score_threshold=5, er_th=0.3, adx_th=20, **_):
    renko_mom_bull = df["renko_mom"] > 0.0
    renko_mom_bear = df["renko_mom"] < 0.0
    hist_bull      = df["hist_slope"] > 0
    hist_bear      = df["hist_slope"] < 0
    rsi_bull       = df["RSI"] > 50
    rsi_bear       = df["RSI"] < 50
    er_ok          = df["ER"] > er_th
    stoch_bull     = df["Stoch_K"] > df["Stoch_D"]
    stoch_bear     = df["Stoch_K"] < df["Stoch_D"]
    obv_bull       = df["OBV_Z"] > 1.0
    obv_bear       = df["OBV_Z"] < -1.0
    
    adx_ok         = df["ADX"] > adx_th
    ema_bull       = (df["Close"] > df["EMA200"]) & (df["EMA200_slope"] > 0)
    ema_bear       = (df["Close"] < df["EMA200"]) & (df["EMA200_slope"] < 0)

    bull_score = (
        renko_mom_bull.astype(int) * 2 +
        hist_bull.astype(int)      * 2 +
        rsi_bull.astype(int)       * 2 +
        er_ok.astype(int)          * 2 +
        stoch_bull.astype(int)     * 2 +
        obv_bull.astype(int)       * 1
    )
    bear_score = (
        renko_mom_bear.astype(int) * 2 +
        hist_bear.astype(int)      * 2 +
        rsi_bear.astype(int)       * 2 +
        er_ok.astype(int)          * 2 +
        stoch_bear.astype(int)     * 2 +
        obv_bear.astype(int)       * 1
    )

    entries = adx_ok & ema_bull & (bull_score >= score_threshold)
    exits   = adx_ok & ema_bear & (bear_score >= score_threshold)
    return entries, exits


# ──────────────── WALK-FORWARD STRATEGY VARIANT ─────────────────── #
class RenkoHybridStrategyWF(RenkoHybridStrategy):
    def _vol_size(self, close, atr):
        if close <= 0: return 1
        return max(1, int(self.equity * 0.90 / close))


from backtesting_engine.walk_forward import run_walk_forward


# ────────────────────────── MAIN ─────────────────────────────── #
def main():
    print("=" * 70)
    print("  Renko + MACD + OBV Hybrid Strategy")
    print("=" * 70)

    from data_ingestion.data_store import load_universe_data
    ohlc_intraday = load_universe_data(TICKERS, interval=INTERVAL)

    tickers = list(ohlc_intraday.keys())
    if not tickers: raise ValueError("No data loaded.")

    PARAM_GRID = dict(
        score_threshold = list(range(4, 9)),
        er_th           = list(np.arange(0.2, 0.6, 0.1)),
        adx_th          = [25, 30],
        tp_atr          = list(np.arange(2.0, 4.0, 0.5)),
        sl_atr          = [1.0, 1.5, 2.0],
        time_stop_bars  = [15, 25],
    )

    run_strategy_pipeline(
        strategy_name="Renko + Hybrid MACD/OBV",
        ohlcv_data=ohlc_intraday,
        strategy_class=RenkoHybridStrategy,
        default_params=DEFAULT_PARAMS,
        param_grid=PARAM_GRID,
        precompute_fn=_precompute_indicators,
        vbt_signal_fn=_generate_vbt_signals,
        cash=CASH,
        commission=COMMISSION,
        freq=INTERVAL,
    )

    print("\n" + "=" * 70)
    print("  Extracting per-ticker best params for walk-forward …")
    
    best_params_map = {}
    for ticker in tickers:
        df_raw = ohlc_intraday[ticker]
        extracted = False
        try:
            proc = _precompute_indicators(df_raw)
            bt   = Backtest(proc, RenkoHybridStrategy, cash=CASH, commission=COMMISSION, trade_on_close=True)
            
            def _tp_gt_sl_local(p): return p.tp_atr > p.sl_atr
            def _maximize_local(stats):
                if stats["# Trades"] < 10: return -9999
                return stats["Sharpe Ratio"]

            opt  = bt.optimize(**PARAM_GRID, maximize=_maximize_local, constraint=_tp_gt_sl_local, return_heatmap=False)

            strat = getattr(opt, '_strategy', None)
            bp, errors = {}, []
            for k in PARAM_GRID:
                val = getattr(strat, k, None) if strat else None
                if val is None and isinstance(opt, pd.Series) and k in opt.index: val = opt[k]
                if val is None: errors.append(k)
                else: bp[k] = int(val) if isinstance(val, (np.integer, int)) else float(val) if isinstance(val, (np.floating, float)) else val

            if errors: raise KeyError(f"params missing: {errors}")
            best_params_map[ticker] = bp
            extracted = True
            print(f"  ✅ {ticker}: {bp}")

        except Exception as e:
            best_params_map[ticker] = DEFAULT_PARAMS.copy()
            print(f"  ⚠️  {ticker}: extraction failed — {e}")

    print("\n" + "=" * 70)
    print("  Walk-Forward Out-of-Sample Validation")
    print("=" * 70)

    for ticker in tickers:
        df = ohlc_intraday[ticker]
        bp = best_params_map.get(ticker, DEFAULT_PARAMS)
        run_walk_forward(ticker, df, RenkoHybridStrategyWF, bp, precompute_fn=_precompute_indicators)

if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            from joblib.externals.loky import get_reusable_executor
            get_reusable_executor().shutdown(wait=True)
        except (ImportError, AttributeError):
            pass
        