"""
Intraday Resistance Breakout Strategy
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import os
os.environ['PYTHONWARNINGS'] = 'ignore:resource_tracker:UserWarning'

import multiprocessing
if os.name == 'posix':
    try:
        if multiprocessing.get_start_method(allow_none=True) is None:
            multiprocessing.set_start_method('spawn', force=True)
    except (RuntimeError, ValueError):
        pass

import backtesting
from backtesting import Strategy, Backtest
backtesting.Pool = multiprocessing.Pool

from indicators.atr import calculate_atr
from backtesting_engine.strategy_runner import run_strategy_pipeline
from config.settings import (
    TICKERS, CASH, COMMISSION, INTERVAL
)

# ─────────────────────────── CONFIG ──────────────────────────── #
MIN_TRADES_VALID = 5

DEFAULT_PARAMS = dict(
    vol_z_threshold    = 0.25,
    climax_z_threshold = 3.0,
    atr_breakout_coef  = 0.1,
    tp_factor          = 3.0,
    sl_factor          = 1.5,
)

PARAM_GRID = dict(
    vol_z_threshold    = [0.0, 0.25, 0.5, 0.75],
    climax_z_threshold = [2.5, 3.0, 3.5, 4.0],
    atr_breakout_coef  = [0.0, 0.1, 0.15, 0.2],
    tp_factor          = [2.5, 3.0, 3.5, 4.0],
    sl_factor          = [1.0, 1.5, 2.0],
)

# ─────────────────────── INDICATOR HELPERS ────────────────────── #
def _precompute_indicators(df, atr_period=14, roll_period=20,
                            ema_fast=20, ema_slow=50, ema_trend=200):
    df = df.copy()
    df = calculate_atr(df, atr_period)

    df['roll_max_cp'] = df['High'].rolling(roll_period).max().shift(1)
    df['roll_min_cp'] = df['Low'].rolling(roll_period).min().shift(1)

    vol_mean = df['Volume'].rolling(roll_period).mean().shift(1)
    vol_std  = df['Volume'].rolling(roll_period).std().shift(1)
    df['vol_zscore'] = (df['Volume'] - vol_mean) / (vol_std + 1e-9)

    df['ema_fast']  = df['Close'].ewm(span=ema_fast,  adjust=False).mean()
    df['ema_slow']  = df['Close'].ewm(span=ema_slow,  adjust=False).mean()
    df['ema_trend'] = df['Close'].ewm(span=ema_trend, adjust=False).mean()

    log_ret = np.log(df['Close'] / df['Close'].shift(1))
    df['rvol'] = log_ret.rolling(roll_period).std() * np.sqrt(252 * 6.5)

    df.dropna(inplace=True)
    return df


# ─────────────────────── STRATEGY CLASS ──────────────────────── #
class BreakoutStrategy(Strategy):
    vol_z_threshold    = 0.25
    climax_z_threshold = 3.0
    atr_breakout_coef  = 0.1
    tp_factor          = 3.0
    sl_factor          = 1.5

    def _vol_size(self, close, atr):
        if close <= 0 or atr <= 0:
            return 1
        equity         = self.equity
        risk_per_trade = equity * 0.01
        shares    = int(risk_per_trade / (atr * self.sl_factor))
        max_shares = max(1, int(equity * 0.20 / close))
        return max(1, min(shares, max_shares))

    def init(self):
        self.atr        = self.I(lambda: self.data.ATR,          name='ATR')
        self.roll_max   = self.I(lambda: self.data.roll_max_cp,  name='Resist')
        self.roll_min   = self.I(lambda: self.data.roll_min_cp,  name='Support')
        self.vol_zscore = self.I(lambda: self.data.vol_zscore,   name='VolZ')
        self.ema_fast   = self.I(lambda: self.data.ema_fast,     name='EMA20')
        self.ema_slow   = self.I(lambda: self.data.ema_slow,     name='EMA50')
        self.ema_trend  = self.I(lambda: self.data.ema_trend,    name='EMA200')

        self._long_stop:  float = -np.inf
        self._short_stop: float =  np.inf

    def next(self):
        close   = self.data.Close[-1]
        atr     = self.atr[-1]
        r_max   = self.roll_max[-1]
        r_min   = self.roll_min[-1]
        vol_z   = self.vol_zscore[-1]
        ema_f   = self.ema_fast[-1]
        ema_s   = self.ema_slow[-1]
        ema_t   = self.ema_trend[-1]

        vol_ok        = self.vol_z_threshold < vol_z < self.climax_z_threshold
        long_trigger  = r_max + atr * self.atr_breakout_coef
        short_trigger = r_min - atr * self.atr_breakout_coef

        trend_up   = close > ema_f > ema_s
        trend_down = close < ema_f < ema_s

        regime_bull = close > ema_t
        regime_bear = close < ema_t

        not_chasing_long  = close <= long_trigger + atr
        not_chasing_short = close >= short_trigger - atr

        shares = self._vol_size(close, atr)

        if not self.position:
            if (close >= long_trigger and vol_ok and trend_up
                    and regime_bull and not_chasing_long):
                self._long_stop = -np.inf
                self.buy(size=shares,
                         sl=close - atr * self.sl_factor,
                         tp=close + atr * self.tp_factor)

            elif (close <= short_trigger and vol_ok and trend_down
                    and regime_bear and not_chasing_short):
                self._short_stop = np.inf
                self.sell(size=shares,
                          sl=close + atr * self.sl_factor,
                          tp=close - atr * self.tp_factor)

        elif self.position.is_long:
            new_stop = close - atr * self.sl_factor
            self._long_stop = max(self._long_stop, new_stop)
            for trade in self.trades:
                if trade.is_long and (trade.sl is None or self._long_stop > trade.sl):
                    trade.sl = self._long_stop

            if (close <= short_trigger and vol_ok and trend_down
                    and regime_bear and not_chasing_short):
                self.position.close()
                shares = self._vol_size(close, atr)
                self._short_stop = np.inf
                self.sell(size=shares,
                          sl=close + atr * self.sl_factor,
                          tp=close - atr * self.tp_factor)

        elif self.position.is_short:
            new_stop = close + atr * self.sl_factor
            self._short_stop = min(self._short_stop, new_stop)
            for trade in self.trades:
                if trade.is_short and (trade.sl is None or self._short_stop < trade.sl):
                    trade.sl = self._short_stop

            if (close >= long_trigger and vol_ok and trend_up
                    and regime_bull and not_chasing_long):
                self.position.close()
                shares = self._vol_size(close, atr)
                self._long_stop = -np.inf
                self.buy(size=shares,
                         sl=close - atr * self.sl_factor,
                         tp=close + atr * self.tp_factor)


# ─────────────────── WF STRATEGY VARIANT ─────────────────────── #
class BreakoutStrategyWF(BreakoutStrategy):
    def _vol_size(self, close, atr):
        if close <= 0: return 1
        return max(1, int(self.equity * 0.90 / close))


# ─────────────────────── VBT SIGNAL HELPER ────────────────────── #
def _generate_vbt_signals(df, vol_z_threshold=0.25, climax_z_threshold=3.0,
                           atr_breakout_coef=0.1, **_):
    vol_ok        = ((df['vol_zscore'] > vol_z_threshold) &
                     (df['vol_zscore'] < climax_z_threshold))
    trend_up      = (df['Close'] > df['ema_fast']) & (df['ema_fast'] > df['ema_slow'])
    trend_down    = (df['Close'] < df['ema_fast']) & (df['ema_fast'] < df['ema_slow'])
    regime_bull   = df['Close'] > df['ema_trend']
    regime_bear   = df['Close'] < df['ema_trend']
    long_trigger  = df['roll_max_cp'] + df['ATR'] * atr_breakout_coef
    short_trigger = df['roll_min_cp'] - df['ATR'] * atr_breakout_coef
    chase_ok_L    = df['Close'] <= long_trigger + df['ATR']
    chase_ok_S    = df['Close'] >= short_trigger - df['ATR']

    entries = (df['Close'] >= long_trigger) & vol_ok & trend_up  & regime_bull & chase_ok_L
    exits   = (df['Close'] <= short_trigger) & vol_ok & trend_down & regime_bear & chase_ok_S
    return entries, exits


from backtesting_engine.walk_forward import run_walk_forward


# ─────────────────────────── MAIN ────────────────────────────── #
def main():
    print("=" * 70)
    print("  Resistance Breakout Strategy — Advanced Backtest")
    print("=" * 70)

    from data_ingestion.data_store import load_universe_data
    raw_ohlcv = load_universe_data(TICKERS, interval=INTERVAL)
    ohlcv = {}
    for ticker, df in raw_ohlcv.items():
        df = df.between_time('09:35', '16:00')
        if not df.empty: ohlcv[ticker] = df

    tickers = list(ohlcv.keys())
    if not tickers: return

    run_strategy_pipeline(
        strategy_name="Resistance Breakout Strategy",
        ohlcv_data=ohlcv,
        strategy_class=BreakoutStrategy,
        default_params=DEFAULT_PARAMS,
        param_grid=PARAM_GRID,
        precompute_fn=_precompute_indicators,
        vbt_signal_fn=_generate_vbt_signals,
        cash=CASH,
        commission=COMMISSION,
        freq='1h',
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("  Extracting per-ticker best params for walk-forward …")
    best_params_map = {}
    for ticker in tickers:
        try:
            proc = _precompute_indicators(ohlcv[ticker])
            bt   = Backtest(proc, BreakoutStrategy, cash=CASH, commission=COMMISSION,
                            trade_on_close=True, finalize_trades=True)

            def _tp_gt_sl(p): return p.tp_factor > p.sl_factor
            opt = bt.optimize(**PARAM_GRID, maximize="Sharpe Ratio",
                              constraint=_tp_gt_sl, return_heatmap=False)

            strat = getattr(opt, '_strategy', None)
            bp, errors = {}, []
            for k in PARAM_GRID:
                val = getattr(strat, k, None) if strat else None
                if val is None and isinstance(opt, pd.Series) and k in opt.index: val = opt[k]
                if val is None: errors.append(k)
                else: bp[k] = int(val) if isinstance(val, (np.integer, int)) else float(val)

            if errors: raise KeyError(f"missing params: {errors}")
            best_params_map[ticker] = bp
            print(f"  ✅ {ticker}: {bp}")

        except Exception as e:
            best_params_map[ticker] = DEFAULT_PARAMS.copy()
            print(f"  ⚠️  {ticker}: extraction failed ({e}) — using defaults")

    print("\n" + "=" * 70)
    print("  Walk-Forward Out-of-Sample Validation")
    wf_results = {}
    for ticker in tickers:
        wf_results[ticker] = run_walk_forward(
            ticker, ohlcv[ticker], BreakoutStrategyWF,
            best_params_map.get(ticker, DEFAULT_PARAMS),
            precompute_fn=_precompute_indicators,
            min_trades_valid=MIN_TRADES_VALID
        )

    print("\n" + "=" * 70)
    print("  CONSOLIDATED WALK-FORWARD SUMMARY")
    summary_rows = []
    for ticker, wf_df in wf_results.items():
        if wf_df.empty: continue
        vdf = wf_df[wf_df["valid"]]
        if vdf.empty:
            summary_rows.append({
                "Ticker": ticker, "Valid WF": 0, "Avg Ret%": "–",
                "Avg Sharpe": "–", "Win Rate%": "–",
                "Win Windows": "–", "Verdict": "Insufficient trades"
            })
            continue
        avg_ret = vdf['return_%'].mean()
        avg_sh  = vdf['sharpe'].mean()
        avg_wr  = vdf['win_rate_%'].mean()
        wins    = (vdf['return_%'] > 0).sum()
        total   = len(vdf)

        if avg_ret > 2 and avg_sh > 0.5 and wins / total >= 0.6:
            verdict = "Strong edge"
        elif avg_ret > 0 and avg_sh > 0:
            verdict = "Marginal edge"
        else:
            verdict = "No edge"

        summary_rows.append({
            "Ticker": ticker,
            "Valid WF": total,
            "Avg Ret%": round(avg_ret, 2),
            "Avg Sharpe": round(avg_sh, 3),
            "Win Rate%": round(avg_wr, 1),
            "Win Windows": f"{wins}/{total}",
            "Verdict": verdict,
        })

    summary_df = pd.DataFrame(summary_rows).set_index("Ticker")
    print(summary_df.to_string())
    print("=" * 70)


if __name__ == '__main__':
    try:
        main()
    finally:
        try:
            from joblib.externals.loky import get_reusable_executor
            get_reusable_executor().shutdown(wait=True)
        except (ImportError, AttributeError):
            pass
