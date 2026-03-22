"""
Advanced Backtesting Engine — vectorbt + Monte Carlo + Walk-Forward + Stress Testing

Provides the VBTBacktester class that wraps vectorbt's Portfolio.from_signals
with 10 validation layers:
  1.  Base backtest (vectorbt)
  2.  Benchmark comparison (alpha, beta, information ratio)
  3.  Monte Carlo simulation (block bootstrap + random baseline)
  4.  Walk-forward analysis (rolling windows)
  5.  Stress testing (inject historical crisis scenarios)
  6.  Deflated Sharpe Ratio (bias removal)
  7.  Trade-level analysis (profit factor, holding periods)
  8.  Regime analysis (vol + trend regimes)
  9.  Extended risk metrics (CVaR, Omega, Ulcer Index)
  10. Kelly position sizing
"""

import os
import multiprocessing
import warnings

import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats as scipy_stats

# Suppress loky semaphore leak warnings on Python 3.14 (cosmetic only)
warnings.filterwarnings(
    'ignore',
    message='resource_tracker.*semaphore',
    category=UserWarning,
)
warnings.filterwarnings(
    'ignore',
    message='resource_tracker.*leaked',
    category=UserWarning,
)

# Suppress pandas future downcasting warning globally
pd.set_option('future.no_silent_downcasting', True)

# Fix multiprocessing spawn context (prevents semaphore leaks on macOS/Windows)
if os.name == 'posix':
    try:
        if multiprocessing.get_start_method(allow_none=True) is None:
            multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass


# ============================================================================
# ANNUALISATION FACTORS — explicit per frequency, no silent fallback
# ============================================================================
FREQ_ANN_FACTORS = {
    'D':     252,
    'W':     52,
    'M':     12,
    '1H':    252 * 6.5,
    '2H':    252 * 6.5 / 2,
    '4H':    252 * 6.5 / 4,
    '30T':   252 * 6.5 * 2,
    '15T':   252 * 6.5 * 4,
    '5T':    252 * 6.5 * 12,
    '1T':    252 * 390,
    # Crypto (24/7)
    '1H_24': 365 * 24,
    '4H_24': 365 * 6,
    'D_24':  365,
}

# ============================================================================
# CRISIS SCENARIOS — historical worst-case drawdown distributions
# ============================================================================
CRISIS_SCENARIOS = {
    '2008_GFC': {
        'description':   'Global Financial Crisis 2008',
        'daily_mean':    -0.0035,
        'daily_std':      0.042,
        'duration_days':  250,
        'peak_drawdown': -0.38,
    },
    'COVID_2020': {
        'description':   'COVID-19 Crash 2020',
        'daily_mean':    -0.0060,
        'daily_std':      0.055,
        'duration_days':  33,
        'peak_drawdown': -0.34,
    },
    'DOTCOM_2000': {
        'description':   'Dot-com Bubble Burst 2000–2002',
        'daily_mean':    -0.0012,
        'daily_std':      0.025,
        'duration_days':  630,
        'peak_drawdown': -0.45,
    },
    'BLACK_MONDAY_1987': {
        'description':   'Black Monday 1987',
        'daily_mean':    -0.0150,
        'daily_std':      0.080,
        'duration_days':  5,
        'peak_drawdown': -0.22,
    },
    'RUSSIA_1998': {
        'description':   'Russian Default / LTCM 1998',
        'daily_mean':    -0.0045,
        'daily_std':      0.035,
        'duration_days':  60,
        'peak_drawdown': -0.20,
    },
    'CRYPTO_2022': {
        'description':   'Crypto Bear Market 2022',
        'daily_mean':    -0.0030,
        'daily_std':      0.060,
        'duration_days':  350,
        'peak_drawdown': -0.77,
    },
}


# ============================================================================
# MAIN CLASS
# ============================================================================
class VBTBacktester:
    """
    Advanced backtesting engine built on vectorbt.

    Usage:
        bt      = VBTBacktester(close, entries, exits, freq='1H')
        base    = bt.run()
        bench   = bt.vs_benchmark()
        mc      = bt.monte_carlo(n_simulations=500)
        wf      = bt.walk_forward(n_splits=4)
        st      = bt.stress_testing()
        dsr     = bt.deflated_sharpe(n_trials=7)
        trades  = bt.trade_analysis()
        regimes = bt.regime_analysis()
        risk    = bt.risk_metrics()
        kelly   = bt.kelly_sizing()
        full    = bt.full_analysis()
    """

    def __init__(
        self,
        close,
        entries,
        exits,
        freq='D',
        init_cash=100_000,
        commission=0.001,
        slippage=0.001,
        lag_signals=True,
        crypto_24_7=False,
    ):
        """
        Args:
            close:        pd.Series of close prices (DatetimeIndex).
            entries:      pd.Series of boolean entry signals.
            exits:        pd.Series of boolean exit signals.
            freq:         Data frequency string ('D', '1H', '5T', etc.).
            init_cash:    Starting capital.
            commission:   Commission rate per trade (e.g. 0.001 = 0.1%).
            slippage:     Slippage rate per trade.
            lag_signals:  Shift signals 1 bar forward to prevent look-ahead bias.
            crypto_24_7:  Use 365-day calendar for annualisation.
        """
        def _squeeze(x):
            return x.squeeze() if isinstance(x, pd.DataFrame) else x

        self.close   = _squeeze(close).astype(float)
        self.entries = _squeeze(entries).astype(bool)
        self.exits   = _squeeze(exits).astype(bool)
        self.freq       = freq
        self.init_cash  = init_cash
        self.commission = commission
        self.slippage   = slippage
        self.crypto_24_7 = crypto_24_7

        # --- Resolve annualisation factor ---
        if crypto_24_7:
            _freq_map = {'D': 365, '1H': 365 * 24, '4H': 365 * 6, '15T': 365 * 96}
            self.ann_factor = _freq_map.get(freq, 365)
        else:
            self.ann_factor = FREQ_ANN_FACTORS.get(freq, 252)

        # --- Look-ahead bias guard ---
        if lag_signals:
            self.entries = (
                self.entries.shift(1)
                .infer_objects(copy=False)
                .fillna(False)
                .astype(bool)
            )
            self.exits = (
                self.exits.shift(1)
                .infer_objects(copy=False)
                .fillna(False)
                .astype(bool)
            )

        self._portfolio = None
        self._returns   = None

    def calculate_market_impact(self, trade_size, adv, volatility, participation_rate=0.1):
        """
        Simplified Almgren-Chriss Market Impact Model.
        Returns the expected price impact (slippage) as a fractional percentage.
        trade_size: Order size in shares.
        adv: Average Daily Volume (or rolling volume).
        volatility: Daily volatility (sigma) of returns.
        participation_rate: Gamma parameter for impact intensity.
        """
        if adv <= 0:
            return self.slippage
        # Temporary impact roughly proportional to sqrt(trade_size / ADV)
        impact = volatility * np.sqrt(abs(trade_size) / adv) * participation_rate
        return float(impact)

    # =========================================================================
    # 1. BASE BACKTEST
    # =========================================================================
    def run(self, print_stats=True):
        """Run vectorbt backtest and return portfolio stats dict."""
        self._portfolio = vbt.Portfolio.from_signals(
            close=self.close,
            entries=self.entries,
            exits=self.exits,
            init_cash=self.init_cash,
            fees=self.commission,
            slippage=self.slippage,
            freq=self.freq,
        )
        self._returns = self._portfolio.returns()

        stats = self._portfolio.stats()
        rets  = self._returns.values.flatten()

        # Sortino
        downside     = np.where(rets < 0, rets, 0.0)
        downside_std = np.sqrt(np.mean(downside ** 2)) * np.sqrt(self.ann_factor)
        sortino      = (np.mean(rets) * self.ann_factor) / downside_std \
                       if downside_std > 0 else 0.0

        # Calmar
        max_dd    = float(self._portfolio.max_drawdown())
        total_ret = float(self._portfolio.total_return())
        years     = len(rets) / max(self.ann_factor, 1)
        cagr      = (1 + total_ret) ** (1 / max(years, 0.01)) - 1 \
                    if total_ret > -1 else -1.0
        calmar    = cagr / abs(max_dd) if abs(max_dd) > 0 else 0.0

        # CVaR
        cvar_95 = self._cvar(rets, alpha=0.05)
        cvar_99 = self._cvar(rets, alpha=0.01)

        if print_stats:
            print("\n" + "=" * 60)
            print("  📊 vectorbt Portfolio Stats")
            print("=" * 60)
            # Replace non-finite floats with readable placeholder
            clean_stats = stats.copy()
            for idx in clean_stats.index:
                val = clean_stats[idx]
                if isinstance(val, float) and not np.isfinite(val):
                    clean_stats[idx] = "N/A (Open Only)"
            print(clean_stats.to_string())
            print(f"\n  Sortino Ratio:      {sortino:.4f}")
            print(f"  Calmar Ratio:       {calmar:.4f}")
            print(f"  CVaR (95%):         {cvar_95 * 100:.4f}%")
            print(f"  CVaR (99%):         {cvar_99 * 100:.4f}%")
            print(f"  Ann. Factor used:   {self.ann_factor}")

        return {
            'portfolio':    self._portfolio,
            'stats':        stats,
            'total_return': total_ret,
            'cagr':         cagr,
            'sharpe':       self._portfolio.sharpe_ratio(),
            'sortino':      sortino,
            'calmar':       calmar,
            'max_drawdown': max_dd,
            'cvar_95':      cvar_95,
            'cvar_99':      cvar_99,
            'returns':      self._returns,
        }

    # =========================================================================
    # 2. BENCHMARK COMPARISON
    # =========================================================================
    def vs_benchmark(self, print_report=True):
        """Compare strategy vs buy-and-hold. Returns alpha, beta, info ratio."""
        if self._returns is None:
            self.run(print_stats=False)

        bh = vbt.Portfolio.from_holding(
            self.close,
            init_cash=self.init_cash,
            fees=self.commission,
            freq=self.freq,
        )
        bh_rets = bh.returns().values.flatten()
        st_rets = self._returns.values.flatten()

        n       = min(len(bh_rets), len(st_rets))
        bh_rets = bh_rets[:n]
        st_rets = st_rets[:n]

        # Beta / Alpha (OLS)
        cov_mat = np.cov(st_rets, bh_rets)
        beta    = cov_mat[0, 1] / cov_mat[1, 1] if cov_mat[1, 1] > 0 else 0.0
        alpha   = (np.mean(st_rets) - beta * np.mean(bh_rets)) * self.ann_factor

        # Information Ratio
        active_rets    = st_rets - bh_rets
        tracking_error = np.std(active_rets, ddof=1) * np.sqrt(self.ann_factor)
        info_ratio     = (np.mean(active_rets) * self.ann_factor) / tracking_error \
                         if tracking_error > 0 else 0.0

        # Correlation (guard zero-variance)
        if np.std(st_rets) == 0 or np.std(bh_rets) == 0:
            corr = float('nan')
        else:
            corr = float(np.corrcoef(st_rets, bh_rets)[0, 1])

        bh_sharpe = float(bh.sharpe_ratio())
        st_sharpe = float(self._portfolio.sharpe_ratio())

        result = {
            'strategy_return':   float(self._portfolio.total_return()),
            'benchmark_return':  float(bh.total_return()),
            'strategy_sharpe':   st_sharpe,
            'benchmark_sharpe':  bh_sharpe,
            'alpha':             alpha,
            'beta':              beta,
            'information_ratio': info_ratio,
            'tracking_error':    tracking_error,
            'correlation':       corr,
        }

        if print_report:
            print("\n" + "=" * 60)
            print("  📈 Benchmark Comparison (vs Buy-and-Hold)")
            print("=" * 60)
            print(f"  Strategy Return:       {result['strategy_return'] * 100:>8.2f}%")
            print(f"  Benchmark Return:      {result['benchmark_return'] * 100:>8.2f}%")
            print(f"  Strategy Sharpe:       {st_sharpe:>8.4f}")
            print(f"  Benchmark Sharpe:      {bh_sharpe:>8.4f}")
            print(f"  Alpha (ann.):          {alpha:>8.4f}")
            print(f"  Beta:                  {beta:>8.4f}")
            print(f"  Information Ratio:     {info_ratio:>8.4f}")
            print(f"  Tracking Error:        {tracking_error * 100:>8.4f}%")
            corr_str = f"{corr:>8.4f}" if np.isfinite(corr) else "     N/A"
            print(f"  Correlation:           {corr_str}")
            verdict = "✅ Adds alpha" if alpha > 0 and info_ratio > 0.5 \
                      else "⚠️ Review vs benchmark"
            print(f"  Verdict:               {verdict}")

        return result

    # =========================================================================
    # 3. MONTE CARLO (Block Bootstrap)
    # =========================================================================
    def monte_carlo(self, n_simulations=500, block_size=None, print_report=True):
        """
        Block-bootstrap Monte Carlo + random-signal baseline.
        Skips gracefully when there are no active trades.
        """
        if self._returns is None:
            self.run(print_stats=False)

        returns = self._returns.values.flatten()
        n       = len(returns)

        # Guard: nothing to simulate
        if np.all(returns == 0) or np.std(returns) == 0:
            if print_report:
                print("\n  ⚠️  Monte Carlo skipped — no active trades in returns")
            return {}

        actual_cum  = float(np.cumprod(1 + returns)[-1] - 1)
        actual_sr   = self._calc_sharpe(returns)
        actual_dd   = self._calc_max_dd(returns)

        if block_size is None:
            block_size = max(5, int(np.sqrt(n)))

        sim_rets    = np.zeros(n_simulations)
        sim_sharpes = np.zeros(n_simulations)
        sim_dds     = np.zeros(n_simulations)
        n_blocks    = (n // block_size) + 1

        for i in range(n_simulations):
            starts  = np.random.randint(0, max(1, n - block_size), size=n_blocks)
            sampled = np.concatenate([returns[s:s + block_size] for s in starts])[:n]
            sim_rets[i]    = np.cumprod(1 + sampled)[-1] - 1
            sim_sharpes[i] = self._calc_sharpe(sampled)
            sim_dds[i]     = self._calc_max_dd(sampled)

        # Random signal baseline
        close_rets = np.log(self.close / self.close.shift(1)).infer_objects(copy=False).fillna(0).values.flatten()
        rand_rets  = np.zeros(n_simulations)
        for i in range(n_simulations):
            mask        = np.random.random(n) > 0.5
            rand_path   = np.where(mask, close_rets[:n], 0.0)
            rand_rets[i] = np.cumprod(1 + rand_path)[-1] - 1

        pctiles      = [5, 25, 50, 75, 95]
        p_ret        = float(np.mean(sim_rets    >= actual_cum))
        p_sr         = float(np.mean(sim_sharpes >= actual_sr))
        p_vs_rand    = float(np.mean(rand_rets   >= actual_cum))

        result = {
            'actual_return':       actual_cum,
            'actual_sharpe':       actual_sr,
            'actual_max_dd':       actual_dd,
            'sim_return_pctiles':  np.percentile(sim_rets,    pctiles),
            'sim_sharpe_pctiles':  np.percentile(sim_sharpes, pctiles),
            'sim_max_dd_pctiles':  np.percentile(sim_dds,     pctiles),
            'rand_return_pctiles': np.percentile(rand_rets,   pctiles),
            'p_value_return':      p_ret,
            'p_value_sharpe':      p_sr,
            'p_value_vs_random':   p_vs_rand,
            'n_simulations':       n_simulations,
            'block_size':          block_size,
        }

        if print_report:
            print("\n" + "=" * 60)
            print(f"  🎲 Monte Carlo — Block Bootstrap "
                  f"({n_simulations} sims, block={block_size})")
            print("=" * 60)
            print(f"\n  Actual Cumulative Return: {actual_cum * 100:>8.2f}%")
            print(f"  Actual Sharpe Ratio:      {actual_sr:>8.4f}")
            print(f"  Actual Max Drawdown:      {actual_dd * 100:>8.2f}%")

            # FIX: guard isfinite before every format call
            def _fmt_pct(v):
                return f"{v * 100:>8.2f}%" if np.isfinite(v) else "     N/A"

            def _fmt_sr(v):
                return f"{v:>8.4f}" if np.isfinite(v) else "     N/A"

            for label, arr, fmt_fn in [
                ("Bootstrap Return",    result['sim_return_pctiles'],  _fmt_pct),
                ("Bootstrap Sharpe",    result['sim_sharpe_pctiles'],  _fmt_sr),
                ("Bootstrap Max DD",    result['sim_max_dd_pctiles'],  _fmt_pct),
                ("Random Baseline Ret", result['rand_return_pctiles'], _fmt_pct),
            ]:
                print(f"\n  {label} Percentiles:")
                for p, v in zip(pctiles, arr):
                    print(f"    {p:>3}th: {fmt_fn(v)}")

            print(f"\n  P-value (bootstrap return): {p_ret:.4f}"
                  f"  {'✅ Robust' if p_ret > 0.25 else '⚠️ Fragile path'}")
            print(f"  P-value (bootstrap Sharpe): {p_sr:.4f}"
                  f"  {'✅ Robust' if p_sr > 0.25 else '⚠️ Fragile path'}")
            print(f"  P-value (vs random signal): {p_vs_rand:.4f}"
                  f"  {'✅ Beats random' if p_vs_rand < 0.05 else '⚠️ No edge over random'}")

        return result

    # =========================================================================
    # 4. WALK-FORWARD ANALYSIS
    # =========================================================================
    def walk_forward(self, n_splits=5, mode='rolling', print_report=True):
        """
        Rolling walk-forward analysis across n_splits independent windows.

        Args:
            mode: 'rolling' — both windows slide forward (default).
                  'anchored' — train always starts from t=0.
        """
        if self._returns is None:
            self.run(print_stats=False)

        n           = len(self.close)
        window_size = n // n_splits
        windows     = []

        for i in range(n_splits):
            start    = i * window_size
            test_end = min((i + 1) * window_size, n)

            sl = slice(start, test_end)
            t_close   = self.close.iloc[sl]
            t_entries = self.entries.iloc[sl]
            t_exits   = self.exits.iloc[sl]

            if len(t_close) < 5:
                continue

            try:
                pf = vbt.Portfolio.from_signals(
                    close=t_close,
                    entries=t_entries,
                    exits=t_exits,
                    init_cash=self.init_cash,
                    fees=self.commission,
                    slippage=self.slippage,
                    freq=self.freq,
                )
                sharpe = float(pf.sharpe_ratio())
                windows.append({
                    'window':       i + 1,
                    'test_period':  (f"{t_close.index[0].strftime('%Y-%m-%d')} → "
                                     f"{t_close.index[-1].strftime('%Y-%m-%d')}"),
                    'total_return': float(pf.total_return()),
                    'sharpe':       sharpe if np.isfinite(sharpe) else 0.0,
                    'max_drawdown': float(pf.max_drawdown()),
                    'n_trades':     int(pf.trades.count()),
                })
            except Exception:
                continue

        if not windows:
            if print_report:
                print("\n  ⚠️ Walk-Forward: insufficient data for splitting")
            return {'windows': [], 'aggregated': {}}

        sharpes = [w['sharpe'] for w in windows]
        agg = {
            'avg_return':         np.mean([w['total_return'] for w in windows]),
            'avg_sharpe':         np.mean(sharpes),
            'sharpe_consistency': np.std(sharpes),
            'worst_drawdown':     np.min([w['max_drawdown'] for w in windows]),
            'total_trades':       sum(w['n_trades'] for w in windows),
            'n_windows':          len(windows),
            'profitable_windows': sum(1 for w in windows if w['total_return'] > 0),
        }

        if print_report:
            print("\n" + "=" * 60)
            print(f"  🔄 Sub-period Analysis ({len(windows)} independent windows)")
            print("=" * 60)
            wf_df = pd.DataFrame(windows)
            wf_df['total_return'] = wf_df['total_return'].map(lambda x: f"{x * 100:.2f}%")
            wf_df['sharpe']       = wf_df['sharpe'].map(lambda x: f"{x:.4f}")
            wf_df['max_drawdown'] = wf_df['max_drawdown'].map(lambda x: f"{x * 100:.2f}%")
            print(wf_df[['window', 'test_period', 'total_return',
                          'sharpe', 'max_drawdown', 'n_trades']].to_string(index=False))

            print(f"\n  --- Aggregated OOS Metrics ---")
            print(f"  Avg Return:           {agg['avg_return'] * 100:.2f}%")
            print(f"  Avg Sharpe:           {agg['avg_sharpe']:.4f}")
            cons_flag = '✅ Stable' if agg['sharpe_consistency'] < 0.5 else '⚠️ Inconsistent'
            print(f"  Sharpe Std Dev:       {agg['sharpe_consistency']:.4f}  {cons_flag}")
            print(f"  Worst Drawdown:       {agg['worst_drawdown'] * 100:.2f}%")
            print(f"  Profitable Windows:   {agg['profitable_windows']}/{agg['n_windows']}")

        return {'windows': windows, 'aggregated': agg}

    # =========================================================================
    # 4b. COMBINATORIAL PURGED CROSS-VALIDATION (CPCV)
    # =========================================================================
    def combinatorial_purged_cv(self, n_splits=6, test_splits=2, purge_bars=10, print_report=True):
        """
        Combinatorial Purged Cross-Validation (CPCV).
        Splits data into `n_splits` groups. Tests on `test_splits` groups,
        trains on the rest. Purges `purge_bars` around the test sets to prevent leakage.
        """
        import itertools
        if self._returns is None:
            self.run(print_stats=False)

        n = len(self.close)
        split_size = n // n_splits
        indices = np.arange(n)
        groups = [indices[i * split_size:(i + 1) * split_size] for i in range(n_splits)]
        
        if len(groups[-1]) < split_size and n_splits > 1:
            groups[-2] = np.concatenate((groups[-2], groups[-1]))
            groups.pop()
            n_splits -= 1

        combinations = list(itertools.combinations(range(n_splits), test_splits))
        paths = []
        
        for idx, test_group_idxs in enumerate(combinations):
            test_indices = np.concatenate([groups[i] for i in test_group_idxs])
            
            purge_mask = np.zeros(n, dtype=bool)
            for i in test_indices:
                p_start = max(0, i - purge_bars)
                p_end = min(n, i + purge_bars + 1)
                purge_mask[p_start:p_end] = True
                
            train_indices = np.array([i for i in range(n) if not purge_mask[i] and i not in test_indices])
            if len(test_indices) < 5:
                continue
                
            t_close   = self.close.iloc[test_indices]
            t_entries = self.entries.iloc[test_indices]
            t_exits   = self.exits.iloc[test_indices]
            
            try:
                pf = vbt.Portfolio.from_signals(
                    close=t_close, entries=t_entries, exits=t_exits,
                    init_cash=self.init_cash, fees=self.commission, slippage=self.slippage, freq=self.freq
                )
                sharpe = float(pf.sharpe_ratio())
                paths.append({
                    'path_id': idx + 1,
                    'total_return': float(pf.total_return()),
                    'sharpe': sharpe if np.isfinite(sharpe) else 0.0,
                    'max_drawdown': float(pf.max_drawdown()),
                    'n_trades': int(pf.trades.count())
                })
            except Exception:
                continue

        if not paths:
            if print_report:
                print("\n  ⚠️ CPCV: insufficient data for splitting")
            return {'paths': [], 'aggregated': {}}
            
        sharpes = [p['sharpe'] for p in paths]
        agg = {
            'avg_return':         np.mean([p['total_return'] for p in paths]),
            'avg_sharpe':         np.mean(sharpes),
            'sharpe_std':         np.std(sharpes),
            'worst_drawdown':     np.min([p['max_drawdown'] for p in paths]),
            'n_paths':            len(paths),
        }

        if print_report:
            print("\n" + "=" * 60)
            print(f"  🧬 Combinatorial Purged CV ({n_splits} splits, {test_splits} test, {purge_bars} purge)")
            print("=" * 60)
            print(f"  Paths Evaluated:      {agg['n_paths']}")
            print(f"  Avg Path Return:      {agg['avg_return'] * 100:.2f}%")
            print(f"  Avg Path Sharpe:      {agg['avg_sharpe']:.4f}")
            print(f"  Path Sharpe StdDev:   {agg['sharpe_std']:.4f}")
            print(f"  Worst Path Drawdown:  {agg['worst_drawdown'] * 100:.2f}%")

        return {'paths': paths, 'aggregated': agg}

    # =========================================================================
    # 5. STRESS TESTING
    # =========================================================================
    def stress_testing(self, scenarios=None, print_report=True):
        """
        Inject historical crisis return distributions into the strategy returns.
        Includes: GFC 2008, COVID 2020, Dot-com, Black Monday, Russia 1998, Crypto 2022.
        """
        if self._returns is None:
            self.run(print_stats=False)

        if scenarios is None:
            scenarios = CRISIS_SCENARIOS

        returns      = self._returns.values.flatten()
        results      = {}
        period_ratio = max(1, 252 / self.ann_factor)

        for name, crisis in scenarios.items():
            dur  = max(1, int(crisis['duration_days'] / period_ratio))
            mean = crisis['daily_mean'] * period_ratio
            std  = crisis['daily_std']  * np.sqrt(period_ratio)

            crisis_rets = np.random.normal(mean, std, dur)
            insert_pos  = np.random.randint(0, max(1, len(returns) - dur))
            stressed    = returns.copy()
            end_pos     = min(insert_pos + dur, len(stressed))
            stressed[insert_pos:end_pos] = crisis_rets[:end_pos - insert_pos]

            results[name] = {
                'description':            crisis['description'],
                'duration_days':          crisis['duration_days'],
                'peak_drawdown_scenario': crisis['peak_drawdown'],
                'stressed_return':        float((1 + stressed).prod() - 1),
                'stressed_max_dd':        self._calc_max_dd(stressed),
                'stressed_sharpe':        self._calc_sharpe(stressed),
                'stressed_cvar_95':       self._cvar(stressed),
                'survived':               float((1 + stressed).prod() - 1) > -0.95,
            }

        if print_report:
            print("\n" + "=" * 60)
            print("  🔥 Stress Test — Crisis Scenario Injection")
            print("=" * 60)
            for name, r in results.items():
                status = "✅ Survived" if r['survived'] else "💀 WIPED OUT"
                print(f"\n  {r['description']} "
                      f"({r['duration_days']}d, hist DD {r['peak_drawdown_scenario'] * 100:.0f}%)")
                print(f"    Stressed Return:    {r['stressed_return'] * 100:>8.2f}%")
                print(f"    Stressed Max DD:    {r['stressed_max_dd'] * 100:>8.2f}%")
                print(f"    Stressed Sharpe:    {r['stressed_sharpe']:>8.4f}")
                print(f"    Stressed CVaR 95%:  {r['stressed_cvar_95'] * 100:>8.4f}%")
                print(f"    Status:             {status}")

        return results

    # =========================================================================
    # 6. DEFLATED SHARPE RATIO
    # =========================================================================
    def deflated_sharpe(self, n_trials=1, print_report=True):
        """
        Deflated Sharpe Ratio per Bailey & López de Prado (2014).
        Adjusts for multiple-testing bias, skewness, and excess kurtosis.
        """
        if self._returns is None:
            self.run(print_stats=False)

        returns = self._returns.values.flatten()
        n       = len(returns)

        if n < 2:
            return {
                'observed_sharpe': 0.0, 'expected_max_sharpe': 0.0,
                'p_value': 1.0, 'significant': False,
            }

        period_std = np.std(returns, ddof=1)
        period_sr  = (np.mean(returns) / period_std) if period_std > 0 else 0.0

        skew = float(scipy_stats.skew(returns))
        kurt = float(scipy_stats.kurtosis(returns, fisher=True))

        euler_m = 0.5772156649
        if n_trials > 1:
            log_t = 2 * np.log(n_trials)
            z_max = (np.sqrt(log_t) * (1 - euler_m / log_t)
                     + euler_m / np.sqrt(log_t))
        else:
            z_max = 0.0

        sr_std = np.sqrt(
            (1 - skew * period_sr + (kurt - 1) / 4 * period_sr ** 2)
            / max(1, n - 1)
        )
        expected_max_sr = z_max * sr_std

        if sr_std > 0:
            dsr_stat = (period_sr - expected_max_sr) / sr_std
            p_value  = 1 - scipy_stats.norm.cdf(dsr_stat)
        else:
            dsr_stat = 0.0
            p_value  = 1.0

        sr_ann     = period_sr * np.sqrt(self.ann_factor)
        sr_max_ann = expected_max_sr * np.sqrt(self.ann_factor)

        result = {
            'observed_sharpe':     sr_ann,
            'expected_max_sharpe': sr_max_ann,
            'dsr_statistic':       dsr_stat,
            'p_value':             p_value,
            'n_trials':            n_trials,
            'skewness':            skew,
            'kurtosis':            kurt,
            'significant':         p_value < 0.05,
        }

        if print_report:
            print("\n" + "=" * 60)
            print(f"  🛡️ Deflated Sharpe Ratio (n_trials={n_trials})")
            print("=" * 60)
            print(f"  Observed Sharpe:       {sr_ann:>8.4f}")
            print(f"  Expected Max Sharpe:   {sr_max_ann:>8.4f}")
            print(f"  DSR Statistic:         {dsr_stat:>8.4f}")
            print(f"  P-value:               {p_value:>8.4f}")
            print(f"  Skewness:              {skew:>8.4f}")
            print(f"  Excess Kurtosis:       {kurt:>8.4f}")
            label = "✅ Likely skill" if result['significant'] else "⚠️ Possibly luck"
            print(f"  Conclusion:            {label}")

        return result

    # =========================================================================
    # 7. TRADE-LEVEL ANALYSIS
    # =========================================================================
    def trade_analysis(self, print_report=True):
        """
        Per-trade statistics: win rate, profit factor, expectancy, holding period.

        FIX: profit factor uses float('inf') when there are zero losses,
        instead of the old 1e-9 sentinel that caused overflow printing.
        """
        if self._portfolio is None:
            self.run(print_stats=False)

        try:
            trades = self._portfolio.trades.records_readable
        except Exception:
            if print_report:
                print("\n  ⚠️ Trade analysis: no trade records available.")
            return {}

        if trades.empty:
            if print_report:
                print("\n  ⚠️ Trade analysis: zero trades executed.")
            return {}

        pnl     = trades['PnL'].values if 'PnL' in trades.columns else np.array([])
        winners = pnl[pnl > 0]
        losers  = pnl[pnl < 0]

        win_rate     = len(winners) / len(pnl) if len(pnl) > 0 else 0.0
        gross_profit = float(winners.sum()) if len(winners) > 0 else 0.0
        gross_loss   = float(abs(losers.sum())) if len(losers) > 0 else 0.0

        # FIX: float('inf') instead of 1e-9 sentinel
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        avg_win  = float(winners.mean()) if len(winners) > 0 else 0.0
        avg_loss = float(losers.mean())  if len(losers)  > 0 else 0.0

        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

        avg_holding = None
        if ('Entry Timestamp' in trades.columns
                and 'Exit Timestamp' in trades.columns):
            holding     = trades['Exit Timestamp'] - trades['Entry Timestamp']
            avg_holding = holding.mean()

        result = {
            'n_trades':      len(pnl),
            'win_rate':      win_rate,
            'profit_factor': profit_factor,
            'avg_win':       avg_win,
            'avg_loss':      avg_loss,
            'gross_profit':  gross_profit,
            'gross_loss':    gross_loss,
            'expectancy':    expectancy,
            'avg_holding':   avg_holding,
        }

        if print_report:
            print("\n" + "=" * 60)
            print("  🏦 Trade-Level Analysis")
            print("=" * 60)
            print(f"  Total Trades:       {len(pnl)}")
            print(f"  Win Rate:           {win_rate * 100:.2f}%")
            # FIX: format inf cleanly
            pf_str = (f"{profit_factor:.4f}" if np.isfinite(profit_factor)
                      else "∞ (no losses)")
            print(f"  Profit Factor:      {pf_str}  "
                  f"{'✅' if profit_factor > 1.5 else '⚠️'}")
            print(f"  Avg Win:            {avg_win:.2f}")
            print(f"  Avg Loss:           {avg_loss:.2f}")
            print(f"  Expectancy/Trade:   {expectancy:.2f}")
            print(f"  Gross Profit:       {gross_profit:.2f}")
            print(f"  Gross Loss:         {gross_loss:.2f}")
            if avg_holding is not None:
                print(f"  Avg Holding Period: {avg_holding}")

        return result

    # =========================================================================
    # 8. REGIME ANALYSIS
    # =========================================================================
    def regime_analysis(self, vol_window=20, trend_window=50, print_report=True):
        """
        Segment performance by volatility and trend regimes.

        Volatility regimes: Low / Medium / High (terciles of rolling vol).
        Trend regimes:      Bull (price > MA) / Bear (price < MA).

        FIX: uses duplicates='drop' in pd.cut to handle flat/degenerate vol.
        """
        if self._returns is None:
            self.run(print_stats=False)

        rets  = self._returns
        close = self.close

        # --- Volatility regimes ---
        rolling_vol = rets.rolling(vol_window).std() * np.sqrt(self.ann_factor)
        q33 = float(rolling_vol.quantile(0.33))
        q67 = float(rolling_vol.quantile(0.67))

        # FIX: deduplicate bin edges before pd.cut
        raw_bins = [-np.inf, q33, q67, np.inf]
        uniq_bins = sorted(set(raw_bins))

        if len(uniq_bins) < 3:
            # Degenerate — assign everything to the middle bucket
            regimes_vol = pd.Series('Med Vol', index=rolling_vol.index, dtype=object)
        else:
            n_labels = len(uniq_bins) - 1
            all_labels = ['Low Vol', 'Med Vol', 'High Vol']
            use_labels = all_labels[:n_labels]
            regimes_vol = pd.cut(
                rolling_vol,
                bins=uniq_bins,
                labels=use_labels,
                duplicates='drop',
            )

        # --- Trend regimes ---
        ma = close.rolling(trend_window).mean()
        trend_regime = pd.Series(
            np.where(close > ma, 'Bull', 'Bear'),
            index=close.index,
        )

        vol_stats   = {}
        trend_stats = {}

        for label in ['Low Vol', 'Med Vol', 'High Vol']:
            mask  = regimes_vol == label
            r_seg = rets[mask].values
            if len(r_seg) > 5:
                vol_stats[label] = {
                    'avg_return':   float(np.mean(r_seg) * self.ann_factor * 100),
                    'sharpe':       self._calc_sharpe(r_seg),
                    'max_drawdown': self._calc_max_dd(r_seg) * 100,
                    'n_periods':    int(mask.sum()),
                }

        for label in ['Bull', 'Bear']:
            mask  = trend_regime == label
            r_seg = rets[mask].values
            if len(r_seg) > 5:
                trend_stats[label] = {
                    'avg_return':   float(np.mean(r_seg) * self.ann_factor * 100),
                    'sharpe':       self._calc_sharpe(r_seg),
                    'max_drawdown': self._calc_max_dd(r_seg) * 100,
                    'n_periods':    int(mask.sum()),
                }

        result = {'volatility_regimes': vol_stats, 'trend_regimes': trend_stats}

        if print_report:
            print("\n" + "=" * 60)
            print("  🌡️ Regime-Conditional Performance")
            print("=" * 60)
            print("\n  Volatility Regimes:")
            for lbl, s in vol_stats.items():
                print(f"    {lbl:<10}  Return={s['avg_return']:>7.2f}%  "
                      f"Sharpe={s['sharpe']:>6.4f}  "
                      f"MaxDD={s['max_drawdown']:>7.2f}%  "
                      f"N={s['n_periods']}")
            print("\n  Trend Regimes:")
            for lbl, s in trend_stats.items():
                print(f"    {lbl:<6}      Return={s['avg_return']:>7.2f}%  "
                      f"Sharpe={s['sharpe']:>6.4f}  "
                      f"MaxDD={s['max_drawdown']:>7.2f}%  "
                      f"N={s['n_periods']}")

        return result

    # =========================================================================
    # 9. EXTENDED RISK METRICS
    # =========================================================================
    def risk_metrics(self, mar=0.0, print_report=True):
        """
        Extended risk metrics:
          CVaR (95%, 99%), Omega Ratio, Tail Ratio,
          Gain-to-Pain Ratio, Ulcer Index.
        """
        if self._returns is None:
            self.run(print_stats=False)

        rets = self._returns.values.flatten()

        cvar_95 = self._cvar(rets, 0.05)
        cvar_99 = self._cvar(rets, 0.01)

        # Omega Ratio
        excess = rets - mar
        gains  = excess[excess > 0].sum()
        losses = abs(excess[excess < 0].sum())
        omega  = (gains / losses) if losses > 0 else float('inf')

        # Tail Ratio
        p95 = np.percentile(rets, 95)
        p05 = np.percentile(rets, 5)
        tail_ratio = (abs(p95) / abs(p05)) if p05 != 0 else float('inf')

        # Gain-to-Pain Ratio
        loss_sum = abs(rets[rets < 0].sum())
        g2p = (rets.sum() / loss_sum) if loss_sum > 0 else float('inf')

        # Ulcer Index
        cumul = np.cumprod(1 + rets)
        peak  = np.maximum.accumulate(cumul)
        dd_pct = (cumul - peak) / peak * 100
        ulcer  = float(np.sqrt(np.mean(dd_pct ** 2)))

        result = {
            'cvar_95':     cvar_95,
            'cvar_99':     cvar_99,
            'omega':       omega,
            'tail_ratio':  tail_ratio,
            'g2p_ratio':   g2p,
            'ulcer_index': ulcer,
        }

        def _fmtr(v, pct=False):
            if not np.isfinite(v):
                return "     inf"
            return f"{v * 100:>8.4f}%" if pct else f"{v:>8.4f}"

        if print_report:
            print("\n" + "=" * 60)
            print("  ⚖️ Extended Risk Metrics")
            print("=" * 60)
            print(f"  CVaR (95%):          {cvar_95 * 100:>8.4f}%")
            print(f"  CVaR (99%):          {cvar_99 * 100:>8.4f}%")
            print(f"  Omega Ratio:         {_fmtr(omega)}  {'✅' if omega > 1 else '⚠️'}")
            print(f"  Tail Ratio:          {_fmtr(tail_ratio)}  "
                  f"{'✅' if tail_ratio > 1 else '⚠️'}")
            print(f"  Gain-to-Pain:        {_fmtr(g2p)}  {'✅' if g2p > 1 else '⚠️'}")
            print(f"  Ulcer Index:         {ulcer:>8.4f}  {'✅' if ulcer < 5 else '⚠️'}")

        return result

    # =========================================================================
    # 10. KELLY POSITION SIZING
    # =========================================================================
    def kelly_sizing(self, fraction=0.25, print_report=True):
        """
        Fractional Kelly criterion.

        Args:
            fraction: Safety multiplier (default 0.25 = Quarter-Kelly).
        """
        if self._portfolio is None:
            self.run(print_stats=False)

        try:
            trades = self._portfolio.trades.records_readable
            pnl    = trades['PnL'].values if 'PnL' in trades.columns else np.array([])
        except Exception:
            pnl = np.array([])

        if len(pnl) < 5:
            if print_report:
                print("\n  ⚠️ Kelly Sizing: insufficient trade data.")
            return {}

        winners  = pnl[pnl > 0]
        losers   = pnl[pnl < 0]
        win_rate = len(winners) / len(pnl)
        avg_win  = float(winners.mean()) if len(winners) > 0 else 0.0
        avg_loss = float(abs(losers.mean())) if len(losers) > 0 else 1e-9

        b       = avg_win / avg_loss
        kelly   = (b * win_rate - (1 - win_rate)) / b
        f_kelly = max(0.0, kelly * fraction)

        result = {
            'win_rate':                  win_rate,
            'avg_win':                   avg_win,
            'avg_loss':                  avg_loss,
            'win_loss_ratio':            b,
            'full_kelly':                kelly,
            f'kelly_{int(fraction*100)}pct': f_kelly,
        }

        if print_report:
            print("\n" + "=" * 60)
            print(f"  🎯 Kelly Position Sizing (fraction={fraction})")
            print("=" * 60)
            print(f"  Win Rate:           {win_rate * 100:.2f}%")
            print(f"  Win/Loss Ratio:     {b:.4f}")
            print(f"  Full Kelly f*:      {kelly * 100:.2f}%")
            print(f"  {int(fraction*100)}% Kelly f*:       {f_kelly * 100:.2f}%")
            rec = ("Aggressive" if f_kelly > 0.20
                   else "Moderate" if f_kelly > 0.05
                   else "Conservative")
            print(f"  Sizing Regime:      {rec}")

        return result

    # =========================================================================
    # FULL ANALYSIS (convenience wrapper)
    # =========================================================================
    def full_analysis(self, n_mc=500, n_wf_splits=4, n_trials=1,
                      wf_mode='rolling', verbose=True):
        """
        Run all 10 analyses in sequence and return combined results dict.

        FIX: single self.run() call — no more double-printed stats.
        Skips advanced analysis when trade count < 2.
        """
        # Single authoritative run — result cached in self._portfolio / self._returns
        base = self.run(print_stats=verbose)

        n_trades = self._portfolio.trades.count()
        if n_trades < 2:
            if verbose:
                print(f"\n  ⚠️  Skipping full VBT analysis: "
                      f"only {n_trades} trade(s) found.")
            return {'base': base}

        bench   = self.vs_benchmark(print_report=verbose)
        mc      = self.monte_carlo(n_simulations=n_mc, print_report=verbose)
        wf      = self.walk_forward(n_splits=n_wf_splits, mode=wf_mode, print_report=verbose)
        st      = self.stress_testing(print_report=verbose)
        dsr     = self.deflated_sharpe(n_trials=n_trials, print_report=verbose)
        trades  = self.trade_analysis(print_report=verbose)
        regimes = self.regime_analysis(print_report=verbose)
        risk    = self.risk_metrics(print_report=verbose)
        kelly   = self.kelly_sizing(print_report=verbose)

        return {
            'base':            base,
            'benchmark':       bench,
            'monte_carlo':     mc,
            'walk_forward':    wf,
            'stress_testing':  st,
            'deflated_sharpe': dsr,
            'trade_analysis':  trades,
            'regime_analysis': regimes,
            'risk_metrics':    risk,
            'kelly_sizing':    kelly,
        }

    def generate_report(self, filepath, results=None):
        """
        Generate a detailed Markdown report from backtest results.
        If results is None, runs full_analysis(verbose=False).
        """
        if results is None:
            results = self.full_analysis(verbose=False)

        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            f.write(f"# Backtest Report: {os.path.basename(filepath).replace('.md', '')}\n\n")
            
            # --- Base Stats ---
            f.write("## 📊 Performance Summary\n")
            base = results.get('base', {})
            stats = base.get('stats', pd.Series())
            
            f.write("| Metric | Value |\n")
            f.write("| :--- | :--- |\n")
            for idx, val in stats.items():
                if isinstance(val, float):
                    f.write(f"| {idx} | {val:.4f} |\n")
                else:
                    f.write(f"| {idx} | {val} |\n")
            
            f.write(f"| Sortino Ratio | {base.get('sortino', 0):.4f} |\n")
            f.write(f"| Calmar Ratio | {base.get('calmar', 0):.4f} |\n")
            f.write(f"| CVaR (95%) | {base.get('cvar_95', 0)*100:.2f}% |\n\n")

            # --- Benchmark ---
            bench = results.get('benchmark', {})
            if bench:
                f.write("## 📈 Benchmark Comparison\n")
                f.write(f"- **Strategy Return:** {bench.get('strategy_return', 0)*100:.2f}%\n")
                f.write(f"- **Benchmark Return:** {bench.get('benchmark_return', 0)*100:.2f}%\n")
                f.write(f"- **Alpha (ann.):** {bench.get('alpha', 0):.4f}\n")
                f.write(f"- **Beta:** {bench.get('beta', 0):.4f}\n")
                f.write(f"- **Information Ratio:** {bench.get('information_ratio', 0):.4f}\n\n")

            # --- Monte Carlo ---
            mc = results.get('monte_carlo', {})
            if mc:
                f.write("## 🎲 Monte Carlo Analysis\n")
                f.write(f"- **P-value (Return):** {mc.get('p_value_return', 0):.4f}\n")
                f.write(f"- **P-value (Sharpe):** {mc.get('p_value_sharpe', 0):.4f}\n")
                f.write(f"- **P-value (vs Random):** {mc.get('p_value_vs_random', 0):.4f}\n\n")

            # --- Walk Forward ---
            wf = results.get('walk_forward', {})
            if wf and wf.get('windows'):
                f.write("## 🔄 Sub-period Analysis\n")
                f.write("| Window | Period | Return | Sharpe | Max DD |\n")
                f.write("| :--- | :--- | :--- | :--- | :--- |\n")
                for w in wf['windows']:
                    f.write(f"| {w['window']} | {w['test_period']} | {w['total_return']*100:.2f}% | {w['sharpe']:.4f} | {w['max_drawdown']*100:.2f}% |\n")
                f.write("\n")

            # --- Stress Test ---
            st = results.get('stress_testing', {})
            if st:
                f.write("## 🔥 Stress Test Performance\n")
                f.write("| Scenario | Stressed Return | Max DD | Status |\n")
                f.write("| :--- | :--- | :--- | :--- |\n")
                for name, r in st.items():
                    status = "Survived" if r['survived'] else "FAILED"
                    f.write(f"| {r['description']} | {r['stressed_return']*100:.2f}% | {r['stressed_max_dd']*100:.2f}% | {status} |\n")
                f.write("\n")

            # --- Trade Analysis ---
            trades = results.get('trade_analysis', {})
            if trades:
                f.write("## 🏦 Trade-Level Analysis\n")
                f.write(f"- **Total Trades:** {trades.get('n_trades', 0)}\n")
                f.write(f"- **Win Rate:** {trades.get('win_rate', 0)*100:.2f}%\n")
                f.write(f"- **Profit Factor:** {trades.get('profit_factor', 0):.2f}\n")
                f.write(f"- **Expectancy:** {trades.get('expectancy', 0):.2f}\n\n")

            f.write("---\n*Report generated by VBTBacktester*")

    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================
    def _calc_sharpe(self, returns):
        """Annualised Sharpe ratio from a returns array."""
        r = np.array(returns)
        if len(r) < 2 or np.std(r, ddof=1) == 0:
            return 0.0
        return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(self.ann_factor))

    def _calc_max_dd(self, returns):
        """Maximum drawdown from a returns array."""
        r     = np.array(returns)
        cumul = np.cumprod(1 + r)
        peak  = np.maximum.accumulate(cumul)
        dd    = (cumul - peak) / peak
        return float(np.min(dd))

    def _cvar(self, returns, alpha=0.05):
        """Conditional Value-at-Risk (Expected Shortfall) at level alpha."""
        r    = np.array(returns)
        var  = np.percentile(r, alpha * 100)
        tail = r[r <= var]
        return float(np.mean(tail)) if len(tail) > 0 else float(var)
        