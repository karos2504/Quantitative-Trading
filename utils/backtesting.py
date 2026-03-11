"""
Advanced Backtesting Engine — vectorbt + Monte Carlo + Walk-Forward + Stress Testing

Provides the VBTBacktester class that wraps vectorbt's Portfolio.from_signals
with 4 validation layers:
  1. Monte Carlo simulation (shuffled returns)
  2. Walk-forward analysis (rolling train/test splits)
  3. Stress testing (inject historical crisis scenarios)
  4. Bias removal (Deflated Sharpe Ratio)
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats as scipy_stats


# ============================================================================
# CRISIS SCENARIOS — historical worst-case drawdown distributions
# ============================================================================
CRISIS_SCENARIOS = {
    '2008_GFC': {
        'description': 'Global Financial Crisis 2008',
        'daily_mean': -0.0035,
        'daily_std': 0.042,
        'duration_days': 250,
        'peak_drawdown': -0.38,
    },
    'COVID_2020': {
        'description': 'COVID-19 Crash 2020',
        'daily_mean': -0.0060,
        'daily_std': 0.055,
        'duration_days': 33,
        'peak_drawdown': -0.34,
    },
    'DOTCOM_2000': {
        'description': 'Dot-com Bubble Burst 2000-2002',
        'daily_mean': -0.0012,
        'daily_std': 0.025,
        'duration_days': 630,
        'peak_drawdown': -0.45,
    },
    'BLACK_MONDAY_1987': {
        'description': 'Black Monday 1987',
        'daily_mean': -0.0150,
        'daily_std': 0.080,
        'duration_days': 5,
        'peak_drawdown': -0.22,
    },
}


class VBTBacktester:
    """
    Advanced backtesting engine built on vectorbt.

    Usage:
        bt = VBTBacktester(close, entries, exits, freq='D')
        result = bt.run()
        mc = bt.monte_carlo(n_simulations=1000)
        wf = bt.walk_forward(n_splits=5)
        st = bt.stress_testing()
        dsr = bt.deflated_sharpe(n_trials=10)
    """

    def __init__(self, close, entries, exits, freq='D',
                 init_cash=100_000, commission=0.001, slippage=0.001):
        """
        Args:
            close: pd.Series of close prices (DatetimeIndex).
            entries: pd.Series of boolean entry signals.
            exits: pd.Series of boolean exit signals.
            freq: Data frequency ('D', '5T', '15T', '1H', etc.).
            init_cash: Initial cash balance.
            commission: Commission percentage per trade.
            slippage: Slippage percentage per trade.
        """
        self.close = close.squeeze() if isinstance(close, pd.DataFrame) else close
        self.entries = entries.squeeze() if isinstance(entries, pd.DataFrame) else entries
        self.exits = exits.squeeze() if isinstance(exits, pd.DataFrame) else exits
        self.freq = freq
        self.init_cash = init_cash
        self.commission = commission
        self.slippage = slippage

        self._portfolio = None
        self._returns = None

    # -----------------------------------------------------------------
    # 1. BASE BACKTEST
    # -----------------------------------------------------------------
    def run(self, print_stats=True):
        """Run vectorbt backtest and return portfolio stats."""
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

        if print_stats:
            print("\n" + "=" * 60)
            print("  📊 vectorbt Portfolio Stats")
            print("=" * 60)
            print(stats.to_string())

        return {
            'portfolio': self._portfolio,
            'stats': stats,
            'total_return': self._portfolio.total_return(),
            'sharpe': self._portfolio.sharpe_ratio(),
            'max_drawdown': self._portfolio.max_drawdown(),
            'returns': self._returns,
        }

    # -----------------------------------------------------------------
    # 2. MONTE CARLO SIMULATION (Block Bootstrap)
    # -----------------------------------------------------------------
    def monte_carlo(self, n_simulations=1000, block_size=None, print_report=True):
        """
        Block-bootstrap Monte Carlo simulation.

        Instead of naively shuffling individual returns (which preserves
        the product), this resamples *blocks* of consecutive returns with
        replacement, producing genuinely different equity paths that
        preserve short-term autocorrelation.

        Additionally compares against a random-signal baseline (buy/sell
        at random) to test if the strategy's signal timing adds value.

        Args:
            n_simulations: Number of bootstrap replications.
            block_size: Size of each block. Defaults to sqrt(n).
            print_report: Whether to print results.
        """
        if self._returns is None:
            self.run(print_stats=False)

        returns = self._returns.values.flatten()
        n = len(returns)
        actual_cum_return = float(np.cumprod(1 + returns)[-1] - 1)
        actual_sharpe = self._calc_sharpe(returns)
        actual_max_dd = self._calc_max_dd(returns)

        if block_size is None:
            block_size = max(5, int(np.sqrt(n)))

        # --- Block Bootstrap ---
        sim_returns = np.zeros(n_simulations)
        sim_sharpes = np.zeros(n_simulations)
        sim_max_dd = np.zeros(n_simulations)

        n_blocks = (n // block_size) + 1

        for i in range(n_simulations):
            # Sample block start indices with replacement
            block_starts = np.random.randint(0, max(1, n - block_size), size=n_blocks)
            sampled = np.concatenate([returns[s:s + block_size] for s in block_starts])[:n]

            sim_returns[i] = np.cumprod(1 + sampled)[-1] - 1
            sim_sharpes[i] = self._calc_sharpe(sampled)
            sim_max_dd[i] = self._calc_max_dd(sampled)

        # --- Random Signal Baseline ---
        rand_returns = np.zeros(n_simulations)
        for i in range(n_simulations):
            random_entries = np.random.random(n) > 0.5
            # When "in market", earn close returns; when out, earn 0
            close_returns = self.close.pct_change().fillna(0).values.flatten()
            rand_path = np.where(random_entries, close_returns, 0.0)
            rand_returns[i] = np.cumprod(1 + rand_path)[-1] - 1

        # P-values
        p_value_return = float(np.mean(sim_returns >= actual_cum_return))
        p_value_sharpe = float(np.mean(sim_sharpes >= actual_sharpe))
        p_value_vs_random = float(np.mean(rand_returns >= actual_cum_return))

        pctiles_list = [5, 25, 50, 75, 95]
        result = {
            'actual_return': actual_cum_return,
            'actual_sharpe': actual_sharpe,
            'actual_max_dd': actual_max_dd,
            'sim_return_pctiles': np.percentile(sim_returns, pctiles_list),
            'sim_sharpe_pctiles': np.percentile(sim_sharpes, pctiles_list),
            'sim_max_dd_pctiles': np.percentile(sim_max_dd, pctiles_list),
            'rand_return_pctiles': np.percentile(rand_returns, pctiles_list),
            'p_value_return': p_value_return,
            'p_value_sharpe': p_value_sharpe,
            'p_value_vs_random': p_value_vs_random,
            'n_simulations': n_simulations,
            'block_size': block_size,
        }

        if print_report:
            print("\n" + "=" * 60)
            print(f"  🎲 Monte Carlo — Block Bootstrap ({n_simulations} sims, "
                  f"block={block_size})")
            print("=" * 60)

            print(f"\n  Actual Cumulative Return: {actual_cum_return * 100:>8.2f}%")
            print(f"  Actual Sharpe Ratio:      {actual_sharpe:>8.4f}")
            print(f"  Actual Max Drawdown:      {actual_max_dd * 100:>8.2f}%")

            print("\n  Bootstrap Return Percentiles:")
            for p, v in zip(pctiles_list, result['sim_return_pctiles']):
                print(f"    {p:>3}th: {v * 100:>8.2f}%")

            print("\n  Bootstrap Sharpe Percentiles:")
            for p, v in zip(pctiles_list, result['sim_sharpe_pctiles']):
                print(f"    {p:>3}th: {v:>8.4f}")

            print("\n  Bootstrap Max Drawdown Percentiles:")
            for p, v in zip(pctiles_list, result['sim_max_dd_pctiles']):
                print(f"    {p:>3}th: {v * 100:>8.2f}%")

            print("\n  Random Signal Baseline Return Percentiles:")
            for p, v in zip(pctiles_list, result['rand_return_pctiles']):
                print(f"    {p:>3}th: {v * 100:>8.2f}%")

            print(f"\n  P-value (bootstrap return): {p_value_return:.4f}"
                  f"  {'✅ Robust' if p_value_return > 0.25 else '⚠️ Fragile path'}")
            print(f"  P-value (bootstrap Sharpe): {p_value_sharpe:.4f}"
                  f"  {'✅ Robust' if p_value_sharpe > 0.25 else '⚠️ Fragile path'}")
            print(f"  P-value (vs random signal): {p_value_vs_random:.4f}"
                  f"  {'✅ Beats random' if p_value_vs_random < 0.05 else '⚠️ No edge over random'}")

        return result

    # -----------------------------------------------------------------
    # 3. WALK-FORWARD ANALYSIS
    # -----------------------------------------------------------------
    def walk_forward(self, n_splits=5, train_ratio=0.75, print_report=True):
        """
        Rolling walk-forward analysis.

        Splits data into `n_splits` windows, each with `train_ratio` in-sample
        and the rest out-of-sample. Runs the backtest ONLY on OOS data to
        prevent look-ahead bias.

        Returns:
            dict with per-window and aggregated OOS stats.
        """
        if self._returns is None:
            self.run(print_stats=False)

        n = len(self.close)
        window_size = n // n_splits
        train_size = int(window_size * train_ratio)
        test_size = window_size - train_size

        windows = []
        for i in range(n_splits):
            start = i * window_size
            train_end = start + train_size
            test_end = min(start + window_size, n)

            if test_end <= train_end:
                continue

            test_close = self.close.iloc[train_end:test_end]
            test_entries = self.entries.iloc[train_end:test_end]
            test_exits = self.exits.iloc[train_end:test_end]

            if len(test_close) < 5:
                continue

            try:
                pf = vbt.Portfolio.from_signals(
                    close=test_close,
                    entries=test_entries,
                    exits=test_exits,
                    init_cash=self.init_cash,
                    fees=self.commission,
                    slippage=self.slippage,
                    freq=self.freq,
                )
                ret = pf.returns()
                windows.append({
                    'window': i + 1,
                    'train_period': f"{self.close.index[start].strftime('%Y-%m-%d')} → "
                                    f"{self.close.index[min(train_end - 1, n - 1)].strftime('%Y-%m-%d')}",
                    'test_period': f"{test_close.index[0].strftime('%Y-%m-%d')} → "
                                   f"{test_close.index[-1].strftime('%Y-%m-%d')}",
                    'total_return': float(pf.total_return()),
                    'sharpe': float(pf.sharpe_ratio()) if not np.isnan(pf.sharpe_ratio()) else 0.0,
                    'max_drawdown': float(pf.max_drawdown()),
                    'n_trades': int(pf.trades.count()),
                })
            except Exception:
                continue

        if not windows:
            if print_report:
                print("\n  ⚠️ Walk-Forward: insufficient data for splitting")
            return {'windows': [], 'aggregated': {}}

        # Aggregated OOS metrics
        agg = {
            'avg_return': np.mean([w['total_return'] for w in windows]),
            'avg_sharpe': np.mean([w['sharpe'] for w in windows]),
            'worst_drawdown': np.min([w['max_drawdown'] for w in windows]),
            'total_trades': sum(w['n_trades'] for w in windows),
            'n_windows': len(windows),
            'profitable_windows': sum(1 for w in windows if w['total_return'] > 0),
        }

        result = {'windows': windows, 'aggregated': agg}

        if print_report:
            print("\n" + "=" * 60)
            print(f"  🔄 Walk-Forward Analysis ({len(windows)} windows, "
                  f"{int(train_ratio * 100)}% train / {int((1 - train_ratio) * 100)}% test)")
            print("=" * 60)

            wf_df = pd.DataFrame(windows)
            wf_df['total_return'] = wf_df['total_return'].map(lambda x: f"{x * 100:.2f}%")
            wf_df['sharpe'] = wf_df['sharpe'].map(lambda x: f"{x:.4f}")
            wf_df['max_drawdown'] = wf_df['max_drawdown'].map(lambda x: f"{x * 100:.2f}%")
            print(wf_df[['window', 'test_period', 'total_return',
                         'sharpe', 'max_drawdown', 'n_trades']].to_string(index=False))

            print(f"\n  --- Aggregated OOS Metrics ---")
            print(f"  Avg Return:         {agg['avg_return'] * 100:.2f}%")
            print(f"  Avg Sharpe:         {agg['avg_sharpe']:.4f}")
            print(f"  Worst Drawdown:     {agg['worst_drawdown'] * 100:.2f}%")
            print(f"  Profitable Windows: {agg['profitable_windows']}/{agg['n_windows']}")

        return result

    # -----------------------------------------------------------------
    # 4. STRESS TESTING
    # -----------------------------------------------------------------
    def stress_testing(self, scenarios=None, print_report=True):
        """
        Inject historical crisis return distributions into the strategy
        returns to measure worst-case performance.

        For each scenario, inserts the crisis returns at a random position
        in the return stream and measures the resulting max drawdown.
        """
        if self._returns is None:
            self.run(print_stats=False)

        if scenarios is None:
            scenarios = CRISIS_SCENARIOS

        returns = self._returns.values.flatten()
        results = {}

        for name, crisis in scenarios.items():
            crisis_returns = np.random.normal(
                crisis['daily_mean'], crisis['daily_std'], crisis['duration_days']
            )

            # Insert crisis at random position
            insert_pos = np.random.randint(0, max(1, len(returns) - crisis['duration_days']))
            stressed = returns.copy()
            end_pos = min(insert_pos + crisis['duration_days'], len(stressed))
            actual_len = end_pos - insert_pos
            stressed[insert_pos:end_pos] = crisis_returns[:actual_len]

            stressed_cumulative = (1 + stressed).prod() - 1
            stressed_max_dd = self._calc_max_dd(stressed)
            stressed_sharpe = self._calc_sharpe(stressed)

            survival = stressed_cumulative > -0.95  # Survived if didn't lose 95%+

            results[name] = {
                'description': crisis['description'],
                'duration_days': crisis['duration_days'],
                'peak_drawdown_scenario': crisis['peak_drawdown'],
                'stressed_return': stressed_cumulative,
                'stressed_max_dd': stressed_max_dd,
                'stressed_sharpe': stressed_sharpe,
                'survived': survival,
            }

        if print_report:
            print("\n" + "=" * 60)
            print("  🔥 Stress Test — Crisis Scenario Injection")
            print("=" * 60)
            for name, r in results.items():
                status = "✅ Survived" if r['survived'] else "💀 WIPED OUT"
                print(f"\n  {r['description']} ({r['duration_days']}d, "
                      f"hist DD {r['peak_drawdown_scenario'] * 100:.0f}%)")
                print(f"    Stressed Return:  {r['stressed_return'] * 100:>8.2f}%")
                print(f"    Stressed Max DD:  {r['stressed_max_dd'] * 100:>8.2f}%")
                print(f"    Stressed Sharpe:  {r['stressed_sharpe']:>8.4f}")
                print(f"    Status:           {status}")

        return results

    # -----------------------------------------------------------------
    # 5. DEFLATED SHARPE RATIO (BIAS REMOVAL)
    # -----------------------------------------------------------------
    def deflated_sharpe(self, n_trials=1, print_report=True):
        """
        Deflated Sharpe Ratio (DSR) per Bailey & López de Prado (2014).

        Adjusts the observed Sharpe for the number of strategy variants
        tested (n_trials), accounting for skewness and kurtosis of returns.

        A DSR p-value < 0.05 means the Sharpe is unlikely due to luck.
        """
        if self._returns is None:
            self.run(print_stats=False)

        returns = self._returns.values.flatten()
        n = len(returns)
        sr = self._calc_sharpe(returns)
        skew = float(scipy_stats.skew(returns))
        kurt = float(scipy_stats.kurtosis(returns, fisher=True))

        # Expected max Sharpe under null (Euler-Mascheroni approximation)
        euler_mascheroni = 0.5772156649
        if n_trials > 1:
            sr_max = np.sqrt(2 * np.log(n_trials)) * (
                1 - euler_mascheroni / (2 * np.log(n_trials))
            ) + euler_mascheroni / np.sqrt(2 * np.log(n_trials))
        else:
            sr_max = 0.0

        # DSR test statistic
        sr_std = np.sqrt(
            (1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / (n - 1)
        )

        if sr_std > 0:
            dsr_stat = (sr - sr_max) / sr_std
            p_value = 1 - scipy_stats.norm.cdf(dsr_stat)
        else:
            dsr_stat = 0.0
            p_value = 1.0

        result = {
            'observed_sharpe': sr,
            'expected_max_sharpe': sr_max,
            'dsr_statistic': dsr_stat,
            'p_value': p_value,
            'n_trials': n_trials,
            'skewness': skew,
            'kurtosis': kurt,
            'significant': p_value < 0.05,
        }

        if print_report:
            print("\n" + "=" * 60)
            print(f"  🛡️ Deflated Sharpe Ratio (n_trials={n_trials})")
            print("=" * 60)
            print(f"  Observed Sharpe:      {sr:>8.4f}")
            print(f"  Expected Max Sharpe:  {sr_max:>8.4f}")
            print(f"  DSR Statistic:        {dsr_stat:>8.4f}")
            print(f"  P-value:              {p_value:>8.4f}")
            print(f"  Skewness:             {skew:>8.4f}")
            print(f"  Excess Kurtosis:      {kurt:>8.4f}")
            sig = "✅ Likely skill" if result['significant'] else "⚠️ Possibly luck"
            print(f"  Conclusion:           {sig}")

        return result

    # -----------------------------------------------------------------
    # FULL ANALYSIS (convenience)
    # -----------------------------------------------------------------
    def full_analysis(self, n_mc=1000, n_wf_splits=5, n_trials=1):
        """Run all 5 analyses in sequence and return combined results."""
        base = self.run(print_stats=True)
        mc = self.monte_carlo(n_simulations=n_mc)
        wf = self.walk_forward(n_splits=n_wf_splits)
        st = self.stress_testing()
        dsr = self.deflated_sharpe(n_trials=n_trials)
        return {
            'base': base,
            'monte_carlo': mc,
            'walk_forward': wf,
            'stress_testing': st,
            'deflated_sharpe': dsr,
        }

    # -----------------------------------------------------------------
    # PRIVATE HELPERS
    # -----------------------------------------------------------------
    @staticmethod
    def _calc_sharpe(returns, periods_per_year=252):
        """Annualized Sharpe ratio from a returns array."""
        r = np.array(returns)
        if len(r) < 2 or np.std(r) == 0:
            return 0.0
        return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(periods_per_year))

    @staticmethod
    def _calc_max_dd(returns):
        """Maximum drawdown from a returns array."""
        r = np.array(returns)
        cumulative = np.cumprod(1 + r)
        peak = np.maximum.accumulate(cumulative)
        dd = (cumulative - peak) / peak
        return float(np.min(dd))
