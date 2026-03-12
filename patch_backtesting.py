import re

with open('utils/backtesting.py', 'r') as f:
    text = f.read()

# 1. Fix _calc_sharpe and _calc_max_dd
text = re.sub(r'@staticmethod\n\s+def _calc_sharpe\(returns, periods_per_year=252\):\n\s+"""Annualized Sharpe ratio from a returns array\."""\n\s+r = np\.array\(returns\)\n\s+if len\(r\) < 2 or np\.std\(r, ddof=1\) == 0:\n\s+return 0\.0\n\s+return float\(np\.mean\(r\) \/ np\.std\(r, ddof=1\) \* np\.sqrt\(periods_per_year\)\)',
r'''def _calc_sharpe(self, returns):
        """Annualized Sharpe ratio from a returns array."""
        r = np.array(returns)
        if len(r) < 2 or np.std(r, ddof=1) == 0:
            return 0.0
        ann_factor = getattr(self._portfolio, 'ann_factor', 252)
        return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(ann_factor))''', text)

text = re.sub(r'@staticmethod\n\s+def _calc_max_dd\(returns\):', r'def _calc_max_dd(self, returns):', text)

# 2. Fix stress_testing
text = re.sub(r'''        for name, crisis in scenarios\.items\(\):\n\s+crisis_returns = np\.random\.normal\(\n\s+crisis\['daily_mean'\], crisis\['daily_std'\], crisis\['duration_days'\]\n\s+\)\n\n\s+# Insert crisis at random position\n\s+insert_pos = np\.random\.randint\(0, max\((.*?)\)\)\n\s+stressed = returns\.copy\(\)\n\s+end_pos = min\(insert_pos \+ crisis\['duration_days'\], len\(stressed\)\)\n\s+actual_len = end_pos - insert_pos\n\s+stressed\[insert_pos:end_pos\] = crisis_returns\[:actual_len\]''',
r'''        for name, crisis in scenarios.items():
            ann_factor = getattr(self._portfolio, 'ann_factor', 252)
            period_ratio = max(1, 252 / ann_factor)
            
            crisis_duration = max(1, int(crisis['duration_days'] / period_ratio))
            crisis_mean = crisis['daily_mean'] * period_ratio
            crisis_std = crisis['daily_std'] * np.sqrt(period_ratio)
            
            crisis_returns = np.random.normal(
                crisis_mean, crisis_std, crisis_duration
            )

            # Insert crisis at random position
            insert_pos = np.random.randint(0, max(1, len(returns) - crisis_duration))
            stressed = returns.copy()
            end_pos = min(insert_pos + crisis_duration, len(stressed))
            actual_len = end_pos - insert_pos
            stressed[insert_pos:end_pos] = crisis_returns[:actual_len]''', text)

# 3. Fix Deflated Sharpe Ratio
text = re.sub(r'''        returns = self\._returns\.values\.flatten\(\)\n\s+n = len\(returns\)\n\s+sr = self\._calc_sharpe\(returns\)\n\s+skew = float\(scipy_stats\.skew\(returns\)\)\n\s+kurt = float\(scipy_stats\.kurtosis\(returns, fisher=True\)\)\n\n\s+# Expected max Sharpe under null \(Euler-Mascheroni approximation\)\n\s+euler_mascheroni = 0\.5772156649\n\s+if n_trials > 1:\n\s+sr_max = np\.sqrt\(2 \* np\.log\(n_trials\)\) \* \(\n\s+1 - euler_mascheroni / \(2 \* np\.log\(n_trials\)\)\n\s+\) \+ euler_mascheroni / np\.sqrt\(2 \* np\.log\(n_trials\)\)\n\s+else:\n\s+sr_max = 0\.0\n\n\s+# DSR test statistic\n\s+sr_std = np\.sqrt\(\n\s+\(1 - skew \* sr \+ \(kurt - 1\) / 4 \* sr \*\* 2\) / \(n - 1\)\n\s+\)\n\n\s+if sr_std > 0:\n\s+dsr_stat = \(sr - sr_max\) / sr_std\n\s+p_value = 1 - scipy_stats\.norm\.cdf\(dsr_stat\)\n\s+else:\n\s+dsr_stat = 0\.0\n\s+p_value = 1\.0\n\n\s+result = \{\n\s+'observed_sharpe': sr,\n\s+'expected_max_sharpe': sr_max,''',
r'''        returns = self._returns.values.flatten()
        n = len(returns)

        period_std = np.std(returns, ddof=1)
        period_sr = np.mean(returns) / period_std if period_std > 0 else 0.0
            
        skew = float(scipy_stats.skew(returns))
        kurt = float(scipy_stats.kurtosis(returns, fisher=True))

        euler_mascheroni = 0.5772156649
        if n_trials > 1:
            z_max = np.sqrt(2 * np.log(n_trials)) * (
                1 - euler_mascheroni / (2 * np.log(n_trials))
            ) + euler_mascheroni / np.sqrt(2 * np.log(n_trials))
        else:
            z_max = 0.0

        sr_std = np.sqrt(
            (1 - skew * period_sr + (kurt - 1) / 4 * period_sr ** 2) / max(1, n - 1)
        )
        
        expected_max_sr = z_max * sr_std

        if sr_std > 0:
            dsr_stat = (period_sr - expected_max_sr) / sr_std
            p_value = 1 - scipy_stats.norm.cdf(dsr_stat)
        else:
            dsr_stat = 0.0
            p_value = 1.0
            
        ann_factor = getattr(self._portfolio, 'ann_factor', 252)
        sr_ann = period_sr * np.sqrt(ann_factor)
        sr_max_ann = expected_max_sr * np.sqrt(ann_factor)

        result = {
            'observed_sharpe': sr_ann,
            'expected_max_sharpe': sr_max_ann,''', text)

text = re.sub(r'''            print\(f"  Observed Sharpe:      \{sr:>8\.4f\}"\)\n\s+print\(f"  Expected Max Sharpe:  \{sr_max:>8\.4f\}"\)''',
r'''            print(f"  Observed Sharpe:      {sr_ann:>8.4f}")
            print(f"  Expected Max Sharpe:  {sr_max_ann:>8.4f}")''', text)

# 4. Fix walk_forward
text = re.sub(r'''        train_size = int\(window_size \* train_ratio\)\n\s+test_size = window_size - train_size\n\n\s+windows = \[\]\n\s+for i in range\(n_splits\):\n\s+start = i \* window_size\n\s+train_end = start \+ train_size\n\s+test_end = min\(start \+ window_size, n\)\n\n\s+if test_end <= train_end:\n\s+continue\n\n\s+test_close = self\.close\.iloc\[train_end:test_end\]\n\s+test_entries = self\.entries\.iloc\[train_end:test_end\]\n\s+test_exits = self\.exits\.iloc\[train_end:test_end\]''',
r'''        windows = []
        for i in range(n_splits):
            start = i * window_size
            test_end = min((i + 1) * window_size, n)

            test_close = self.close.iloc[start:test_end]
            test_entries = self.entries.iloc[start:test_end]
            test_exits = self.exits.iloc[start:test_end]''', text)

text = re.sub(r'''                    'train_period': f"\{self\.close\.index\[start\]\.strftime\('%Y-%m-%d'\)\} → "\n\s+f"\{self\.close\.index\[min\(train_end - 1, n - 1\)\]\.strftime\('%Y-%m-%d'\)\}",\n\s+'test_period':''', r'''                    'test_period':''', text)

text = re.sub(r'''            print\(f"  🔄 Walk-Forward Analysis \(\{len\(windows\)\} windows, "\n\s+f"\{int\(train_ratio \* 100\)\}% train / \{int\(\(1 - train_ratio\) \* 100\)\}% test\)"\)\n\s+print\("=" \* 60\)''',
r'''            print(f"  🔄 Sub-period Analysis ({len(windows)} independent windows)")
            print("=" * 60)''', text)

with open('utils/backtesting.py', 'w') as f:
    f.write(text)
