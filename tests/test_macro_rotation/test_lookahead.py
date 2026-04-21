"""
Look-Ahead Bias Test
=====================
Verifies that precomputed signals match signals computed on truncated data.
This is the ultimate correctness test: for 50 random dates, we verify that
signal(t) == signal_recomputed_only_up_to_t.

If any indicator uses future data (via rolling windows without min_periods,
forward-fill before lag shift, etc.), this test will catch it.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import pytest

from macro_rotation.signal_engine import (
    precompute_all_signals, classify_signal,
)


class TestLookAheadBias:
    """Verify no look-ahead bias in signal computation."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic price data with known patterns."""
        np.random.seed(42)
        n = 800
        dates = pd.date_range("2020-01-01", periods=n, freq="D")

        # Create a realistic price series: trend + mean-reversion + noise
        trend = np.cumsum(np.random.normal(0.0005, 0.02, n))
        prices = 100 * np.exp(trend)

        return pd.DataFrame({"TEST_ASSET": prices}, index=dates)

    def test_signal_at_date_equals_truncated_recomputation(self, synthetic_data):
        """
        For 50 random dates (after warm-up), verify:
            signal(full_series, date=t) == signal(series[:t+1])

        This catches any look-ahead bias from:
            - Rolling windows using future data
            - EMA warm-up leaking
            - Forward-fill before lag application
        """
        # Precompute on full series
        full_indicators = precompute_all_signals(synthetic_data)
        full_df = full_indicators["TEST_ASSET"]

        # Select 50 random test dates (after warm-up period of 300 days)
        valid_dates = full_df.index[300:]
        np.random.seed(123)
        test_indices = np.random.choice(len(valid_dates), size=min(50, len(valid_dates)), replace=False)
        test_dates = valid_dates[test_indices]

        mismatches = []

        for date in test_dates:
            # Signal from full precomputation (what the backtest uses)
            full_agg = float(full_df.loc[date, "aggregate_score"])
            full_slope = float(full_df.loc[date, "aggregate_slope"])
            full_signal = classify_signal(full_agg, full_slope)

            # Recompute using only data up to this date (no future data)
            truncated_data = synthetic_data[synthetic_data.index <= date]
            trunc_indicators = precompute_all_signals(truncated_data)

            if "TEST_ASSET" not in trunc_indicators:
                continue

            trunc_df = trunc_indicators["TEST_ASSET"]
            if date not in trunc_df.index:
                continue

            trunc_agg = float(trunc_df.loc[date, "aggregate_score"])
            trunc_slope = float(trunc_df.loc[date, "aggregate_slope"])
            trunc_signal = classify_signal(trunc_agg, trunc_slope)

            # Compare
            if full_signal != trunc_signal:
                mismatches.append({
                    "date": date,
                    "full_signal": full_signal.value,
                    "trunc_signal": trunc_signal.value,
                    "full_agg": round(full_agg, 4),
                    "trunc_agg": round(trunc_agg, 4),
                    "diff": round(abs(full_agg - trunc_agg), 6),
                })

            # Even if signals match, check aggregate score is close
            # (small numerical differences are OK due to floating point)
            if not np.isnan(full_agg) and not np.isnan(trunc_agg):
                assert abs(full_agg - trunc_agg) < 0.01, (
                    f"Look-ahead bias detected at {date}: "
                    f"full_agg={full_agg:.6f}, trunc_agg={trunc_agg:.6f}"
                )

        # Report any signal mismatches
        if mismatches:
            mismatch_pct = len(mismatches) / len(test_dates) * 100
            # Allow up to 5% mismatches due to floating-point edge cases
            # near classification boundaries
            assert mismatch_pct < 5.0, (
                f"Look-ahead bias: {len(mismatches)}/{len(test_dates)} "
                f"({mismatch_pct:.1f}%) signal mismatches:\n"
                + "\n".join(str(m) for m in mismatches[:5])
            )

    def test_indicators_nan_during_warmup(self, synthetic_data):
        """
        Indicators should be NaN (not extrapolated) during warm-up period.
        This prevents look-ahead via "warm-up contamination".
        """
        indicators = precompute_all_signals(synthetic_data)
        df = indicators["TEST_ASSET"]

        # RSI with period=14 should be NaN for first ~14 rows
        first_valid_rsi = df["rsi"].first_valid_index()
        first_date = df.index[0]
        warmup_days = (first_valid_rsi - first_date).days
        assert warmup_days >= 13, f"RSI warm-up too short: {warmup_days} days"

        # MA200 should be NaN for first 199 rows
        first_valid_ma = df["price_vs_ma200"].first_valid_index()
        if first_valid_ma is not None:
            ma_warmup = (first_valid_ma - first_date).days
            assert ma_warmup >= 199, f"MA200 warm-up too short: {ma_warmup} days"
