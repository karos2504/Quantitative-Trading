"""Tests for drift_weights — highest execution frequency function in the system."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pytest

from macro_rotation.allocation import drift_weights


class TestDriftWeightsEdgeCases:
    """
    Comprehensive tests for drift_weights.  This function runs on EVERY day
    of the backtest (not just rebalance events), so errors compound fast.
    """

    def test_identity_with_zero_returns(self):
        """Zero returns and zero cash rate → weights unchanged."""
        w = {"A": 0.3, "B": 0.5}
        cash = 0.2
        ret = {"A": 0.0, "B": 0.0}
        new_w, new_c = drift_weights(w, ret, cash, 0.0)
        assert abs(new_w["A"] - 0.3) < 1e-12
        assert abs(new_w["B"] - 0.5) < 1e-12
        assert abs(new_c - 0.2) < 1e-12

    def test_total_always_one(self):
        """Invariant: drifted weights + cash must sum to exactly 1.0."""
        rng = np.random.RandomState(42)
        for _ in range(100):
            n_assets = rng.randint(1, 6)
            raw = rng.dirichlet(np.ones(n_assets + 1))  # Includes cash
            w = {f"A{i}": raw[i] for i in range(n_assets)}
            cash = raw[-1]
            ret = {f"A{i}": rng.normal(0, 0.05) for i in range(n_assets)}
            rate = rng.uniform(0, 0.0003)

            new_w, new_c = drift_weights(w, ret, cash, rate)
            total = sum(new_w.values()) + new_c
            assert abs(total - 1.0) < 1e-10, f"Total = {total}"

    def test_positive_return_increases_relative_weight(self):
        """Asset with positive return gets a larger share."""
        w = {"X": 0.5, "Y": 0.5}
        ret = {"X": 0.10, "Y": 0.0}
        new_w, _ = drift_weights(w, ret, 0.0, 0.0)
        assert new_w["X"] > new_w["Y"]

    def test_negative_return_decreases_relative_weight(self):
        """Asset with negative return gets a smaller share."""
        w = {"X": 0.5, "Y": 0.5}
        ret = {"X": -0.10, "Y": 0.0}
        new_w, _ = drift_weights(w, ret, 0.0, 0.0)
        assert new_w["X"] < new_w["Y"]

    def test_cash_grows_with_yield(self):
        """With only cash and a positive rate, cash should grow relative to assets."""
        w = {"X": 0.5}
        cash = 0.5
        ret = {"X": 0.0}
        rate = 0.001  # 0.1% daily

        new_w, new_c = drift_weights(w, ret, cash, rate)
        # Cash grew: new_c > 0.5 (slightly)
        assert new_c > cash

    def test_multi_day_compound_analytical(self):
        """Verify N-day compounding matches analytical solution."""
        w = {"A": 0.7}
        cash = 0.3
        r_a = 0.02      # 2% daily
        r_cash = 0.0002  # Cash rate

        for _ in range(20):
            w, cash = drift_weights(w, {"A": r_a}, cash, r_cash)

        # Analytical
        a_val = 0.7 * (1.02 ** 20)
        c_val = 0.3 * (1.0002 ** 20)
        total = a_val + c_val
        expected_a = a_val / total
        expected_c = c_val / total

        assert abs(w["A"] - expected_a) < 1e-8
        assert abs(cash - expected_c) < 1e-8

    def test_asset_drops_100_pct(self):
        """If an asset goes to zero, its weight becomes 0."""
        w = {"A": 0.5, "B": 0.5}
        ret = {"A": -1.0, "B": 0.0}
        new_w, new_c = drift_weights(w, ret, 0.0, 0.0)
        assert new_w["A"] == 0.0
        assert new_w["B"] == 1.0
        assert new_c == 0.0

    def test_all_assets_drop_100_pct_with_cash(self):
        """If all assets go to zero but cash exists, cash absorbs everything."""
        w = {"A": 0.4, "B": 0.4}
        cash = 0.2
        ret = {"A": -1.0, "B": -1.0}
        new_w, new_c = drift_weights(w, ret, cash, 0.0)
        # A and B are worth 0, cash is 0.2
        total = 0 + 0 + 0.2
        assert abs(new_c - 1.0) < 1e-9  # Cash becomes 100%

    def test_missing_return_treated_as_zero(self):
        """Assets not in returns dict should be treated as 0% return."""
        w = {"A": 0.5, "B": 0.5}
        ret = {"A": 0.05}  # B missing
        new_w, _ = drift_weights(w, ret, 0.0, 0.0)
        # B had 0% return → its value unchanged
        # A grew → A weight increases
        assert new_w["A"] > new_w["B"]

    def test_weights_nonzero_remain_nonnegative(self):
        """Weights should never become negative after drift."""
        rng = np.random.RandomState(123)
        for _ in range(200):
            n = rng.randint(1, 5)
            raw = rng.dirichlet(np.ones(n + 1))
            w = {f"A{i}": raw[i] for i in range(n)}
            cash = raw[-1]
            ret = {f"A{i}": rng.uniform(-0.5, 0.5) for i in range(n)}

            new_w, new_c = drift_weights(w, ret, cash, 0.0001)
            for v in new_w.values():
                assert v >= -1e-12, f"Negative weight: {v}"
            assert new_c >= -1e-12, f"Negative cash: {new_c}"
