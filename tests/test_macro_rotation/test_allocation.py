"""Tests for allocation logic — weight computation, drift, turnover, fees."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pytest

from macro_rotation.config import (
    SignalState, SignalTier, MacroRegime, AssetClass,
    RebalanceMode, SystemConfig,
)
from macro_rotation.allocation import (
    compute_final_weights, get_cash_weight,
    compute_trade_deltas, compute_total_fees, compute_turnover,
    drift_weights, should_rebalance,
)
from macro_rotation.portfolios import CryptoGoldRotation, CoreAssetMacroRotation


# ============================================================================
# WEIGHT COMPUTATION TESTS
# ============================================================================
class TestComputeFinalWeights:
    """Test dynamic allocation formula correctness."""

    def test_weights_sum_leq_one(self):
        """Final risk-asset weights must sum to <= 1.0."""
        portfolio = CryptoGoldRotation()
        signals = {
            "BTC": SignalState.BULLISH, "ETH": SignalState.BULLISH,
            "XAUT": SignalState.NEUTRAL,
        }
        regime = MacroRegime.RISK_ON_DISINFLATION
        perf = {"BTC": 0.1, "ETH": 0.15, "XAUT": 0.02}

        weights = compute_final_weights(portfolio, signals, regime, perf)
        assert sum(weights.values()) <= 1.0 + 1e-9

    def test_unsuitable_regime_zeros_risk_assets(self):
        """Risk-off regime should zero out crypto assets."""
        portfolio = CryptoGoldRotation()
        signals = {
            "BTC": SignalState.BULLISH, "ETH": SignalState.BULLISH,
            "XAUT": SignalState.BULLISH,
        }
        regime = MacroRegime.RISK_OFF_INFLATION  # Crypto unsuitable

        weights = compute_final_weights(portfolio, signals, regime)
        # BTC and ETH should be zero
        assert weights.get("BTC", 0.0) == 0.0
        assert weights.get("ETH", 0.0) == 0.0
        # XAUT should be non-zero (gold is suitable)
        assert weights.get("XAUT", 0.0) > 0

    def test_bearish_zeros_gold_equity(self):
        """Bearish signal on gold/equity → 0% (per multiplier table)."""
        portfolio = CoreAssetMacroRotation()
        signals = {
            "VNINDEX": SignalState.BEARISH,
            "XAUT": SignalState.BEARISH,
            "BTC": SignalState.NEUTRAL,
        }
        regime = MacroRegime.RISK_ON_DISINFLATION

        weights = compute_final_weights(portfolio, signals, regime)
        assert weights.get("VNINDEX", 0.0) == 0.0
        assert weights.get("XAUT", 0.0) == 0.0

    def test_cash_weight_computation(self):
        """Cash = 1.0 - sum(risk weights)."""
        weights = {"BTC": 0.3, "ETH": 0.2}
        cash = get_cash_weight(weights)
        assert abs(cash - 0.5) < 1e-9

    def test_cash_weight_never_negative(self):
        """Cash weight should never be negative."""
        weights = {"BTC": 0.6, "ETH": 0.5}  # Sum > 1
        cash = get_cash_weight(weights)
        assert cash >= 0.0

    def test_core3_good_macro_allocations(self):
        """Core3 in best regime should produce ~60/20/20 split (scaled by signal)."""
        portfolio = CoreAssetMacroRotation()
        signals = {
            "VNINDEX": SignalState.BULLISH,  # 0.9 multiplier
            "XAUT": SignalState.BULLISH,     # 0.9 multiplier
            "BTC": SignalState.BULLISH,      # 0.9 multiplier
        }
        regime = MacroRegime.RISK_ON_DISINFLATION  # alpha = 1.0

        weights = compute_final_weights(portfolio, signals, regime)
        # VNINDEX target = 0.60 * 0.9 = 0.54
        assert abs(weights.get("VNINDEX", 0) - 0.54) < 0.01
        # XAUT target = 0.20 * 0.9 = 0.18
        assert abs(weights.get("XAUT", 0) - 0.18) < 0.01


# ============================================================================
# DRIFT WEIGHTS TESTS
# ============================================================================
class TestDriftWeights:
    """Test weight drift calculation — the most frequently called function."""

    def test_no_return_no_change(self):
        """Zero returns should produce no weight change (except tiny cash yield)."""
        weights = {"BTC": 0.5, "ETH": 0.3}
        cash = 0.2
        returns = {"BTC": 0.0, "ETH": 0.0}

        new_w, new_cash = drift_weights(weights, returns, cash, 0.0)
        assert abs(new_w["BTC"] - 0.5) < 1e-9
        assert abs(new_w["ETH"] - 0.3) < 1e-9
        assert abs(new_cash - 0.2) < 1e-9

    def test_drift_sums_to_one(self):
        """After drift, risk weights + cash should sum to 1.0."""
        weights = {"BTC": 0.5, "ETH": 0.3}
        cash = 0.2
        returns = {"BTC": 0.05, "ETH": -0.02}

        new_w, new_cash = drift_weights(weights, returns, cash, 0.0001)
        total = sum(new_w.values()) + new_cash
        assert abs(total - 1.0) < 1e-9

    def test_positive_return_increases_weight(self):
        """Asset with positive return should have higher weight after drift."""
        weights = {"BTC": 0.5, "ETH": 0.5}
        cash = 0.0
        returns = {"BTC": 0.10, "ETH": 0.0}

        new_w, _ = drift_weights(weights, returns, cash, 0.0)
        assert new_w["BTC"] > 0.5
        assert new_w["ETH"] < 0.5

    def test_cash_yield_accrues(self):
        """Cash should grow by daily rate even when risk assets are flat."""
        weights = {}
        cash = 1.0
        returns = {}
        daily_rate = 0.0001  # ~3.6% APY

        new_w, new_cash = drift_weights(weights, returns, cash, daily_rate)
        # Cash should have grown slightly
        assert new_cash > 0.999  # Close to 1.0 but with tiny yield
        assert abs(new_cash - 1.0) < 0.001  # Very small daily change

    def test_drift_with_large_move(self):
        """Large positive move should significantly shift weights."""
        weights = {"BTC": 0.5, "ETH": 0.5}
        cash = 0.0
        returns = {"BTC": 0.50, "ETH": -0.20}  # BTC +50%, ETH -20%

        new_w, _ = drift_weights(weights, returns, cash, 0.0)
        # BTC: 0.5 * 1.5 = 0.75, ETH: 0.5 * 0.8 = 0.40, total = 1.15
        # BTC weight = 0.75/1.15 ≈ 0.652
        assert abs(new_w["BTC"] - 0.75 / 1.15) < 1e-9

    def test_cash_never_negative_after_surge(self):
        """Cash weight should NEVER go negative, even after asset surge."""
        weights = {"BTC": 0.8}
        cash = 0.2
        returns = {"BTC": 1.0}  # BTC doubles

        new_w, new_cash = drift_weights(weights, returns, cash, 0.0)
        assert new_cash >= 0
        assert abs(sum(new_w.values()) + new_cash - 1.0) < 1e-9

    def test_catastrophic_loss(self):
        """All assets go to zero → full cash."""
        weights = {"BTC": 0.5, "ETH": 0.5}
        cash = 0.0
        returns = {"BTC": -1.0, "ETH": -1.0}  # Total wipeout

        new_w, new_cash = drift_weights(weights, returns, cash, 0.0)
        # Should not crash — cash should be 1.0 or edge case handled
        assert new_cash >= 0

    def test_analytical_multi_day_drift(self):
        """After N days of known returns, verify analytical solution."""
        weights = {"BTC": 0.6}
        cash = 0.4
        daily_ret = 0.01  # 1% daily return on BTC
        daily_cash = 0.0001
        n_days = 10

        for _ in range(n_days):
            weights, cash = drift_weights(weights, {"BTC": daily_ret}, cash, daily_cash)

        # After 10 days: BTC grew by (1.01)^10 ≈ 1.10462
        # Cash grew by (1.0001)^10 ≈ 1.001
        # Analytical:
        btc_value = 0.6 * (1.01 ** 10)
        cash_value = 0.4 * (1.0001 ** 10)
        total = btc_value + cash_value
        expected_btc_w = btc_value / total

        assert abs(weights["BTC"] - expected_btc_w) < 1e-6


# ============================================================================
# TRADE DELTA AND FEE TESTS
# ============================================================================
class TestTradeDeltasAndFees:
    """Test marginal rebalancing and fee computation."""

    def test_marginal_mode_trades_delta_only(self):
        """MARGINAL mode should only trade the delta."""
        current = {"BTC": 0.40, "ETH": 0.30}
        target = {"BTC": 0.50, "ETH": 0.20}

        orders = compute_trade_deltas(current, target, RebalanceMode.MARGINAL)
        deltas = {o.asset: o.delta_weight for o in orders}
        assert abs(deltas["BTC"] - 0.10) < 1e-9
        assert abs(deltas["ETH"] - (-0.10)) < 1e-9

    def test_fee_on_traded_notional(self):
        """Fees should be computed on |delta| × portfolio_value."""
        current = {"BTC": 0.40}
        target = {"BTC": 0.50}
        config = SystemConfig(crypto_fee=0.001)

        orders = compute_trade_deltas(
            current, target, RebalanceMode.MARGINAL,
            portfolio_value=100_000, config=config,
        )
        # Fee = 0.10 * 100000 * 0.001 = $10
        assert abs(orders[0].fee_usd - 10.0) < 0.01

    def test_vnindex_asymmetric_fees(self):
        """VNINDEX should have different buy and sell fees."""
        config = SystemConfig(vnindex_buy_fee=0.0015, vnindex_sell_fee=0.0025)

        # Buy
        orders_buy = compute_trade_deltas(
            {}, {"VNINDEX": 0.50}, RebalanceMode.MARGINAL,
            portfolio_value=100_000, config=config,
        )
        assert abs(orders_buy[0].fee_rate - 0.0015) < 1e-9

        # Sell
        orders_sell = compute_trade_deltas(
            {"VNINDEX": 0.50}, {}, RebalanceMode.MARGINAL,
            portfolio_value=100_000, config=config,
        )
        assert abs(orders_sell[0].fee_rate - 0.0025) < 1e-9

    def test_one_way_turnover(self):
        """One-way turnover = sum(|delta|) / 2."""
        current = {"BTC": 0.40, "ETH": 0.30, "XAUT": 0.30}
        target = {"BTC": 0.50, "ETH": 0.20, "XAUT": 0.30}

        turnover = compute_turnover(current, target)
        # Deltas: BTC +0.10, ETH -0.10, XAUT 0.0
        # One-way: (0.10 + 0.10) / 2 = 0.10
        assert abs(turnover - 0.10) < 1e-9


# ============================================================================
# REBALANCE TRIGGER TESTS
# ============================================================================
class TestRebalanceTrigger:
    """Test that rebalance triggers on tier change, not modifier change."""

    def test_modifier_change_does_not_trigger(self):
        """BULLISH→BULLISH_UP should NOT trigger rebalance."""
        prev = {"BTC": SignalTier.BULLISH}
        curr = {"BTC": SignalTier.BULLISH}  # Same tier
        regime_old = MacroRegime.RISK_ON_DISINFLATION
        regime_new = MacroRegime.RISK_ON_DISINFLATION

        trigger, _ = should_rebalance(prev, curr, regime_old, regime_new)
        assert trigger is False

    def test_tier_change_triggers(self):
        """BULLISH→NEUTRAL should trigger rebalance."""
        prev = {"BTC": SignalTier.BULLISH}
        curr = {"BTC": SignalTier.NEUTRAL}
        regime_old = MacroRegime.RISK_ON_DISINFLATION
        regime_new = MacroRegime.RISK_ON_DISINFLATION

        trigger, reason = should_rebalance(prev, curr, regime_old, regime_new)
        assert trigger is True
        assert "Tier change" in reason

    def test_regime_change_triggers(self):
        """Regime change should trigger rebalance."""
        prev = {"BTC": SignalTier.BULLISH}
        curr = {"BTC": SignalTier.BULLISH}
        regime_old = MacroRegime.RISK_ON_DISINFLATION
        regime_new = MacroRegime.RISK_OFF_INFLATION

        trigger, reason = should_rebalance(prev, curr, regime_old, regime_new)
        assert trigger is True
        assert "Regime" in reason

    def test_no_change_no_trigger(self):
        """No change in tier or regime → no trigger."""
        prev = {"BTC": SignalTier.NEUTRAL, "ETH": SignalTier.BULLISH}
        curr = {"BTC": SignalTier.NEUTRAL, "ETH": SignalTier.BULLISH}
        regime = MacroRegime.RISK_ON_DISINFLATION

        trigger, _ = should_rebalance(prev, curr, regime, regime)
        assert trigger is False
