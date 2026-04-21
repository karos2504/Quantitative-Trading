"""Tests for the Signal Engine — signal classification and persistence filter."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import pytest

from macro_rotation.config import SignalState, SignalTier, SIGNAL_TO_TIER
from macro_rotation.signal_engine import (
    classify_signal, get_signal_tier, PersistenceState,
    precompute_all_signals, classify_all_signals_at_date,
    _rsi, _macd, _ema, _sma,
)


# ============================================================================
# SIGNAL CLASSIFICATION TESTS
# ============================================================================
class TestClassifySignal:
    """Test the pure signal classification function."""

    def test_strong_bullish(self):
        """Aggregate > 0.5 with positive slope → BULLISH_UP."""
        state = classify_signal(0.7, 0.05)
        assert state == SignalState.BULLISH_UP

    def test_bullish_flat(self):
        """Aggregate > 0.5 with flat slope → BULLISH."""
        state = classify_signal(0.6, 0.01)
        assert state == SignalState.BULLISH

    def test_bullish_weakening(self):
        """Aggregate > 0.5 with negative slope → BULLISH_DOWN."""
        state = classify_signal(0.6, -0.05)
        assert state == SignalState.BULLISH_DOWN

    def test_neutral(self):
        """Aggregate between -0.2 and 0.5 → NEUTRAL tier."""
        state = classify_signal(0.1, 0.0)
        assert state == SignalState.NEUTRAL

    def test_neutral_improving(self):
        state = classify_signal(0.2, 0.05)
        assert state == SignalState.NEUTRAL_UP

    def test_neutral_weakening(self):
        state = classify_signal(0.0, -0.05)
        assert state == SignalState.NEUTRAL_DOWN

    def test_bearish(self):
        """Aggregate <= -0.2 → BEARISH tier."""
        state = classify_signal(-0.5, 0.0)
        assert state == SignalState.BEARISH

    def test_bearish_recovering(self):
        state = classify_signal(-0.3, 0.05)
        assert state == SignalState.BEARISH_UP

    def test_bearish_worsening(self):
        state = classify_signal(-0.8, -0.05)
        assert state == SignalState.BEARISH_DOWN

    def test_nan_returns_neutral(self):
        """NaN inputs should return NEUTRAL (safe fallback)."""
        assert classify_signal(np.nan, 0.0) == SignalState.NEUTRAL
        assert classify_signal(0.5, np.nan) == SignalState.NEUTRAL

    def test_boundary_bullish(self):
        """Exactly 0.5 should be Neutral (> 0.5 needed for Bullish)."""
        state = classify_signal(0.5, 0.0)
        assert get_signal_tier(state) == SignalTier.NEUTRAL

    def test_boundary_bearish(self):
        """Exactly -0.2 should be Bearish (<= -0.2)."""
        state = classify_signal(-0.2, 0.0)
        assert get_signal_tier(state) == SignalTier.BEARISH


# ============================================================================
# TIER MAPPING TESTS
# ============================================================================
class TestSignalTierMapping:
    """Verify all 9 states map correctly to 3 tiers."""

    def test_all_states_have_tiers(self):
        for state in SignalState:
            assert state in SIGNAL_TO_TIER, f"{state} missing from SIGNAL_TO_TIER"

    def test_bullish_states_map_to_bullish(self):
        for state in [SignalState.BULLISH_UP, SignalState.BULLISH, SignalState.BULLISH_DOWN]:
            assert SIGNAL_TO_TIER[state] == SignalTier.BULLISH

    def test_neutral_states_map_to_neutral(self):
        for state in [SignalState.NEUTRAL_UP, SignalState.NEUTRAL, SignalState.NEUTRAL_DOWN]:
            assert SIGNAL_TO_TIER[state] == SignalTier.NEUTRAL

    def test_bearish_states_map_to_bearish(self):
        for state in [SignalState.BEARISH_UP, SignalState.BEARISH, SignalState.BEARISH_DOWN]:
            assert SIGNAL_TO_TIER[state] == SignalTier.BEARISH


# ============================================================================
# PERSISTENCE FILTER TESTS
# ============================================================================
class TestPersistenceState:
    """Test the candidate-tracking persistence filter (fixes off-by-one bug)."""

    def test_initial_state(self):
        ps = PersistenceState(SignalTier.NEUTRAL, SignalTier.NEUTRAL, 0)
        assert ps.confirmed == SignalTier.NEUTRAL

    def test_same_signal_resets_candidate(self):
        """If raw == confirmed, candidate is reset."""
        ps = PersistenceState(SignalTier.NEUTRAL, SignalTier.BULLISH, 2)
        ps = ps.update(SignalTier.NEUTRAL, min_bars=3)
        assert ps.confirmed == SignalTier.NEUTRAL
        assert ps.candidate_bars == 0

    def test_new_candidate_starts_at_1(self):
        """New candidate starts with count=1."""
        ps = PersistenceState(SignalTier.NEUTRAL, SignalTier.NEUTRAL, 0)
        ps = ps.update(SignalTier.BULLISH, min_bars=3)
        assert ps.confirmed == SignalTier.NEUTRAL
        assert ps.candidate == SignalTier.BULLISH
        assert ps.candidate_bars == 1

    def test_candidate_increments(self):
        """Same candidate increments counter."""
        ps = PersistenceState(SignalTier.NEUTRAL, SignalTier.BULLISH, 1)
        ps = ps.update(SignalTier.BULLISH, min_bars=3)
        assert ps.confirmed == SignalTier.NEUTRAL
        assert ps.candidate == SignalTier.BULLISH
        assert ps.candidate_bars == 2

    def test_candidate_confirms_at_min_bars(self):
        """Candidate confirms after min_bars consecutive observations."""
        ps = PersistenceState(SignalTier.NEUTRAL, SignalTier.BULLISH, 2)
        ps = ps.update(SignalTier.BULLISH, min_bars=3)
        assert ps.confirmed == SignalTier.BULLISH
        assert ps.candidate_bars == 0

    def test_different_candidate_resets(self):
        """If raw != candidate AND raw != confirmed → new candidate."""
        ps = PersistenceState(SignalTier.NEUTRAL, SignalTier.BULLISH, 2)
        ps = ps.update(SignalTier.BEARISH, min_bars=3)
        assert ps.confirmed == SignalTier.NEUTRAL  # Unchanged
        assert ps.candidate == SignalTier.BEARISH   # New candidate
        assert ps.candidate_bars == 1

    def test_noisy_oscillation_does_not_confirm(self):
        """A→B→A→B should NOT confirm B — counter resets each time."""
        ps = PersistenceState(SignalTier.NEUTRAL, SignalTier.NEUTRAL, 0)

        # Bar 1: B appears
        ps = ps.update(SignalTier.BULLISH, min_bars=3)
        assert ps.confirmed == SignalTier.NEUTRAL

        # Bar 2: A reappears — resets
        ps = ps.update(SignalTier.NEUTRAL, min_bars=3)
        assert ps.confirmed == SignalTier.NEUTRAL
        assert ps.candidate_bars == 0

        # Bar 3: B again
        ps = ps.update(SignalTier.BULLISH, min_bars=3)
        assert ps.confirmed == SignalTier.NEUTRAL
        assert ps.candidate_bars == 1

        # Bar 4: A again
        ps = ps.update(SignalTier.NEUTRAL, min_bars=3)
        assert ps.confirmed == SignalTier.NEUTRAL

    def test_clean_transition_3_bars(self):
        """Three consecutive bars of new state → confirmed transition."""
        ps = PersistenceState(SignalTier.NEUTRAL, SignalTier.NEUTRAL, 0)

        ps = ps.update(SignalTier.BULLISH, min_bars=3)
        assert ps.confirmed == SignalTier.NEUTRAL

        ps = ps.update(SignalTier.BULLISH, min_bars=3)
        assert ps.confirmed == SignalTier.NEUTRAL

        ps = ps.update(SignalTier.BULLISH, min_bars=3)
        assert ps.confirmed == SignalTier.BULLISH  # Confirmed!


# ============================================================================
# VECTORIZED INDICATOR TESTS
# ============================================================================
class TestVectorizedIndicators:
    """Test that vectorized indicator computations are correct."""

    @pytest.fixture
    def synthetic_prices(self):
        """Create synthetic price series with known characteristics."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=500, freq="D")
        # Trending up then down
        trend = np.concatenate([
            np.linspace(100, 200, 250),
            np.linspace(200, 120, 250),
        ])
        noise = np.random.normal(0, 2, 500)
        prices = pd.Series(trend + noise, index=dates, name="TEST")
        return prices

    def test_rsi_range(self, synthetic_prices):
        """RSI should be between 0 and 100."""
        rsi = _rsi(synthetic_prices)
        valid = rsi.dropna()
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_rsi_high_in_uptrend(self, synthetic_prices):
        """RSI should be high during uptrend."""
        rsi = _rsi(synthetic_prices)
        # Middle of uptrend (day 100-200)
        uptrend_rsi = rsi.iloc[100:200].mean()
        assert uptrend_rsi > 50

    def test_macd_structure(self, synthetic_prices):
        """MACD histogram should be positive in uptrend, negative in downtrend."""
        _, _, hist = _macd(synthetic_prices)
        # During strong uptrend (after warm-up)
        uptrend_hist = hist.iloc[50:200].dropna().mean()
        assert uptrend_hist > 0

    def test_ema_follows_price(self, synthetic_prices):
        """EMA should lag but follow the price."""
        ema20 = _ema(synthetic_prices, 20)
        # Correlation should be very high
        common = pd.concat([synthetic_prices, ema20], axis=1).dropna()
        corr = common.corr().iloc[0, 1]
        assert corr > 0.95

    def test_precompute_returns_all_assets(self, synthetic_prices):
        """precompute_all_signals should return indicators for each asset."""
        prices_df = pd.DataFrame({
            "ASSET_A": synthetic_prices.values,
            "ASSET_B": synthetic_prices.values * 1.5,
        }, index=synthetic_prices.index)

        result = precompute_all_signals(prices_df)
        assert "ASSET_A" in result
        assert "ASSET_B" in result
        assert "aggregate_score" in result["ASSET_A"].columns
        assert "aggregate_slope" in result["ASSET_A"].columns
