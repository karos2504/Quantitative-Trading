"""
Dynamic Allocation Logic (Module 4)
=====================================
Computes final position weights by applying signal multipliers and macro
suitability to portfolio target weights.  Handles marginal rebalancing
(trade deltas only) and fee computation on traded notional.

Formula:
    final_weight[asset] = target_weight × modifier_multiplier × macro_suitability

Unallocated weight rolls into cash (a deliberate money-market position,
not a residual).
"""

import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass

from macro_rotation.config import (
    SignalState, SignalTier, MacroRegime, AssetClass, RebalanceMode,
    SIGNAL_MULTIPLIERS, SIGNAL_TO_TIER, REGIME_SUITABILITY,
    ASSET_CLASSES, SystemConfig, CONFIG, logger,
)
from macro_rotation.portfolios import AbstractPortfolio


# ============================================================================
# WEIGHT COMPUTATION
# ============================================================================
def compute_final_weights(
    portfolio: AbstractPortfolio,
    signals: dict[str, SignalState],
    regime: MacroRegime,
    performance_30d: dict[str, float] | None = None,
    current_date: pd.Timestamp | None = None,
    volumes: pd.Series | None = None,
    stats: dict | None = None,
) -> dict[str, float]:
    """
    Compute the final allocation weights for a portfolio given current
    signals and macro regime.

    Steps:
        1. Portfolio selects active universe (momentum filtering, etc.)
        2. Portfolio provides base target weights
        3. Each weight is scaled by signal modifier × macro suitability
        4. Residual (1.0 - sum) becomes cash allocation

    Returns:
        dict mapping asset → weight (0.0 to 1.0).
        Does NOT include cash key — cash is the implicit residual.
        All returned weights are non-negative and sum to <= 1.0.
    """
    # 1. Universe selection (passes current_date for suspension filtering)
    active_universe = portfolio.select_universe(signals, regime, performance_30d, current_date, volumes)

    # 2. Base target weights
    target_weights = portfolio.get_target_weights(active_universe, signals, regime, stats=stats)

    # 3. Apply signal multiplier × macro suitability
    final_weights = {}
    regime_suitability = REGIME_SUITABILITY.get(regime, {})

    for asset, target_w in target_weights.items():
        if target_w <= 0:
            continue

        # Macro suitability gate (binary: 0 or 1)
        is_suitable = regime_suitability.get(asset, True)

        if not is_suitable:
            # Unsuitable macro → close position entirely
            final_weights[asset] = 0.0
            continue

        # Signal modifier
        signal = signals.get(asset, SignalState.NEUTRAL)
        asset_class = ASSET_CLASSES.get(asset, AssetClass.CRYPTO)
        multiplier = SIGNAL_MULTIPLIERS.get(
            (signal, asset_class), 0.5  # Default to neutral multiplier
        )

        final_weights[asset] = target_w * multiplier

    # Remove zero weights
    final_weights = {a: w for a, w in final_weights.items() if w > 1e-6}

    return final_weights


def get_cash_weight(risk_weights: dict[str, float]) -> float:
    """
    Cash allocation = 1.0 - sum(risk asset weights).
    Cash is a deliberate money-market position, not a residual.
    """
    total_risk = sum(risk_weights.values())
    return max(0.0, 1.0 - total_risk)


# ============================================================================
# MARGINAL REBALANCE — DELTA COMPUTATION
# ============================================================================
@dataclass
class TradeOrder:
    """Representation of a single rebalance trade."""
    asset: str
    delta_weight: float    # Positive = buy, negative = sell
    fee_rate: float        # Fee rate applied to |delta| × portfolio_value
    fee_usd: float = 0.0  # Computed fee in USD

    @property
    def is_buy(self) -> bool:
        return self.delta_weight > 0

    @property
    def action(self) -> str:
        if self.delta_weight > 0.01:
            return "BUY"
        elif self.delta_weight < -0.01:
            return "SELL"
        return "HOLD"


def compute_trade_deltas(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    mode: RebalanceMode = RebalanceMode.MARGINAL,
    portfolio_value: float = 100_000,
    config: SystemConfig = CONFIG,
) -> list[TradeOrder]:
    """
    Compute the trades needed to move from current → target weights.

    MARGINAL mode: trade only the weight delta (default, more realistic).
    FULL mode: sell everything, rebuy at target (higher friction).

    Fees are computed on TRADED NOTIONAL (|delta| × portfolio_value),
    NOT on total position value.
    """
    all_assets = set(current_weights.keys()) | set(target_weights.keys())
    orders = []

    for asset in all_assets:
        current_w = current_weights.get(asset, 0.0)
        target_w = target_weights.get(asset, 0.0)

        if mode == RebalanceMode.MARGINAL:
            delta = target_w - current_w
        else:
            # FULL mode: conceptually sell current, buy target
            # WARNING: This overstates friction significantly
            warnings.warn(
                "FULL rebalance mode computes fees on total position value, "
                "not traded notional. Results will not reflect realistic costs. "
                "Use MARGINAL mode (default) for production backtests.",
                UserWarning,
                stacklevel=2,
            )
            delta = target_w  # Simplified: the full target is the "buy"

        if abs(delta) < 1e-6:
            continue

        # Determine fee rate based on asset and direction
        asset_class = ASSET_CLASSES.get(asset, AssetClass.CRYPTO)
        if asset_class == AssetClass.EQUITY:
            fee_rate = config.vnindex_buy_fee if delta > 0 else config.vnindex_sell_fee
        elif asset_class == AssetClass.GOLD:
            fee_rate = config.gold_fee
        else:
            fee_rate = config.crypto_fee

        # Fee on traded notional
        fee_usd = abs(delta) * portfolio_value * fee_rate
        
        # --- Cost-Benefit Gate ---
        # Skip trade if it's too small (unless it's a full close of a position)
        if abs(delta) < 0.01 and target_w > 0:
            continue

        orders.append(TradeOrder(
            asset=asset,
            delta_weight=delta,
            fee_rate=fee_rate,
            fee_usd=fee_usd,
        ))
        
    # Rebalance Gate: If total turnover is less than threshold, skip all trades
    total_delta = sum(abs(o.delta_weight) for o in orders)
    if total_delta < config.min_weight_delta_to_rebalance:
        return []

    return orders


def compute_total_fees(orders: list[TradeOrder]) -> float:
    """Total fees from all trade orders in USD."""
    return sum(o.fee_usd for o in orders)


def compute_turnover(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
) -> float:
    """
    One-way portfolio turnover: sum(|delta_weight_i|) / 2.
    Standard portfolio convention.
    """
    all_assets = set(current_weights.keys()) | set(target_weights.keys())
    total_delta = sum(
        abs(target_weights.get(a, 0.0) - current_weights.get(a, 0.0))
        for a in all_assets
    )
    return total_delta / 2.0


# ============================================================================
# WEIGHT DRIFT (runs every day, not just rebalance events)
# ============================================================================
def drift_weights(
    weights: dict[str, float],
    returns: dict[str, float],
    cash_weight: float,
    daily_cash_rate: float,
) -> tuple[dict[str, float], float]:
    """
    Update weights to reflect price drift without trading.

    After a day of returns, each asset's value changes proportionally.
    Cash accrues at the daily rate.  The total portfolio value changes,
    so weights are renormalized to sum to 1.0 (including cash).

    This prevents cash_weight from going negative after a strong up day.

    Args:
        weights: Current risk-asset weights (sum <= 1.0).
        returns: Single-day returns per asset {'BTC': 0.02, ...}.
        cash_weight: Current cash allocation (1.0 - sum(weights)).
        daily_cash_rate: Daily compound rate from APY.

    Returns:
        (new_risk_weights, new_cash_weight) — both normalized so total = 1.0.
    """
    # New asset values (proportional)
    new_values = {
        asset: w * (1 + returns.get(asset, 0.0))
        for asset, w in weights.items()
    }

    # Cash value after daily yield
    new_cash_value = cash_weight * (1 + daily_cash_rate)

    # Total portfolio value (proportional)
    total = sum(new_values.values()) + new_cash_value

    if total <= 0:
        # Catastrophic scenario — all assets went to zero
        return {}, 1.0

    # Renormalize
    drifted_weights = {asset: v / total for asset, v in new_values.items()}
    drifted_cash = new_cash_value / total

    return drifted_weights, drifted_cash


# ============================================================================
# REBALANCE TRIGGER CHECK
# ============================================================================
def should_rebalance(
    current_tiers: dict[str, SignalTier],
    new_tiers: dict[str, SignalTier],
    current_regime: MacroRegime,
    new_regime: MacroRegime,
) -> tuple[bool, str]:
    """
    Check if a rebalance should be triggered.

    Triggers ONLY on:
        1. Weight tier change for any held asset (Bullish↔Neutral↔Bearish)
        2. Macro regime change

    Does NOT trigger on ↑/↓ modifier changes within a tier.

    Returns:
        (should_rebalance: bool, reason: str)
    """
    # Check regime change
    if current_regime != new_regime:
        return True, f"Regime: {current_regime.value} → {new_regime.value}"

    # Check tier changes
    all_assets = set(current_tiers.keys()) | set(new_tiers.keys())
    changed_assets = []
    for asset in all_assets:
        old_tier = current_tiers.get(asset, SignalTier.NEUTRAL)
        new_tier = new_tiers.get(asset, SignalTier.NEUTRAL)
        if old_tier != new_tier:
            changed_assets.append(f"{asset}: {old_tier.value}→{new_tier.value}")

    if changed_assets:
        return True, f"Tier change: {', '.join(changed_assets)}"

    return False, ""
