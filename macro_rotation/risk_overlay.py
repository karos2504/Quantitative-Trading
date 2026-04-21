"""
Risk Management & Sentiment Overlays (Module 5)
=================================================
Counter-trades extreme sentiment and optimizes entry/exit timing.

Components:
    1. Bitcoin Risk Metric (0.0–1.0) — cycle position indicator
    2. Sentiment Regime Classifier (Apathy→FOMO→Panic→Despair) with persistence
    3. Live Buy/Sell Signals (aggregate extremes)
    4. Opportunity Zones (tactical DCA/profit adjustments)
    5. Asset Performance Tracker (30-day vs USD and vs BTC)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

from macro_rotation.config import (
    SentimentRegime, MacroRegime, SignalState, SignalTier,
    SIGNAL_TO_TIER, REGIME_SUITABILITY,
    SystemConfig, CONFIG, logger,
)
from macro_rotation.signal_engine import PersistenceState


# Type alias for sentiment-specific persistence state.
# Using the generic PersistenceState[SentimentRegime] prevents accidental
# mixing with PersistenceState[SignalTier] — the == comparisons would
# silently never match and the filter would never confirm transitions.
SentimentPersistenceState = PersistenceState[SentimentRegime]


def _make_default_sentiment_state() -> PersistenceState[SentimentRegime]:
    """Create a default sentiment persistence state (Apathy)."""
    return PersistenceState(SentimentRegime.APATHY, SentimentRegime.APATHY, 0)


# ============================================================================
# BITCOIN RISK METRIC (0.0 → 1.0)
# ============================================================================
def compute_btc_risk_metric(
    btc_prices: pd.Series,
    window: int = 730,
) -> pd.Series:
    """
    Bitcoin Risk Metric: composite score from 0.0 (max opportunity) to 1.0 (cycle peak).

    Components:
        1. Price vs 200-week (~1400 day) MA ratio — MVRV proxy
        2. Monthly RSI position within historical range
        3. Pi Cycle approximation: 111-day MA vs 350-day MA × 2

    Each component is normalized to [0, 1] over the trailing window,
    then averaged.

    Action zones:
        0.0–0.3: Strong accumulation (max opportunity)
        0.3–0.5: Main DCA zone
        0.5–0.6: Neutral
        0.6–0.8: Prepare to exit / DCE
        0.8–1.0: Cycle peak (aggressive risk reduction)
    """
    if btc_prices.empty or len(btc_prices) < 252:
        return pd.Series(0.5, index=btc_prices.index, name="btc_risk")

    close = btc_prices

    # Component 1: Price vs 200-week MA (MVRV proxy)
    ma_200w = close.rolling(window=1400, min_periods=252).mean()
    price_ratio = close / ma_200w
    # Normalize to [0, 1] via percentile rank over trailing window
    ratio_rank = price_ratio.rolling(window=window, min_periods=min(window, 252)).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    # Component 2: Monthly RSI (30-day RSI)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 30, min_periods=30).mean()
    avg_loss = loss.ewm(alpha=1 / 30, min_periods=30).mean()
    rs = avg_gain / avg_loss
    rsi_30 = 100 - (100 / (1 + rs))
    rsi_rank = rsi_30 / 100  # RSI already [0, 100] → normalize to [0, 1]

    # Component 3: Pi Cycle indicator
    # When 111-day MA crosses ABOVE 350-day MA × 2, historically signals cycle tops
    ma_111 = close.rolling(window=111, min_periods=111).mean()
    ma_350x2 = close.rolling(window=350, min_periods=350).mean() * 2
    pi_ratio = ma_111 / ma_350x2
    pi_rank = pi_ratio.rolling(window=window, min_periods=min(window, 252)).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    # Weighted average
    risk = (ratio_rank * 0.40 + rsi_rank * 0.30 + pi_rank * 0.30)
    risk = risk.clip(0.0, 1.0)
    risk.name = "btc_risk"

    return risk


def get_btc_risk_zone(risk: float) -> str:
    """Map BTC risk metric to an action zone description."""
    if risk <= 0.3:
        return "Strong Accumulation (0.0–0.3)"
    elif risk <= 0.5:
        return "DCA Zone (0.3–0.5)"
    elif risk <= 0.6:
        return "Neutral (0.5–0.6)"
    elif risk <= 0.8:
        return "DCE / Prepare Exit (0.6–0.8)"
    else:
        return "Cycle Peak (0.8–1.0)"


# ============================================================================
# SENTIMENT REGIME CLASSIFIER
# ============================================================================
def _classify_raw_sentiment(
    rsi: float,
    volatility_rank: float,
    drawdown_pct: float,
    btc_risk: float,
) -> SentimentRegime:
    """
    Raw sentiment classification from market indicators.
    No persistence filtering — persistence is applied by the caller.

    Logic:
        FOMO:    RSI > 75 AND btc_risk > 0.65 AND low vol
        PANIC:   Sharp drawdown (>15%) AND RSI < 35 AND high vol
        DESPAIR: Extended drawdown (>30%) AND RSI < 25
        APATHY:  Default / low activity / small moves
    """
    if drawdown_pct < -0.30 and rsi < 25:
        return SentimentRegime.DESPAIR
    elif drawdown_pct < -0.15 and rsi < 35 and volatility_rank > 0.7:
        return SentimentRegime.PANIC
    elif rsi > 75 and btc_risk > 0.65 and volatility_rank < 0.5:
        return SentimentRegime.FOMO
    else:
        return SentimentRegime.APATHY


def precompute_sentiment_series(
    btc_prices: pd.Series,
    btc_risk: pd.Series,
    config: SystemConfig = CONFIG,
) -> pd.DataFrame:
    """
    Pre-compute raw sentiment indicators for every date.
    Persistence filtering is applied in the backtest loop.
    """
    close = btc_prices
    if close.empty:
        return pd.DataFrame()

    # RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Trailing drawdown
    cummax = close.cummax()
    drawdown = (close - cummax) / cummax

    # Volatility rank (20-day rolling vol, percentile over 252 days)
    daily_ret = close.pct_change()
    vol_20 = daily_ret.rolling(20, min_periods=20).std()
    vol_rank = vol_20.rolling(252, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    return pd.DataFrame({
        "rsi": rsi,
        "drawdown": drawdown,
        "vol_rank": vol_rank,
        "btc_risk": btc_risk.reindex(close.index).ffill(),
    }, index=close.index)


# ============================================================================
# LIVE BUY/SELL SIGNALS
# ============================================================================
class SignalType:
    STRONG_BUY = "Strong Buy"
    RE_ENTRY = "Re-Entry"
    STRONG_SELL = "Strong Sell"
    NONE = "None"


def classify_live_signal(
    btc_risk: float,
    sentiment: SentimentRegime,
    regime: MacroRegime,
) -> str:
    """
    Aggregate derivative + macro data to identify extremes.

    Strong Buy:  BTC Risk < 0.3 + Despair + macro suitable for crypto
    Re-Entry:    BTC Risk < 0.5 + recovering from Panic
    Strong Sell: BTC Risk > 0.8 + FOMO
    """
    crypto_suitable = REGIME_SUITABILITY.get(regime, {}).get("BTC", True)

    if btc_risk < 0.3 and sentiment == SentimentRegime.DESPAIR and crypto_suitable:
        return SignalType.STRONG_BUY
    elif btc_risk < 0.5 and sentiment == SentimentRegime.PANIC:
        return SignalType.RE_ENTRY
    elif btc_risk > 0.8 and sentiment == SentimentRegime.FOMO:
        return SignalType.STRONG_SELL
    return SignalType.NONE


# ============================================================================
# OPPORTUNITY ZONES
# ============================================================================
@dataclass
class OpportunityZone:
    """Tactical adjustment to allocation weights."""
    zone: str               # "accumulation" or "risk_reduction"
    adjustment_pct: float   # e.g., 0.05 = increase DCA by 5% of cash

    @classmethod
    def from_risk(cls, btc_risk: float, sentiment: SentimentRegime) -> "OpportunityZone | None":
        """
        Determine if we're in an opportunity zone.

        Accumulation Zone: BTC Risk < 0.4 → increase DCA by 4–6% of cash
        Risk Reduction Zone: BTC Risk > 0.7 → take 1–2% profit into stablecoins
        """
        if btc_risk < 0.3:
            return cls("accumulation", 0.06)  # 6% of cash into risk
        elif btc_risk < 0.4:
            return cls("accumulation", 0.04)  # 4% of cash into risk
        elif btc_risk > 0.85:
            return cls("risk_reduction", 0.02)  # 2% profit take
        elif btc_risk > 0.7:
            return cls("risk_reduction", 0.01)  # 1% profit take
        return None


def apply_opportunity_zone(
    weights: dict[str, float],
    cash_weight: float,
    zone: OpportunityZone | None,
    anchor_asset: str = "BTC",
) -> tuple[dict[str, float], float]:
    """
    Apply tactical opportunity zone adjustments to weights.

    Accumulation: move cash → anchor asset
    Risk Reduction: move anchor asset → cash
    """
    if zone is None:
        return weights, cash_weight

    adjusted = dict(weights)
    adj_cash = cash_weight

    if zone.zone == "accumulation" and anchor_asset in adjusted:
        # Move % of cash into anchor
        move = min(adj_cash * zone.adjustment_pct, adj_cash * 0.5)
        adjusted[anchor_asset] = adjusted.get(anchor_asset, 0.0) + move
        adj_cash -= move
    elif zone.zone == "risk_reduction" and anchor_asset in adjusted:
        # Take profit from anchor into cash
        move = adjusted.get(anchor_asset, 0.0) * zone.adjustment_pct
        adjusted[anchor_asset] = max(0.0, adjusted[anchor_asset] - move)
        adj_cash += move

    return adjusted, adj_cash


# ============================================================================
# ASSET PERFORMANCE TRACKER
# ============================================================================
def rank_assets_by_performance(
    prices: pd.DataFrame,
    date: pd.Timestamp,
    window: int = 30,
) -> pd.DataFrame:
    """
    Rank crypto assets by 30-day performance vs USD and vs BTC.
    Assets negative vs BTC are flagged for dropping from rotation pool.

    Returns DataFrame with columns:
        asset, return_usd, return_vs_btc, drop_flag
    """
    records = []
    btc_col = "BTC"

    for asset in prices.columns:
        series = prices[asset].dropna()
        valid = series[series.index <= date]
        if len(valid) < window:
            continue

        ret_usd = float(valid.iloc[-1] / valid.iloc[-window] - 1)
        btc_series = prices[btc_col].dropna() if btc_col in prices.columns else None

        if btc_series is not None:
            btc_valid = btc_series[btc_series.index <= date]
            if len(btc_valid) >= window:
                btc_ret = float(btc_valid.iloc[-1] / btc_valid.iloc[-window] - 1)
                ret_vs_btc = ret_usd - btc_ret
            else:
                ret_vs_btc = 0.0
        else:
            ret_vs_btc = 0.0

        records.append({
            "asset": asset,
            "return_usd": round(ret_usd * 100, 2),
            "return_vs_btc": round(ret_vs_btc * 100, 2),
            "drop_flag": ret_vs_btc < 0,
        })

    df = pd.DataFrame(records).sort_values("return_usd", ascending=False)
    return df


# ============================================================================
# COMBINED RISK OVERLAY APPLICATION
# ============================================================================
def apply_risk_overlay(
    weights: dict[str, float],
    cash_weight: float,
    btc_risk: float,
    sentiment: SentimentRegime,
    regime: MacroRegime,
    anchor_asset: str = "BTC",
) -> tuple[dict[str, float], float, str]:
    """
    Apply all risk overlays to the portfolio weights.

    Returns:
        (adjusted_weights, adjusted_cash, live_signal_type)
    """
    # 1. Opportunity zone adjustment
    zone = OpportunityZone.from_risk(btc_risk, sentiment)
    adjusted, adj_cash = apply_opportunity_zone(weights, cash_weight, zone, anchor_asset)

    # 2. Live signal — informational only (logged, doesn't auto-trade)
    live_signal = classify_live_signal(btc_risk, sentiment, regime)

    # 3. Cycle peak override — graduated risk reduction starting at BTC Risk > 0.85
    if btc_risk > 0.85:
        # Linear scale: 1.0 at 0.85, 0.75 at 0.90, 0.50 at 0.95, 0.25 at 1.0
        multiplier = max(0.0, 5.25 - 5 * btc_risk)
        
        for asset in list(adjusted.keys()):
            if asset != "XAUT":  # Keep gold
                orig_w = adjusted[asset]
                new_w = orig_w * multiplier
                reduction = orig_w - new_w
                adjusted[asset] = new_w
                adj_cash += reduction

    return adjusted, adj_cash, live_signal
