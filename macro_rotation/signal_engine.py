"""
Quantitative Signal Engine (Module 2)
======================================
Vectorized indicator pre-computation for all assets, then per-date row lookup
for signal classification.  NO re-computation in the backtest loop.

Signal flow:
    1. precompute_all_signals(prices) → full indicator DataFrame (once)
    2. classify_signal(indicator_row)  → SignalState (pure function, no data access)
    3. get_signal_tier(state)          → SignalTier (for rebalance trigger check)

Look-ahead bias safeguards:
    - All rolling operations use min_periods=window
    - Higher-high/lower-low uses left-side pivots (confirmed N bars ago)
    - No future data in any EMA (adjust=False not needed; standard pandas EMA is causal)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import TypeVar, Generic

from macro_rotation.config import (
    SignalState, SignalTier, SIGNAL_TO_TIER, logger,
)


T = TypeVar('T')  # Generic over SignalTier or SentimentRegime


@dataclass
class PersistenceState(Generic[T]):
    """
    Tracks confirmed vs candidate state for persistence filtering.
    Prevents noisy oscillation — state change only confirms after N
    consecutive bars of the new classification.

    Generic over T to prevent accidental mixing of SignalTier with
    SentimentRegime — passing the wrong type will never match ==
    comparisons and the filter would silently never confirm.

    Usage:
        signal_state: PersistenceState[SignalTier] = PersistenceState(...)
        sentiment_state: PersistenceState[SentimentRegime] = PersistenceState(...)
    """
    confirmed: T
    candidate: T
    candidate_bars: int = 0

    def update(self, raw: T, min_bars: int) -> "PersistenceState[T]":
        """
        Process a new raw classification.  Returns a NEW PersistenceState.

        Rules:
            - If raw == confirmed → reset candidate (signal reverted)
            - If raw == candidate → increment counter, confirm if >= min_bars
            - If raw is new → start fresh candidate tracking
        """
        if raw == self.confirmed:
            # Signal reverted to confirmed state — reset candidate
            return PersistenceState(self.confirmed, raw, 0)
        elif raw == self.candidate:
            # Same candidate seen again
            new_count = self.candidate_bars + 1
            if new_count >= min_bars:
                # Confirmed transition
                return PersistenceState(raw, raw, 0)
            return PersistenceState(self.confirmed, raw, new_count)
        else:
            # New candidate state — start fresh count
            return PersistenceState(self.confirmed, raw, 1)


# ============================================================================
# VECTORIZED INDICATOR COMPUTATION
# ============================================================================
def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average — causal, no future data."""
    return series.ewm(span=span, min_periods=span).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI — fully vectorized."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line, histogram — vectorized."""
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index — vectorized Wilder's smoothing."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period).mean()

    up_move = high.diff()
    down_move = -low.diff()
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    smooth_pos = pd.Series(pos_dm, index=close.index).ewm(alpha=1 / period, min_periods=period).mean()
    smooth_neg = pd.Series(neg_dm, index=close.index).ewm(alpha=1 / period, min_periods=period).mean()

    pos_di = 100 * smooth_pos / atr
    neg_di = 100 * smooth_neg / atr
    dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di)
    adx = dx.ewm(alpha=1 / period, min_periods=period).mean()
    return adx


def _bollinger_width(close: pd.Series, window: int = 20, std_dev: float = 2.0) -> pd.Series:
    """Bollinger Band width as a percentage of middle band."""
    sma = _sma(close, window)
    std = close.rolling(window=window, min_periods=window).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    width = (upper - lower) / sma
    return width

def _pct_rank(series: pd.Series, window: int) -> pd.Series:
    """Rolling percentile rank over a trailing window mapping to [0, 1]."""
    return series.rolling(window, min_periods=max(window//2, 30)).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

def _left_pivot_highs(high: pd.Series, lookback: int = 5) -> pd.Series:
    """
    Left-side pivot highs: a bar is a pivot high if it's the max of the
    preceding `lookback` bars.  No future bars used — confirmed N bars ago.
    Returns 1.0 at confirmed pivot high, 0.0 otherwise.
    """
    rolling_max = high.rolling(window=lookback, min_periods=lookback).max()
    return (high.shift(lookback) == rolling_max.shift(lookback)).astype(float)


def _left_pivot_lows(low: pd.Series, lookback: int = 5) -> pd.Series:
    """Left-side pivot lows: confirmed N bars ago."""
    rolling_min = low.rolling(window=lookback, min_periods=lookback).min()
    return (low.shift(lookback) == rolling_min.shift(lookback)).astype(float)


def _higher_highs_ratio(high: pd.Series, lookback: int = 5, window: int = 60) -> pd.Series:
    """
    Ratio of higher-highs to total pivot highs in a trailing window.
    Uses left-side pivots only.  Returns value in [0, 1].
    """
    pivots = _left_pivot_highs(high, lookback)
    # At each pivot, compare to previous pivot
    pivot_values = high.where(pivots.shift(lookback) == 1.0)
    pivot_values = pivot_values.ffill()
    is_higher = (pivot_values > pivot_values.shift(1)).astype(float)
    # Rolling mean over window
    ratio = is_higher.rolling(window=window, min_periods=lookback * 2).mean()
    return ratio.fillna(0.5)


# ============================================================================
# PRECOMPUTE ALL INDICATORS (once per backtest)
# ============================================================================
def precompute_all_signals(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Pre-compute all technical indicators for every asset in the price matrix.

    Args:
        prices: DataFrame with columns = asset names, index = DatetimeIndex.

    Returns:
        dict mapping asset_name → DataFrame of indicator values at each date.
        Each DataFrame has columns:
            rsi, macd_hist, macd_slope, adx, ema_cross, price_vs_ma200,
            bb_width_pctile, roc_20, hh_ratio, aggregate_score
    """
    result = {}
    for asset in prices.columns:
        close = prices[asset].dropna()
        if len(close) < 250:
            logger.warning(f"  ⚠️ {asset}: only {len(close)} rows, skipping indicators")
            continue

        # We use close for high/low approximation (daily data without OHLC)
        high = close  # Approximation for daily close-only data
        low = close

        # --- Trend indicators ---
        ema20 = _ema(close, 20)
        ema50 = _ema(close, 50)
        ma200 = _sma(close, 200)

        # EMA cross: normalized distance (>0 = bullish)
        ema_cross = (ema20 - ema50) / ema50
        # Price vs MA200: normalized distance
        price_vs_ma200 = (close - ma200) / ma200

        # --- Momentum indicators ---
        rsi = _rsi(close, 14)
        macd_line, macd_signal, macd_hist = _macd(close)
        macd_slope = macd_hist.diff(5)  # 5-day slope of histogram

        roc_20 = close.pct_change(20)  # 20-day rate of change

        # --- Trend strength ---
        adx = _adx(high, low, close, 14)

        # --- Volatility ---
        bb_width = _bollinger_width(close, 20)
        # Percentile rank of BB width over 60 days
        bb_width_pctile = bb_width.rolling(60, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        # --- Structure (left-side pivots) ---
        hh_ratio = _higher_highs_ratio(high, lookback=5, window=60)

        # --- Aggregate score [-1, +1] ---
        # Each sub-indicator maps to [-1, +1]
        scores = pd.DataFrame(index=close.index)

        # Percentile rank key unlimited series over 252 days to fix mixed-distribution problem
        ema_cross_rank = _pct_rank(ema_cross, 252)
        price_ma200_rank = _pct_rank(price_vs_ma200, 252)
        roc_20_rank = _pct_rank(roc_20, 252)

        # Trend score: EMA cross direction + ADX strength
        scores["trend"] = (ema_cross_rank * 2 - 1) * np.clip(adx / 40, 0.3, 1.0)

        # Price structure: price vs MA200
        scores["structure"] = (price_ma200_rank * 2 - 1)

        # Momentum: RSI z-score + MACD histogram slope + ROC
        rsi_centered = (rsi - 50) / 50  # [-1, +1]
        macd_slope_norm = np.clip(macd_slope / close.rolling(20).std().clip(lower=1e-8), -1, 1)
        roc_norm = (roc_20_rank * 2 - 1)
        scores["momentum"] = (rsi_centered * 0.4 + macd_slope_norm * 0.3 + roc_norm * 0.3)

        # Volatility: inverted BB width (narrow = uncertainty, wide = trend)
        scores["volatility"] = np.clip(1 - bb_width_pctile * 2, -1, 1) * 0.3

        # Structure: higher-highs ratio
        scores["hh_score"] = (hh_ratio - 0.5) * 2  # Center and scale to [-1, 1]

        # Weighted aggregate
        weights = {"trend": 0.30, "structure": 0.25, "momentum": 0.25,
                   "volatility": 0.10, "hh_score": 0.10}
        aggregate = sum(scores[col] * w for col, w in weights.items())

        # --- Build indicator DataFrame ---
        indicators = pd.DataFrame({
            "rsi": rsi,
            "macd_hist": macd_hist,
            "macd_slope": macd_slope,
            "adx": adx,
            "ema_cross": ema_cross,
            "price_vs_ma200": price_vs_ma200,
            "bb_width_pctile": bb_width_pctile,
            "roc_20": roc_20,
            "hh_ratio": hh_ratio,
            "aggregate_score": aggregate,
        }, index=close.index)

        # 5-period slope of aggregate (for ↑/↓ modifier)
        indicators["aggregate_slope"] = aggregate.diff(5)

        result[asset] = indicators

    logger.info(f"  ✅ Pre-computed signals for {len(result)} assets")
    return result


# ============================================================================
# SIGNAL CLASSIFICATION (pure function — no data access)
# ============================================================================
def classify_signal(aggregate_score: float, aggregate_slope: float) -> SignalState:
    """
    Classify a single observation into a SignalState.
    Pure function: takes pre-computed values, returns classification.

    Args:
        aggregate_score: Weighted indicator composite, range ~[-1, +1].
        aggregate_slope: 5-period change in aggregate score.

    Returns:
        SignalState enum value.
    """
    if np.isnan(aggregate_score) or np.isnan(aggregate_slope):
        return SignalState.NEUTRAL

    # Determine tier
    if aggregate_score > 0.5:
        # Bullish tier
        if aggregate_slope > 0.02:
            return SignalState.BULLISH_UP
        elif aggregate_slope < -0.02:
            return SignalState.BULLISH_DOWN
        return SignalState.BULLISH
    elif aggregate_score > -0.2:
        # Neutral tier
        if aggregate_slope > 0.02:
            return SignalState.NEUTRAL_UP
        elif aggregate_slope < -0.02:
            return SignalState.NEUTRAL_DOWN
        return SignalState.NEUTRAL
    else:
        # Bearish tier
        if aggregate_slope > 0.02:
            return SignalState.BEARISH_UP
        elif aggregate_slope < -0.02:
            return SignalState.BEARISH_DOWN
        return SignalState.BEARISH


def get_signal_tier(state: SignalState) -> SignalTier:
    """Extract the weight tier from a full SignalState."""
    return SIGNAL_TO_TIER[state]


def classify_all_signals_at_date(
    indicator_dfs: dict[str, pd.DataFrame],
    date: pd.Timestamp,
) -> dict[str, SignalState]:
    """
    Classify all assets at a single date using pre-computed indicators.
    Pure lookup — no indicator computation happens here.
    """
    signals = {}
    for asset, df in indicator_dfs.items():
        if date not in df.index:
            # Find nearest prior date
            valid_dates = df.index[df.index <= date]
            if len(valid_dates) == 0:
                signals[asset] = SignalState.NEUTRAL
                continue
            date_to_use = valid_dates[-1]
        else:
            date_to_use = date

        row = df.loc[date_to_use]
        agg = row.get("aggregate_score", 0.0)
        slope = row.get("aggregate_slope", 0.0)
        signals[asset] = classify_signal(
            float(agg) if not pd.isna(agg) else 0.0,
            float(slope) if not pd.isna(slope) else 0.0,
        )
    return signals


def compute_30d_performance(prices: pd.DataFrame, date: pd.Timestamp) -> dict[str, float]:
    """
    Compute 30-day return for each asset as of `date`.
    Used for momentum ranking in CryptoGoldRotation.select_universe().
    """
    result = {}
    for asset in prices.columns:
        series = prices[asset].dropna()
        valid = series[series.index <= date]
        if len(valid) < 30:
            result[asset] = 0.0
            continue
        result[asset] = float(valid.iloc[-1] / valid.iloc[-30] - 1)
    return result
