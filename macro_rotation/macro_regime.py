"""
Macro Market Regime Filter (Module 3)
======================================
Scores and classifies the current global macro environment using composite
indices built from FRED data and market proxies.

Regime quadrants:
    RISK_ON_DISINFLATION   Growth↑  Inflation↓   (best for risk assets)
    RISK_ON_INFLATION      Growth↑  Inflation↑   (good, but with drag)
    RISK_OFF_INFLATION     Growth↓  Inflation↑   (worst — stagflation)
    RISK_OFF_DISINFLATION  Growth↓  Inflation↓   (defensive)

All composites use cross-sectional percentile rank over trailing window
to avoid the z-score mean-reversion problem in prolonged bear markets.

FRED Step-Function Note:
    Monthly FRED data (M2 money supply, PMI, etc.) is forward-filled to daily
    frequency.  This means the GLI/FCI composites update in discrete monthly
    steps — 21 trading days of flat signal followed by a jump on the release
    day.  If this jump crosses a percentile-rank threshold, the regime
    classifier will fire a rebalance on the exact day of the FRED data
    release.  This is expected behavior, not signal noise.  Monthly
    rebalances triggered by macro data releases are inherently step-function
    in nature and do not indicate overfitting.

    The signal_persistence_bars filter (default 3 bars) provides a natural
    buffer against single-day macro noise — a regime must persist for 3+
    consecutive days to be confirmed.
"""

import numpy as np
import pandas as pd

from macro_rotation.config import (
    MacroRegime, SystemConfig, CONFIG, logger,
)
from macro_rotation.quant_utils import apply_holt_winters


# ============================================================================
# COMPOSITE INDEX CONSTRUCTION
# ============================================================================
def _percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling percentile rank over trailing window.
    Returns values in [0, 1].  0.5 = median of trailing window.

    This is preferred over z-score because it doesn't mean-revert
    during prolonged unidirectional moves (review item #6).
    """
    return series.rolling(window=window, min_periods=max(window // 2, 30)).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )


def build_gli(fred_df: pd.DataFrame, window_days: int = 1260) -> pd.Series:
    """
    Global Liquidity Index (GLI): Normalized composite of M2 money supply
    from US, Eurozone, Japan, China.

    Uses percentile rank over trailing window to avoid z-score mean-reversion.
    Rising GLI = Risk-On.
    """
    m2_cols = ["M2_US", "M2_EUROZONE", "M2_JAPAN", "M2_CHINA"]
    available = [c for c in m2_cols if c in fred_df.columns]

    if not available:
        logger.warning("  ⚠️ No M2 data available for GLI construction")
        return pd.Series(dtype=float, name="GLI")

    # Percentile rank each M2 series independently
    ranks = pd.DataFrame(index=fred_df.index)
    for col in available:
        ranks[col] = _percentile_rank(fred_df[col], window=window_days)

    # GLI = equal-weighted mean of percentile ranks
    gli = ranks.mean(axis=1)
    gli.name = "GLI"
    return gli


def build_glti(gli: pd.Series, roc_periods: int = 130) -> pd.Series:
    """
    Global Liquidity Rate of Change (GLTI): 6-month (~130 trading days) ROC of GLI.
    Positive = liquidity improving.

    GLI is already in daily frequency (forward-filled from weekly-interpolated
    monthly data), so ROC in daily units is straightforward.
    """
    if gli.empty:
        return pd.Series(dtype=float, name="GLTI")

    glti = gli.pct_change(roc_periods)
    glti.name = "GLTI"
    return glti


def build_fci(
    fred_df: pd.DataFrame,
    proxy_prices: pd.DataFrame,
    window_days: int = 1260,
) -> pd.Series:
    """
    Financial Conditions Index (FCI): 3-component composite.
        1. VIX (percentile rank) — volatility fear gauge
        2. HY-IG spread (credit risk premium)
        3. 2s10s Treasury spread (yield curve — recession predictor)

    FCI > 0.5 = tight / defensive
    FCI < 0.5 = loose / risk-on
    """
    components = []

    # Component 1: VIX — from FRED or proxy_prices
    if "VIX" in fred_df.columns:
        vix_rank = _percentile_rank(fred_df["VIX"], window_days)
        components.append(("VIX", vix_rank, 0.40))
    elif proxy_prices is not None and "VIX" in proxy_prices.columns:
        vix_rank = _percentile_rank(proxy_prices["VIX"], window_days)
        components.append(("VIX", vix_rank, 0.40))

    # Component 2: HY-IG credit spread
    if "HY_SPREAD" in fred_df.columns and "IG_SPREAD" in fred_df.columns:
        credit_spread = fred_df["HY_SPREAD"] - fred_df["IG_SPREAD"]
        credit_rank = _percentile_rank(credit_spread, window_days)
        components.append(("Credit", credit_rank, 0.35))

    # Component 3: 2s10s Treasury spread (inverted — flat/inverted = tight)
    if "T10Y2Y" in fred_df.columns:
        # Invert: low/negative 2s10s = tight conditions = high FCI
        t10y2y_inverted = -fred_df["T10Y2Y"]
        yield_curve_rank = _percentile_rank(t10y2y_inverted, window_days)
        components.append(("YieldCurve", yield_curve_rank, 0.25))

    if not components:
        logger.warning("  ⚠️ No FCI components available")
        return pd.Series(dtype=float, name="FCI")

    # Weighted average, renormalized to actual available weights
    total_weight = sum(w for _, _, w in components)
    fci = sum(rank * (w / total_weight) for _, rank, w in components)
    fci.name = "FCI"
    return fci


def build_gei(
    proxy_prices: pd.DataFrame,
    window_days: int = 1260,
) -> pd.Series:
    """
    Global Economic Index (GEI): Composite of growth proxies.
        1. Copper/Gold ratio (Dr. Copper as growth barometer)
        2. DXY inverse (dollar weakness = global growth)

    GEI > 0.5 = expansion.
    """
    components = []

    if proxy_prices is None or proxy_prices.empty:
        return pd.Series(dtype=float, name="GEI")

    # Component 1: Copper/Gold ratio
    if "COPPER" in proxy_prices.columns and "GOLD_FUTURES" in proxy_prices.columns:
        cu_au = proxy_prices["COPPER"] / proxy_prices["GOLD_FUTURES"]
        cu_au_rank = _percentile_rank(cu_au, window_days)
        components.append(("Cu/Au", cu_au_rank, 0.50))

    # Component 2: DXY inverse (dollar weakness = global growth)
    if "DXY" in proxy_prices.columns:
        dxy_inv = 1 / proxy_prices["DXY"]
        dxy_rank = _percentile_rank(dxy_inv, window_days)
        components.append(("DXY_Inv", dxy_rank, 0.50))

    if not components:
        return pd.Series(dtype=float, name="GEI")

    total_weight = sum(w for _, _, w in components)
    gei = sum(rank * (w / total_weight) for _, rank, w in components)
    gei.name = "GEI"
    return gei


def build_expansion_tracker(
    sector_prices: pd.DataFrame,
    window: int = 60,
) -> pd.Series:
    """
    Economic Expansion Tracker: Fraction of 12 sector ETFs with positive
    N-day momentum.  Higher = broader expansion.

    Returns value in [0, 1]:
        > 0.7  = Expansion
        0.4-0.7 = Recovery / Cautious
        < 0.4  = Recession
    """
    if sector_prices is None or sector_prices.empty:
        return pd.Series(dtype=float, name="ExpansionTracker")

    momentums = sector_prices.pct_change(window)
    breadth = (momentums > 0).mean(axis=1)  # Fraction positive
    breadth.name = "ExpansionTracker"
    return breadth


# ============================================================================
# REGIME CLASSIFICATION
# ============================================================================
def classify_regime(
    growth_score: float,
    inflation_score: float,
) -> MacroRegime:
    """
    Map 2D (growth, inflation) scores to a regime quadrant.

    Args:
        growth_score:    Percentile rank [0, 1]. > 0.5 = growth above trailing median.
        inflation_score: Percentile rank [0, 1]. > 0.5 = inflation above trailing median.

    Returns:
        MacroRegime enum value.
    """
    growth_up = growth_score > 0.5
    inflation_up = inflation_score > 0.5

    if growth_up and not inflation_up:
        return MacroRegime.RISK_ON_DISINFLATION
    elif growth_up and inflation_up:
        return MacroRegime.RISK_ON_INFLATION
    elif not growth_up and inflation_up:
        return MacroRegime.RISK_OFF_INFLATION
    else:
        return MacroRegime.RISK_OFF_DISINFLATION


def build_inflation_proxy(
    fred_df: pd.DataFrame,
    window_days: int = 1260,
) -> pd.Series:
    """
    Robust inflation proxy using CPI and market breakevens.
    Combines backward-looking sticky inflation (CPI 12m momentum) with
    forward-looking real-time market expectations (T10YIE, T5YIFR).
    
    Weights: 50% CPI, 50% Breakeven composite.
    """
    if fred_df is None or fred_df.empty:
        return pd.Series(dtype=float, name="InflationProxy")

    components = []

    # 1. CPI Momentum (12-month trailing)
    if "CPI" in fred_df.columns:
        cpi = fred_df["CPI"]
        # CPI is monthly, ffilled to daily.
        # 12-month momentum is roughly 252 trading days.
        cpi_mom12 = cpi.pct_change(252)
        # Percentile rank the momentum to avoid mean-reversion traps
        cpi_rank = _percentile_rank(cpi_mom12, window_days)
        components.append(cpi_rank * 0.50)

    # 2. Breakevens Composite (T10YIE and T5YIFR)
    breakevens = []
    if "T10YIE" in fred_df.columns:
        breakevens.append(_percentile_rank(fred_df["T10YIE"], window_days))
    if "T5YIFR" in fred_df.columns:
        breakevens.append(_percentile_rank(fred_df["T5YIFR"], window_days))
        
    if breakevens:
        # Equal weight average of available breakevens, collectively forming 50%
        bk_avg = pd.concat(breakevens, axis=1).mean(axis=1)
        components.append(bk_avg * 0.50)

    if not components:
        return pd.Series(dtype=float, name="InflationProxy")

    # Sum available components, re-normalize if some are missing
    total_weight = len(components) * 0.50
    proxy = sum(components) / total_weight if total_weight > 0 else pd.Series(0.0, index=fred_df.index)
    
    proxy.name = "InflationProxy"
    return proxy


# ============================================================================
# FULL REGIME SERIES PRE-COMPUTATION
# ============================================================================
def precompute_macro_regimes(
    fred_df: pd.DataFrame,
    proxy_prices: pd.DataFrame,
    sector_prices: pd.DataFrame,
    config: SystemConfig = CONFIG,
) -> pd.DataFrame:
    """
    Pre-compute macro composite indices and regime classification for every date.

    Returns DataFrame with columns:
        GLI, GLTI, FCI, GEI, InflationProxy, ExpansionTracker,
        growth_score, inflation_score, regime
    """
    window = config.macro_zscore_window_years * 252  # Trading days

    # OPTIONAL: Apply Holt-Winters forecasting to anticipate lagging macro data
    fred_forecast = fred_df.copy()
    
    # We only forecast series with monthly frequency or clear lag to keep it manageable
    lagging_cols = ["M2_US", "M2_EUROZONE", "M2_JAPAN", "M2_CHINA", "CPI"]
    
    for col in lagging_cols:
        if col not in fred_forecast.columns:
            continue
            
        s = fred_forecast[col].dropna()
        if len(s) < 100:
            continue
            
        # To avoid extremely slow execution, we compute the forecast every 21 days (roughly monthly)
        # and forward-fill for daily signal use.
        forecasted_values = pd.Series(index=fred_df.index, dtype=float)
        
        # Step through the series in roughly monthly increments to update the forecast
        for j in range(252, len(fred_df), 21):
            idx = fred_df.index[j]
            sub_s = fred_df[col].iloc[:j+1].dropna()
            if len(sub_s) < 100: continue
            
            # Forecast 3 periods ahead
            val = apply_holt_winters(
                sub_s, 
                forecast_periods=config.macro_forecast_periods,
                fit_window_years=config.macro_fit_window_years
            )
            forecasted_values.loc[idx] = val
            
        fred_forecast[col] = forecasted_values.ffill().fillna(fred_df[col])

    # Build composites using the forecasted data
    gli = build_gli(fred_forecast, window_days=window)
    glti = build_glti(gli, roc_periods=config.gli_roc_months * 22)
    fci = build_fci(fred_forecast, proxy_prices, window_days=window)
    gei = build_gei(proxy_prices, window_days=window)
    inflation = build_inflation_proxy(fred_forecast, window_days=window)
    expansion = build_expansion_tracker(sector_prices, window=60)

    # Combine into DataFrame
    macro_df = pd.DataFrame({
        "GLI": gli,
        "GLTI": glti,
        "FCI": fci,
        "GEI": gei,
        "InflationProxy": inflation,
        "ExpansionTracker": expansion,
    }).ffill().dropna(how="all")

    if macro_df.empty:
        logger.warning("  ⚠️ No macro data available. Using default Risk-On regime.")
        return pd.DataFrame()

    # Growth score: average of GLI percentile + GEI + (1 - FCI) + expansion
    growth_components = []
    if "GLI" in macro_df.columns and macro_df["GLI"].notna().any():
        growth_components.append(macro_df["GLI"])
    if "GEI" in macro_df.columns and macro_df["GEI"].notna().any():
        growth_components.append(macro_df["GEI"])
    if "FCI" in macro_df.columns and macro_df["FCI"].notna().any():
        growth_components.append(1 - macro_df["FCI"])  # Invert: low FCI = growth
    if "ExpansionTracker" in macro_df.columns and macro_df["ExpansionTracker"].notna().any():
        growth_components.append(macro_df["ExpansionTracker"])

    if growth_components:
        macro_df["growth_score"] = pd.concat(growth_components, axis=1).mean(axis=1)
    else:
        macro_df["growth_score"] = 0.5  # Neutral default

    # Inflation score
    if "InflationProxy" in macro_df.columns and macro_df["InflationProxy"].notna().any():
        macro_df["inflation_score"] = macro_df["InflationProxy"]
    else:
        macro_df["inflation_score"] = 0.5

    # Classify regime at each date
    macro_df["regime"] = macro_df.apply(
        lambda row: classify_regime(
            row.get("growth_score", 0.5),
            row.get("inflation_score", 0.5),
        ),
        axis=1,
    )

    logger.info(f"  ✅ Macro regimes computed: {len(macro_df)} days")
    regime_counts = macro_df["regime"].value_counts()
    for regime, count in regime_counts.items():
        logger.info(f"     {regime.value}: {count} days ({count / len(macro_df) * 100:.1f}%)")

    return macro_df
