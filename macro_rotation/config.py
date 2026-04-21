"""
Central Configuration for the Macro Rotation System
=====================================================
All constants, enums, FRED series definitions, and system parameters.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
import warnings
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("macro_rotation")

# ---------------------------------------------------------------------------
# Directory Paths
# ---------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
CACHE_DIR = PACKAGE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
REPORTS_DIR = PACKAGE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR = PACKAGE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


# ============================================================================
# ENUMS
# ============================================================================
class SignalTier(Enum):
    """Weight-tier classification — rebalance triggers on tier change only."""
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"


class SignalState(Enum):
    """
    Full 9-state signal classification.
    Modifiers (↑/↓) affect sizing within tier but do NOT trigger rebalancing.
    """
    BULLISH_UP   = "Green↑"
    BULLISH      = "Green"
    BULLISH_DOWN = "Green↓"
    NEUTRAL_UP   = "Yellow↑"
    NEUTRAL      = "Yellow"
    NEUTRAL_DOWN = "Yellow↓"
    BEARISH_UP   = "Red↑"
    BEARISH      = "Red"
    BEARISH_DOWN = "Red↓"


# Mapping from full state to weight tier
SIGNAL_TO_TIER = {
    SignalState.BULLISH_UP:   SignalTier.BULLISH,
    SignalState.BULLISH:      SignalTier.BULLISH,
    SignalState.BULLISH_DOWN: SignalTier.BULLISH,
    SignalState.NEUTRAL_UP:   SignalTier.NEUTRAL,
    SignalState.NEUTRAL:      SignalTier.NEUTRAL,
    SignalState.NEUTRAL_DOWN: SignalTier.NEUTRAL,
    SignalState.BEARISH_UP:   SignalTier.BEARISH,
    SignalState.BEARISH:      SignalTier.BEARISH,
    SignalState.BEARISH_DOWN: SignalTier.BEARISH,
}


class MacroRegime(Enum):
    """4-quadrant macro regime classification."""
    RISK_ON_DISINFLATION  = "Growth↑ Inflation↓"
    RISK_ON_INFLATION     = "Growth↑ Inflation↑"
    RISK_OFF_INFLATION    = "Growth↓ Inflation↑"
    RISK_OFF_DISINFLATION = "Growth↓ Inflation↓"


class SentimentRegime(Enum):
    """Market sentiment cycle states."""
    APATHY  = "apathy"   # Early cycle — low attention
    FOMO    = "fomo"     # Market peak — greed extremes
    PANIC   = "panic"    # Reversal — sudden fear
    DESPAIR = "despair"  # Bottom — capitulation


class RebalanceMode(Enum):
    """
    How rebalancing trades are computed.

    MARGINAL (default): Trade only the weight delta. Fees computed on
        traded notional — realistic and recommended.
    FULL: Sell all positions, rebuy at new target weights. Fees computed
        on total position value — significantly overstates friction.
        Retained for comparison/testing purposes only.
    """
    MARGINAL = "marginal"
    FULL     = "full"


class AssetClass(Enum):
    """Asset class for fee/multiplier selection."""
    CRYPTO  = "crypto"
    GOLD    = "gold"
    EQUITY  = "equity"
    CASH    = "cash"


# ============================================================================
# SIGNAL MULTIPLIER TABLES
# ============================================================================
# Final Weight = Target% × modifier_multiplier × macro_suitability(0 or 1)
#
# Modifiers within tier affect sizing but don't trigger rebalance.
# Key: (SignalState, AssetClass) → multiplier

SIGNAL_MULTIPLIERS: dict[tuple[SignalState, AssetClass], float] = {
    # --- Crypto ---
    (SignalState.BULLISH_UP,   AssetClass.CRYPTO): 1.00,
    (SignalState.BULLISH,      AssetClass.CRYPTO): 0.90,
    (SignalState.BULLISH_DOWN, AssetClass.CRYPTO): 0.75,
    (SignalState.NEUTRAL_UP,   AssetClass.CRYPTO): 0.50,
    (SignalState.NEUTRAL,      AssetClass.CRYPTO): 0.45,
    (SignalState.NEUTRAL_DOWN, AssetClass.CRYPTO): 0.40,
    (SignalState.BEARISH_UP,   AssetClass.CRYPTO): 0.25,
    (SignalState.BEARISH,      AssetClass.CRYPTO): 0.15,
    (SignalState.BEARISH_DOWN, AssetClass.CRYPTO): 0.00,
    # --- Gold ---
    (SignalState.BULLISH_UP,   AssetClass.GOLD): 1.00,
    (SignalState.BULLISH,      AssetClass.GOLD): 0.90,
    (SignalState.BULLISH_DOWN, AssetClass.GOLD): 0.75,
    (SignalState.NEUTRAL_UP,   AssetClass.GOLD): 0.50,
    (SignalState.NEUTRAL,      AssetClass.GOLD): 0.45,
    (SignalState.NEUTRAL_DOWN, AssetClass.GOLD): 0.40,
    (SignalState.BEARISH_UP,   AssetClass.GOLD): 0.00,
    (SignalState.BEARISH,      AssetClass.GOLD): 0.00,
    (SignalState.BEARISH_DOWN, AssetClass.GOLD): 0.00,
    # --- Equity ---
    (SignalState.BULLISH_UP,   AssetClass.EQUITY): 1.00,
    (SignalState.BULLISH,      AssetClass.EQUITY): 0.90,
    (SignalState.BULLISH_DOWN, AssetClass.EQUITY): 0.75,
    (SignalState.NEUTRAL_UP,   AssetClass.EQUITY): 0.50,
    (SignalState.NEUTRAL,      AssetClass.EQUITY): 0.45,
    (SignalState.NEUTRAL_DOWN, AssetClass.EQUITY): 0.40,
    (SignalState.BEARISH_UP,   AssetClass.EQUITY): 0.00,
    (SignalState.BEARISH,      AssetClass.EQUITY): 0.00,
    (SignalState.BEARISH_DOWN, AssetClass.EQUITY): 0.00,
}


# ============================================================================
# MACRO REGIME → ASSET SUITABILITY MATRIX
# ============================================================================
# True = asset is suitable for this regime, False = force close
ASSET_TICKERS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "BNB": "BNB-USD",
    "XRP": "XRP-USD",
    "LINK": "LINK-USD",
    "DOGE": "DOGE-USD",
    "XAUT": "PAXG-USD",  # Gold proxy (PAXG for 2018+, GLD for pre-2018)
    "VNINDEX": "VNM",    # VNM ETF as fallback proxy
}

# Which assets are suitable under each macro regime
REGIME_SUITABILITY: dict[MacroRegime, dict[str, bool]] = {
    MacroRegime.RISK_ON_DISINFLATION: {
        "BTC": True, "ETH": True, "BNB": True, "XRP": True,
        "LINK": True, "DOGE": True, "XAUT": True, "VNINDEX": True,
    },
    MacroRegime.RISK_ON_INFLATION: {
        "BTC": True, "ETH": True, "BNB": True, "XRP": True,
        "LINK": True, "DOGE": True, "XAUT": True, "VNINDEX": True,
    },
    MacroRegime.RISK_OFF_INFLATION: {
        "BTC": False, "ETH": False, "BNB": False, "XRP": False,
        "LINK": False, "DOGE": False, "XAUT": True, "VNINDEX": False,
    },
    MacroRegime.RISK_OFF_DISINFLATION: {
        "BTC": False, "ETH": False, "BNB": False, "XRP": False,
        "LINK": False, "DOGE": False, "XAUT": True, "VNINDEX": False,
    },
}

# Asset class mapping
ASSET_CLASSES: dict[str, AssetClass] = {
    "BTC": AssetClass.CRYPTO, "ETH": AssetClass.CRYPTO,
    "BNB": AssetClass.CRYPTO, "XRP": AssetClass.CRYPTO,
    "LINK": AssetClass.CRYPTO, "DOGE": AssetClass.CRYPTO,
    "XAUT": AssetClass.GOLD, "VNINDEX": AssetClass.EQUITY,
}

ASSET_CURRENCIES: dict[str, str] = {
    "BTC": "USD", "ETH": "USD", "BNB": "USD", "XRP": "USD",
    "LINK": "USD", "DOGE": "USD", "XAUT": "USD",
    "VNINDEX": "VND",
}


# ============================================================================
# ASSET SUSPENSION WINDOWS — periods where a USD-based portfolio could not
# trade an asset despite price data existing (SEC halts, exchange delistings)
# ============================================================================
ASSET_SUSPENSION_WINDOWS: dict[str, list[tuple[str, str]]] = {
    "XRP": [("2021-01-19", "2021-09-17")],  # Coinbase/US trading halt (SEC lawsuit)
}

# Minimum years of macro data needed BEFORE backtest_start for valid
# trailing percentile rank windows in the regime classifier.
MACRO_WARMUP_YEARS = 5


# ============================================================================
# FRED SERIES MAP — Explicit IDs, units, expected release lag
# ============================================================================
FRED_SERIES = {
    # Money Supply (Global Liquidity Index components)
    "M2_US":       {"id": "M2SL",            "unit": "billions_usd", "freq": "M", "lag_weeks": 4},
    "M2_EUROZONE": {"id": "MABMM301EZM189S", "unit": "index",       "freq": "M", "lag_weeks": 6},
    "M2_JAPAN":    {"id": "MYAGM2JPM189S",   "unit": "index",       "freq": "M", "lag_weeks": 6},
    "M2_CHINA":    {"id": "MABMM301CNM189S", "unit": "index",       "freq": "M", "lag_weeks": 6},
    # Financial Conditions Index components
    "HY_SPREAD":   {"id": "BAMLH0A0HYM2",    "unit": "percent",     "freq": "D", "lag_weeks": 1},
    "IG_SPREAD":   {"id": "BAMLC0A0CM",       "unit": "percent",     "freq": "D", "lag_weeks": 1},
    "T10Y2Y":      {"id": "T10Y2Y",           "unit": "percent",     "freq": "D", "lag_weeks": 0},
    "VIX":         {"id": "VIXCLS",            "unit": "index",       "freq": "D", "lag_weeks": 0},
    # Inflation indicators
    "CPI":         {"id": "CPIAUCSL",          "unit": "index",       "freq": "M", "lag_weeks": 6},
    "T10YIE":      {"id": "T10YIE",            "unit": "percent",     "freq": "D", "lag_weeks": 0},
    "T5YIFR":      {"id": "T5YIFR",            "unit": "percent",     "freq": "D", "lag_weeks": 0},
}


# ============================================================================
# SYSTEM CONFIG
# ============================================================================
@dataclass(frozen=True)
class SystemConfig:
    """Central configuration for the entire macro rotation system."""
    # Capital & Yield
    initial_capital: float = 100_000
    cash_yield_apy: float = 0.0525       # 5.25% APY — realistic T-bill rate

    # Fee Structure
    crypto_fee: float = 0.001            # 10 bps (Binance-like)
    vnindex_buy_fee: float = 0.0015      # 15 bps
    vnindex_sell_fee: float = 0.0025     # 25 bps
    gold_fee: float = 0.001             # 10 bps

    # Settlement
    vnindex_settlement_lag: int = 2      # T+1.5 → modeled as 2-day execution lag
    crypto_settlement_lag: int = 0       # Instant settlement

    # Rebalancing
    rebalance_mode: RebalanceMode = RebalanceMode.MARGINAL
    min_rebalance_interval_days: int = 5
    min_weight_delta_to_rebalance: float = 0.05

    # Persistence Filters (bars before state change confirms)
    equity_persistence_bars: int = 5
    crypto_persistence_bars: int = 3
    sentiment_persistence_bars: int = 3

    # BTC Risk Metric
    btc_risk_window: int = 730           # 2-year normalization window

    # Crypto Universe Filters
    crypto_backtest_start: str = "2021-01-01"  # Avoids survivorship bias
    min_daily_volume_usd: float = 10_000_000   # $10M minimum daily volume
    max_rotation_positions: int = 3             # Max simultaneous rotation alts

    # Macro Regime
    macro_zscore_window_years: int = 5  # Trailing window for z-score / percentile
    gli_roc_months: int = 6             # Rate of change period for GLTI

    # Backtester
    backtest_start: str = "2021-01-01"
    backtest_end: str | None = None
    annualization_factor: int = 365

    # Institutional Grade Upgrades
    target_aum: float = 10_000_000       # $10M baseline for impact modeling
    execution_max_days: int = 5          # Max days to spread a rebalance
    lambda_risk_aversion: float = 1e-6   # Almgren-Chriss risk aversion
    kelly_max_fraction: float = 0.5      # Half-Kelly for tail-risk safety
    kelly_ewma_span: int = 60            # Responsive window for mu/var
    macro_forecast_periods: int = 3      # Ahead-of-the-curve projection
    macro_fit_window_years: int = 5      # Trailing fit for Holt-Winters

    # Checkpointing
    checkpoint_interval_days: int = 60
    resume_from_checkpoint: bool = False

    # Data sources
    vnindex_csv_path: str = "data/vnindex.csv"          # CSV override for VNINDEX (recommended)
    fred_api_key: str = os.getenv("FRED_API_KEY")              # FRED API key (for fredapi)
    fred_cache_ttl_days: int = 7        # Cache TTL for FRED data

    @property
    def daily_cash_rate(self) -> float:
        """Daily compounding rate from APY (365-day year for crypto)."""
        return (1 + self.cash_yield_apy) ** (1 / 365) - 1

    def clone_with(self, **kwargs) -> 'SystemConfig':
        """Return a new SystemConfig instance with specified overrides."""
        from dataclasses import replace
        return replace(self, **kwargs)


# Default system configuration
CONFIG = SystemConfig()
