"""
Portfolio Definitions (Module 1)
================================
Two primary portfolio models with a common AbstractPortfolio interface.
Split interface: select_universe() for asset selection, get_target_weights() for weight computation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from macro_rotation.config import (
    AssetClass, MacroRegime, SignalState, SignalTier,
    SIGNAL_TO_TIER, REGIME_SUITABILITY, ASSET_CLASSES,
    ASSET_SUSPENSION_WINDOWS,
    SystemConfig, CONFIG, logger,
)


# ============================================================================
# SUSPENSION CHECK HELPER
# ============================================================================
def _is_suspended(asset: str, date: pd.Timestamp) -> bool:
    """
    Check if an asset is under regulatory suspension at a given date.
    Uses ASSET_SUSPENSION_WINDOWS from config.py.
    """
    windows = ASSET_SUSPENSION_WINDOWS.get(asset, [])
    for start_str, end_str in windows:
        start = pd.Timestamp(start_str)
        end = pd.Timestamp(end_str)
        if start <= date <= end:
            return True
    return False


# ============================================================================
# ABSTRACT PORTFOLIO INTERFACE
# ============================================================================
class AbstractPortfolio(ABC):
    """Base class for all portfolio models."""

    @abstractmethod
    def select_universe(
        self,
        signals: dict[str, SignalState],
        regime: MacroRegime,
        performance_30d: dict[str, float] | None = None,
        current_date: pd.Timestamp | None = None,
        volumes: pd.Series | None = None,
    ) -> list[str]:
        pass

    @abstractmethod
    def get_target_weights(
        self,
        active_universe: list[str],
        signals: dict[str, SignalState],
        regime: MacroRegime,
        stats: dict | None = None,
    ) -> dict[str, float]:
        pass

    @abstractmethod
    def get_benchmark_ticker(self) -> str:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


# ============================================================================
# PORTFOLIO 1: CRYPTO + GOLD ROTATION
# ============================================================================
@dataclass
class CryptoGoldRotation(AbstractPortfolio):
    anchor: str = "BTC"
    rotation_pool: list[str] = field(
        default_factory=lambda: ["ETH", "BNB", "XRP", "LINK", "DOGE"]
    )
    defense: str = "XAUT"
    max_rotation: int = 3

    # Institutional Grade Upgrades
    use_kelly_sizing: bool = field(default=True)
    anchor_base_weight: float = 0.40
    rotation_base_weight: float = 0.40
    defense_base_weight: float = 0.20

    def get_name(self) -> str:
        return "Crypto + Gold Rotation"

    def get_benchmark_ticker(self) -> str:
        return "BTC-USD"

    def select_universe(self, signals, regime, performance_30d=None, current_date=None, volumes=None):
        universe = [self.anchor]
        available_pool = [a for a in self.rotation_pool if not _is_suspended(a, current_date)] if current_date else list(self.rotation_pool)
        
        if performance_30d:
            btc_perf = performance_30d.get(self.anchor, 0.0)
            eligible = [a for a in available_pool if performance_30d.get(a, 0.0) > btc_perf]
            if volumes is not None:
                eligible = [a for a in eligible if volumes.get(a, 1e9) >= CONFIG.min_daily_volume_usd]
            ranked = sorted(eligible, key=lambda a: performance_30d.get(a, 0.0), reverse=True)
            universe.extend(ranked[:self.max_rotation])
        else:
            universe.extend(available_pool[:self.max_rotation])
            
        if self.defense not in universe:
            universe.append(self.defense)
        return universe

    def get_target_weights(self, active_universe, signals, regime, stats=None):
        weights = {}
        if self.anchor in active_universe:
            weight = self.anchor_base_weight
            if self.use_kelly_sizing and stats and "kelly_fractions" in stats:
                weight = stats["kelly_fractions"].get(self.anchor, weight)
            weights[self.anchor] = weight
            
        rotation_assets = [a for a in active_universe if a in self.rotation_pool]
        if rotation_assets:
            per_alt = self.rotation_base_weight / len(rotation_assets)
            for asset in rotation_assets:
                weights[asset] = per_alt
        
        if self.defense in active_universe:
            weights[self.defense] = self.defense_base_weight
            
        return weights


# ============================================================================
# PORTFOLIO 2: CORE ASSET MACRO-ROTATION
# ============================================================================
@dataclass
class CoreAssetMacroRotation(AbstractPortfolio):
    good_macro_weights: dict[str, float] = field(
        default_factory=lambda: {"VNINDEX": 0.60, "XAUT": 0.20, "BTC": 0.20}
    )
    bad_macro_weights: dict[str, float] = field(
        default_factory=lambda: {"XAUT": 1.00, "VNINDEX": 0.00, "BTC": 0.00}
    )
    regime_interpolation: dict[MacroRegime, float] = field(
        default_factory=lambda: {
            MacroRegime.RISK_ON_DISINFLATION:  1.0,
            MacroRegime.RISK_ON_INFLATION:     0.7,
            MacroRegime.RISK_OFF_DISINFLATION: 0.3,
            MacroRegime.RISK_OFF_INFLATION:    0.0,
        }
    )
    use_risk_parity: bool = field(default=True)

    def get_name(self) -> str:
        return "Core Asset Portfolio 3 (Macro-Rotation)"

    def get_benchmark_ticker(self) -> str:
        return "VNM"

    def select_universe(self, signals, regime, performance_30d=None, current_date=None, volumes=None):
        return ["VNINDEX", "XAUT", "BTC"]

    def get_target_weights(self, active_universe, signals, regime, stats=None):
        if self.use_risk_parity and stats and "cov_matrix" in stats:
            from macro_rotation.quant_utils import optimize_risk_parity
            cov = stats["cov_matrix"]
            active_assets = [a for a in active_universe if a in cov.columns]
            if active_assets:
                rp_weights = optimize_risk_parity(cov.loc[active_assets, active_assets], target_total_weight=1.0)
                alpha = self.regime_interpolation.get(regime, 0.5)
                final_weights = {}
                for asset in active_universe:
                    w_rp = rp_weights.get(asset, 0.0)
                    w_bad = self.bad_macro_weights.get(asset, 0.0)
                    final_weights[asset] = alpha * w_rp + (1 - alpha) * w_bad
                return final_weights

        alpha = self.regime_interpolation.get(regime, 0.5)
        weights = {}
        for asset in active_universe:
            good_w = self.good_macro_weights.get(asset, 0.0)
            bad_w = self.bad_macro_weights.get(asset, 0.0)
            weights[asset] = alpha * good_w + (1 - alpha) * bad_w
        return weights


# ============================================================================
# PORTFOLIO 3: NAIVE BITCOIN + GOLD BASELINE
# ============================================================================
@dataclass
class NaiveBitcoinGoldPortfolio(AbstractPortfolio):
    def get_name(self) -> str:
        return "Naive 50/50 BTC/Gold Baseline"

    def get_benchmark_ticker(self) -> str:
        return "BTC-USD"

    def select_universe(self, signals, regime, performance_30d=None, current_date=None, volumes=None):
        return ["BTC", "XAUT"]

    def get_target_weights(self, active_universe, signals, regime, stats=None):
        weights = {}
        if "BTC" in active_universe: weights["BTC"] = 0.50
        if "XAUT" in active_universe: weights["XAUT"] = 0.50
        return weights
