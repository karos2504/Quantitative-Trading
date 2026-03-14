"""
Monthly Portfolio Rebalancing — v6: Markowitz Mean-Variance Optimization
========================================================================
Replaces inverse-vol weighting with proper Markowitz optimization.

HOW IT WORKS:
  1. Momentum selects the CANDIDATE universe (top 30 by dual 12-1 / 6-1 score)
  2. Markowitz solves for the OPTIMAL WEIGHTS within those 30 candidates
  3. Three optimization objectives available (set OBJECTIVE below):
       "max_sharpe"   — maximize Sharpe ratio (return/risk tradeoff)
       "min_vol"      — minimize portfolio volatility
       "efficient_risk" — maximize return for a target volatility

COVARIANCE ESTIMATION:
  Raw sample covariance is noisy with 30 stocks. We use the
  Ledoit-Wolf shrinkage estimator (built into PyPortfolioOpt) which
  shrinks the covariance matrix toward a structured target — this is
  the industry standard for portfolio optimization with limited history.

EXPECTED RETURNS:
  We use momentum score as the return forecast rather than raw
  historical mean returns. Historical mean returns on monthly data
  are extremely noisy (require 100+ years to be reliable). The
  momentum score is a better forward-return predictor.

CONSTRAINTS:
  - Long-only (no shorting)
  - 2% minimum weight per stock (avoids 0-weight solutions)
  - 15% maximum weight per stock
  - Sector cap: max 25% in any single sector
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import datetime as dt

from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.exceptions import OptimizationError

from utils.data import fetch_ohlcv_data
from utils.backtesting import VBTBacktester

# ============================================================
#  CONFIG
# ============================================================
OBJECTIVE       = "max_sharpe"   # "max_sharpe" | "min_vol" | "efficient_risk"
TARGET_VOL      = 0.12           # Only used if OBJECTIVE = "efficient_risk" (12% annual)

CANDIDATE_SIZE  = 30   # Momentum pre-selects top 30 candidates
PORTFOLIO_SIZE  = 20   # Markowitz then picks optimal weights (some may go to min weight)
MIN_WEIGHT      = 0.02 # 2% floor — avoids near-zero allocations
MAX_WEIGHT      = 0.15 # 15% ceiling
MAX_SECTOR_W    = 0.25 # 25% max in any single sector

COV_LOOKBACK    = 36   # Months of history for covariance estimation (3 years)
VOL_LOOKBACK    = 6
MOMENTUM_SKIP   = 1

START_DATE = dt.datetime.today() - dt.timedelta(days=365 * 13)  # Extra for COV_LOOKBACK warm-up
END_DATE   = dt.datetime.today()

# ============================================================
#  UNIVERSE WITH SECTORS (same as v5)
# ============================================================
UNIVERSE_WITH_SECTORS = {
    "AAPL":"IT","MSFT":"IT","NVDA":"IT","AVGO":"IT","ORCL":"IT",
    "CRM":"IT","AMD":"IT","QCOM":"IT","TXN":"IT","CSCO":"IT",
    "INTU":"IT","IBM":"IT","NOW":"IT","AMAT":"IT","MU":"IT",
    "INTC":"IT","ADBE":"IT","KLAC":"IT","LRCX":"IT","ADI":"IT",
    "UNH":"HC","JNJ":"HC","LLY":"HC","ABBV":"HC","MRK":"HC",
    "TMO":"HC","ABT":"HC","DHR":"HC","BMY":"HC","AMGN":"HC",
    "PFE":"HC","SYK":"HC","ISRG":"HC","MDT":"HC","CI":"HC",
    "ELV":"HC","HCA":"HC","VRTX":"HC",
    "BRK-B":"FIN","JPM":"FIN","BAC":"FIN","WFC":"FIN","GS":"FIN",
    "MS":"FIN","BLK":"FIN","SCHW":"FIN","AXP":"FIN","CB":"FIN",
    "MMC":"FIN","TRV":"FIN","PNC":"FIN","USB":"FIN","MET":"FIN",
    "PRU":"FIN","ICE":"FIN","CME":"FIN",
    "AMZN":"CD","TSLA":"CD","HD":"CD","MCD":"CD","NKE":"CD",
    "LOW":"CD","SBUX":"CD","TJX":"CD","BKNG":"CD","MAR":"CD",
    "F":"CD","GM":"CD","ORLY":"CD","AZO":"CD",
    "PG":"CS","KO":"CS","PEP":"CS","COST":"CS","WMT":"CS",
    "PM":"CS","MO":"CS","CL":"CS","KMB":"CS","GIS":"CS","SYY":"CS",
    "GE":"IND","CAT":"IND","HON":"IND","UNP":"IND","RTX":"IND",
    "LMT":"IND","DE":"IND","BA":"IND","UPS":"IND","FDX":"IND",
    "EMR":"IND","ETN":"IND","ITW":"IND","MMM":"IND","NSC":"IND","WM":"IND",
    "GOOGL":"COM","META":"COM","NFLX":"COM","DIS":"COM","CMCSA":"COM",
    "T":"COM","VZ":"COM","TMUS":"COM","EA":"COM","TTWO":"COM",
    "XOM":"EN","CVX":"EN","COP":"EN","EOG":"EN","SLB":"EN",
    "MPC":"EN","PSX":"EN","VLO":"EN","OXY":"EN","HAL":"EN","DVN":"EN",
    "NEE":"UT","DUK":"UT","SO":"UT","D":"UT","AEP":"UT",
    "EXC":"UT","SRE":"UT","XEL":"UT","ED":"UT","PEG":"UT",
    "PLD":"RE","AMT":"RE","EQIX":"RE","CCI":"RE",
    "PSA":"RE","SPG":"RE","O":"RE","WELL":"RE",
    "LIN":"MAT","APD":"MAT","SHW":"MAT","FCX":"MAT","NEM":"MAT",
    "NUE":"MAT","VMC":"MAT","MLM":"MAT","PPG":"MAT","ECL":"MAT",
}
SP500_UNIVERSE = list(UNIVERSE_WITH_SECTORS.keys())


# ============================================================
#  DATA
# ============================================================
def build_prices(data: dict) -> pd.DataFrame:
    prices = pd.DataFrame(
        {t: df['Adj Close'] for t, df in data.items()}
    ).dropna(how='all').ffill(limit=2)
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change()


# ============================================================
#  MOMENTUM SIGNALS (same as v5)
# ============================================================
def compute_12_1(prices: pd.DataFrame) -> pd.DataFrame:
    return (prices.shift(1) / prices.shift(13)) - 1.0

def compute_6_1(prices: pd.DataFrame) -> pd.DataFrame:
    return (prices.shift(1) / prices.shift(7)) - 1.0


# ============================================================
#  MARKOWITZ OPTIMIZER
# ============================================================
def markowitz_weights(
    candidates:    list,
    returns_hist:  pd.DataFrame,  # Historical returns for cov matrix (COV_LOOKBACK months)
    momentum_scores: dict,         # Momentum score per ticker → used as return forecast
    objective:     str = OBJECTIVE,
) -> dict:
    """
    Solve Markowitz optimization for the given candidate list.

    Returns a dict of {ticker: weight} or falls back to equal-weight
    if the optimizer fails (singular matrix, insufficient history, etc.)
    """
    # Need at least 12 months of history per ticker
    hist = returns_hist[candidates].dropna()
    if len(hist) < 12 or len(candidates) < 3:
        # Fallback: equal weight
        w = 1.0 / len(candidates)
        return {t: w for t in candidates}

    try:
        # ---- Covariance: Ledoit-Wolf shrinkage ----
        # Converts monthly returns → annualised covariance
        S = risk_models.CovarianceShrinkage(
            hist, returns_data=True, frequency=12
        ).ledoit_wolf()

        # ---- Expected returns: momentum score scaled to annual ----
        # We use the 12-1 momentum score directly as the mu vector.
        # This is a forward-return estimate — better than historical mean on monthly data.
        mu = pd.Series({t: momentum_scores.get(t, 0.0) for t in candidates})

        # ---- Build EfficientFrontier ----
        ef = EfficientFrontier(
            mu, S,
            weight_bounds=(MIN_WEIGHT, MAX_WEIGHT),
            solver="CLARABEL",          # Fast, robust solver
        )

        # ---- Sector constraints ----
        sectors = [UNIVERSE_WITH_SECTORS.get(t, "OTHER") for t in candidates]
        unique_sectors = list(set(sectors))
        for sec in unique_sectors:
            sector_mask = [1.0 if UNIVERSE_WITH_SECTORS.get(t) == sec else 0.0
                           for t in candidates]
            if sum(sector_mask) > 1:
                ef.add_constraint(
                    lambda w, m=sector_mask: sum(w[i] * m[i] for i in range(len(m))) <= MAX_SECTOR_W
                )

        # ---- Optimize ----
        if objective == "max_sharpe":
            ef.max_sharpe(risk_free_rate=0.04)   # 4% risk-free rate
        elif objective == "min_vol":
            ef.min_volatility()
        elif objective == "efficient_risk":
            ef.efficient_risk(target_volatility=TARGET_VOL)
        else:
            ef.max_sharpe(risk_free_rate=0.04)

        cleaned = ef.clean_weights(cutoff=MIN_WEIGHT, rounding=4)
        # Remove zero-weight tickers
        return {t: w for t, w in cleaned.items() if w > 0.0}

    except (OptimizationError, Exception):
        # Fallback: inverse-vol weights (robust alternative)
        vols = returns_hist[candidates].std()
        inv  = {t: 1.0 / vols[t] if vols[t] > 0 else 1.0 for t in candidates}
        tot  = sum(inv.values())
        raw  = {t: v / tot for t, v in inv.items()}
        cap  = {t: min(w, MAX_WEIGHT) for t, w in raw.items()}
        tot2 = sum(cap.values())
        return {t: w / tot2 for t, w in cap.items()}


# ============================================================
#  STRATEGY
# ============================================================
def run_strategy(
    prices:   pd.DataFrame,
    returns:  pd.DataFrame,
    mom_12_1: pd.DataFrame,
    mom_6_1:  pd.DataFrame,
) -> pd.Series:
    """
    Each month:
      Step 1 — Momentum filter: select top-CANDIDATE_SIZE stocks
               where BOTH 12-1 and 6-1 scores are positive
      Step 2 — Markowitz: solve for optimal weights within those candidates
      Step 3 — Hold the optimized portfolio for the month
    """
    monthly_returns = []
    current_weights: dict = {}

    for i in range(len(returns)):
        # ---- Realise P&L ----
        if current_weights:
            pnl = sum(
                returns[t].iloc[i] * w
                for t, w in current_weights.items()
                if t in returns.columns and not pd.isna(returns[t].iloc[i])
            )
            monthly_returns.append(pnl)
        else:
            monthly_returns.append(0.0)

        # ---- Step 1: Momentum candidate selection ----
        s12 = mom_12_1.iloc[i]
        s6  = mom_6_1.iloc[i]

        eligible = {
            t: float(s12[t])
            for t in SP500_UNIVERSE
            if t in s12.index
            and not pd.isna(s12[t]) and float(s12[t]) > 0.0
            and t in s6.index
            and not pd.isna(s6[t]) and float(s6[t]) > 0.0
        }

        if len(eligible) < 5:
            # Relax: drop 6-1 requirement
            eligible = {
                t: float(s12[t])
                for t in SP500_UNIVERSE
                if t in s12.index and not pd.isna(s12[t])
            }

        if not eligible:
            continue

        # Top CANDIDATE_SIZE by momentum score → feed into Markowitz
        ranked     = sorted(eligible, key=lambda t: eligible[t], reverse=True)
        candidates = ranked[:CANDIDATE_SIZE]

        # ---- Step 2: Markowitz optimization ----
        # Use COV_LOOKBACK months of history ending at month i-1 (no look-ahead)
        hist_start = max(0, i - COV_LOOKBACK)
        returns_window = returns.iloc[hist_start:i]

        # Only keep candidates with enough history in this window
        available = [
            t for t in candidates
            if t in returns_window.columns
            and returns_window[t].notna().sum() >= 12
        ]

        if len(available) < 3:
            # Not enough history — equal weight top 10
            top = candidates[:10]
            current_weights = {t: 1.0 / len(top) for t in top}
            continue

        current_weights = markowitz_weights(
            candidates     = available,
            returns_hist   = returns_window,
            momentum_scores= eligible,
            objective      = OBJECTIVE,
        )

    return pd.Series(monthly_returns, index=returns.index, name="Monthly Return")


# ============================================================
#  MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  Monthly Rebalancing — v6: Markowitz Optimization")
    print("=" * 60)
    print(f"\n  Universe:    {len(SP500_UNIVERSE)} S&P 500 stocks")
    print(f"  Candidates:  Top {CANDIDATE_SIZE} by dual 12-1/6-1 momentum")
    print(f"  Optimizer:   Markowitz [{OBJECTIVE}], Ledoit-Wolf covariance")
    print(f"  Constraints: [{MIN_WEIGHT*100:.0f}%, {MAX_WEIGHT*100:.0f}%] per stock | "
          f"{MAX_SECTOR_W*100:.0f}% per sector\n")

    print("  Fetching data...")
    ohlcv   = fetch_ohlcv_data(SP500_UNIVERSE, start=START_DATE, end=END_DATE, interval='1mo')
    prices  = build_prices(ohlcv)
    returns = compute_returns(prices).dropna(how='all')
    prices  = prices.reindex(returns.index)

    # Quality filter
    coverage = prices.notna().mean()
    good     = coverage[coverage > 0.80].index
    prices   = prices[good]
    returns  = returns[good]
    print(f"  After quality filter: {len(prices.columns)} tickers retained\n")

    # Signals
    mom_12_1 = compute_12_1(prices)
    mom_6_1  = compute_6_1(prices)

    print("  Running strategy with Markowitz optimization each month...")
    print("  (Solving ~140 optimization problems — takes ~2 min)\n")

    strategy_returns = run_strategy(
        prices=prices, returns=returns,
        mom_12_1=mom_12_1, mom_6_1=mom_6_1,
    )

    strategy_close = (1 + strategy_returns).cumprod() * 100

    # Always invested — measure selection + sizing skill
    entries = pd.Series(True,  index=strategy_returns.index)
    exits   = pd.Series(False, index=strategy_returns.index)
    exits.iloc[-1] = True

    bt = VBTBacktester(
        close=strategy_close, entries=entries, exits=exits,
        freq='30D', init_cash=100_000, commission=0.001,
    )
    bt.full_analysis(n_mc=1000, n_wf_splits=5, n_trials=1)


if __name__ == '__main__':
    main()
    