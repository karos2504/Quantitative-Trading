"""
Monthly Portfolio Rebalancing — v5: Pure Cross-Sectional Momentum
=================================================================
Key architectural changes from v4:

  PROBLEM DIAGNOSED: The macro filter (golden cross / moving average)
  was the primary return killer. It kept the strategy in defensive assets
  during bear signals, but the market recovered faster than the slow MA
  signal could re-trigger. Result: sell the dip, buy the top — every time.

  v5 APPROACH — Test the SIGNAL independently of market timing:
  1. NO macro filter — always fully invested in equities
  2. Pure 12-1 momentum rotation: top 20 stocks rebalanced monthly
  3. ALWAYS_INVESTED flag: entries=True every bar so VBT never exits to cash
     This ensures the backtest measures stock selection skill, not timing luck
  4. Sector diversification constraint: max 3 stocks per sector
  5. Dual momentum confirmation: both 12-1 AND 6-1 must be positive
     (reduces exposure to one-sided momentum that reverses quickly)

  If pure momentum STILL shows P-value > 0.20 vs random, the signal
  itself has no edge and we should pivot to a different alpha source.
  If it shows edge, we add timing back carefully as a separate layer.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt

from utils.data import fetch_ohlcv_data
from utils.backtesting import VBTBacktester

# ============================================================
#  UNIVERSE — same 150-stock S&P 500 universe from v4
#  with sector tags for the diversification constraint
# ============================================================
UNIVERSE_WITH_SECTORS = {
    # Information Technology
    "AAPL": "IT", "MSFT": "IT", "NVDA": "IT", "AVGO": "IT", "ORCL": "IT",
    "CRM": "IT", "AMD": "IT", "QCOM": "IT", "TXN": "IT", "CSCO": "IT",
    "INTU": "IT", "IBM": "IT", "NOW": "IT", "AMAT": "IT", "MU": "IT",
    "INTC": "IT", "ADBE": "IT", "KLAC": "IT", "LRCX": "IT", "ADI": "IT",
    # Health Care
    "UNH": "HC", "JNJ": "HC", "LLY": "HC", "ABBV": "HC", "MRK": "HC",
    "TMO": "HC", "ABT": "HC", "DHR": "HC", "BMY": "HC", "AMGN": "HC",
    "PFE": "HC", "SYK": "HC", "ISRG": "HC", "MDT": "HC", "CI": "HC",
    "ELV": "HC", "HCA": "HC", "VRTX": "HC",
    # Financials
    "BRK-B": "FIN", "JPM": "FIN", "BAC": "FIN", "WFC": "FIN", "GS": "FIN",
    "MS": "FIN", "BLK": "FIN", "SCHW": "FIN", "AXP": "FIN", "CB": "FIN",
    "MMC": "FIN", "TRV": "FIN", "PNC": "FIN", "USB": "FIN", "MET": "FIN",
    "PRU": "FIN", "ICE": "FIN", "CME": "FIN",
    # Consumer Discretionary
    "AMZN": "CD", "TSLA": "CD", "HD": "CD", "MCD": "CD", "NKE": "CD",
    "LOW": "CD", "SBUX": "CD", "TJX": "CD", "BKNG": "CD", "MAR": "CD",
    "F": "CD", "GM": "CD", "ORLY": "CD", "AZO": "CD",
    # Consumer Staples
    "PG": "CS", "KO": "CS", "PEP": "CS", "COST": "CS", "WMT": "CS",
    "PM": "CS", "MO": "CS", "CL": "CS", "KMB": "CS", "GIS": "CS", "SYY": "CS",
    # Industrials
    "GE": "IND", "CAT": "IND", "HON": "IND", "UNP": "IND", "RTX": "IND",
    "LMT": "IND", "DE": "IND", "BA": "IND", "UPS": "IND", "FDX": "IND",
    "EMR": "IND", "ETN": "IND", "ITW": "IND", "MMM": "IND", "NSC": "IND", "WM": "IND",
    # Communication Services
    "GOOGL": "CS2", "META": "CS2", "NFLX": "CS2", "DIS": "CS2", "CMCSA": "CS2",
    "T": "CS2", "VZ": "CS2", "TMUS": "CS2", "EA": "CS2", "TTWO": "CS2",
    # Energy
    "XOM": "EN", "CVX": "EN", "COP": "EN", "EOG": "EN", "SLB": "EN",
    "MPC": "EN", "PSX": "EN", "VLO": "EN", "OXY": "EN", "HAL": "EN", "DVN": "EN",
    # Utilities
    "NEE": "UT", "DUK": "UT", "SO": "UT", "D": "UT", "AEP": "UT",
    "EXC": "UT", "SRE": "UT", "XEL": "UT", "ED": "UT", "PEG": "UT",
    # Real Estate
    "PLD": "RE", "AMT": "RE", "EQIX": "RE", "CCI": "RE",
    "PSA": "RE", "SPG": "RE", "O": "RE", "WELL": "RE",
    # Materials
    "LIN": "MAT", "APD": "MAT", "SHW": "MAT", "FCX": "MAT", "NEM": "MAT",
    "NUE": "MAT", "VMC": "MAT", "MLM": "MAT", "PPG": "MAT", "ECL": "MAT",
}

SP500_UNIVERSE = list(UNIVERSE_WITH_SECTORS.keys())

START_DATE = dt.datetime.today() - dt.timedelta(days=365 * 12)
END_DATE   = dt.datetime.today()

PORTFOLIO_SIZE    = 20    # Top 20 momentum stocks
MAX_WEIGHT        = 0.10  # 10% cap per stock
MAX_PER_SECTOR    = 3     # Max 3 stocks per sector (diversification constraint)
VOL_LOOKBACK      = 6     # Months for volatility estimate

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
#  SIGNALS
# ============================================================
def compute_rolling_sharpe(returns: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """12-month Rolling Sharpe Ratio (Mean / Std) excluding recent month."""
    # We shift returns by 1 to skip the current month (reversal avoidance)
    shifted_ret_series = returns.shift(1)
    roll_mean = shifted_ret_series.rolling(window).mean()
    roll_std  = shifted_ret_series.rolling(window).std()
    
    # Avoid div by zero
    sharpe = roll_mean / roll_std.replace(0, np.nan)
    return sharpe


def compute_6_1_momentum(prices: pd.DataFrame) -> pd.DataFrame:
    """6-month return skipping most recent month (medium-term confirmation)."""
    return (prices.shift(1) / prices.shift(7)) - 1.0


def compute_vol(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.rolling(VOL_LOOKBACK).std() * np.sqrt(12)


# ============================================================
#  STRATEGY — ALWAYS INVESTED, pure stock selection
# ============================================================
def run_strategy(
    prices:   pd.DataFrame,
    returns:  pd.DataFrame,
    sharpe_12: pd.DataFrame,
    mom_6_1:   pd.DataFrame,
    vols:      pd.DataFrame,
) -> pd.Series:
    """
    Each month: select top-PORTFOLIO_SIZE stocks that pass 12-month Rolling Sharpe
    AND 6-month momentum filters, subject to sector cap. Always fully invested.
    If fewer than 5 pass, use remaining highest-Sharpe stocks as fallback.
    """
    monthly_returns = []
    current_weights: dict = {}

    for i in range(len(returns)):
        # ---- Realise monthly P&L ----
        if current_weights:
            pnl = sum(
                returns[t].iloc[i] * w
                for t, w in current_weights.items()
                if t in returns.columns and not pd.isna(returns[t].iloc[i])
            )
            monthly_returns.append(pnl)
        else:
            monthly_returns.append(0.0)

        # ---- Build next month's portfolio ----
        s12 = sharpe_12.iloc[i]
        s6  = mom_6_1.iloc[i]
        v   = vols.iloc[i]

        # Dual confirmation: Positive 12m Sharpe AND positive 6-1 momentum
        eligible = {
            t: s12[t]
            for t in SP500_UNIVERSE
            if t in s12.index
            and not pd.isna(s12[t]) and s12[t] > 0.0
            and not pd.isna(s6.get(t)) and s6.get(t, -1) > 0.0
            and t in v.index and not pd.isna(v[t]) and v[t] > 0.0
        }

        if len(eligible) < 5:
            # Fallback: top-N by Sharpe regardless of 6-1 sign
            eligible = {
                t: s12[t]
                for t in SP500_UNIVERSE
                if t in s12.index and not pd.isna(s12[t])
                and t in v.index and not pd.isna(v[t]) and v[t] > 0.0
            }

        if not eligible:
            # Extreme fallback: hold previous portfolio
            monthly_returns[-1] = monthly_returns[-1]  # already captured
            continue

        # Rank by 12-month Sharpe Ratio
        ranked = sorted(eligible, key=lambda t: eligible[t], reverse=True)

        # Apply sector cap: pick greedily, skip if sector is full
        selected = []
        sector_count: dict = {}
        for t in ranked:
            sector = UNIVERSE_WITH_SECTORS.get(t, "OTHER")
            if sector_count.get(sector, 0) < MAX_PER_SECTOR:
                selected.append(t)
                sector_count[sector] = sector_count.get(sector, 0) + 1
            if len(selected) >= PORTFOLIO_SIZE:
                break

        if not selected:
            continue

        # True Risk Parity weighting (Inverse Variance: 1 / Vol^2)
        inv_vars = {t: 1.0 / (v[t] ** 2) for t in selected}
        total_iv = sum(inv_vars.values())
        raw_w    = {t: iv / total_iv for t, iv in inv_vars.items()}
        capped   = {t: min(w, MAX_WEIGHT) for t, w in raw_w.items()}
        total_c  = sum(capped.values())
        current_weights = {t: w / total_c for t, w in capped.items()}

    return pd.Series(monthly_returns, index=returns.index, name="Monthly Return")


# ============================================================
#  MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  Monthly Rebalancing — v6: Rolling Sharpe & True Risk Parity")
    print("=" * 60)
    print(f"\n  Universe:  {len(SP500_UNIVERSE)} S&P 500 stocks, 11 sectors")
    print(f"  Signal:    12-month Rolling Sharpe Ratio > 0 AND 6-1 Momentum > 0")
    print(f"  Holdings:  Top {PORTFOLIO_SIZE}, max {MAX_PER_SECTOR}/sector, Risk Parity (1/Vol^2) weighted")
    print(f"  Timing:    NONE — always fully invested\n")

    print("  Fetching data...")
    ohlcv   = fetch_ohlcv_data(SP500_UNIVERSE, start=START_DATE, end=END_DATE, interval='1mo')
    prices  = build_prices(ohlcv)
    returns = compute_returns(prices).dropna(how='all')
    prices  = prices.reindex(returns.index)

    # Quality filter: keep tickers with >80% data coverage
    coverage = prices.notna().mean()
    good     = coverage[coverage > 0.80].index
    prices   = prices[good]
    returns  = returns[good]
    print(f"  After quality filter: {len(prices.columns)} tickers retained\n")

    # Signals
    sharpe_12 = compute_rolling_sharpe(returns, window=12)
    mom_6_1   = compute_6_1_momentum(prices)
    vols      = compute_vol(returns)

    print("  Running strategy...\n")
    strategy_returns = run_strategy(
        prices=prices, returns=returns,
        sharpe_12=sharpe_12, mom_6_1=mom_6_1,
        vols=vols,
    )

    # ALWAYS INVESTED: entries=True every bar
    # This tells VBT the strategy holds continuously — measures selection skill only
    entries = pd.Series(True,  index=strategy_returns.index)
    exits   = pd.Series(False, index=strategy_returns.index)
    exits.iloc[-1] = True  # Close at end

    strategy_close = (1 + strategy_returns).cumprod() * 100

    bt = VBTBacktester(
        close=strategy_close, entries=entries, exits=exits,
        freq='30D', init_cash=100_000, commission=0.001,
    )
    bt.full_analysis(n_mc=1000, n_wf_splits=5, n_trials=1)


if __name__ == '__main__':
    main()