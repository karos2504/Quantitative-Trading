"""
Monthly Portfolio Rebalancing
"""

import sys
import os
os.environ['PYTHONWARNINGS'] = 'ignore:resource_tracker:UserWarning'
import multiprocessing
from pathlib import Path

if os.name == 'posix':
    try:
        if multiprocessing.get_start_method(allow_none=True) is None:
            multiprocessing.set_start_method('spawn', force=True)
    except (RuntimeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict

from pypfopt import EfficientFrontier, risk_models

from data_ingestion.data import fetch_ohlcv_data
from backtesting_engine.backtesting import VBTBacktester
from config.settings import CASH, COMMISSION


# ============================================================
#  CONFIG
# ============================================================
RISK_FREE_RATE      = 0.04
TARGET_VOL          = 0.20
CANDIDATE_SIZE      = 10
MIN_WEIGHT          = 0.07
MAX_WEIGHT          = 0.30
MAX_SECTOR_W_BULL   = 0.30
MAX_SECTOR_W_TOP    = 0.30
MAX_SECTOR_W_BEAR   = 0.20
COV_LOOKBACK        = 36

# Momentum
MOM_6_1_VETO        = 0.00

# Regime dead-band
REGIME_BULL_BAND    = 1.05
REGIME_BEAR_BAND    = 0.95

# Crash protection
DEFENSIVE_SECTORS   = {'UT', 'CS', 'HC'}
BEAR_DEF_BOOST      = 1.5
CRASH_CASH_RATIO    = 0.70

# Peak-drawdown circuit
DD_REDUCE_THRESH    = -0.12
DD_RESTORE_THRESH   = -0.06
DD_EXPOSURE_SCALE   = 0.70

# Downside-vol penalty
DOWNSIDE_VOL_MULT   = 3.0

# Rebalance threshold — skip trade if drift < this and no entry/exit
REBAL_THRESHOLD     = 0.03

# Fast bear trigger — override regime to bear if SPY fell > this last month
SPY_CRASH_THRESH    = -0.05

# SPY anchor weight in bull regime
SPY_ANCHOR_W        = 0.10

TXN_COST_BPS        = 10
MIN_BACKTEST_MONTHS = 24

START_DATE = dt.datetime.today() - dt.timedelta(days=365 * 13)
END_DATE   = dt.datetime.today()

REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# ============================================================
#  UNIVERSE
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

SECTOR_COLORS = {
    "IT":"#4C78A8","HC":"#72B7B2","FIN":"#F58518","CD":"#E45756",
    "CS":"#54A24B","IND":"#B279A2","COM":"#FF9DA6","EN":"#9D755D",
    "UT":"#BAB0AC","RE":"#EECA3B","MAT":"#76B7B2","OTHER":"#aaaaaa",
}


# ============================================================
#  HELPERS
# ============================================================
def _to_period_index(s: pd.Series) -> pd.Series:
    if not isinstance(s.index, pd.PeriodIndex):
        s = s.copy()
        s.index = pd.PeriodIndex(s.index, freq='M')
    return s


# ============================================================
#  DATA
# ============================================================
def build_prices(data: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {t: df['Close'] for t, df in data.items()}
    ).dropna(how='all').ffill(limit=2)

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1))

def compute_12_1(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices.shift(1) / prices.shift(13))

def compute_6_1(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices.shift(1) / prices.shift(7))


# ============================================================
#  RISK METRICS
# ============================================================
def _sortino(returns: pd.Series, rf_monthly: float = RISK_FREE_RATE / 12) -> float:
    excess  = returns - rf_monthly
    downside = excess[excess < 0]
    if len(downside) < 3:
        return np.nan
    semi_dev = np.sqrt(np.mean(downside ** 2)) * np.sqrt(12)
    if semi_dev == 0:
        return np.nan
    return (excess.mean() * 12) / semi_dev


def _calmar(returns: pd.Series) -> float:
    if len(returns) < 6:
        return np.nan
    equity  = (1 + returns).cumprod()
    peak    = equity.cummax()
    dd      = (equity - peak) / peak
    max_dd  = dd.min()
    if max_dd == 0:
        return np.nan
    n_years = len(returns) / 12
    cagr    = equity.iloc[-1] ** (1 / n_years) - 1
    return cagr / abs(max_dd)


def _information_ratio(strat: pd.Series, bench: pd.Series) -> float:
    active = strat - bench.reindex(strat.index).fillna(0)
    if active.std() == 0 or len(active) < 6:
        return np.nan
    return (active.mean() * 12) / (active.std() * np.sqrt(12))


def compute_full_metrics(strat: pd.Series, bench: pd.Series) -> dict:
    strat  = strat.dropna()
    n_mo   = len(strat)
    n_yrs  = n_mo / 12

    equity  = (1 + strat).cumprod()
    peak    = equity.cummax()
    dd      = (equity - peak) / peak
    max_dd  = dd.min()

    cagr    = equity.iloc[-1] ** (1 / n_yrs) - 1 if n_yrs > 0 else np.nan
    ann_vol = strat.std() * np.sqrt(12)
    sharpe  = (cagr - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else np.nan
    sortino = _sortino(strat)
    calmar  = cagr / abs(max_dd) if max_dd != 0 else np.nan

    bench_a = bench.reindex(strat.index).fillna(0)
    active  = strat - bench_a
    te      = active.std() * np.sqrt(12)
    ir      = (active.mean() * 12) / te if te > 0 else np.nan

    in_dd   = (dd < 0).astype(int)
    runs    = []
    count   = 0
    for v in in_dd:
        if v:
            count += 1
        else:
            if count:
                runs.append(count)
            count = 0
    if count:
        runs.append(count)
    max_recovery = max(runs) if runs else 0

    win_rate  = (strat > 0).mean()
    avg_win   = strat[strat > 0].mean() if (strat > 0).any() else 0
    avg_loss  = strat[strat < 0].mean() if (strat < 0).any() else 0
    gain_pain = avg_win / abs(avg_loss) if avg_loss != 0 else np.nan

    return {
        "CAGR (%)":            round(cagr * 100, 2),
        "Ann. Vol (%)":        round(ann_vol * 100, 2),
        "Sharpe":              round(sharpe, 3),
        "Sortino":             round(sortino, 3),
        "Calmar":              round(calmar, 3),
        "Max Drawdown (%)":    round(max_dd * 100, 2),
        "Max Recovery (mo)":   max_recovery,
        "Info Ratio vs SPY":   round(ir, 3),
        "Tracking Error (%)":  round(te * 100, 2),
        "Win Rate (%)":        round(win_rate * 100, 1),
        "Gain/Pain":           round(gain_pain, 2),
        "Months":              n_mo,
    }


# ============================================================
#  REGIME DETECTION
# ============================================================
def _detect_regime(spy_prices_daily: pd.Series):
    if spy_prices_daily is None or len(spy_prices_daily) < 220:
        return None
    try:
        from portfolio_construction.regime_hmm import fit_hmm_regimes
        regime_labels = fit_hmm_regimes(spy_prices_daily, n_components=2)
        rets  = np.log(spy_prices_daily / spy_prices_daily.shift(1))
        vol_0 = rets[regime_labels == 0].std()
        vol_1 = rets[regime_labels == 1].std()
        bull  = 0 if vol_0 < vol_1 else 1
        print("  Regime: HMM ✅")
        return regime_labels.map({bull: 1, 1 - bull: -1})
    except Exception as e:
        print(f"  ⚠️  HMM failed ({e}). Using 200-MA dead-band.")

    ma200    = spy_prices_daily.rolling(200).mean()
    regime   = pd.Series(0, index=spy_prices_daily.index, dtype=int)
    regime[spy_prices_daily > ma200 * REGIME_BULL_BAND]  =  1
    regime[spy_prices_daily < ma200 * REGIME_BEAR_BAND]  = -1
    regime   = regime.replace(0, np.nan).ffill().fillna(1).astype(int)
    bull_pct = (regime == 1).mean() * 100
    print(f"  Regime (200-MA dead-band): bull {bull_pct:.1f}%  bear {100-bull_pct:.1f}%  ✅")
    return regime


def _get_regime_state(spy_regime, date) -> int:
    if spy_regime is None:
        return 1
    # Ensure date is Timestamp for lookup in daily spy_regime index
    lookup_date = date.to_timestamp() if hasattr(date, 'to_timestamp') else date
    try:
        idx = spy_regime.index.get_indexer([lookup_date], method='ffill')[0]
        if idx >= 0:
            return int(spy_regime.iloc[idx])
    except Exception:
        pass
    return 1


# ============================================================
#  PORTFOLIO CONSTRUCTION
# ============================================================
def _downside_adjusted_scores(candidates: list,
                               momentum_scores: dict,
                               returns_window: pd.DataFrame) -> dict:
    if returns_window.empty or len(returns_window) < 6:
        return momentum_scores

    semi_devs = {}
    for t in candidates:
        if t not in returns_window.columns:
            continue
        r   = returns_window[t].dropna()
        neg = r[r < 0]
        semi_devs[t] = np.sqrt(np.mean(neg ** 2)) if len(neg) >= 3 else 0.0

    if not semi_devs:
        return momentum_scores

    median_sd = np.median(list(semi_devs.values()))
    adjusted  = {}
    for t in candidates:
        base = momentum_scores.get(t, 0.0)
        sd   = semi_devs.get(t, 0.0)
        if median_sd > 0 and sd > DOWNSIDE_VOL_MULT * median_sd:
            adjusted[t] = base * (median_sd / sd)
        else:
            adjusted[t] = base
    return adjusted


def _momentum_proportional_weights(candidates: list,
                                    adj_scores: dict) -> dict:
    raw   = {t: max(adj_scores.get(t, 0.0), 0.0) for t in candidates}
    total = sum(raw.values())
    if total == 0:
        eq = 1.0 / len(candidates)
        return {t: eq for t in candidates}
    w = {t: v / total for t, v in raw.items()}
    for _ in range(3):
        w = {t: min(max(v, MIN_WEIGHT), MAX_WEIGHT) for t, v in w.items()}
        s = sum(w.values())
        w = {t: v / s for t, v in w.items()}
    return w


def _get_dynamic_sector_caps(candidates: list,
                              momentum_scores: dict) -> dict:
    """Top-2 momentum sectors get MAX_SECTOR_W_TOP; others get MAX_SECTOR_W_BULL."""
    sec_scores = defaultdict(list)
    for t in candidates:
        sec = UNIVERSE_WITH_SECTORS.get(t, "OTHER")
        sec_scores[sec].append(momentum_scores.get(t, 0.0))
    sec_avg = {s: np.mean(v) for s, v in sec_scores.items()}
    top2    = sorted(sec_avg, key=lambda s: sec_avg[s], reverse=True)[:2]
    return {s: (MAX_SECTOR_W_TOP if s in top2 else MAX_SECTOR_W_BULL)
            for s in sec_avg}


def _markowitz_weights(candidates, returns_hist, momentum_scores,
                        objective: str, is_bull: bool) -> dict:
    hist = returns_hist[candidates].dropna()
    if len(hist) < 12 or len(candidates) < 3:
        eq = 1.0 / len(candidates)
        return {t: eq for t in candidates}

    dyn_caps   = _get_dynamic_sector_caps(candidates, momentum_scores)
    sector_cap = MAX_SECTOR_W_BEAR if not is_bull else MAX_SECTOR_W_BULL

    try:
        S  = risk_models.CovarianceShrinkage(
                hist, returns_data=True, frequency=12).ledoit_wolf()
        mu = pd.Series({t: momentum_scores.get(t, 0.0) for t in candidates})
        ef = EfficientFrontier(mu, S,
                               weight_bounds=(MIN_WEIGHT, MAX_WEIGHT),
                               solver="CLARABEL")

        for sec in set(UNIVERSE_WITH_SECTORS.get(t, "OTHER") for t in candidates):
            cap  = dyn_caps.get(sec, sector_cap) if is_bull else MAX_SECTOR_W_BEAR
            mask = [
                1.0 if UNIVERSE_WITH_SECTORS.get(t, "OTHER") == sec else 0.0
                for t in candidates
            ]
            if sum(mask) > 1:
                ef.add_constraint(
                    lambda w, m=mask, c=cap:
                        sum(w[i] * m[i] for i in range(len(m))) <= c
                )

        if objective == "min_vol":
            ef.min_volatility()
        elif objective == "efficient_risk":
            ef.efficient_risk(target_volatility=TARGET_VOL)
        else:
            ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)

        cleaned = ef.clean_weights(cutoff=MIN_WEIGHT, rounding=4)
        return {t: w for t, w in cleaned.items() if w > 0.0}

    except Exception:
        vols = returns_hist[candidates].std()
        inv  = {t: 1.0 / vols[t] if vols[t] > 0 else 1.0 for t in candidates}
        tot  = sum(inv.values())
        raw  = {t: v / tot for t, v in inv.items()}
        cap  = {t: min(w, MAX_WEIGHT) for t, w in raw.items()}
        tot2 = sum(cap.values())
        return {t: w / tot2 for t, w in cap.items()}


# ============================================================
#  STRATEGY HELPERS
# ============================================================
def _apply_regime_tilt(weights: dict, regime: int) -> dict:
    if regime >= 0:
        return weights
    adj   = {
        t: w * BEAR_DEF_BOOST if UNIVERSE_WITH_SECTORS.get(t, 'OTHER') in DEFENSIVE_SECTORS
           else w * 0.8
        for t, w in weights.items()
    }
    total = sum(adj.values())
    return {t: w / total for t, w in adj.items()} if total > 0 else weights


def _apply_crash_protection(weights: dict, crash_mode: bool) -> dict:
    if crash_mode:
        return {t: w * CRASH_CASH_RATIO for t, w in weights.items()}
    return weights


def _apply_dd_circuit(weights: dict, active: bool) -> dict:
    if not active:
        return weights
    return {t: w * DD_EXPOSURE_SCALE for t, w in weights.items()}


def _apply_spy_anchor(weights: dict, regime: int, spy_available: bool) -> dict:
    """Reserve SPY_ANCHOR_W for SPY itself in bull regime to reduce tracking error."""
    if regime != 1 or not spy_available:
        return weights
    scaled = {t: w * (1.0 - SPY_ANCHOR_W) for t, w in weights.items()}
    scaled['SPY'] = SPY_ANCHOR_W
    return scaled


def _weights_changed_enough(prev: dict, new: dict) -> bool:
    """Return True if rebalance is warranted (entry/exit or drift > REBAL_THRESHOLD)."""
    if set(prev.keys()) != set(new.keys()):
        return True
    return any(
        abs(new.get(t, 0.0) - prev.get(t, 0.0)) > REBAL_THRESHOLD
        for t in set(prev) | set(new)
    )


def _compute_turnover_cost(prev_weights: dict, new_weights: dict) -> float:
    all_t    = set(prev_weights) | set(new_weights)
    turnover = sum(abs(new_weights.get(t, 0.0) - prev_weights.get(t, 0.0)) for t in all_t)
    return turnover * TXN_COST_BPS / 10_000


# ============================================================
#  STRATEGY
# ============================================================
def run_strategy(prices, returns, mom_12_1, mom_6_1,
                 spy_regime=None, spy_monthly_returns=None):
    monthly_returns  = []
    current_weights  = {}
    prev_weights     = {}
    order_book_rows  = []
    weights_history  = {}
    total_turnover   = 0.0
    total_txn_cost   = 0.0
    regime_counts    = {1: 0, 0: 0, -1: 0}
    rebal_skipped    = 0

    equity_value = 1.0
    equity_peak  = 1.0
    dd_circuit_on = False

    spy_in_universe = 'SPY' in returns.columns

    for i in range(len(returns)):
        date = returns.index[i]

        # ── Mark-to-market ─────────────────────────────────────────────
        if current_weights:
            pnl = sum(
                returns[t].iloc[i] * w
                for t, w in current_weights.items()
                if t in returns.columns and not pd.isna(returns[t].iloc[i])
            )
            monthly_returns.append(pnl)
            equity_value *= (1 + pnl)
        else:
            monthly_returns.append(0.0)

        # ── Peak-drawdown circuit ───────────────────────────────────────
        equity_peak = max(equity_peak, equity_value)
        current_dd  = (equity_value - equity_peak) / equity_peak
        if not dd_circuit_on and current_dd < DD_REDUCE_THRESH:
            dd_circuit_on = True
        elif dd_circuit_on and current_dd > DD_RESTORE_THRESH:
            dd_circuit_on = False

        s12 = mom_12_1.iloc[i]
        s6  = mom_6_1.iloc[i]

        # ── Eligibility ────────────────────────────────────────────────
        eligible = {}
        for t in SP500_UNIVERSE:
            if t not in s12.index or t not in s6.index:
                continue
            v12, v6 = s12.get(t), s6.get(t)
            if pd.isna(v12) or pd.isna(v6):
                continue
            if float(v12) > 0.0 and float(v6) > MOM_6_1_VETO:
                eligible[t] = float(v12)

        if len(eligible) < 5:
            eligible = {
                t: float(s12[t])
                for t in SP500_UNIVERSE
                if t in s12.index and not pd.isna(s12.get(t)) and float(s12[t]) > 0.0
            }
        if len(eligible) < 3:
            weights_history[date] = {}
            current_weights       = {}
            prev_weights          = {}
            continue

        avg_momentum = np.mean(list(eligible.values()))
        crash_mode   = avg_momentum < 0

        ranked     = sorted(eligible, key=lambda t: eligible[t], reverse=True)
        candidates = ranked[:CANDIDATE_SIZE]

        hist_start     = max(0, i - COV_LOOKBACK)
        returns_window = returns.iloc[hist_start:i]
        available      = [
            t for t in candidates
            if t in returns_window.columns
            and returns_window[t].notna().sum() >= 12
        ]

        # ── Regime: base from 200-MA/HMM + fast crash override ─────────
        regime = _get_regime_state(spy_regime, date)

        # Fast bear override — if SPY fell > SPY_CRASH_THRESH last month
        if spy_monthly_returns is not None and i > 0:
            try:
                spy_mo = spy_monthly_returns.iloc[i - 1]
                if not pd.isna(spy_mo) and float(spy_mo) < SPY_CRASH_THRESH:
                    regime = -1
            except Exception:
                pass

        regime_counts[regime if regime in regime_counts else 0] += 1

        adj_scores = _downside_adjusted_scores(candidates, eligible, returns_window)

        # ── Regime-conditional construction ────────────────────────────
        if regime == 1:
            new_weights = _momentum_proportional_weights(candidates, adj_scores)
        elif regime == -1:
            if len(available) >= 3:
                new_weights = _markowitz_weights(
                    available, returns_window, eligible,
                    objective="min_vol", is_bull=False
                )
            else:
                top = candidates[:10]
                new_weights = {t: 1.0 / len(top) for t in top}
        else:
            if len(available) >= 3:
                new_weights = _markowitz_weights(
                    available, returns_window, eligible,
                    objective="efficient_risk", is_bull=True
                )
            else:
                top = candidates[:10]
                new_weights = {t: 1.0 / len(top) for t in top}

        new_weights = _apply_regime_tilt(new_weights, regime)
        new_weights = _apply_crash_protection(new_weights, crash_mode)
        new_weights = _apply_dd_circuit(new_weights, dd_circuit_on)
        new_weights = _apply_spy_anchor(new_weights, regime, spy_in_universe)

        # ── Skip rebalance if drift is negligible ──────────────────
        if prev_weights and not _weights_changed_enough(prev_weights, new_weights):
            rebal_skipped += 1
            weights_history[date] = dict(current_weights)
            continue

        # ── Transaction costs ───────────────────────────────────────────
        txn_cost = _compute_turnover_cost(prev_weights, new_weights)
        turnover = sum(
            abs(new_weights.get(t, 0.0) - prev_weights.get(t, 0.0))
            for t in set(prev_weights) | set(new_weights)
        )
        total_turnover += turnover
        total_txn_cost += txn_cost
        if monthly_returns:
            monthly_returns[-1] -= txn_cost

        # ── Order book ──────────────────────────────────────────────────
        prev_set = set(prev_weights)
        new_set  = set(new_weights)
        for t in new_set - prev_set:
            order_book_rows.append({
                "Date": str(date), "Ticker": t,
                "Sector": UNIVERSE_WITH_SECTORS.get(t, "?"),
                "Action": "BUY",
                "Weight_%": round(new_weights[t] * 100, 2),
                "Mom_12_1_%": round(eligible.get(t, 0) * 100, 2),
                "Regime": {1: "Bull", 0: "Neutral", -1: "Bear"}.get(regime, "?"),
                "DD_Circuit": dd_circuit_on,
                "Price": round(float(prices[t].iloc[i]), 2) if t in prices.columns else None,
            })
        for t in prev_set - new_set:
            order_book_rows.append({
                "Date": str(date), "Ticker": t,
                "Sector": UNIVERSE_WITH_SECTORS.get(t, "?"),
                "Action": "SELL",
                "Weight_%": 0.0,
                "Mom_12_1_%": round(eligible.get(t, 0) * 100, 2),
                "Regime": {1: "Bull", 0: "Neutral", -1: "Bear"}.get(regime, "?"),
                "DD_Circuit": dd_circuit_on,
                "Price": round(float(prices[t].iloc[i]), 2) if t in prices.columns else None,
            })
        for t in prev_set & new_set:
            delta = abs(new_weights[t] - prev_weights[t])
            if delta > 0.005:
                action = "ADD" if new_weights[t] > prev_weights[t] else "TRIM"
                order_book_rows.append({
                    "Date": str(date), "Ticker": t,
                    "Sector": UNIVERSE_WITH_SECTORS.get(t, "?"),
                    "Action": action,
                    "Weight_%": round(new_weights[t] * 100, 2),
                    "Mom_12_1_%": round(eligible.get(t, 0) * 100, 2),
                    "Regime": {1: "Bull", 0: "Neutral", -1: "Bear"}.get(regime, "?"),
                    "DD_Circuit": dd_circuit_on,
                    "Price": round(float(prices[t].iloc[i]), 2) if t in prices.columns else None,
                })

        current_weights       = new_weights
        prev_weights          = dict(new_weights)
        weights_history[date] = dict(new_weights)

    strategy_returns = pd.Series(monthly_returns, index=returns.index, name="Monthly Return")
    order_book_df    = pd.DataFrame(order_book_rows)

    total = sum(regime_counts.values()) or 1
    print(f"  Regime: Bull {regime_counts[1]/total*100:.0f}%  "
          f"Neutral {regime_counts[0]/total*100:.0f}%  "
          f"Bear {regime_counts[-1]/total*100:.0f}%")
    print(f"  Total turnover:   {total_turnover:.2f}  (skipped {rebal_skipped} rebalances)")
    print(f"  Total txn costs:  {total_txn_cost * 100:.3f}% of equity")
    return strategy_returns, order_book_df, weights_history


# ============================================================
#  ROLLING RISK METRICS
# ============================================================
def rolling_risk_metrics(strat_ts: pd.Series,
                          spy_ts: pd.Series,
                          window: int = 24) -> pd.DataFrame:
    idx     = strat_ts.index
    sharpes, sortinos, calmars, irs, tes = [], [], [], [], []

    for i in range(len(idx)):
        if i < window:
            sharpes.append(np.nan); sortinos.append(np.nan)
            calmars.append(np.nan); irs.append(np.nan); tes.append(np.nan)
            continue
        s  = strat_ts.iloc[i - window: i]
        b  = spy_ts.reindex(s.index).fillna(0)
        mu = s.mean() * 12
        sv = s.std() * np.sqrt(12)

        sharpes.append((mu - RISK_FREE_RATE) / sv if sv > 0 else np.nan)
        sortinos.append(_sortino(s))
        calmars.append(_calmar(s))
        irs.append(_information_ratio(s, b))
        active = s - b
        tes.append(active.std() * np.sqrt(12))

    return pd.DataFrame({
        "Sharpe":  sharpes,
        "Sortino": sortinos,
        "Calmar":  calmars,
        "IR":      irs,
        "TE":      tes,
    }, index=idx)


# ============================================================
#  DASHBOARD
# ============================================================
def build_dashboard(
    strategy_returns: pd.Series,
    order_book_df:    pd.DataFrame,
    weights_history:  dict,
    spy_returns:      pd.Series,
    metrics:          dict,
):
    strategy_returns = _to_period_index(strategy_returns)
    spy_returns      = _to_period_index(spy_returns)
    spy_aligned = spy_returns.reindex(strategy_returns.index, method='ffill').fillna(0)

    strat_ts       = strategy_returns.copy()
    strat_ts.index = strategy_returns.index.to_timestamp()
    spy_ts         = spy_aligned.copy()
    spy_ts.index   = spy_aligned.index.to_timestamp()

    equity   = (1 + strat_ts).cumprod() * 100_000
    spy_eq   = (1 + spy_ts).cumprod() * 100_000
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max * 100

    df_ret          = strat_ts.to_frame("ret")
    df_ret["year"]  = df_ret.index.year
    df_ret["month"] = df_ret.index.month
    heat_pivot      = df_ret.pivot(index="year", columns="month", values="ret") * 100
    heat_pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                          "Jul","Aug","Sep","Oct","Nov","Dec"]

    roll = rolling_risk_metrics(strat_ts, spy_ts, window=24)

    wh_ts = {
        k.to_timestamp() if isinstance(k, pd.Period) else k: v
        for k, v in weights_history.items()
    }
    wh_df = pd.DataFrame(wh_ts).T.fillna(0) * 100

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "📈 Equity Curve vs SPY",
            "🌊 Underwater Drawdown",
            "📅 Monthly Returns Heatmap (%)",
            "📊 Rolling Sharpe / Sortino / Calmar (24-mo)",
            "🥧 Portfolio Composition Over Time",
            "📋 Risk Summary + Rolling IR vs SPY",
        ),
        row_heights=[0.33, 0.33, 0.34],
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
        specs=[
            [{"type": "xy"},    {"type": "xy"}],
            [{"type": "xy"},    {"type": "xy"}],
            [{"type": "xy"},    {"type": "xy", "secondary_y": True}],
        ],
    )

    # Panel 1 — Equity
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values, name="Strategy",
        line=dict(color="#4C78A8", width=2.5),
        hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra>Strategy</extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=spy_eq.index, y=spy_eq.values, name="SPY",
        line=dict(color="#F58518", width=1.8, dash="dot"),
        hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra>SPY</extra>",
    ), row=1, col=1)
    fig.add_annotation(x=equity.index[-1], y=float(equity.iloc[-1]),
        text=f"  ${float(equity.iloc[-1]):,.0f}",
        showarrow=False, font=dict(color="#4C78A8", size=11), row=1, col=1)
    fig.add_annotation(x=spy_eq.index[-1], y=float(spy_eq.iloc[-1]),
        text=f"  ${float(spy_eq.iloc[-1]):,.0f}",
        showarrow=False, font=dict(color="#F58518", size=11), row=1, col=1)

    # Panel 2 — Drawdown
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        fill="tozeroy", fillcolor="rgba(229,115,115,0.25)",
        line=dict(color="#E45756", width=1.5), name="Drawdown",
        hovertemplate="%{x|%b %Y}<br>%{y:.1f}%<extra>Drawdown</extra>",
    ), row=1, col=2)

    # Panel 3 — Heatmap
    valid = heat_pivot.values[~np.isnan(heat_pivot.values)]
    zmax  = max(abs(valid.max()), abs(valid.min())) if len(valid) else 1.0
    fig.add_trace(go.Heatmap(
        z=heat_pivot.values,
        x=heat_pivot.columns.tolist(),
        y=[str(y) for y in heat_pivot.index.tolist()],
        colorscale=[[0.0,"#c0392b"],[0.5,"#f7f7f7"],[1.0,"#27ae60"]],
        zmid=0, zmin=-zmax, zmax=zmax,
        text=np.round(heat_pivot.values, 1), texttemplate="%{text}",
        textfont=dict(size=9),
        colorbar=dict(len=0.28, y=0.50, thickness=12, title="%"),
        hovertemplate="<b>%{y} %{x}</b><br>%{z:.2f}%<extra></extra>",
        name="Monthly Ret",
    ), row=2, col=1)

    # Panel 4 — Rolling ratios
    colors = {"Sharpe": "#4C78A8", "Sortino": "#72B7B2", "Calmar": "#F58518"}
    for metric, color in colors.items():
        fig.add_trace(go.Scatter(
            x=roll.index, y=roll[metric].values,
            line=dict(color=color, width=1.8), name=metric,
            hovertemplate=f"%{{x|%b %Y}}<br>{metric}: %{{y:.2f}}<extra></extra>",
        ), row=2, col=2)
    fig.add_hline(y=0, line=dict(color="gray",    dash="dot",  width=1), row=2, col=2)
    fig.add_hline(y=1, line=dict(color="#27ae60", dash="dash", width=1), row=2, col=2)

    # Panel 5 — Sector composition
    sector_wh = pd.DataFrame(index=wh_df.index)
    for sec in sorted(set(UNIVERSE_WITH_SECTORS.values())):
        tks = [t for t in wh_df.columns if UNIVERSE_WITH_SECTORS.get(t) == sec]
        if tks:
            sector_wh[sec] = wh_df[tks].sum(axis=1)
    for sec in sector_wh.columns:
        color = SECTOR_COLORS.get(sec, "#aaaaaa")
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fig.add_trace(go.Scatter(
            x=sector_wh.index, y=sector_wh[sec].values,
            stackgroup="one", name=sec,
            line=dict(width=0.5, color=color),
            fillcolor=f"rgba({r},{g},{b},0.7)",
            hovertemplate=f"<b>{sec}</b><br>%{{x|%b %Y}}<br>%{{y:.1f}}%<extra></extra>",
            legendgroup=sec,
        ), row=3, col=1)

    # Panel 6 — Rolling IR + Tracking Error
    ir_colors = ["#27ae60" if v >= 0 else "#c0392b"
                 for v in roll["IR"].fillna(0)]
    fig.add_trace(go.Bar(
        x=roll.index, y=roll["IR"].values,
        marker_color=ir_colors, name="Rolling IR vs SPY",
        hovertemplate="%{x|%b %Y}<br>IR: %{y:.2f}<extra></extra>",
        opacity=0.75,
    ), row=3, col=2)
    fig.add_trace(go.Scatter(
        x=roll.index, y=(roll["TE"] * 100).values,
        line=dict(color="#9D755D", width=1.5, dash="dot"),
        name="Tracking Error (%)",
        hovertemplate="%{x|%b %Y}<br>TE: %{y:.1f}%<extra></extra>",
        yaxis="y6",
    ), row=3, col=2, secondary_y=True)
    fig.add_hline(y=0, line=dict(color="gray", dash="dot", width=1), row=3, col=2)

    # Metrics annotation
    metric_lines = [
        "<b>Risk-Adjusted Summary</b>",
        f"CAGR:           {metrics['CAGR (%)']:.1f}%",
        f"Ann. Vol:        {metrics['Ann. Vol (%)']:.1f}%",
        f"Sharpe:          {metrics['Sharpe']:.2f}",
        f"Sortino:          {metrics['Sortino']:.2f}",
        f"Calmar:          {metrics['Calmar']:.2f}",
        f"Max DD:         {metrics['Max Drawdown (%)']:.1f}%",
        f"Max Recovery: {metrics['Max Recovery (mo)']}mo",
        f"IR vs SPY:       {metrics['Info Ratio vs SPY']:.2f}",
        f"Tracking Err:   {metrics['Tracking Error (%)']:.1f}%",
        f"Win Rate:        {metrics['Win Rate (%)']:.0f}%",
        f"Gain/Pain:       {metrics['Gain/Pain']:.2f}",
    ]
    fig.add_annotation(
        x=0.99, y=0.12,
        xref="paper", yref="paper",
        text="<br>".join(metric_lines),
        align="left", showarrow=False,
        font=dict(size=10, family="monospace"),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#cccccc", borderwidth=1,
        xanchor="right",
    )

    fig.update_layout(
        title=dict(
            text="<b>Markowitz Momentum Portfolio — Risk-Adjusted Dashboard</b>",
            font=dict(size=20), x=0.5,
        ),
        height=1350, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, font=dict(size=10)),
        margin=dict(l=60, r=60, t=80, b=40),
        hovermode="x unified",
        barmode="relative",
    )
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1,
                     tickprefix="$", tickformat=",")
    fig.update_yaxes(title_text="Drawdown (%)",   row=1, col=2)
    fig.update_yaxes(title_text="Return (%)",     row=2, col=1)
    fig.update_yaxes(title_text="Ratio",          row=2, col=2)
    fig.update_yaxes(title_text="Allocation (%)", row=3, col=1)
    fig.update_yaxes(title_text="Info Ratio",     row=3, col=2)
    fig.update_yaxes(title_text="Tracking Error (%)", row=3, col=2, secondary_y=True)

    return fig


# ============================================================
#  MAIN
# ============================================================
def main():
    print("=" * 62)
    print("  Monthly Rebalancing: Risk-Adjusted Optimisation")
    print("=" * 62)
    print(f"\n  Target: Calmar + Sortino + Information Ratio vs SPY")
    print(f"  Construction:")
    print(f"    Bull    → Momentum-proportional (downside-vol penalised)")
    print(f"    Neutral → Markowitz efficient_risk (target vol {TARGET_VOL*100:.0f}%)")
    print(f"    Bear    → Markowitz min_vol")
    print(f"  Candidates: top {CANDIDATE_SIZE}  |  bounds: [{MIN_WEIGHT*100:.0f}%, {MAX_WEIGHT*100:.0f}%]")
    print(f"  Sector cap: bull {MAX_SECTOR_W_BULL*100:.0f}% (top-2 sectors {MAX_SECTOR_W_TOP*100:.0f}%)  /  bear {MAX_SECTOR_W_BEAR*100:.0f}%")
    print(f"  Peak-DD circuit: fires at {DD_REDUCE_THRESH*100:.0f}%"
          f" → {DD_EXPOSURE_SCALE*100:.0f}% exposure"
          f" | restores at {DD_RESTORE_THRESH*100:.0f}%")
    print(f"  Regime: 200-MA dead-band ±{int((REGIME_BULL_BAND-1)*100)}%"
          f"  +  fast bear if SPY 1-mo < {SPY_CRASH_THRESH*100:.0f}%")
    print(f"  SPY anchor: {SPY_ANCHOR_W*100:.0f}% in bull regime")
    print(f"  Rebalance threshold: {REBAL_THRESHOLD*100:.0f}% drift")
    print(f"  Cov lookback: {COV_LOOKBACK} months  |  Txn cost: {TXN_COST_BPS} bps\n")

    # ── SPY benchmark ──────────────────────────────────────────────────
    print("  Fetching SPY benchmark...")
    spy_raw = yf.download("SPY", start=START_DATE, end=END_DATE,
                          interval="1mo", auto_adjust=True, progress=False)
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_price = spy_raw["Close"].squeeze()
    elif "Close" in spy_raw.columns:
        spy_price = spy_raw["Close"]
    else:
        spy_price = spy_raw['Close']
    if isinstance(spy_price, pd.DataFrame):
        spy_price = spy_price.iloc[:, 0]
    spy_rets = np.log(spy_price / spy_price.shift(1)).dropna()
    spy_rets = _to_period_index(spy_rets)
    print(f"  SPY: {len(spy_rets)} monthly bars | mean {spy_rets.mean()*100:.2f}%/mo  ✅")

    # ── SPY monthly returns aligned for fast-bear trigger ──────────────
    spy_monthly_aligned = spy_rets.copy()

    # ── Regime ─────────────────────────────────────────────────────────
    print("  Fetching daily SPY for regime detection...")
    spy_regime = None
    try:
        spy_daily = yf.download("SPY", start=START_DATE, end=END_DATE,
                                interval="1d", auto_adjust=True, progress=False)
        if isinstance(spy_daily.columns, pd.MultiIndex):
            spy_dc = spy_daily["Close"].squeeze()
        elif "Close" in spy_daily.columns:
            spy_dc = spy_daily["Close"]
        else:
            spy_dc = spy_daily['Close']
        if isinstance(spy_dc, pd.DataFrame):
            spy_dc = spy_dc.iloc[:, 0]
        spy_regime = _detect_regime(spy_dc)
    except Exception as e:
        print(f"  ⚠️  Regime detection skipped: {e}")

    # ── Price data ──────────────────────────────────────────────────────
    print("  Fetching price data from store...")
    from data_ingestion.data_store import load_universe_data, update_universe_data
    update_universe_data(SP500_UNIVERSE, start=START_DATE, end=END_DATE, interval='1mo')
    ohlcv   = load_universe_data(SP500_UNIVERSE, interval='1mo')
    prices  = build_prices(ohlcv)
    returns = compute_returns(prices).dropna(how='all')
    prices  = prices.reindex(returns.index)

    # Ensure monthly alignment with PeriodIndex
    if not isinstance(returns.index, pd.PeriodIndex):
        returns.index = pd.PeriodIndex(returns.index, freq='M')
    if not isinstance(prices.index, pd.PeriodIndex):
        prices.index = pd.PeriodIndex(prices.index, freq='M')

    coverage = prices.notna().mean()
    good     = coverage[coverage > 0.80].index
    prices   = prices[good]
    returns  = returns[good]
    print(f"  After quality filter: {len(prices.columns)} tickers retained\n")

    if len(returns) < MIN_BACKTEST_MONTHS:
        print(f"⚠️  Only {len(returns)} months — need ≥ {MIN_BACKTEST_MONTHS}. Exiting.")
        return

    # Align SPY monthly returns to the same PeriodIndex
    spy_mo_aligned = spy_monthly_aligned.reindex(returns.index, method='ffill').fillna(0)

    mom_12_1 = compute_12_1(prices)
    mom_6_1  = compute_6_1(prices)

    print("  Running strategy ...\n")
    strategy_returns, order_book_df, weights_history = run_strategy(
        prices, returns, mom_12_1, mom_6_1,
        spy_regime=spy_regime,
        spy_monthly_returns=spy_mo_aligned,
    )
    strategy_returns = _to_period_index(strategy_returns)

    # ── VBT backtest ────────────────────────────────────────────────────
    strat_for_vbt       = strategy_returns.copy()
    strat_for_vbt.index = strategy_returns.index.to_timestamp()
    strategy_close = (1 + strat_for_vbt).cumprod() * 100
    entries = pd.Series(True,  index=strat_for_vbt.index)
    exits   = pd.Series(False, index=strat_for_vbt.index)
    exits.iloc[-1] = True
    bt = VBTBacktester(
        close=strategy_close, entries=entries, exits=exits,
        freq='30D', init_cash=CASH, commission=COMMISSION,
    )
    bt.full_analysis(n_mc=1000, n_wf_splits=5, n_trials=1)

    # ── Full risk-adjusted metrics ──────────────────────────────────────
    spy_aligned = spy_rets.reindex(strategy_returns.index, method='ffill').fillna(0)
    metrics     = compute_full_metrics(strategy_returns, spy_aligned)
    spy_metrics = compute_full_metrics(spy_aligned, spy_aligned)

    print("\n" + "=" * 62)
    print("  RISK-ADJUSTED METRICS COMPARISON")
    print("=" * 62)
    print(f"  {'Metric':<26} {'Strategy':>12} {'SPY':>10}")
    print(f"  {'-'*26} {'-'*12} {'-'*10}")
    for k in metrics:
        sv     = metrics[k]
        bv     = spy_metrics.get(k, "—")
        sv_str = f"{sv:>12}"
        bv_str = f"{bv:>10}"
        print(f"  {k:<26} {sv_str} {bv_str}")
    print("=" * 62 + "\n")

    # ── Save outputs ────────────────────────────────────────────────────
    ob_path = REPORTS_DIR / "order_book.csv"
    order_book_df.to_csv(ob_path, index=False)
    print(f"  ✅ Order book saved → {ob_path}")

    metrics_path = REPORTS_DIR / "metrics.csv"
    pd.DataFrame([metrics, spy_metrics],
                 index=["Strategy", "SPY"]).to_csv(metrics_path)
    print(f"  ✅ Metrics saved    → {metrics_path}")

    print("\n  Building dashboard...")
    fig = build_dashboard(
        strategy_returns, order_book_df, weights_history, spy_rets, metrics
    )
    dash_path = REPORTS_DIR / "portfolio_dashboard.html"
    fig.write_html(str(dash_path), include_plotlyjs="cdn", full_html=True)
    print(f"  ✅ Dashboard saved  → {dash_path}")
    print("     Open in any browser to explore interactively.\n")


if __name__ == '__main__':
    try:
        main()
    finally:
        try:
            from joblib.externals.loky import get_reusable_executor
            get_reusable_executor().shutdown(wait=True)
        except (ImportError, AttributeError):
            pass
