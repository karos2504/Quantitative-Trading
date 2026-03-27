"""
Monthly Portfolio Rebalancing
"""

import sys
import os
import multiprocessing
from pathlib import Path

# Suppress only the noisy resource-tracker warning from joblib subprocesses
os.environ['PYTHONWARNINGS'] = 'ignore:resource_tracker:UserWarning'

if os.name == 'posix':
    try:
        if multiprocessing.get_start_method(allow_none=True) is None:
            multiprocessing.set_start_method('spawn', force=True)
    except (RuntimeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtesting_engine.backtesting import VBTBacktester
from config.settings import CASH, COMMISSION
from portfolio_construction import kpi
from portfolio_construction.weight_allocators import (
    downside_adjusted_scores,
    momentum_proportional_weights,
    risk_parity_momentum_weights,
    markowitz_weights,
)


# ============================================================
#  CONFIG
# ============================================================
RISK_FREE_RATE      = 0.07
TARGET_VOL          = 0.10
CANDIDATE_SIZE      = 10
MIN_WEIGHT          = 0.05
MAX_WEIGHT          = 0.15
MAX_SECTOR_W_BULL   = 0.15
MAX_SECTOR_W_TOP    = 0.15
MAX_SECTOR_W_BEAR   = 0.10
COV_LOOKBACK        = 36

# Momentum
MOM_6_1_VETO        = 0.00

# Regime dead-band (200-MA fallback)
REGIME_BULL_BAND    = 1.05
REGIME_BEAR_BAND    = 0.95

# Crash protection
DEFENSIVE_SECTORS   = {'UT', 'CS', 'HC'}
BEAR_DEF_BOOST      = 1.5
CRASH_CASH_RATIO    = 0.90

# Peak-drawdown circuit
DD_REDUCE_THRESH    = -0.12
DD_RESTORE_THRESH   = -0.06
DD_EXPOSURE_SCALE   = 0.90

# Downside-vol penalty
DOWNSIDE_VOL_MULT   = 3.0

# Rebalance threshold — skip trade if max weight drift < this
REBAL_THRESHOLD     = 0.03

# Fast bear trigger — override regime to bear if SPY fell > this last month
SPY_CRASH_THRESH    = -0.05

# SPY anchor weight in bull regime (only applied when SPY momentum > 0)
SPY_ANCHOR_W        = 0.10

# Minimum absolute weight change to log ADD/TRIM in order book
ORDER_BOOK_MIN_DELTA = 0.005

TXN_COST_BPS        = 10
MIN_BACKTEST_MONTHS = 24

START_DATE = dt.datetime.today() - dt.timedelta(days=365 * 10)
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
    "TRV":"FIN","PNC":"FIN","USB":"FIN","MET":"FIN",
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
    "SPY":"OTHER",
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
    """Ensure Series carries a monthly PeriodIndex."""
    if not isinstance(s.index, pd.PeriodIndex):
        s = s.copy()
        s.index = pd.PeriodIndex(s.index, freq='M')
    return s


def _extract_close(raw: pd.DataFrame) -> pd.Series:
    """
    Safely extract the Close column from a yfinance DataFrame regardless
    of whether it has a MultiIndex (multi-ticker download) or flat columns.
    """
    close = raw["Close"] if not isinstance(raw.columns, pd.MultiIndex) \
            else raw["Close"].squeeze()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close


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
    """12-minus-1 month momentum (skips most recent month)."""
    return np.log(prices.shift(1) / prices.shift(13))


def compute_6_1(prices: pd.DataFrame) -> pd.DataFrame:
    """6-minus-1 month momentum."""
    return np.log(prices.shift(1) / prices.shift(7))


# ============================================================
#  RISK METRICS
# ============================================================
def compute_full_metrics(strat: pd.Series, bench: pd.Series) -> dict:
    strat        = strat.dropna()
    bench_aligned = bench.reindex(strat.index).fillna(0)

    cagr         = kpi.cagr_from_returns(strat, periods_per_year=12)
    ann_vol      = kpi.volatility(strat, periods_per_year=12)
    sharpe       = kpi.sharpe_ratio(strat, risk_free_rate=RISK_FREE_RATE, periods_per_year=12)
    sortino      = kpi.sortino_ratio(strat, risk_free_rate=RISK_FREE_RATE, periods_per_year=12)
    max_dd       = kpi.max_drawdown(strat)
    calmar       = kpi.calmar_ratio(strat, periods_per_year=12)
    ir           = kpi.information_ratio(strat, bench_aligned, periods_per_year=12)
    gain_pain    = kpi.gain_pain_ratio(strat)
    max_recovery = kpi.max_recovery_period(strat)

    active = strat - bench_aligned
    te     = kpi.volatility(active, periods_per_year=12)

    return {
        "CAGR (%)":           round(cagr * 100, 2),
        "Ann. Vol (%)":       round(ann_vol * 100, 2),
        "Sharpe":             round(sharpe, 3),
        "Sortino":            round(sortino, 3),
        "Calmar":             round(calmar, 3),
        "Max Drawdown (%)":   round(max_dd * 100, 2),
        "Max Recovery (mo)":  max_recovery,
        "Info Ratio vs SPY":  round(ir, 3),
        "Tracking Error (%)": round(te * 100, 2),
        "Win Rate (%)":       round((strat > 0).mean() * 100, 1),
        "Gain/Pain":          round(gain_pain, 2),
        "Months":             len(strat),
    }


# ============================================================
#  REGIME DETECTION
# ============================================================
def _detect_regime(spy_prices_daily: pd.Series, vix_prices_daily: pd.Series = None) -> pd.Series:
    """
    Returns a daily Series of {1 = bull, 0 = neutral, -1 = bear}.

    Primary:  3-state Gaussian HMM (State 0 = low vol = bull, already
              guaranteed by fit_hmm_regimes ordering).
    Fallback: 200-MA dead-band (±5%).
    """
    if spy_prices_daily is None or len(spy_prices_daily) < 220:
        return None

    # ── HMM path ───────────────────────────────────────────────────────
    try:
        from portfolio_construction.regime_hmm import fit_hmm_regimes

        # fit_hmm_regimes guarantees State 0 = lowest-vol regime = bull.
        regime_labels = fit_hmm_regimes(spy_prices_daily, vix_prices=vix_prices_daily, n_components=3)
        mapped = regime_labels.map({0: 1, 1: 0, 2: -1})
        bull_pct = (mapped == 1).mean() * 100
        print(f"  Regime: HMM ✅  bull {bull_pct:.1f}%  bear {100-bull_pct:.1f}%")
        return mapped

    except Exception as e:
        print(f"  ⚠️  HMM failed ({e}). Falling back to 200-MA dead-band.")

    # ── 200-MA dead-band fallback ───────────────────────────────────────
    ma200  = spy_prices_daily.rolling(200).mean()
    regime = pd.Series(0, index=spy_prices_daily.index, dtype=int)
    regime[spy_prices_daily > ma200 * REGIME_BULL_BAND] =  1
    regime[spy_prices_daily < ma200 * REGIME_BEAR_BAND] = -1
    # Neutral (0) → forward-fill last known regime, default to bull at start
    regime = regime.replace(0, np.nan).ffill().fillna(1).astype(int)

    bull_pct = (regime == 1).mean() * 100
    print(f"  Regime (200-MA dead-band): bull {bull_pct:.1f}%  bear {100-bull_pct:.1f}%  ✅")
    return regime


def _get_regime_state(spy_regime: pd.Series | None, date) -> int:
    """Look up the regime label on or before `date`. Returns 1 (bull) if unknown."""
    if spy_regime is None:
        return 1
    lookup = date.to_timestamp() if hasattr(date, 'to_timestamp') else date
    try:
        idx = spy_regime.index.get_indexer([lookup], method='ffill')[0]
        if idx >= 0:
            return int(spy_regime.iloc[idx])
    except Exception:
        pass
    return 1


def _smooth_regime(regime: pd.Series, min_duration: int = 5) -> pd.Series:
    """Suppress regime flips shorter than min_duration days."""
    smoothed = regime.copy()
    state    = regime.iloc[0]
    count    = 0
    pending  = None
    for i, val in enumerate(regime):
        if val == state:
            count  += 1
            pending = None
        else:
            if pending is None:
                pending       = val
                pending_count = 1
            elif val == pending:
                pending_count += 1
                if pending_count >= min_duration:
                    state   = pending
                    pending = None
            else:
                pending       = val
                pending_count = 1
        smoothed.iloc[i] = state
    return smoothed


# ============================================================
#  STRATEGY HELPERS
# ============================================================
def _apply_regime_tilt(weights: dict, regime: int) -> dict:
    """In bear regime: boost defensive sectors, trim cyclicals."""
    if regime >= 0:
        return weights
    adj   = {
        t: w * BEAR_DEF_BOOST
           if UNIVERSE_WITH_SECTORS.get(t, 'OTHER') in DEFENSIVE_SECTORS
           else w * 0.8
        for t, w in weights.items()
    }
    total = sum(adj.values())
    return {t: w / total for t, w in adj.items()} if total > 0 else weights


def _apply_crash_protection(weights: dict, crash_mode: bool) -> dict:
    """Scale all weights down, implicitly raising cash."""
    if not crash_mode:
        return weights
    return {t: w * CRASH_CASH_RATIO for t, w in weights.items()}


def _apply_dd_circuit(weights: dict, active: bool) -> dict:
    """Reduce overall exposure when peak-drawdown circuit is tripped."""
    if not active:
        return weights
    return {t: w * DD_EXPOSURE_SCALE for t, w in weights.items()}


def _apply_spy_anchor(
    weights: dict,
    regime: int,
    spy_available: bool,
    spy_mom_12_1: float | None,
) -> dict:
    """
    In bull regime only, anchor SPY_ANCHOR_W to SPY — but only when SPY's
    own 12-1 momentum is positive (avoids anchoring a lagging index).
    """
    if regime != 1 or not spy_available:
        return weights
    if spy_mom_12_1 is None or pd.isna(spy_mom_12_1) or spy_mom_12_1 <= 0:
        return weights
    scaled = {t: w * (1.0 - SPY_ANCHOR_W) for t, w in weights.items()}
    scaled['SPY'] = SPY_ANCHOR_W
    return scaled


def _weights_changed_enough(prev: dict, new: dict) -> bool:
    """True if any position drifted beyond REBAL_THRESHOLD."""
    if set(prev.keys()) != set(new.keys()):
        return True
    return any(
        abs(new.get(t, 0.0) - prev.get(t, 0.0)) > REBAL_THRESHOLD
        for t in set(prev) | set(new)
    )


# ============================================================
#  STRATEGY
# ============================================================
def run_strategy(
    prices:              pd.DataFrame,
    returns:             pd.DataFrame,
    mom_12_1:            pd.DataFrame,
    mom_6_1:             pd.DataFrame,
    spy_regime:          pd.Series | None = None,
    spy_monthly_returns: pd.Series | None = None,
) -> tuple[pd.Series, pd.DataFrame, dict]:

    monthly_returns = []
    current_weights: dict = {}
    prev_weights:    dict = {}
    order_book_rows: list = []
    weights_history: dict = {}
    total_turnover  = 0.0
    total_txn_cost  = 0.0
    regime_counts   = {1: 0, 0: 0, -1: 0}
    rebal_skipped   = 0

    equity_value  = 1.0
    equity_peak   = 1.0
    dd_circuit_on = False

    spy_in_universe = 'SPY' in returns.columns

    for i in range(len(returns)):
        date = returns.index[i]

        # ── Mark-to-market ──────────────────────────────────────────────
        if current_weights:
            pnl = sum(
                returns[t].iloc[i] * w
                for t, w in current_weights.items()
                if t in returns.columns and not pd.isna(returns[t].iloc[i])
            )
            # Drift weights based on monthly performance
            current_weights = {
                t: w * (1 + returns[t].iloc[i]) / (1 + pnl)
                for t, w in current_weights.items()
                if (1 + pnl) != 0
            }
            monthly_returns.append(pnl)
            equity_value *= (1 + pnl)
        else:
            monthly_returns.append(0.0)

        # ── Peak-drawdown circuit ────────────────────────────────────────
        equity_peak = max(equity_peak, equity_value)
        current_dd  = (equity_value - equity_peak) / equity_peak
        if not dd_circuit_on and current_dd < DD_REDUCE_THRESH:
            dd_circuit_on = True
        elif dd_circuit_on and current_dd > DD_RESTORE_THRESH:
            dd_circuit_on = False

        s12 = mom_12_1.iloc[i]
        s6  = mom_6_1.iloc[i]

        # ── Crash mode: use SPY's own 12-1 momentum ─────────────────────
        # The eligible-pool average is always positive by construction, so
        # it is not a reliable crash signal. SPY's momentum is.
        spy_12_1: float | None = (
            float(s12['SPY'])
            if 'SPY' in s12.index and not pd.isna(s12.get('SPY'))
            else None
        )
        crash_mode = spy_12_1 is not None and spy_12_1 < 0

        # ── Eligibility ─────────────────────────────────────────────────
        eligible: dict[str, float] = {}
        for t in SP500_UNIVERSE:
            if t not in s12.index or t not in s6.index:
                continue
            v12, v6 = s12.get(t), s6.get(t)
            if pd.isna(v12) or pd.isna(v6):
                continue
            if float(v12) > 0.0 and float(v6) > MOM_6_1_VETO:
                eligible[t] = float(v12)

        # Widen pool if too few pass the dual filter
        if len(eligible) < 5:
            eligible = {
                t: float(s12[t])
                for t in SP500_UNIVERSE
                if t in s12.index
                and not pd.isna(s12.get(t))
                and float(s12[t]) > 0.0
            }
        if len(eligible) < 3:
            weights_history[date] = {}
            current_weights       = {}
            prev_weights          = {}
            continue

        ranked     = sorted(eligible, key=lambda t: eligible[t], reverse=True)
        candidates = ranked[:CANDIDATE_SIZE]

        hist_start     = max(0, i - COV_LOOKBACK)
        returns_window = returns.iloc[hist_start:i]
        available      = [
            t for t in candidates
            if t in returns_window.columns
            and returns_window[t].notna().sum() >= 12
        ]

        # ── Regime: base from HMM/200-MA + fast SPY crash override ──────
        regime = _get_regime_state(spy_regime, date)

        if spy_monthly_returns is not None and i > 0:
            try:
                spy_mo = float(spy_monthly_returns.iloc[i - 1])
                if not pd.isna(spy_mo) and spy_mo < SPY_CRASH_THRESH:
                    regime = -1
            except Exception:
                pass

        # Clamp to valid keys; unknown values count as neutral
        regime_counts[regime if regime in (-1, 0, 1) else 0] += 1

        adj_scores = downside_adjusted_scores(
            candidates, eligible, returns_window, DOWNSIDE_VOL_MULT
        )

        # ── Regime-conditional portfolio construction ────────────────────
        markowitz_kwargs = dict(
            universe_with_sectors=UNIVERSE_WITH_SECTORS,
            min_weight=MIN_WEIGHT,
            max_weight=MAX_WEIGHT,
            max_sector_w_top=MAX_SECTOR_W_TOP,
            max_sector_w_bull=MAX_SECTOR_W_BULL,
            max_sector_w_bear=MAX_SECTOR_W_BEAR,
            target_vol=TARGET_VOL,
            risk_free_rate=RISK_FREE_RATE,
        )

        if regime == 1:
            # Bull: Risk-Parity Momentum (vol-adjusted for better Sharpe)
            new_weights = risk_parity_momentum_weights(
                candidates, adj_scores, returns_window, MIN_WEIGHT, MAX_WEIGHT
            )

        elif regime == -1:
            # Bear: minimum-variance Markowitz
            if len(available) >= 3:
                new_weights = markowitz_weights(
                    available, returns_window, eligible,
                    objective="min_vol", is_bull=False,
                    **markowitz_kwargs,
                )
            else:
                top = candidates[:10]
                new_weights = {t: 1.0 / len(top) for t in top}

        else:
            # Neutral (regime == 0): efficient-risk Markowitz
            if len(available) >= 3:
                new_weights = markowitz_weights(
                    available, returns_window, eligible,
                    objective="efficient_risk", is_bull=True,
                    **markowitz_kwargs,
                )
            else:
                top = candidates[:10]
                new_weights = {t: 1.0 / len(top) for t in top}

        # ── Overlays (order matters) ─────────────────────────────────────
        new_weights = _apply_regime_tilt(new_weights, regime)
        new_weights = _apply_crash_protection(new_weights, crash_mode)
        new_weights = _apply_dd_circuit(new_weights, dd_circuit_on)
        new_weights = _apply_spy_anchor(
            new_weights, regime, spy_in_universe, spy_12_1
        )

        # ── Skip rebalance if drift is negligible ────────────────────────
        if current_weights and not _weights_changed_enough(current_weights, new_weights):
            rebal_skipped += 1
            weights_history[date] = dict(current_weights)
            continue

        # ── Transaction costs ────────────────────────────────────────────
        all_tickers = set(prev_weights) | set(new_weights)
        turnover    = sum(
            abs(new_weights.get(t, 0.0) - prev_weights.get(t, 0.0))
            for t in all_tickers
        )
        txn_cost        = turnover * TXN_COST_BPS / 10_000
        total_turnover += turnover
        total_txn_cost += txn_cost

        # Deduct cost from equity and from this month's already-booked return
        equity_value           *= (1 - txn_cost)
        if monthly_returns:
            monthly_returns[-1] -= txn_cost

        # ── Order book ───────────────────────────────────────────────────
        regime_label = {1: "Bull", 0: "Neutral", -1: "Bear"}.get(regime, "?")
        prev_set, new_set = set(prev_weights), set(new_weights)

        def _ob_row(ticker, action, weight):
            return {
                "Date":        str(date),
                "Ticker":      ticker,
                "Sector":      UNIVERSE_WITH_SECTORS.get(ticker, "?"),
                "Action":      action,
                "Weight_%":    round(weight * 100, 2),
                "Mom_12_1_%":  round(eligible.get(ticker, 0) * 100, 2),
                "Regime":      regime_label,
                "DD_Circuit":  dd_circuit_on,
                "Price":       round(float(prices[ticker].iloc[i]), 2)
                               if ticker in prices.columns else None,
            }

        for t in new_set - prev_set:
            order_book_rows.append(_ob_row(t, "BUY", new_weights[t]))
        for t in prev_set - new_set:
            order_book_rows.append(_ob_row(t, "SELL", 0.0))
        for t in prev_set & new_set:
            delta = new_weights[t] - prev_weights[t]
            if abs(delta) > ORDER_BOOK_MIN_DELTA:
                action = "ADD" if delta > 0 else "TRIM"
                order_book_rows.append(_ob_row(t, action, new_weights[t]))

        current_weights       = new_weights
        prev_weights          = dict(new_weights)
        weights_history[date] = dict(new_weights)

    strategy_returns = pd.Series(
        monthly_returns, index=returns.index, name="Monthly Return"
    )
    order_book_df = pd.DataFrame(order_book_rows)

    total = sum(regime_counts.values()) or 1
    print(
        f"  Regime distribution:  "
        f"Bull {regime_counts[1]/total*100:.0f}%  "
        f"Neutral {regime_counts[0]/total*100:.0f}%  "
        f"Bear {regime_counts[-1]/total*100:.0f}%"
    )
    print(f"  Total turnover:   {total_turnover:.2f}  (skipped {rebal_skipped} rebalances)")
    print(f"  Total txn costs:  {total_txn_cost * 100:.3f}% of equity")
    return strategy_returns, order_book_df, weights_history


# ============================================================
#  ROLLING RISK METRICS
# ============================================================
def rolling_risk_metrics(
    strat_ts: pd.Series,
    spy_ts:   pd.Series,
    window:   int = 24,
) -> pd.DataFrame:
    idx = strat_ts.index
    sharpes, sortinos, calmars, irs, tes = [], [], [], [], []

    for i in range(len(idx)):
        if i < window:
            sharpes.append(np.nan);  sortinos.append(np.nan)
            calmars.append(np.nan);  irs.append(np.nan)
            tes.append(np.nan)
            continue
        s = strat_ts.iloc[i - window: i]
        b = spy_ts.reindex(s.index).fillna(0)
        sharpes.append(kpi.sharpe_ratio(s,  risk_free_rate=RISK_FREE_RATE, periods_per_year=12))
        sortinos.append(kpi.sortino_ratio(s, risk_free_rate=RISK_FREE_RATE, periods_per_year=12))
        calmars.append(kpi.calmar_ratio(s,  periods_per_year=12))
        irs.append(kpi.information_ratio(s, b, periods_per_year=12))
        tes.append(kpi.volatility(s - b, periods_per_year=12))

    return pd.DataFrame(
        {"Sharpe": sharpes, "Sortino": sortinos,
         "Calmar": calmars, "IR": irs, "TE": tes},
        index=idx,
    )


# ============================================================
#  DASHBOARD
# ============================================================
def build_dashboard(
    strategy_returns: pd.Series,
    order_book_df:    pd.DataFrame,
    weights_history:  dict,
    spy_returns:      pd.Series,
    metrics:          dict,
) -> go.Figure:

    strategy_returns = _to_period_index(strategy_returns)
    spy_returns      = _to_period_index(spy_returns)
    spy_aligned      = spy_returns.reindex(strategy_returns.index, method='ffill').fillna(0)

    strat_ts       = strategy_returns.copy()
    strat_ts.index = strategy_returns.index.to_timestamp()
    spy_ts         = spy_aligned.copy()
    spy_ts.index   = spy_aligned.index.to_timestamp()

    equity   = (1 + strat_ts).cumprod() * 100_000
    spy_eq   = (1 + spy_ts).cumprod()   * 100_000
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
            [{"type": "xy"},  {"type": "xy"}],
            [{"type": "xy"},  {"type": "xy"}],
            [{"type": "xy"},  {"type": "xy", "secondary_y": True}],
        ],
    )

    # ── Panel 1: Equity ─────────────────────────────────────────────────
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
    for series, color in [(equity, "#4C78A8"), (spy_eq, "#F58518")]:
        fig.add_annotation(
            x=series.index[-1], y=float(series.iloc[-1]),
            text=f"  ${float(series.iloc[-1]):,.0f}",
            showarrow=False, font=dict(color=color, size=11),
            row=1, col=1,
        )

    # ── Panel 2: Drawdown ────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        fill="tozeroy", fillcolor="rgba(229,115,115,0.25)",
        line=dict(color="#E45756", width=1.5), name="Drawdown",
        hovertemplate="%{x|%b %Y}<br>%{y:.1f}%<extra>Drawdown</extra>",
    ), row=1, col=2)

    # ── Panel 3: Heatmap ─────────────────────────────────────────────────
    valid = heat_pivot.values[~np.isnan(heat_pivot.values)]
    zmax  = max(abs(valid.max()), abs(valid.min())) if len(valid) else 1.0
    fig.add_trace(go.Heatmap(
        z=heat_pivot.values,
        x=heat_pivot.columns.tolist(),
        y=[str(y) for y in heat_pivot.index.tolist()],
        colorscale=[[0.0, "#c0392b"], [0.5, "#f7f7f7"], [1.0, "#27ae60"]],
        zmid=0, zmin=-zmax, zmax=zmax,
        text=np.round(heat_pivot.values, 1), texttemplate="%{text}",
        textfont=dict(size=9),
        colorbar=dict(len=0.28, y=0.50, thickness=12, title="%"),
        hovertemplate="<b>%{y} %{x}</b><br>%{z:.2f}%<extra></extra>",
        name="Monthly Ret",
    ), row=2, col=1)

    # ── Panel 4: Rolling ratios ──────────────────────────────────────────
    for metric, color in [("Sharpe", "#4C78A8"), ("Sortino", "#72B7B2"), ("Calmar", "#F58518")]:
        fig.add_trace(go.Scatter(
            x=roll.index, y=roll[metric].values,
            line=dict(color=color, width=1.8), name=metric,
            hovertemplate=f"%{{x|%b %Y}}<br>{metric}: %{{y:.2f}}<extra></extra>",
        ), row=2, col=2)
    fig.add_hline(y=0, line=dict(color="gray",    dash="dot",  width=1), row=2, col=2)
    fig.add_hline(y=1, line=dict(color="#27ae60", dash="dash", width=1), row=2, col=2)

    # ── Panel 5: Sector composition ──────────────────────────────────────
    sector_wh = pd.DataFrame(index=wh_df.index)
    for sec in sorted(set(UNIVERSE_WITH_SECTORS.values())):
        tks = [t for t in wh_df.columns if UNIVERSE_WITH_SECTORS.get(t) == sec]
        if tks:
            sector_wh[sec] = wh_df[tks].sum(axis=1)
    for sec in sector_wh.columns:
        color    = SECTOR_COLORS.get(sec, "#aaaaaa")
        r, g, b  = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fig.add_trace(go.Scatter(
            x=sector_wh.index, y=sector_wh[sec].values,
            stackgroup="one", name=sec,
            line=dict(width=0.5, color=color),
            fillcolor=f"rgba({r},{g},{b},0.7)",
            hovertemplate=f"<b>{sec}</b><br>%{{x|%b %Y}}<br>%{{y:.1f}}%<extra></extra>",
            legendgroup=sec,
        ), row=3, col=1)

    # ── Panel 6: Rolling IR + Tracking Error ────────────────────────────
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

    # ── Metrics annotation ───────────────────────────────────────────────
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
    fig.update_yaxes(title_text="Drawdown (%)",       row=1, col=2)
    fig.update_yaxes(title_text="Return (%)",         row=2, col=1)
    fig.update_yaxes(title_text="Ratio",              row=2, col=2)
    fig.update_yaxes(title_text="Allocation (%)",     row=3, col=1)
    fig.update_yaxes(title_text="Info Ratio",         row=3, col=2)
    fig.update_yaxes(title_text="Tracking Error (%)", row=3, col=2, secondary_y=True)

    return fig


# ============================================================
#  MAIN
# ============================================================
def main() -> None:
    print("=" * 62)
    print("  Monthly Rebalancing: Risk-Adjusted Optimisation")
    print("=" * 62)
    print(f"\n  Target: Calmar + Sortino + Information Ratio vs SPY")
    print(f"  Construction:")
    print(f"    Bull    → Momentum-proportional (downside-vol penalised)")
    print(f"    Neutral → Markowitz efficient_risk (target vol {TARGET_VOL*100:.0f}%)")
    print(f"    Bear    → Markowitz min_vol")
    print(f"  Candidates: top {CANDIDATE_SIZE}  |  bounds: [{MIN_WEIGHT*100:.0f}%, {MAX_WEIGHT*100:.0f}%]")
    print(f"  Sector cap: bull {MAX_SECTOR_W_BULL*100:.0f}% "
          f"(top-2 sectors {MAX_SECTOR_W_TOP*100:.0f}%)  /  bear {MAX_SECTOR_W_BEAR*100:.0f}%")
    print(f"  Peak-DD circuit: fires at {DD_REDUCE_THRESH*100:.0f}%"
          f" → {DD_EXPOSURE_SCALE*100:.0f}% exposure"
          f" | restores at {DD_RESTORE_THRESH*100:.0f}%")
    print(f"  Regime: 200-MA dead-band ±{int((REGIME_BULL_BAND-1)*100)}%"
          f"  +  fast bear if SPY 1-mo < {SPY_CRASH_THRESH*100:.0f}%")
    print(f"  SPY anchor: {SPY_ANCHOR_W*100:.0f}% in bull regime (when SPY momentum > 0)")
    print(f"  Rebalance threshold: {REBAL_THRESHOLD*100:.0f}% drift")
    print(f"  Cov lookback: {COV_LOOKBACK} months  |  Txn cost: {TXN_COST_BPS} bps\n")

    # ── SPY benchmark ────────────────────────────────────────────────────
    print("  Fetching SPY benchmark...")
    spy_raw    = yf.download("SPY", start=START_DATE, end=END_DATE,
                             interval="1mo", auto_adjust=True, progress=False)
    spy_price  = _extract_close(spy_raw)
    spy_rets   = _to_period_index(np.log(spy_price / spy_price.shift(1)).dropna())
    print(f"  SPY: {len(spy_rets)} monthly bars | mean {spy_rets.mean()*100:.2f}%/mo  ✅")

    # ── Regime detection (daily SPY) ─────────────────────────────────────
    print("  Fetching daily SPY for regime detection...")
    spy_regime = None
    try:
        spy_daily_raw = yf.download("SPY", start=START_DATE, end=END_DATE,
                                    interval="1d", auto_adjust=True, progress=False)
        spy_daily_close = _extract_close(spy_daily_raw).dropna()                            
        
        print("  Fetching daily VIX for regime detection...")
        vix_raw = yf.download("^VIX", start=START_DATE, end=END_DATE,
                              interval="1d", auto_adjust=True, progress=False)
        vix_close = _extract_close(vix_raw).dropna()
        
        spy_regime = _detect_regime(spy_daily_close, vix_prices_daily=vix_close)
    except Exception as e:
        print(f"  ⚠️  Regime detection skipped: {e}")

    # ── Price data ───────────────────────────────────────────────────────
    print("  Fetching price data from store...")
    from data_ingestion.data_store import load_universe_data, update_universe_data

    update_universe_data(SP500_UNIVERSE, start=START_DATE, end=END_DATE, interval='1mo')
    ohlcv   = load_universe_data(SP500_UNIVERSE, interval='1mo')
    prices  = build_prices(ohlcv)
    returns = compute_returns(prices).dropna(how='all')
    prices  = prices.reindex(returns.index)

    if not isinstance(returns.index, pd.PeriodIndex):
        returns.index = pd.PeriodIndex(returns.index, freq='M')
    if not isinstance(prices.index, pd.PeriodIndex):
        prices.index = pd.PeriodIndex(prices.index, freq='M')

    # Quality filter: keep tickers with ≥ 80% price coverage
    coverage = prices.notna().mean()
    good     = coverage[coverage > 0.80].index
    prices   = prices[good]
    returns  = returns[good]
    print(f"  After quality filter: {len(prices.columns)} tickers retained\n")

    if len(returns) < MIN_BACKTEST_MONTHS:
        print(f"⚠️  Only {len(returns)} months — need ≥ {MIN_BACKTEST_MONTHS}. Exiting.")
        return

    spy_mo_aligned = spy_rets.reindex(returns.index, method='ffill').fillna(0)
    mom_12_1       = compute_12_1(prices)
    mom_6_1        = compute_6_1(prices)

    # ── Run strategy ──────────────────────────────────────────────────────
    print("  Running strategy ...\n")
    strategy_returns, order_book_df, weights_history = run_strategy(
        prices, returns, mom_12_1, mom_6_1,
        spy_regime=spy_regime,
        spy_monthly_returns=spy_mo_aligned,
    )
    strategy_returns = _to_period_index(strategy_returns)

    # ── VBT backtest ──────────────────────────────────────────────────────
    # weights_df is constructed from the dict and aligned to the full price index
    weights_df = pd.DataFrame.from_dict(weights_history, orient="index").sort_index()
    weights_df = weights_df.reindex(prices.index, method="ffill").fillna(0.0)
    
    weights_df.index = pd.to_datetime([d.to_timestamp() if hasattr(d, 'to_timestamp') else d for d in weights_df.index])
    
    # Ensure prices matches weights index for VBT
    vbt_prices = prices.copy()
    vbt_prices.index = pd.to_datetime([d.to_timestamp() if hasattr(d, 'to_timestamp') else d for d in vbt_prices.index])
    
    bt = VBTBacktester(
        close=vbt_prices,
        freq='30D', init_cash=CASH, commission=COMMISSION,
    )
    bt.run_from_weights(weights_df)
    bt.full_analysis(benchmark_series=spy_daily_close, n_mc=1000, n_wf_splits=5, n_trials=1)

    # ── Risk-adjusted metrics ─────────────────────────────────────────────
    spy_aligned = spy_rets.reindex(strategy_returns.index, method='ffill').fillna(0)
    metrics     = compute_full_metrics(strategy_returns, spy_aligned)
    spy_metrics = compute_full_metrics(spy_aligned, spy_aligned)

    print("\n" + "=" * 62)
    print("  RISK-ADJUSTED METRICS COMPARISON")
    print("=" * 62)
    print(f"  {'Metric':<26} {'Strategy':>12} {'SPY':>10}")
    print(f"  {'-'*26} {'-'*12} {'-'*10}")
    for k in metrics:
        print(f"  {k:<26} {str(metrics[k]):>12} {str(spy_metrics.get(k, '—')):>10}")
    print("=" * 62 + "\n")

    # ── Save outputs ──────────────────────────────────────────────────────
    ob_path = REPORTS_DIR / "order_book.csv"
    order_book_df.to_csv(ob_path, index=False)
    print(f"  ✅ Order book saved → {ob_path}")

    metrics_path = REPORTS_DIR / "metrics.csv"
    pd.DataFrame([metrics, spy_metrics], index=["Strategy", "SPY"]).to_csv(metrics_path)
    print(f"  ✅ Metrics saved    → {metrics_path}")

    print("\n  Building dashboard...")
    fig       = build_dashboard(strategy_returns, order_book_df, weights_history,
                                spy_rets, metrics)
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
