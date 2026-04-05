"""
Monthly Portfolio Rebalancing
"""

import sys
import os
import multiprocessing
from pathlib import Path

os.environ['PYTHONWARNINGS'] = 'ignore:resource_tracker:UserWarning'

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
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtesting_engine.backtesting import VBTBacktester
from config.settings import CASH, COMMISSION
from portfolio_construction import kpi
from pit_universe import PointInTimeUniverse
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from dataclasses import dataclass, field
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ============================================================
#  CONFIG & METADATA
# ============================================================
@dataclass(frozen=True)
class StrategyConfig:
    risk_free_rate: float = 0.03
    min_backtest_months: int = 24
    cash: float = CASH
    commission: float = 0.001  # 10 bps
    slippage: float = 0.0005   # 5 bps
    order_book_min_delta: float = 0.005
    rebal_freq: int = 3
    min_weight_delta: float = 0.03    # 3.0% Turnover filter
    min_score_count: int = 5
    target_vol: float = 0.15
    vol_spike_threshold: float = 2.0
    n_select: int = 15
    txn_cost: float = commission + slippage


@dataclass(frozen=True)
class UniverseMetadata:
    universe_with_sectors: dict[str, str] = field(default_factory=lambda: {
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
    })
    
    sector_colors: dict[str, str] = field(default_factory=lambda: {
        "IT":"#4C78A8","HC":"#72B7B2","FIN":"#F58518","CD":"#E45756",
        "CS":"#54A24B","IND":"#B279A2","COM":"#FF9DA6","EN":"#9D755D",
        "UT":"#BAB0AC","RE":"#EECA3B","MAT":"#76B7B2","OTHER":"#aaaaaa",
    })

    @property
    def tickers(self) -> list[str]:
        return list(self.universe_with_sectors.keys())


# Default configuration
CONFIG = StrategyConfig()
UNIVERSE = UniverseMetadata()

START_DATE = dt.datetime.today() - dt.timedelta(days=365 * 10)
END_DATE   = dt.datetime.today()

REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# ============================================================
#  HELPERS
# ============================================================
def _to_period_index(s: pd.Series) -> pd.Series:
    if not isinstance(s.index, pd.PeriodIndex):
        s = s.copy()
        s.index = pd.PeriodIndex(s.index, freq='M')
    return s


def _extract_close(raw: pd.DataFrame) -> pd.Series:
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
    return (prices / prices.shift(1) - 1)


def compute_12_1(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute the 12-month momentum (excluding the most recent month)."""
    return (prices.shift(1) / prices.shift(13) - 1)


def select_top_robust(scores: dict, universe: UniverseMetadata, n: int = 15):
    """Select the top N tickers globally to allow for higher conviction and alpha."""
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [t for t, _ in ranked[:n]]


def compute_weights_mvo_constrained(
    selected: list, 
    scores: dict, 
    historical_returns: pd.DataFrame, 
    universe_sectors: dict,
    prev_weights: dict = None,
    txn_cost: float = 0.0015  # 15 bps total friction (commission + slippage)
) -> dict:
    """
    Mean-Variance Optimization with Ledoit-Wolf Shrinkage, Sector Constraints, 
    and L1-Norm Transaction Cost Penalization.
    """
    valid_assets = [t for t in selected if t in historical_returns.columns]
    if len(valid_assets) < 2:
        return {t: 1.0 / len(valid_assets) for t in valid_assets}

    rets = historical_returns[valid_assets]
    mu = np.array([scores[t] for t in valid_assets])
    
    # Align previous weights to the current valid_assets vector
    if prev_weights is None:
        prev_weights = {}
    w_prev_arr = np.array([prev_weights.get(t, 0.0) for t in valid_assets])

    # Normalize previous weights to 1.0 to isolate compositional turnover
    total_prev = np.sum(w_prev_arr)
    if total_prev > 0:
        w_prev_arr = w_prev_arr / total_prev

    # 1. Ledoit-Wolf Covariance Shrinkage
    try:
        lw = LedoitWolf().fit(rets.values)
        cov_matrix = lw.covariance_ * 12
    except ValueError:
        cov_matrix = rets.cov().values * 12

    # 2. Sector Constraints Setup
    sector_indices = {}
    for i, ticker in enumerate(valid_assets):
        sector = universe_sectors.get(ticker, "OTHER")
        if sector not in sector_indices:
            sector_indices[sector] = []
        sector_indices[sector].append(i)

    max_sector_weight = 0.30  
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

    for sector, indices in sector_indices.items():
        constraints.append({
            'type': 'ineq',
            'fun': lambda w, idxs=indices: max_sector_weight - np.sum(w[idxs])
        })

    # 3. Objective Function: Friction-Adjusted Sharpe
    def neg_friction_adjusted_sharpe(w):
        # Base portfolio return
        port_return = w @ mu
        
        # Calculate L1-norm turnover (sum of absolute weight changes)
        turnover_drag = np.sum(np.abs(w - w_prev_arr)) * txn_cost
        
        # Net expected return
        net_return = port_return - turnover_drag
        
        port_vol = np.sqrt(w.T @ cov_matrix @ w)
        if port_vol == 0:
            return 0
            
        return -(net_return / port_vol)

    # 4. Optimization Execution
    init_guess = np.ones(len(valid_assets)) / len(valid_assets)
    bounds = tuple((0.0, 1.0) for _ in range(len(valid_assets)))

    try:
        result = minimize(
            neg_friction_adjusted_sharpe,
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-6, 'disp': False}
        )
        weights = result.x if result.success else init_guess
    except Exception:
        weights = init_guess

    # 5. Clean up and normalize
    weights = np.where(weights < 0.01, 0, weights)
    total_w = np.sum(weights)
    if total_w > 0:
        weights = weights / total_w
    else:
        weights = init_guess

    return dict(zip(valid_assets, weights))


def compute_hw_forecasts(prices: pd.DataFrame, window: int = 24) -> dict[str, float]:
    """
    Generates 1-step-ahead expected returns using Double Exponential Smoothing.
    This creates a forward-looking mu vector for the optimizer.
    """
    forecasts = {}
    
    # Suppress statsmodels convergence warnings to keep logs clean
    warnings.filterwarnings("ignore", category=UserWarning)

    for ticker in prices.columns:
        ts = prices[ticker].dropna()
        if len(ts) < window:
            continue

        # We use Double Exponential Smoothing (Trend, no Seasonality for equities)
        try:
            model = ExponentialSmoothing(
                ts.values[-window:],
                trend='add',
                seasonal=None,
                initialization_method="estimated"
            ).fit(optimized=True)
            
            # Forecast 1 period ahead
            next_price = model.forecast(1)[0]
            current_price = ts.iloc[-1]
            
            # Convert to expected return format
            expected_return = (next_price / current_price) - 1
            forecasts[ticker] = expected_return
            
        except Exception as e:
            # Fallback to 0 expectation if the model fails to converge
            forecasts[ticker] = 0.0 

    return forecasts


def compute_full_metrics(strat: pd.Series, bench: pd.Series, config: StrategyConfig = CONFIG) -> dict:
    """Compute comprehensive risk/return metrics for the strategy vs benchmark."""
    strat = strat.dropna()
    bench_aligned = bench.reindex(strat.index).fillna(0)

    cagr = kpi.cagr_from_returns(strat, periods_per_year=12)
    ann_vol = kpi.volatility(strat, periods_per_year=12)
    sharpe = kpi.sharpe_ratio(strat, risk_free_rate=config.risk_free_rate, periods_per_year=12)
    sortino = kpi.sortino_ratio(strat, risk_free_rate=config.risk_free_rate, periods_per_year=12)
    max_dd = kpi.max_drawdown(strat)
    calmar = kpi.calmar_ratio(strat, periods_per_year=12)
    ir = kpi.information_ratio(strat, bench_aligned, periods_per_year=12)
    gain_pain = kpi.gain_pain_ratio(strat)
    max_recovery = kpi.max_recovery_period(strat)

    active = strat - bench_aligned
    te = kpi.volatility(active, periods_per_year=12)
    
    # Add Alpha, Beta, Correlation
    try:
        st_arr, bh_arr = strat.values, bench_aligned.values
        risk_free_rate = config.risk_free_rate
        cov_mat = np.cov(st_arr, bh_arr)
        beta = cov_mat[0, 1] / np.var(bh_arr) if np.var(bh_arr) > 0 else 0.0
        alpha = (st_arr.mean() - (risk_free_rate + beta * (bh_arr.mean() * 12 - risk_free_rate)))
        corr = float(np.corrcoef(st_arr, bh_arr)[0, 1])
    except Exception:
        alpha, beta, corr = 0.0, 0.0, 0.0

    return {
        "CAGR (%)": round(cagr * 100, 2),
        "Ann. Vol (%)": round(ann_vol * 100, 2),
        "Sharpe": round(sharpe, 3),
        "Sortino": round(sortino, 3),
        "Calmar": round(calmar, 3),
        "Max Drawdown (%)": round(max_dd * 100, 2),
        "Max Recovery (mo)": max_recovery,
        "Alpha (ann.)": round(alpha, 4),
        "Beta": round(beta, 4),
        "Correlation": round(corr, 4),
        "Info Ratio vs SPY": round(ir, 3),
        "Tracking Error (%)": round(te * 100, 2),
        "Win Rate (%)": round((strat > 0).mean() * 100, 1),
        "Gain/Pain": round(gain_pain, 2),
        "Months": len(strat),
    }


def _handle_pnl_and_weights(
    current_weights: dict,
    returns_row: pd.Series
) -> tuple[float, dict]:
    """Calculate PnL for the period and update weights for mark-to-market."""
    pnl = 0.0
    if not current_weights:
        return 0.0, {}

    # Calculate PnL based on returns of held assets
    pnl = sum(
        returns_row[t] * w
        for t, w in current_weights.items()
        if t in returns_row.index and not pd.isna(returns_row[t])
    )
    
    # Update weights (mark-to-market)
    divisor = (1 + pnl)
    if divisor == 0:
        return pnl, {}
        
    updated_weights = {
        t: w * (1 + returns_row[t]) / divisor
        for t, w in current_weights.items()
        if t in returns_row.index and not pd.isna(returns_row[t])
    }
    return pnl, updated_weights


def _get_active_tickers(date, returns_columns, pit_engine) -> list[str]:
    """Determine the valid universe of tickers for a given date."""
    active_tickers = []
    if pit_engine:
        try:
            valid_tickers = pit_engine.get_universe_for_date(date.to_timestamp())
            active_tickers = [t for t in valid_tickers if t in returns_columns]
        except Exception:
            pass
            
    # FAILSAFE: If PiT engine fails or returns too few, default to full universe
    if len(active_tickers) < 50:
        active_tickers = [t for t in returns_columns if t != "SPY"]
    return active_tickers


def _apply_vol_target(raw_weights: dict, hist_rets: pd.DataFrame, target_vol: float) -> dict:
    """Scale portfolio exposure to meet a specific volatility target."""
    if not raw_weights:
        return {}

    w_arr = np.array(list(raw_weights.values()))
    tickers = list(raw_weights.keys())
    
    # Calculate the annualized covariance matrix
    sub_cov = hist_rets[tickers].cov() * 12
    port_variance = w_arr.T @ sub_cov @ w_arr
    port_vol = np.sqrt(port_variance) if port_variance > 0 else 0.001
    
    # Scale exposure (cap at 1.0)
    vol_scalar = target_vol / port_vol
    exposure_cap = min(vol_scalar, 1.0)
    
    return {t: w * exposure_cap for t, w in raw_weights.items()}


def _apply_turnover_filter(prev_weights: dict, new_weights: dict, min_delta: float) -> dict:
    """Filter out small weight changes to reduce unnecessary turnover and costs."""
    all_tickers = set(prev_weights.keys()) | set(new_weights.keys())
    potential_weights = dict(new_weights)
    
    for t in all_tickers:
        old_w = prev_weights.get(t, 0.0)
        new_w = new_weights.get(t, 0.0)
        
        # If weight delta is too small, hold previous weight
        if old_w > 0 and new_w > 0:
            if abs(new_w - old_w) < min_delta:
                potential_weights[t] = old_w
        # Purge dust weights
        elif old_w == 0 and new_w > 0:
            if new_w < 0.01: 
                potential_weights[t] = 0.0
                
    return {t: w for t, w in potential_weights.items() if w >= 0.01}


def run_strategy(prices: pd.DataFrame, mom_12_1: pd.DataFrame, universe: UniverseMetadata, config: StrategyConfig = CONFIG, pit_engine=None):
    """
    Main strategy loop with institutional-grade MVO and Dynamic Volatility Targeting.
    Replaces heuristic regime overlays with mathematical covariance and vol scaling.
    """
    returns = prices.pct_change().dropna(how='all')
    
    current_weights, prev_weights, weights_history = {}, {}, {}
    order_book_rows = []
    
    total_turnover = 0.0
    txn_costs_series = pd.Series(0.0, index=returns.index)

    for i in range(len(returns)):
        date = returns.index[i]
        
        # Skip initial period for forecasting warm-up
        if i < config.min_backtest_months: 
            weights_history[date] = {}
            continue

        # Check rebalance frequency
        if i % config.rebal_freq != 0:
            weights_history[date] = dict(current_weights)
            continue

        # 1. Asset Selection & Signal Generation
        active_tickers = _get_active_tickers(date, returns.columns, pit_engine)
        window_prices = prices.iloc[max(0, i-config.min_backtest_months):i+1]
        hw_forecasts = compute_hw_forecasts(window_prices[active_tickers], window=config.min_backtest_months)
        
        # Filter for positive expected returns (long-only)
        valid_scores = {k: v for k, v in hw_forecasts.items() if v > 0.0}
        
        if not valid_scores:
            new_weights = {}
        else:
            # 2. Select Top N and Optimize
            selected = select_top_robust(valid_scores, universe, n=config.n_select)
            hist_rets = returns.iloc[max(0, i-12):i]
            
            raw_weights = compute_weights_mvo_constrained(
                selected=selected, 
                scores=valid_scores, 
                historical_returns=hist_rets, 
                universe_sectors=universe.universe_with_sectors,
                prev_weights=prev_weights,
                txn_cost=config.txn_cost
            )

            # 3. Dynamic Volatility Targeting
            new_weights = _apply_vol_target(raw_weights, hist_rets, config.target_vol)

        # 4. Smart Turnover Filtering
        new_weights = _apply_turnover_filter(prev_weights, new_weights, config.min_weight_delta)

        # 5. Rebalance Execution & Logging
        all_tickers = set(prev_weights.keys()) | set(new_weights.keys())
        turnover = sum(abs(new_weights.get(t, 0) - prev_weights.get(t, 0)) for t in all_tickers)
        txn_cost_val = turnover * config.txn_cost
        total_turnover += turnover
        txn_costs_series.loc[date] = txn_cost_val
        
        prev_set, new_set = set(prev_weights), set(new_weights)
        for t in new_set - prev_set:
            order_book_rows.append({"Date": str(date), "Ticker": t, "Action": "BUY", "Weight_%": round(new_weights[t]*100,2), "Price": round(float(prices[t].iloc[i]), 2)})
        for t in prev_set - new_set:
            order_book_rows.append({"Date": str(date), "Ticker": t, "Action": "SELL", "Weight_%": 0.0, "Price": round(float(prices[t].iloc[i]), 2)})
        for t in prev_set & new_set:
            delta = new_weights[t] - prev_weights[t]
            if abs(delta) > config.order_book_min_delta:
                order_book_rows.append({"Date": str(date), "Ticker": t, "Action": "ADD" if delta > 0 else "TRIM", "Weight_%": round(new_weights[t]*100, 2), "Price": round(float(prices[t].iloc[i]), 2)})

        current_weights = dict(new_weights)
        prev_weights = dict(new_weights)
        weights_history[date] = dict(new_weights)

    # --- E. REALISM & PERFORMANCE CALCULATION ---
    weights_df = pd.DataFrame(weights_history).fillna(0.0).T.reindex(returns.index).fillna(0.0)
    
    # Lag weights by T+1 to prevent lookahead bias in returns calculation
    weights_df = weights_df.shift(1).fillna(0.0) 

    logger.info(f"  Target Accuracy Check (T+1 Lagged): {len(weights_df)} periods")
    logger.info(f"  Avg Exposure (%): {(weights_df.sum(axis=1).mean() * 100):.2f}%")
    logger.info(f"  Total strategy turnover: {total_turnover:.2f}")

    # Calculate Synthetic Cash Yield for uninvested capital
    cash_yield_monthly = (1 + config.risk_free_rate) ** (1/12) - 1
    invested_capital = weights_df.sum(axis=1)
    cash_weights = np.clip(1.0 - invested_capital, 0.0, 1.0)
    
    # Final Portfolio Return: Equity returns + Cash Yield - Transaction Costs
    monthly_rets = returns.multiply(weights_df).sum(axis=1) + (cash_weights * cash_yield_monthly) - txn_costs_series.shift(1).fillna(0.0)

    return monthly_rets, pd.DataFrame(order_book_rows), weights_history


# ============================================================
#  ROLLING RISK METRICS
# ============================================================
def rolling_risk_metrics(
    strat_ts: pd.Series,
    spy_ts:   pd.Series,
    window:   int = 24,
    config:   StrategyConfig = CONFIG
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
        sharpes.append(kpi.sharpe_ratio(s,  risk_free_rate=config.risk_free_rate, periods_per_year=12))
        sortinos.append(kpi.sortino_ratio(s, risk_free_rate=config.risk_free_rate, periods_per_year=12))
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

def _resample_spy_to_monthly(spy_daily_close: pd.Series) -> pd.Series:
    """
    Convert daily SPY prices to monthly prices on month-START timestamps.
    vs_benchmark() does pct_change().reindex(self._returns.index).
    When self._returns is monthly (155 rows), benchmark_series must also be
    monthly prices so pct_change() yields monthly returns that reindex cleanly.
    Uses month-start timestamps to match Period('M').to_timestamp() output.
    """
    try:
        monthly = spy_daily_close.resample('ME').last().dropna()
    except ValueError:
        monthly = spy_daily_close.resample('M').last().dropna()
    monthly.index = monthly.index.to_period('M').to_timestamp()
    return monthly


def _add_equity_traces(fig: go.Figure, equity: pd.Series, spy_eq: pd.Series, row: int, col: int) -> None:
    """Add strategy and benchmark equity curves to the dashboard."""
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values, name="Strategy",
        line=dict(color="#4C78A8", width=2.5),
        hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra>Strategy</extra>",
    ), row=row, col=col)
    
    fig.add_trace(go.Scatter(
        x=spy_eq.index, y=spy_eq.values, name="SPY",
        line=dict(color="#F58518", width=1.8, dash="dot"),
        hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra>SPY</extra>",
    ), row=row, col=col)
    
    for series, color in [(equity, "#4C78A8"), (spy_eq, "#F58518")]:
        fig.add_annotation(
            x=series.index[-1], y=float(series.iloc[-1]),
            text=f"  ${float(series.iloc[-1]):,.0f}",
            showarrow=False, font=dict(color=color, size=11),
            row=row, col=col,
        )


def _add_drawdown_trace(fig: go.Figure, drawdown: pd.Series, row: int, col: int) -> None:
    """Add the underwater drawdown area plot to the dashboard."""
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        fill="tozeroy", fillcolor="rgba(229,115,115,0.25)",
        line=dict(color="#E45756", width=1.5), name="Drawdown",
        hovertemplate="%{x|%b %Y}<br>%{y:.1f}%<extra>Drawdown</extra>",
    ), row=row, col=col)


def _add_heatmap_trace(fig: go.Figure, heat_pivot: pd.DataFrame, row: int, col: int) -> None:
    """Add the monthly returns heatmap to the dashboard."""
    valid = heat_pivot.values[~np.isnan(heat_pivot.values)]
    zmax = max(abs(valid.max()), abs(valid.min())) if len(valid) else 1.0
    
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
    ), row=row, col=col)


def _add_rolling_metrics_traces(fig: go.Figure, roll: pd.DataFrame, row: int, col: int) -> None:
    """Add rolling risk/return ratios to the dashboard."""
    for metric, color in [("Sharpe", "#4C78A8"), ("Sortino", "#72B7B2"), ("Calmar", "#F58518")]:
        fig.add_trace(go.Scatter(
            x=roll.index, y=roll[metric].values,
            line=dict(color=color, width=1.8), name=metric,
            hovertemplate=f"%{{x|%b %Y}}<br>{metric}: %{{y:.2f}}<extra></extra>",
        ), row=row, col=col)
    fig.add_hline(y=0, line=dict(color="gray", dash="dot", width=1), row=row, col=col)
    fig.add_hline(y=1, line=dict(color="#27ae60", dash="dash", width=1), row=row, col=col)


def _add_portfolio_composition_traces(
    fig: go.Figure, 
    wh_df: pd.DataFrame, 
    row: int, 
    col: int,
    universe: UniverseMetadata = UNIVERSE
) -> None:
    """Add the stacked sector allocation plot to the dashboard."""
    sector_wh = pd.DataFrame(index=wh_df.index)
    for sec in sorted(set(universe.universe_with_sectors.values())):
        tks = [t for t in wh_df.columns if universe.universe_with_sectors.get(t) == sec]
        if tks:
            sector_wh[sec] = wh_df[tks].sum(axis=1)
            
    for sec in sector_wh.columns:
        color = universe.sector_colors.get(sec, "#aaaaaa")
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fig.add_trace(go.Scatter(
            x=sector_wh.index, y=sector_wh[sec].values,
            stackgroup="one", name=sec,
            line=dict(width=0.5, color=color),
            fillcolor=f"rgba({r},{g},{b},0.7)",
            hovertemplate=f"<b>{sec}</b><br>%{{x|%b %Y}}<br>%{{y:.1f}}%<extra></extra>",
            legendgroup=sec,
        ), row=row, col=col)


def _add_risk_summary_traces(fig: go.Figure, roll: pd.DataFrame, metrics: dict, row: int, col: int) -> None:
    """Add the rolling IR, TE, and a text summary annotation to the dashboard."""
    ir_colors = ["#27ae60" if v >= 0 else "#c0392b" for v in roll["IR"].fillna(0)]
    
    fig.add_trace(go.Bar(
        x=roll.index, y=roll["IR"].values,
        marker_color=ir_colors, name="Rolling IR vs SPY",
        hovertemplate="%{x|%b %Y}<br>IR: %{y:.2f}<extra></extra>",
        opacity=0.75,
    ), row=row, col=col)
    
    fig.add_trace(go.Scatter(
        x=roll.index, y=(roll["TE"] * 100).values,
        line=dict(color="#9D755D", width=1.5, dash="dot"),
        name="Tracking Error (%)",
        hovertemplate="%{x|%b %Y}<br>TE: %{y:.1f}%<extra></extra>",
        yaxis="y6",
    ), row=row, col=col, secondary_y=True)
    
    fig.add_hline(y=0, line=dict(color="gray", dash="dot", width=1), row=row, col=col)

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
        x=0.99, y=0.12, xref="paper", yref="paper",
        text="<br>".join(metric_lines),
        align="left", showarrow=False,
        font=dict(size=10, family="monospace"),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#cccccc", borderwidth=1,
        xanchor="right",
    )


def build_dashboard(
    strategy_returns: pd.Series,
    order_book_df:    pd.DataFrame,
    weights_history:  dict,
    spy_returns:      pd.Series,
    metrics:          dict,
) -> go.Figure:
    """Construct the interactive Plotly dashboard for strategy analysis."""
    # Data Preparation
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

    # Initialize Figure
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
        vertical_spacing=0.10, horizontal_spacing=0.08,
        specs=[
            [{"type": "xy"},  {"type": "xy"}],
            [{"type": "xy"},  {"type": "xy"}],
            [{"type": "xy"},  {"type": "xy", "secondary_y": True}],
        ],
    )

    # Populate Subplots
    _add_equity_traces(fig, equity, spy_eq, row=1, col=1)
    _add_drawdown_trace(fig, drawdown, row=1, col=2)
    _add_heatmap_trace(fig, heat_pivot, row=2, col=1)
    _add_rolling_metrics_traces(fig, roll, row=2, col=2)
    _add_portfolio_composition_traces(fig, wh_df, row=3, col=1)
    _add_risk_summary_traces(fig, roll, metrics, row=3, col=2)

    # Layout Customization
    fig.update_layout(
        title=dict(
            text="<b>Markowitz Momentum Portfolio — Risk-Adjusted Dashboard</b>",
            font=dict(size=20), x=0.5,
        ),
        height=1350, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, font=dict(size=10)),
        margin=dict(l=60, r=60, t=80, b=40),
        hovermode="x unified", barmode="relative",
    )
    
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1, tickprefix="$", tickformat=",")
    fig.update_yaxes(title_text="Drawdown (%)",       row=1, col=2)
    fig.update_yaxes(title_text="Return (%)",         row=2, col=1)
    fig.update_yaxes(title_text="Ratio",              row=2, col=2)
    fig.update_yaxes(title_text="Allocation (%)",     row=3, col=1)
    fig.update_yaxes(title_text="Info Ratio",         row=3, col=2)
    fig.update_yaxes(title_text="Tracking Error (%)", row=3, col=2, secondary_y=True)

    return fig


def _print_risk_metrics_comparison(metrics: dict, spy_metrics: dict) -> None:
    """Print a formatted comparison table of risk metrics."""
    print("\n" + "=" * 62)
    print("  RISK-ADJUSTED METRICS COMPARISON")
    print("=" * 62)
    print(f"  {'Metric':<26} {'Strategy':>12} {'SPY':>10}")
    print(f"  {'-'*26} {'-'*12} {'-'*10}")
    for k in metrics:
        print(f"  {k:<26} {str(metrics[k]):>12} {str(spy_metrics.get(k, '—')):>10}")
    print("=" * 62 + "\n")


def _save_outputs(order_book_df: pd.DataFrame, metrics: dict, spy_metrics: dict) -> None:
    """Save strategy outputs to CSV files."""
    ob_path = REPORTS_DIR / "order_book.csv"
    order_book_df.to_csv(ob_path, index=False)
    logger.info(f"  ✅ Order book saved → {ob_path}")

    metrics_path = REPORTS_DIR / "metrics.csv"
    pd.DataFrame([metrics, spy_metrics], index=["Strategy", "SPY"]).to_csv(metrics_path)
    logger.info(f"  ✅ Metrics saved    → {metrics_path}")


def _load_market_data(config: StrategyConfig, start_date: dt.datetime, end_date: dt.datetime):
    """Fetch benchmark and universe data, handling caching and alignment."""
    # 1. Benchmark Data
    spy_start = end_date - dt.timedelta(days=365 * 14)
    spy_raw = yf.download("SPY", start=spy_start, end=end_date, interval="1mo", auto_adjust=True, progress=False)
    spy_price = _extract_close(spy_raw)
    spy_price = _to_period_index(spy_price).groupby(level=0).last()
    spy_rets = spy_price.pct_change().dropna()
    
    # 2. Universe Data (Cache-Aware)
    from data_ingestion.data_store import load_universe_data, update_universe_data, DATA_DIR
    import pickle
    
    CACHE_FILE = DATA_DIR / "pit_cache.pkl"
    pit_engine = None
    master_tickers_list = []
    
    if CACHE_FILE.exists():
        logger.info(f"  📂 PiT cache found. Skipping download phase...")
        with open(CACHE_FILE, 'rb') as f:
            cache_data = pickle.load(f)
            pit_engine = cache_data.get('pit_engine')
            master_tickers_list = cache_data.get('master_tickers_list')
            cache_ts = cache_data.get('timestamp')
            logger.info(f"  PiT metadata loaded from {cache_ts.strftime('%Y-%m-%d %H:%M')}")
    else:
        logger.info("  ⚠️  PiT cache not found. Re-scraping Wikipedia...")
        from pit_universe import PointInTimeUniverse
        pit_engine = PointInTimeUniverse()
        master_tickers = set(pit_engine.current_sp500)
        for t in pit_engine.changes_df['Removed_Ticker']:
            if str(t) != 'nan':
                master_tickers.add(t)
        master_tickers_list = list(master_tickers)
        update_universe_data(master_tickers_list, start=start_date, end=end_date, interval='1mo')
    
    ohlcv = load_universe_data(master_tickers_list, interval='1mo')
    prices = build_prices(ohlcv)
    prices.index = pd.PeriodIndex(prices.index, freq='M')
    prices = prices.groupby(level=0).last()
    
    returns = compute_returns(prices).dropna(how='all')
    returns.index = pd.PeriodIndex(returns.index, freq='M')
    returns = returns.groupby(level=0).last()
    prices = prices.reindex(returns.index)

    # Filter for data quality
    good_tickers = prices.notna().mean()[lambda x: x > 0.80].index
    prices, returns = prices[good_tickers], returns[good_tickers]
    
    # Ensure SPY is in prices for alignment
    if "SPY" not in prices.columns:
        spy_aligned = spy_price.reindex(prices.index, method='ffill').fillna(0.0)
        prices["SPY"] = spy_aligned

    return spy_price, spy_rets, prices, returns, pit_engine


def _run_vbt_diagnostics(weights_history: dict, prices: pd.DataFrame, config: StrategyConfig):
    """Run comprehensive VectorBT diagnostics and stress tests."""
    logger.info("\n  Running Portfolio Backtest (T+1 Lagged & 15bps Cost) ...")
    
    if not weights_history:
        logger.warning("⚠️  Weights history is empty.")
        return
        
    weights_df = pd.DataFrame.from_dict(weights_history, orient="index").sort_index()
    weights_df = weights_df.reindex(columns=prices.columns).fillna(0.0)
    
    if not isinstance(weights_df.index, pd.PeriodIndex):
        weights_df = weights_df[~weights_df.index.duplicated(keep='last')]
        weights_df.index = pd.PeriodIndex(weights_df.index, freq='M')
        
    weights_df = weights_df.reindex(prices.index, method="ffill").fillna(0.0)
    weights_df.index = pd.to_datetime([d.to_timestamp() if hasattr(d, 'to_timestamp') else d for d in weights_df.index])
    
    vbt_prices = prices.copy()
    vbt_prices.index = pd.to_datetime([d.to_timestamp() if hasattr(d, 'to_timestamp') else d for d in vbt_prices.index])

    bt = VBTBacktester(close=vbt_prices, freq='30D', init_cash=config.cash, commission=config.commission)
    bt.run_from_weights(weights_df)
    
    # Analyze
    bt._returns = bt._portfolio.returns(group_by=True)
    if hasattr(bt._returns, 'columns'): 
        bt._returns = bt._returns.iloc[:, 0]
    
    bt.monte_carlo(n_simulations=1000, print_report=True)
    bt.walk_forward(n_splits=5, print_report=True)
    bt.stress_testing(print_report=True)
    bt.deflated_sharpe(n_trials=1, print_report=True)
    bt.trade_analysis(print_report=True)
    bt.regime_analysis(print_report=True)
    bt.risk_metrics(print_report=True)
    bt.kelly_sizing(print_report=True)


def main(config: StrategyConfig = CONFIG, universe: UniverseMetadata = UNIVERSE) -> None:
    """Main execution flow for the rebalancing strategy."""
    logger.info("=" * 62)
    logger.info("  Monthly Rebalancing: Risk-Adjusted Optimisation")
    logger.info("=" * 62)
    logger.info(f"  Min backtest months: {config.min_backtest_months}")
    logger.info(f"  Rebalance Freq: {config.rebal_freq} months")
    logger.info(f"  Turnover Threshold: {config.min_weight_delta*100:.0f}%")
    logger.info(f"  Txn cost assumed: 10 bps\n")

    # 1. Fetch Data
    spy_price, spy_rets, prices, returns, pit_engine = _load_market_data(config, START_DATE, END_DATE)

    if len(returns) < config.min_backtest_months:
        logger.warning(f"⚠️  Only {len(returns)} months — need ≥ {config.min_backtest_months}. Exiting.")
        return

    # 2. Strategy Execution
    logger.info("  Running strategy (Institutional Adaptive Engine) ...\n")
    mom_12_1 = compute_12_1(prices)
    strategy_returns, order_book_df, weights_history = run_strategy(
        prices=prices, 
        mom_12_1=mom_12_1, 
        universe=universe, 
        config=config,
        pit_engine=pit_engine
    )

    # 3. VBT Analysis
    _run_vbt_diagnostics(weights_history, prices, config)

    # 4. Final Reporting
    spy_simple_aligned = spy_rets.reindex(strategy_returns.index, method='ffill').fillna(0)
    strat_simple_aligned = strategy_returns.reindex(spy_simple_aligned.index).fillna(0.0)
    
    metrics = compute_full_metrics(strat_simple_aligned, spy_simple_aligned, config)
    spy_metrics = compute_full_metrics(spy_simple_aligned, spy_simple_aligned, config)
    
    _print_risk_metrics_comparison(metrics, spy_metrics)
    _save_outputs(order_book_df, metrics, spy_metrics)

    # Holdout validation
    try:
        ho = strategy_returns.dropna().iloc[-24:]
        if len(ho) >= 6:
            spy_ho = spy_rets.reindex(ho.index, method='ffill').fillna(0)
            ho_m = compute_full_metrics(ho, spy_ho, config)
            logger.info("\n  🔒 Frozen Holdout (last 24 months) Metrics")
            for k, v in ho_m.items(): logger.info(f"    {k}: {v}")
    except Exception: pass

    logger.info("\n  Building dashboard...")
    fig = build_dashboard(strategy_returns, order_book_df, weights_history, spy_rets, metrics)
    dash_path = REPORTS_DIR / "portfolio_dashboard.html"
    fig.write_html(str(dash_path), include_plotlyjs="cdn", full_html=True)
    logger.info(f"  ✅ Dashboard saved  → {dash_path}\n")


if __name__ == '__main__':
    try:
        main()
    finally:
        try:
            from joblib.externals.loky import get_reusable_executor
            get_reusable_executor().shutdown(wait=True)
        except (ImportError, AttributeError):
            pass
