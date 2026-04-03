"""
Monthly Portfolio Rebalancing
"""

import sys
import os
import multiprocessing
from pathlib import Path

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
from scipy.stats import zscore
from pit_universe import PointInTimeUniverse

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
    risk_free_rate: float = 0.07
    min_backtest_months: int = 24
    cash: float = CASH
    commission: float = 0.001  # 10 bps
    slippage: float = 0.0005   # 5 bps
    order_book_min_delta: float = 0.005
    rebal_freq: int = 1
    min_weight_delta: float = 0.02    # 2.0% Turnover filter
    min_score_count: int = 5
    target_vol: float = 0.12
    vol_spike_threshold: float = 2.0


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
    return (prices.shift(1) / prices.shift(13) - 1)


def compute_6_1(prices: pd.DataFrame) -> pd.DataFrame:
    return (prices.shift(1) / prices.shift(7) - 1)


def compute_3_1(prices: pd.DataFrame) -> pd.DataFrame:
    return (prices.shift(1) / prices.shift(4) - 1)


def compute_scores_cs(
    mom_1m: pd.Series,
    mom_3m: pd.Series,
    mom_6m: pd.Series,
    mom_12m: pd.Series,
    vol: pd.Series,
    stability: pd.Series,
    drawdown: pd.Series,
    regime_type: str = "normal",
    power: float = 1.3
) -> dict[str, float]:
    """
    Institutional Alpha Ranking Engine:
    - 30% Trend Consistency (Ratio of positive returns)
    - 20% Volatility Inversion (Signal-to-noise optimization)
    - 15% Drawdown Penalty (Peak-to-current risk)
    - 35% Multi-Horizon Momentum (3m, 6m, 12m blend)
    - Cross-Sectional Z-Score Normalization
    """
    factors = {
        "m1": mom_1m, "m3": mom_3m, "m6": mom_6m, "m12": mom_12m,
        "v": vol, "s": stability, "d": drawdown
    }
    
    # 1. Align across all factors
    valid = None
    for f in factors.values():
        valid = f.dropna().index if valid is None else valid.intersection(f.dropna().index)
    
    if valid is None or valid.empty:
        return {}
    
    # Extract valid slice
    m1, m3, m6, m12 = mom_1m.loc[valid], mom_3m.loc[valid], mom_6m.loc[valid], mom_12m.loc[valid]
    v, s, d = vol.loc[valid], stability.loc[valid], drawdown.loc[valid]
    
    # 2. Factor Normalization (Z-Score)
    def z(series):
        return (series - series.mean()) / (series.std() + 1e-12)
    
    # Risk-Adjusted Momentum Base
    zm = z(0.4 * z(m6) + 0.3 * z(m3) + 0.2 * z(m12) + 0.1 * z(m1))
    
    # Consistency & Quality Factors
    zs = z(s)         # Trend Stability
    zv = z(1.0 / (v + 1e-12)) # Vol Inversion
    zd = z(-d)        # Drawdown Penalty (negative of DD)
    acc = z(m3 - m6)  # Momentum Acceleration (change in trend)
    
    # 3. Composite Alpha Score (The Alpha Engine)
    # Concentration on trend quality and acceleration
    composite = (0.30 * zm) + (0.25 * zs) + (0.15 * zv) + (0.10 * zd) + (0.20 * acc)
    
    # 4. Regime-Aware Macro Overlay
    if regime_type == "high_vol":
        composite = composite - 0.25 * z(v) # Aggressive vol penalty in stress
        
    # 5. Selection Filtering (Handled in loop for tri-mode flexibility)
    if composite.empty:
        return {}
        
    # 6. Apply final convexity power (Z^power)
    # Concentrates weights into best names
    base = composite - composite.min() + 1e-6
    convex_scores = base ** power
    
    return convex_scores.to_dict()


def select_top_robust(scores: dict, universe: UniverseMetadata, n: int = 15):
    """Select the top N tickers globally to allow for higher conviction and alpha."""
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [t for t, _ in ranked[:n]]


def compute_weights(selected: list, scores: dict, temperature: float = 0.5):
    """
    Conviction-based Softmax Weighting.
    Concentrates capital into the highest alpha scores with numeric stability.
    """
    if not selected or not scores:
        return {}
        
    # Extract alpha scores for selected tickers
    scores_arr = np.array([scores.get(t, 0.0) for t in selected])
    
    # Numeric stable softmax with temperature control
    x = scores_arr / temperature
    x = x - np.max(x) # Shift for stability
    w = np.exp(x)
    weights_arr = w / w.sum()

    return dict(zip(selected, weights_arr))


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
        risk_free_rate=config.risk_free_rate
        cov_mat = np.cov(st_arr, bh_arr)
        beta = cov_mat[0, 1] / np.var(bh_arr) if np.var(bh_arr) > 0 else 0.0
        alpha = (st_arr.mean() - risk_free_rate + beta * (bh_arr.mean() - risk_free_rate)) * 12
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


def run_strategy(prices: pd.DataFrame, mom_12_1: pd.DataFrame, universe: UniverseMetadata, config: StrategyConfig = CONFIG, pit_engine=None):
    """
    Main strategy loop with institutional-grade adaptive risk controls.
    - Continuous regime score (inverse-vol weighted)
    - Blended rolling Z-normalization
    - Adaptive EMA smoothing & non-linear risk caps
    - Proportional shock factor (memory-aware)
    - T+1 execution lag and turnover-scaled costs (15 bps)
    """
    # 0. Pre-calculate SPY / Benchmark Stats
    spy_prices = prices['SPY'] if 'SPY' in prices.columns else None
    spy_rets = spy_prices.pct_change() if spy_prices is not None else None
    
    # Adaptive Threshold Components
    spy_vol_3m = spy_rets.rolling(3).std() * np.sqrt(12).astype(float) if spy_rets is not None else None
    if spy_vol_3m is not None:
        # Floor vol at 10th percentile to prevent inverse-vol explosion
        spy_vol_3m = np.maximum(spy_vol_3m, spy_vol_3m.quantile(0.1))
        
    spy_ma_3m = spy_prices.rolling(3).mean() if spy_prices is not None else None
    spy_ma_6m = spy_prices.rolling(6).mean() if spy_prices is not None else None
    
    # 1. Prepare Returns & Signals
    returns = prices.pct_change().dropna(how='all')
    mom_12_1 = mom_12_1.reindex(returns.index)

    current_weights = {}
    prev_weights = {}
    weights_history = {}
    order_book_rows = []
    
    # State tracking for adaptive logic
    prev_regime_strength = 0.5
    prev_shock_factor = 0.0
    regime_score_buffer = []

    total_turnover = 0.0
    txn_costs_series = pd.Series(0.0, index=returns.index)

    for i in range(len(returns)):
        date = returns.index[i]
        
        # --- STRATEGY EXECUTION ---
        # Warmup: need 12 months for long-term stats (M freq)
        if i < 13: 
            weights_history[date] = {}
            continue

        if i % config.rebal_freq != 0:
            weights_history[date] = dict(current_weights)
            continue

        # A. ADAPTIVE REGIME SCORE (Self-Weighting Horizons)
        # Calculate local trends (Months: 1, 3, 12)
        m1 = spy_prices.iloc[i] / spy_prices.iloc[i-1] - 1
        m3 = spy_prices.iloc[i] / spy_prices.iloc[i-3] - 1
        m12 = spy_prices.iloc[i] / spy_prices.iloc[i-12] - 1
        
        # Multi-horizon local vol (risk scaling - Monthly data)
        v1 = spy_rets.iloc[max(0, i-3):i].std() * np.sqrt(12)  # Use at least 3m for 1m vol
        v3 = spy_rets.iloc[max(0, i-6):i].std() * np.sqrt(12)
        v12 = spy_rets.iloc[max(0, i-12):i].std() * np.sqrt(12)
        
        # Inverse-vol weighting (capped at 10x to prevent horizon takeover)
        inv_v = 1.0 / np.maximum([v1, v3, v12], 0.01)
        inv_v = np.clip(inv_v, 0, 10.0)
        h_weights = inv_v / inv_v.sum()
        regime_score = (h_weights[0] * m1) + (h_weights[1] * m3) + (h_weights[2] * m12)
        
        # B. BLENDED Z-NORMALIZATION
        regime_score_buffer.append(regime_score)
        if len(regime_score_buffer) > 12:
            s_buf_1y = np.array(regime_score_buffer[-12:])
            s_buf_3y = np.array(regime_score_buffer[-36:])
            
            # Blended distribution (70% 1y / 30% 3y)
            mean_blended = 0.7 * s_buf_1y.mean() + 0.3 * s_buf_3y.mean()
            std_blended  = 0.7 * s_buf_1y.std() + 0.3 * s_buf_3y.std()
            z_score = (regime_score - mean_blended) / (std_blended + 1e-12)
            new_regime_strength = np.clip((z_score + 1) / 2, 0, 1)
        else:
            new_regime_strength = 0.5 # Neutral during early buffer buildup
            
        # C. RECOVERY BOOST (Trend Confirmation)
        if spy_prices.iloc[i] > spy_ma_3m.iloc[i]:
            new_regime_strength = max(new_regime_strength, 0.5)

        # D. ADAPTIVE EMA SMOOTHING (Fixed responsiveness 0.3)
        alpha = 0.3
        regime_strength = (1 - alpha) * prev_regime_strength + alpha * new_regime_strength
        prev_regime_strength = regime_strength

        # E. CONTINUOUS SHOCK FACTOR (Relaxed suppression max 50%)
        # Intensity scaled by standard deviations (adjusted for monthly)
        s1 = m1 / (-2.5 * (v1/np.sqrt(1) + 1e-12)) # Monthly shock
        s3 = m3 / (-3.0 * (v3/np.sqrt(1) + 1e-12))
        raw_shock = np.clip(max(s1, s3, 0), 0, 1)
        
        # Memory-aware shock recovery
        shock_factor = 0.7 * prev_shock_factor + 0.3 * raw_shock
        prev_shock_factor = shock_factor
        shock_suppression = max(0.5, (1.0 - 0.5 * shock_factor)) # Proportional reduction (50% max cut)

        # F. EXPOSURE & CONVEX SCALING (TRI-MODE REGIME ENGINE)
        base_exposure = 0.75
        # Adaptive base exposure (50% floor + 50% regime)
        regime_cap = base_exposure * (0.5 + 0.5 * regime_strength)
        
        if regime_strength < 0.4:
             # mode 1: DEFENSIVE (Low trend quality, preserve capital)
             n_select = 10
             score_threshold = 0.2
             target_exposure = regime_cap * 0.6
        elif regime_strength < 0.7:
             # mode 2: NEUTRAL (Stable conditions, balanced breadth)
             n_select = 20
             score_threshold = 0.0
             target_exposure = regime_cap
        else:
             # mode 3: AGGRESSIVE (High acceleration, high conviction)
             n_select = 15
             score_threshold = -0.2
             target_exposure = min(regime_cap * 1.2, 1.0) # Capped at 1.0 (No Leverage)

        # Apply shock suppression overlay
        target_exposure = target_exposure * shock_suppression

        # G. ASSET SELECTION (INSTITUTIONAL ALPHA BRAIN)
        vol = returns.iloc[max(0, i - 12):i].std()
        
        # Trend Consistency (Positive Return Ratio)
        hist_rets = returns.iloc[max(0, i-12):i]
        stability = (hist_rets > 0).sum() / len(hist_rets) if len(hist_rets) > 0 else pd.Series(0, index=returns.columns)
        
        # Max Drawdown Penalty (12-month peak-to-current)
        window_prices = prices.iloc[max(0, i-12):i+1]
        peak = window_prices.max()
        drawdown = (peak - prices.iloc[i]) / (peak + 1e-12)
        
        # Multi-horizon data points
        m1_row  = prices.iloc[i] / prices.iloc[i-1] - 1
        m3_row  = prices.iloc[i] / prices.iloc[i-3] - 1
        m6_row  = prices.iloc[i] / prices.iloc[i-6] - 1
        m12_row = prices.iloc[i] / prices.iloc[i-12] - 1
        
        # --- NEW: POINT-IN-TIME FILTERING ---
        if pit_engine:
            valid_tickers = pit_engine.get_universe_for_date(date.to_timestamp())
            # Intersection of what was officially in the index AND what we actually have data for
            active_tickers = [t for t in valid_tickers if t in returns.columns]
            
            m1_row = m1_row.loc[active_tickers]
            m3_row = m3_row.loc[active_tickers]
            m6_row = m6_row.loc[active_tickers]
            m12_row = m12_row.loc[active_tickers]
            vol = vol.loc[active_tickers]
            stability = stability.loc[active_tickers]
            drawdown = drawdown.loc[active_tickers]
        # ------------------------------------

        regime_type = "high_vol" if spy_vol_3m.iloc[i] > 0.25 else "normal"
        power = 1.2 + (0.3 * regime_strength) # Signal convexity
        
        scores = compute_scores_cs(
            mom_1m=m1_row, 
            mom_3m=m3_row, 
            mom_6m=m6_row,
            mom_12m=m12_row, 
            vol=vol, 
            stability=stability,
            drawdown=drawdown,
            regime_type=regime_type,
            power=power
        )
        
        if not scores:
            new_weights = {} # Only move to cash if no data
        else:
            # 1. First-Stage Filter: Regime-specific score threshold
            valid_scores = {k: v for k, v in scores.items() if v >= score_threshold}
            
            if not valid_scores:
                new_weights = {}
            else:
                # 2. Second-Stage Filter: Top 30% Cutoff (70th Percentile)
                cutoff = np.percentile(list(valid_scores.values()), 70)
                final_scores = {k: v for k, v in valid_scores.items() if v >= cutoff}
                
                # 3. Final Selection (N-selection for breadth)
                selected = select_top_robust(final_scores, universe, n=n_select)
                raw_weights = compute_weights(selected, final_scores, temperature=0.5)
                
                # H. STRICT NORMALIZATION
                total_raw_w = sum(raw_weights.values())
                if total_raw_w > 0:
                    new_weights = {t: (w / total_raw_w) * target_exposure for t, w in raw_weights.items()}
                else:
                    new_weights = {}

        # I. DATA VALIDATION FAIL-SAFE
        nw_array = np.array(list(new_weights.values()))
        if np.isnan(nw_array).any() or np.isinf(nw_array).any():
            logger.error(f"  [{date}] Data Stability Failure: NaN detected. Fallback to cash.")
            new_weights = {}

        # J. TURNOVER FILTER (2% Noise Reduction)
        all_tickers = set(prev_weights.keys()) | set(new_weights.keys())
        potential_weights = dict(new_weights)
        for t in all_tickers:
            old_w = prev_weights.get(t, 0.0)
            new_w = new_weights.get(t, 0.0)
            if abs(new_w - old_w) < config.min_weight_delta:
                potential_weights[t] = old_w
        
        # Base filter for dust weights
        new_weights = {t: w for t, w in potential_weights.items() if w > 0.001}

        # STRICT RE-NORMALIZATION
        current_exposure = sum(new_weights.values())
        # If the turnover filter pushed us over our target exposure, scale everything back down proportionately.
        if current_exposure > target_exposure:
            scale = target_exposure / current_exposure
            new_weights = {t: w * scale for t, w in new_weights.items()}

        # K. REBALANCE LOGGING & COSTING
        # Calculate Turnover-Scaled Costs (10 bps fees + 5 bps slippage = 15 bps)
        turnover = sum(abs(new_weights.get(t, 0) - prev_weights.get(t, 0)) for t in all_tickers)
        txn_cost = turnover * 0.0015
        total_turnover += turnover
        txn_costs_series[date] = txn_cost
        
        # Order Book
        def _get_ob_row(ticker, action, weight):
            sector = universe.universe_with_sectors.get(ticker, "UNKNOWN") # Handled gracefully
            return {
                "Date": str(date), "Ticker": ticker, "Sector": sector,
                "Action": action, "Weight_%": round(weight * 100, 2),
                "Mom_12_1_%": round(mom_12_1.iloc[i].get(ticker, 0) * 100, 2),
                "Price": round(float(prices[ticker].iloc[i]), 2) if ticker in prices.columns else None,
            }
        prev_set, new_set = set(prev_weights), set(new_weights)
        for t in new_set - prev_set:
            order_book_rows.append(_get_ob_row(t, "BUY", new_weights[t]))
        for t in prev_set - new_set:
            order_book_rows.append(_get_ob_row(t, "SELL", 0.0))
        for t in prev_set & new_set:
            delta = new_weights[t] - prev_weights[t]
            if abs(delta) > config.order_book_min_delta:
                order_book_rows.append(_get_ob_row(t, "ADD" if delta > 0 else "TRIM", new_weights[t]))

        current_weights = new_weights
        prev_weights = dict(new_weights)
        weights_history[date] = dict(new_weights)

    # L. EXECUTION REALISM: T+1 SIGNAL LAG
    # Signal at Close(t) -> Trade at Open(t+1). We shift weights by 1 period before backtesting.
    all_dates = returns.index
    weights_df = pd.DataFrame(weights_history).fillna(0.0).T.reindex(all_dates).fillna(0.0)
    weights_df = weights_df.shift(1).fillna(0.0) # Apply the Institutional T+1 Lag

    logger.info(f"  Target Accuracy Check (T+1 Lagged): {len(weights_df)} periods")
    logger.info(f"  Avg Exposure (%): {(weights_df.sum(axis=1).mean() * 100):.2f}%")
    logger.info(f"  Total strategy turnover: {total_turnover:.2f}")
    logger.info(f"  Total strategy costs: {txn_costs_series.sum()*100:.2f}%")

    # M. SYNTHETIC CASH YIELD (Approximating ~3% annual on uninvested capital)
    cash_yield_annual = 0.03
    cash_yield_monthly = (1 + cash_yield_annual) ** (1/12) - 1
    
    # Calculate how much of the portfolio was in cash each month (using lagged weights_df)
    invested_capital = weights_df.sum(axis=1)
    cash_weights = np.clip(1.0 - invested_capital, 0.0, 1.0)
    
    # Generate the return stream for the cash portion
    cash_returns = cash_weights * cash_yield_monthly

    # Re-extract monthly returns for internal stats (accurately subtracting costs per period)
    # Add the cash returns to the equity returns
    monthly_rets = returns.multiply(weights_df).sum(axis=1) + cash_returns - txn_costs_series.shift(1).fillna(0.0)

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


def main(config: StrategyConfig = CONFIG, universe: UniverseMetadata = UNIVERSE) -> None:
    """Main execution flow for the rebalancing strategy."""
    logger.info("=" * 62)
    logger.info("  Monthly Rebalancing: Risk-Adjusted Optimisation")
    logger.info("=" * 62)
    logger.info(f"  Min backtest months: {config.min_backtest_months}")
    logger.info(f"  Rebalance Freq: {config.rebal_freq} months")
    logger.info(f"  Turnover Threshold: {config.min_weight_delta*100:.0f}%")
    logger.info(f"  Txn cost assumed: 10 bps\n")

    # 1. Benchmark Data
    logger.info("  Fetching benchmarks...")
    spy_start = END_DATE - dt.timedelta(days=365 * 14)
    spy_raw = yf.download("SPY", start=spy_start, end=END_DATE, interval="1mo", auto_adjust=True, progress=False)
    spy_price = _extract_close(spy_raw)
    spy_price = _to_period_index(spy_price).groupby(level=0).last()
    
    spy_rets = spy_price.pct_change().dropna()
    
    # 2. Universe Data (Cache-Aware)
    from data_ingestion.data_store import load_universe_data, update_universe_data, DATA_DIR
    import pickle
    
    CACHE_FILE = DATA_DIR / "pit_cache.pkl"
    
    if CACHE_FILE.exists():
        logger.info(f"  📂 PiT cache found. Skipping download phase...")
        with open(CACHE_FILE, 'rb') as f:
            cache_data = pickle.load(f)
            pit_engine = cache_data.get('pit_engine')
            master_tickers_list = cache_data.get('master_tickers_list')
            cache_ts = cache_data.get('timestamp')
            logger.info(f"  PiT metadata loaded from {cache_ts.strftime('%Y-%m-%d %H:%M')}")
    else:
        logger.info("  ⚠️  PiT cache not found. Re-scraping Wikipedia (this may be slow)...")
        pit_engine = PointInTimeUniverse()
        master_tickers = set(pit_engine.current_sp500)
        for t in pit_engine.changes_df['Removed_Ticker']:
            if t != 'nan':
                master_tickers.add(t)
        master_tickers_list = list(master_tickers)
        
        logger.info(f"  Total historical universe size: {len(master_tickers_list)} tickers")
        logger.info("  Fetching universe data...")
        update_universe_data(master_tickers_list, start=START_DATE, end=END_DATE, interval='1mo')
    
    ohlcv = load_universe_data(master_tickers_list, interval='1mo')
    prices = build_prices(ohlcv)
    prices.index = pd.PeriodIndex(prices.index, freq='M')
    prices = prices.groupby(level=0).last()
    
    returns = compute_returns(prices).dropna(how='all')
    
    # Final alignment
    returns.index = pd.PeriodIndex(returns.index, freq='M')
    returns = returns.groupby(level=0).last()
    prices = prices.reindex(returns.index)

    # Filtering
    good_tickers = prices.notna().mean()[lambda x: x > 0.80].index
    prices, returns = prices[good_tickers], returns[good_tickers]

    if len(returns) < config.min_backtest_months:
        logger.warning(f"⚠️  Only {len(returns)} months — need ≥ {config.min_backtest_months}. Exiting.")
        return

    # 3. Strategy Execution
    # Ensure SPY is in prices for regime detection
    if "SPY" not in prices.columns:
        spy_aligned = spy_price.reindex(prices.index, method='ffill').fillna(0.0)
        prices["SPY"] = spy_aligned

    logger.info("  Running strategy (Institutional Adaptive Engine) ...\n")

    mom_12_1 = compute_12_1(prices)

    strategy_returns, order_book_df, weights_history = run_strategy(
        prices=prices, 
        mom_12_1=mom_12_1, 
        universe=universe, 
        config=config,
        pit_engine=pit_engine
    )

    # 4. VBT Backtest (T+1 Lagged Weights)
    logger.info("\n  Running Portfolio Backtest (T+1 Lagged & 15bps Cost) ...")
    if not weights_history:
        logger.warning("⚠️  Weights history is empty. Strategy did not enter any positions.")
        weights_df = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    else:
        weights_df = pd.DataFrame.from_dict(weights_history, orient="index").sort_index()
        # Ensure all universe columns exist
        weights_df = weights_df.reindex(columns=prices.columns).fillna(0.0)
        # Ensure PeriodIndex if created from dict
        if not isinstance(weights_df.index, pd.PeriodIndex):
            try:
                # Deduplicate and ensure period index
                weights_df = weights_df[~weights_df.index.duplicated(keep='last')]
                weights_df.index = pd.PeriodIndex(weights_df.index, freq='M')
            except Exception:
                pass
        weights_df = weights_df.reindex(prices.index, method="ffill").fillna(0.0)
    weights_df.index = pd.to_datetime([d.to_timestamp() if hasattr(d, 'to_timestamp') else d for d in weights_df.index])
    
    vbt_prices = prices.copy()
    vbt_prices.index = pd.to_datetime([d.to_timestamp() if hasattr(d, 'to_timestamp') else d for d in vbt_prices.index])

    bt = VBTBacktester(close=vbt_prices, freq='30D', init_cash=config.cash, commission=config.commission)
    bt.run_from_weights(weights_df)
    
    # Analyze
    bt._returns = bt._portfolio.returns(group_by=True)
    if hasattr(bt._returns, 'columns'): bt._returns = bt._returns.iloc[:, 0]
    
    bt.monte_carlo(n_simulations=1000, print_report=True)
    bt.walk_forward(n_splits=5, print_report=True)
    bt.stress_testing(print_report=True)
    bt.deflated_sharpe(n_trials=1, print_report=True)
    bt.trade_analysis(print_report=True)
    bt.regime_analysis(print_report=True)
    bt.risk_metrics(print_report=True)
    bt.kelly_sizing(print_report=True)

    # 5. Final Reporting
    spy_simple_aligned = spy_rets.reindex(strategy_returns.index, method='ffill').fillna(0)
    # The Strategy output is already simple returns, so we just align it
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
            # Both strategy 'ho' and SPY 'spy_ho' are now simple returns
            ho_m = compute_full_metrics(ho, spy_ho, config)
            logger.info("\n  🔒 Frozen Holdout (last 24 months) Metrics")
            for k, v in ho_m.items(): logger.info(f"    {k}: {v}")
    except Exception: pass

    logger.info("\n  Building dashboard...")
    # SPY rets are already simple returns (pct_change)
    spy_simple = spy_rets
    
    fig = build_dashboard(strategy_returns, order_book_df, weights_history, spy_simple, metrics)
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
