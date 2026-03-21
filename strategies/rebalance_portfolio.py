"""
Monthly Portfolio Rebalancing — v7: Markowitz + Regime + Dashboard
==================================================================
Improvements over v6:
  1. Regime detection — SPY 200-day MA to detect bull/bear regimes.
     In bear regimes, shifts toward defensive sectors.
  2. Momentum crash protection — When composite momentum turns negative,
     reduces position sizes by 50%.
  3. Transaction cost modeling — Tracks rebalance turnover and deducts
     proportional costs for realistic PnL.
  4. SPY benchmark PeriodIndex fix preserved from v6.
"""

import sys
import os
# Suppress resource_tracker warnings across all processes (especially on macOS/Python 3.14)
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
import plotly.express as px
from plotly.subplots import make_subplots

from pypfopt import EfficientFrontier, risk_models
from pypfopt.exceptions import OptimizationError

from data_ingestion.data import fetch_ohlcv_data
from backtesting_engine.backtesting import VBTBacktester
from config.settings import CASH, COMMISSION

# ============================================================
#  CONFIG
# ============================================================
OBJECTIVE       = "max_sharpe"
TARGET_VOL      = 0.12
CANDIDATE_SIZE  = 30
MIN_WEIGHT      = 0.02
MAX_WEIGHT      = 0.15
MAX_SECTOR_W    = 0.25
COV_LOOKBACK    = 36
VOL_LOOKBACK    = 6

# Regime & crash protection
DEFENSIVE_SECTORS  = {'UT', 'CS', 'HC'}
BEAR_DEF_BOOST     = 1.5
CRASH_CASH_RATIO   = 0.50
TXN_COST_BPS       = 10

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
#  HELPERS — format-agnostic index normalisation
# ============================================================
def _to_period_index(s: pd.Series) -> pd.Series:
    """
    Normalise a Series with any monthly timestamp format to
    PeriodIndex(freq='M').  Safe to call multiple times.
    """
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
    return prices.pct_change()

def compute_12_1(prices: pd.DataFrame) -> pd.DataFrame:
    return (prices.shift(1) / prices.shift(13)) - 1.0

def compute_6_1(prices: pd.DataFrame) -> pd.DataFrame:
    return (prices.shift(1) / prices.shift(7)) - 1.0


# ============================================================
#  MARKOWITZ
# ============================================================
def markowitz_weights(candidates, returns_hist, momentum_scores):
    hist = returns_hist[candidates].dropna()
    if len(hist) < 12 or len(candidates) < 3:
        w = 1.0 / len(candidates)
        return {t: w for t in candidates}
    try:
        S  = risk_models.CovarianceShrinkage(
                hist, returns_data=True, frequency=12).ledoit_wolf()
        mu = pd.Series({t: momentum_scores.get(t, 0.0) for t in candidates})
        ef = EfficientFrontier(mu, S,
                               weight_bounds=(MIN_WEIGHT, MAX_WEIGHT),
                               solver="CLARABEL")
        for sec in set(UNIVERSE_WITH_SECTORS.get(t, "OTHER") for t in candidates):
            mask = [1.0 if UNIVERSE_WITH_SECTORS.get(t) == sec else 0.0
                    for t in candidates]
            if sum(mask) > 1:
                ef.add_constraint(
                    lambda w, m=mask: sum(w[i]*m[i] for i in range(len(m))) <= MAX_SECTOR_W)
        if OBJECTIVE == "min_vol":
            ef.min_volatility()
        elif OBJECTIVE == "efficient_risk":
            ef.efficient_risk(target_volatility=TARGET_VOL)
        else:
            ef.max_sharpe(risk_free_rate=0.04)
        cleaned = ef.clean_weights(cutoff=MIN_WEIGHT, rounding=4)
        return {t: w for t, w in cleaned.items() if w > 0.0}
    except Exception:
        vols = returns_hist[candidates].std()
        inv  = {t: 1.0/vols[t] if vols[t] > 0 else 1.0 for t in candidates}
        tot  = sum(inv.values())
        raw  = {t: v/tot for t,v in inv.items()}
        cap  = {t: min(w, MAX_WEIGHT) for t,w in raw.items()}
        tot2 = sum(cap.values())
        return {t: w/tot2 for t,w in cap.items()}


# ============================================================
#  STRATEGY
# ============================================================
def _detect_regime(spy_prices_daily):
    """
    Detect bull/bear regime using Gaussian HMM.
    Returns a Series of booleans (True = bull/low-vol, False = bear/high-vol).
    """
    if spy_prices_daily is None or len(spy_prices_daily) < 50:
        return None
        
    try:
        from portfolio_construction.regime_hmm import fit_hmm_regimes
        regime_labels = fit_hmm_regimes(spy_prices_daily, n_components=2)
        
        # We need to map the hidden state (0 or 1) to Bull (True) or Bear (False).
        # Low volatility is historically associated with Bull markets.
        returns = spy_prices_daily.pct_change()
        vol_0 = returns[regime_labels == 0].std()
        vol_1 = returns[regime_labels == 1].std()
        
        bull_state = 0 if vol_0 < vol_1 else 1
        
        return regime_labels == bull_state
    except Exception as e:
        print(f"  ⚠️ HMM Regime detection failed: {e}. Falling back to 200-MA.")
        ma200 = spy_prices_daily.rolling(200).mean()
        return spy_prices_daily > ma200


def run_strategy(prices, returns, mom_12_1, mom_6_1, spy_regime=None):
    monthly_returns  = []
    current_weights  = {}
    prev_weights     = {}
    order_book_rows  = []
    weights_history  = {}
    total_turnover   = 0.0
    total_txn_cost   = 0.0

    for i in range(len(returns)):
        date = returns.index[i]

        if current_weights:
            pnl = sum(
                returns[t].iloc[i] * w
                for t, w in current_weights.items()
                if t in returns.columns and not pd.isna(returns[t].iloc[i])
            )
            monthly_returns.append(pnl)
        else:
            monthly_returns.append(0.0)

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
            eligible = {
                t: float(s12[t])
                for t in SP500_UNIVERSE
                if t in s12.index and not pd.isna(s12[t])
            }
        if not eligible:
            weights_history[date] = {}
            continue

        # --- Momentum crash protection ---
        avg_momentum = np.mean(list(eligible.values()))
        crash_mode = avg_momentum < 0

        ranked     = sorted(eligible, key=lambda t: eligible[t], reverse=True)
        candidates = ranked[:CANDIDATE_SIZE]

        hist_start     = max(0, i - COV_LOOKBACK)
        returns_window = returns.iloc[hist_start:i]
        available      = [
            t for t in candidates
            if t in returns_window.columns
            and returns_window[t].notna().sum() >= 12
        ]
        if len(available) < 3:
            top = candidates[:10]
            new_weights = {t: 1.0/len(top) for t in top}
        else:
            new_weights = markowitz_weights(available, returns_window, eligible)

        # --- Regime detection: boost defensive sectors in bear ---
        is_bull = True
        if spy_regime is not None:
            try:
                regime_dates = spy_regime.index
                closest_idx = regime_dates.get_indexer([date], method='ffill')[0]
                if closest_idx >= 0:
                    is_bull = bool(spy_regime.iloc[closest_idx])
            except Exception:
                pass

        if not is_bull:
            adjusted = {}
            for t, w in new_weights.items():
                sector = UNIVERSE_WITH_SECTORS.get(t, 'OTHER')
                if sector in DEFENSIVE_SECTORS:
                    adjusted[t] = w * BEAR_DEF_BOOST
                else:
                    adjusted[t] = w * 0.8
            total = sum(adjusted.values())
            new_weights = {t: w / total for t, w in adjusted.items()} if total > 0 else new_weights

        # --- Crash protection: scale down all positions ---
        if crash_mode:
            new_weights = {t: w * CRASH_CASH_RATIO for t, w in new_weights.items()}

        # --- Transaction cost modeling ---
        turnover = 0.0
        for t in set(list(prev_weights.keys()) + list(new_weights.keys())):
            old_w = prev_weights.get(t, 0.0)
            new_w = new_weights.get(t, 0.0)
            turnover += abs(new_w - old_w)
        txn_cost = turnover * TXN_COST_BPS / 10000
        total_turnover += turnover
        total_txn_cost += txn_cost

        # Deduct transaction cost from this period's return
        if monthly_returns:
            monthly_returns[-1] -= txn_cost

        # Log order book
        prev_set = set(prev_weights.keys())
        new_set  = set(new_weights.keys())
        for t in new_set - prev_set:
            order_book_rows.append({
                "Date": str(date), "Ticker": t,
                "Sector": UNIVERSE_WITH_SECTORS.get(t, "?"),
                "Action": "BUY",
                "Weight_%": round(new_weights[t] * 100, 2),
                "Mom_12_1_%": round(eligible.get(t, 0) * 100, 2),
                "Price": round(float(prices[t].iloc[i]), 2) if t in prices.columns else None,
            })
        for t in prev_set - new_set:
            order_book_rows.append({
                "Date": str(date), "Ticker": t,
                "Sector": UNIVERSE_WITH_SECTORS.get(t, "?"),
                "Action": "SELL",
                "Weight_%": 0.0,
                "Mom_12_1_%": round(eligible.get(t, 0) * 100, 2),
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
                    "Price": round(float(prices[t].iloc[i]), 2) if t in prices.columns else None,
                })

        current_weights = new_weights
        prev_weights    = dict(new_weights)
        weights_history[date] = dict(new_weights)

    strategy_returns = pd.Series(monthly_returns, index=returns.index, name="Monthly Return")
    order_book_df    = pd.DataFrame(order_book_rows)

    print(f"  Total turnover: {total_turnover:.2f} | "
          f"Total txn costs: {total_txn_cost * 100:.3f}% of equity")

    return strategy_returns, order_book_df, weights_history


# ============================================================
#  DASHBOARD
# ============================================================
def build_dashboard(
    strategy_returns: pd.Series,
    order_book_df:    pd.DataFrame,
    weights_history:  dict,
    spy_returns:      pd.Series,
):
    # ── FIX: normalise both series to PeriodIndex before any alignment ──
    strategy_returns = _to_period_index(strategy_returns)
    spy_returns      = _to_period_index(spy_returns)

    # Align SPY to strategy index — now guaranteed to match because
    # both use PeriodIndex(freq='M') regardless of original timestamp format
    spy_aligned = spy_returns.reindex(strategy_returns.index, method='ffill').fillna(0)

    # Convert back to timestamps for Plotly (PeriodIndex plots as text otherwise)
    strat_ts = strategy_returns.copy()
    strat_ts.index = strategy_returns.index.to_timestamp()
    spy_ts = spy_aligned.copy()
    spy_ts.index = spy_aligned.index.to_timestamp()

    equity   = (1 + strat_ts).cumprod() * 100_000
    spy_eq   = (1 + spy_ts).cumprod() * 100_000
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max * 100
    roll_std = strat_ts.rolling(12).std()
    roll_sharpe = (
        strat_ts.rolling(12).mean() / roll_std.replace(0, np.nan)
    ) * np.sqrt(12)
    roll_sharpe = roll_sharpe.fillna(0)

    # Monthly returns heatmap
    df_ret = strat_ts.to_frame("ret")
    df_ret["year"]  = df_ret.index.year
    df_ret["month"] = df_ret.index.month
    heat_pivot = df_ret.pivot(index="year", columns="month", values="ret") * 100
    heat_pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                          "Jul","Aug","Sep","Oct","Nov","Dec"]

    # Weights area chart — convert dict keys to timestamps
    wh_ts = {
        k.to_timestamp() if isinstance(k, pd.Period) else k: v
        for k, v in weights_history.items()
    }
    all_tickers = sorted({t for wts in wh_ts.values() for t in wts})
    wh_df = pd.DataFrame(wh_ts).T.fillna(0)
    if all_tickers:
        wh_df = wh_df[[t for t in all_tickers if t in wh_df.columns]] * 100

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "📈 Equity Curve vs SPY",
            "🌊 Underwater Drawdown",
            "📅 Monthly Returns Heatmap (%)",
            "📊 Rolling 12-Month Sharpe",
            "🥧 Portfolio Composition Over Time",
            "📋 Order Book (last 60 events)",
        ),
        row_heights=[0.35, 0.30, 0.35],
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
        specs=[
            [{"type": "xy"},    {"type": "xy"}],
            [{"type": "xy"},    {"type": "xy"}],
            [{"type": "xy"},    {"type": "table"}],
        ],
    )

    # ── Panel 1: Equity curve ─────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        name="Strategy", line=dict(color="#4C78A8", width=2.5),
        hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra>Strategy</extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=spy_eq.index, y=spy_eq.values,
        name="SPY", line=dict(color="#F58518", width=1.8, dash="dot"),
        hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra>SPY</extra>",
    ), row=1, col=1)
    fig.add_annotation(
        x=equity.index[-1], y=float(equity.iloc[-1]),
        text=f"  ${float(equity.iloc[-1]):,.0f}",
        showarrow=False, font=dict(color="#4C78A8", size=11), row=1, col=1,
    )
    fig.add_annotation(
        x=spy_eq.index[-1], y=float(spy_eq.iloc[-1]),
        text=f"  ${float(spy_eq.iloc[-1]):,.0f}",
        showarrow=False, font=dict(color="#F58518", size=11), row=1, col=1,
    )

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        fill="tozeroy", fillcolor="rgba(229,115,115,0.25)",
        line=dict(color="#E45756", width=1.5), name="Drawdown",
        hovertemplate="%{x|%b %Y}<br>%{y:.1f}%<extra>Drawdown</extra>",
    ), row=1, col=2)

    # ── Panel 3: Monthly returns heatmap ─────────────────────────────────
    valid = heat_pivot.values[~np.isnan(heat_pivot.values)]
    zmax  = max(abs(valid.max()), abs(valid.min())) if len(valid) else 1.0
    fig.add_trace(go.Heatmap(
        z=heat_pivot.values,
        x=heat_pivot.columns.tolist(),
        y=[str(y) for y in heat_pivot.index.tolist()],
        colorscale=[[0.0,"#c0392b"],[0.5,"#f7f7f7"],[1.0,"#27ae60"]],
        zmid=0, zmin=-zmax, zmax=zmax,
        text=np.round(heat_pivot.values, 1),
        texttemplate="%{text}",
        textfont=dict(size=9),
        colorbar=dict(len=0.3, y=0.38, thickness=12, title="%"),
        hovertemplate="<b>%{y} %{x}</b><br>%{z:.2f}%<extra></extra>",
        name="Monthly Ret",
    ), row=2, col=1)

    # ── Panel 4: Rolling Sharpe ───────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=roll_sharpe.index, y=roll_sharpe.values,
        line=dict(color="#72B7B2", width=2), name="Rolling Sharpe",
        hovertemplate="%{x|%b %Y}<br>Sharpe: %{y:.2f}<extra></extra>",
    ), row=2, col=2)
    # Reference lines via shapes (avoids duplicate legend entries)
    for y_val, color, dash in [(0, "gray", "dot"), (1, "#27ae60", "dash")]:
        fig.add_hline(y=y_val,
                      line=dict(color=color, dash=dash, width=1),
                      row=2, col=2)

    # ── Panel 5: Sector composition stacked area ──────────────────────────
    sector_wh = pd.DataFrame(index=wh_df.index)
    for sec in sorted(set(UNIVERSE_WITH_SECTORS.values())):
        sec_tickers = [t for t in wh_df.columns if UNIVERSE_WITH_SECTORS.get(t) == sec]
        if sec_tickers:
            sector_wh[sec] = wh_df[sec_tickers].sum(axis=1)

    for sec in sector_wh.columns:
        color = SECTOR_COLORS.get(sec, "#aaaaaa")
        r,g,b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
        fig.add_trace(go.Scatter(
            x=sector_wh.index, y=sector_wh[sec].values,
            stackgroup="one", name=sec,
            line=dict(width=0.5, color=color),
            fillcolor=f"rgba({r},{g},{b},0.7)",
            hovertemplate=f"<b>{sec}</b><br>%{{x|%b %Y}}<br>%{{y:.1f}}%<extra></extra>",
            legendgroup=sec,
        ), row=3, col=1)

    # ── Panel 6: Order book table ─────────────────────────────────────────
    if not order_book_df.empty:
        ob_display = order_book_df.tail(60).iloc[::-1].reset_index(drop=True)
        action_colors = {
            "BUY": "#d4efdf", "SELL": "#fadbd8",
            "ADD": "#d6eaf8", "TRIM": "#fdebd0",
        }
        fill_colors = []
        for col in ob_display.columns:
            if col == "Action":
                fill_colors.append([action_colors.get(a, "#ffffff") for a in ob_display["Action"]])
            else:
                fill_colors.append(["#f9f9f9" if i % 2 == 0 else "#ffffff"
                                     for i in range(len(ob_display))])
        fig.add_trace(go.Table(
            header=dict(
                values=[f"<b>{c}</b>" for c in ob_display.columns],
                fill_color="#2c3e50",
                font=dict(color="white", size=11),
                align="center", height=28,
            ),
            cells=dict(
                values=[ob_display[c].tolist() for c in ob_display.columns],
                fill_color=fill_colors,
                font=dict(size=10),
                align=["center","center","center","center","right","right","right"],
                height=22,
            ),
        ), row=3, col=2)

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text="<b>Markowitz Momentum Portfolio — Full Dashboard</b>",
            font=dict(size=20), x=0.5,
        ),
        height=1300,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, font=dict(size=10)),
        margin=dict(l=60, r=60, t=80, b=40),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1,
                     tickprefix="$", tickformat=",")
    fig.update_yaxes(title_text="Drawdown (%)",  row=1, col=2)
    fig.update_yaxes(title_text="Return (%)",    row=2, col=1)
    fig.update_yaxes(title_text="Sharpe Ratio",  row=2, col=2)
    fig.update_yaxes(title_text="Allocation (%)",row=3, col=1)

    return fig


# ============================================================
#  MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  Monthly Rebalancing — v7: Markowitz + Regime + Dashboard")
    print("=" * 60)
    print(f"\n  Universe:    {len(SP500_UNIVERSE)} S&P 500 stocks")
    print(f"  Optimizer:   Markowitz [{OBJECTIVE}], Ledoit-Wolf shrinkage")
    print(f"  Constraints: [{MIN_WEIGHT*100:.0f}%, {MAX_WEIGHT*100:.0f}%] / stock | "
          f"{MAX_SECTOR_W*100:.0f}% / sector")
    print(f"  Regime:      SPY 200-day MA | Crash protection: {CRASH_CASH_RATIO*100:.0f}% reduction")
    print(f"  Txn cost:    {TXN_COST_BPS} bps per rebalance\n")

    # ── SPY benchmark ─────────────────────────────────────────────────────
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

    spy_rets = spy_price.pct_change().dropna()
    spy_rets = _to_period_index(spy_rets)
    print(f"  SPY: {len(spy_rets)} monthly bars | mean {spy_rets.mean()*100:.2f}%/mo  ✅")

    # ── Regime detection (daily SPY for 200-day MA) ──
    print("  Fetching daily SPY for regime detection...")
    spy_regime = None
    try:
        spy_daily = yf.download("SPY", start=START_DATE, end=END_DATE,
                                interval="1d", auto_adjust=True, progress=False)
        if isinstance(spy_daily.columns, pd.MultiIndex):
            spy_daily_close = spy_daily["Close"].squeeze()
        elif "Close" in spy_daily.columns:
            spy_daily_close = spy_daily["Close"]
        else:
            spy_daily_close = spy_daily['Close']
        if isinstance(spy_daily_close, pd.DataFrame):
            spy_daily_close = spy_daily_close.iloc[:, 0]
        spy_regime = _detect_regime(spy_daily_close)
        if spy_regime is not None:
            bull_pct = spy_regime.mean() * 100
            print(f"  Regime: {bull_pct:.1f}% bull / {100-bull_pct:.1f}% bear  ✅")
    except Exception as e:
        print(f"  ⚠️ Regime detection skipped: {e}")

    # ── Strategy data ─────────────────────────────────────────────────────
    print("  Fetching price data from store...")
    from data_ingestion.data_store import load_universe_data, update_universe_data
    update_universe_data(SP500_UNIVERSE, start=START_DATE, end=END_DATE, interval='1mo')
    ohlcv   = load_universe_data(SP500_UNIVERSE, interval='1mo')
    prices  = build_prices(ohlcv)
    returns = compute_returns(prices).dropna(how='all')
    prices  = prices.reindex(returns.index)

    coverage = prices.notna().mean()
    good     = coverage[coverage > 0.80].index
    prices   = prices[good]
    returns  = returns[good]
    print(f"  After quality filter: {len(prices.columns)} tickers retained\n")

    mom_12_1 = compute_12_1(prices)
    mom_6_1  = compute_6_1(prices)

    print("  Running Markowitz optimization with regime detection (~2 min)...\n")
    strategy_returns, order_book_df, weights_history = run_strategy(
        prices, returns, mom_12_1, mom_6_1, spy_regime=spy_regime,
    )

    # Normalise strategy index too — safe for both reindex and VBT
    strategy_returns = _to_period_index(strategy_returns)

    # ── VBT backtest ──────────────────────────────────────────────────────
    # VBT needs a DatetimeIndex — convert back to timestamps just for this
    strat_for_vbt = strategy_returns.copy()
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

    # ── Save order book ───────────────────────────────────────────────────
    ob_path = REPORTS_DIR / "order_book.csv"
    order_book_df.to_csv(ob_path, index=False)
    print(f"\n  ✅ Order book saved → {ob_path}")
    print(f"     {len(order_book_df)} total events | "
          f"{len(order_book_df[order_book_df.Action=='BUY'])} buys | "
          f"{len(order_book_df[order_book_df.Action=='SELL'])} sells")

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print("\n  📋 Order Book — last 15 events:")
    print(order_book_df.tail(15).to_string(index=False))

    # ── Dashboard ─────────────────────────────────────────────────────────
    print("\n  Building dashboard...")
    fig = build_dashboard(strategy_returns, order_book_df, weights_history, spy_rets)
    dash_path = REPORTS_DIR / "portfolio_dashboard.html"
    fig.write_html(str(dash_path), include_plotlyjs="cdn", full_html=True)
    print(f"  ✅ Dashboard saved → {dash_path}")
    print("     Open in any browser to explore interactively.\n")


if __name__ == '__main__':
    try:
        main()
    finally:
        # Explicitly shut down loky to prevent leaked semaphore warnings on exit
        try:
            from joblib.externals.loky import get_reusable_executor
            get_reusable_executor().shutdown(wait=True)
        except (ImportError, AttributeError):
            pass
    