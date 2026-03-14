"""
Monthly Portfolio Rebalancing — v6: Markowitz + Interactive Dashboard
=====================================================================
Additions over v6-base:
  - Full Plotly dashboard saved to  reports/portfolio_dashboard.html
  - Live order book: every BUY/SELL/HOLD event logged with
    ticker, date, action, weight, momentum score, estimated price
  - Order book saved to  reports/order_book.csv
  - Dashboard panels:
      1. Equity curve vs SPY benchmark
      2. Underwater drawdown chart
      3. Monthly returns heatmap (calendar view)
      4. Rolling 12-month Sharpe ratio
      5. Portfolio composition over time (stacked area)
      6. Order book table (last 50 trades)
"""

import sys
from pathlib import Path
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

from utils.data import fetch_ohlcv_data
from utils.backtesting import VBTBacktester

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
#  DATA
# ============================================================
def build_prices(data: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {t: df['Adj Close'] for t, df in data.items()}
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
        sectors = [UNIVERSE_WITH_SECTORS.get(t, "OTHER") for t in candidates]
        for sec in set(sectors):
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
#  STRATEGY  (returns monthly_returns + order_book + weights_history)
# ============================================================
def run_strategy(prices, returns, mom_12_1, mom_6_1):
    monthly_returns  = []
    current_weights  = {}
    prev_weights     = {}

    # Order book: list of dicts
    order_book_rows  = []
    # Weights over time: {date: {ticker: weight}}
    weights_history  = {}

    for i in range(len(returns)):
        date = returns.index[i]

        # ---- P&L ----
        if current_weights:
            pnl = sum(
                returns[t].iloc[i] * w
                for t, w in current_weights.items()
                if t in returns.columns and not pd.isna(returns[t].iloc[i])
            )
            monthly_returns.append(pnl)
        else:
            monthly_returns.append(0.0)

        # ---- Momentum candidates ----
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

        ranked     = sorted(eligible, key=lambda t: eligible[t], reverse=True)
        candidates = ranked[:CANDIDATE_SIZE]

        hist_start = max(0, i - COV_LOOKBACK)
        returns_window = returns.iloc[hist_start:i]
        available = [
            t for t in candidates
            if t in returns_window.columns
            and returns_window[t].notna().sum() >= 12
        ]
        if len(available) < 3:
            top = candidates[:10]
            new_weights = {t: 1.0/len(top) for t in top}
        else:
            new_weights = markowitz_weights(available, returns_window, eligible)

        # ---- Log order book ----
        prev_set = set(prev_weights.keys())
        new_set  = set(new_weights.keys())

        for t in new_set - prev_set:
            order_book_rows.append({
                "Date":       date.strftime("%Y-%m-%d"),
                "Ticker":     t,
                "Sector":     UNIVERSE_WITH_SECTORS.get(t, "?"),
                "Action":     "BUY",
                "Weight_%":   round(new_weights[t] * 100, 2),
                "Mom_12_1_%": round(eligible.get(t, 0) * 100, 2),
                "Price":      round(float(prices[t].iloc[i]), 2) if t in prices.columns else None,
            })
        for t in prev_set - new_set:
            order_book_rows.append({
                "Date":       date.strftime("%Y-%m-%d"),
                "Ticker":     t,
                "Sector":     UNIVERSE_WITH_SECTORS.get(t, "?"),
                "Action":     "SELL",
                "Weight_%":   0.0,
                "Mom_12_1_%": round(eligible.get(t, 0) * 100, 2),
                "Price":      round(float(prices[t].iloc[i]), 2) if t in prices.columns else None,
            })
        for t in prev_set & new_set:
            delta = abs(new_weights[t] - prev_weights[t])
            if delta > 0.005:  # log only meaningful rebalances
                action = "ADD" if new_weights[t] > prev_weights[t] else "TRIM"
                order_book_rows.append({
                    "Date":       date.strftime("%Y-%m-%d"),
                    "Ticker":     t,
                    "Sector":     UNIVERSE_WITH_SECTORS.get(t, "?"),
                    "Action":     action,
                    "Weight_%":   round(new_weights[t] * 100, 2),
                    "Mom_12_1_%": round(eligible.get(t, 0) * 100, 2),
                    "Price":      round(float(prices[t].iloc[i]), 2) if t in prices.columns else None,
                })

        current_weights = new_weights
        prev_weights    = dict(new_weights)
        weights_history[date] = dict(new_weights)

    strategy_returns = pd.Series(monthly_returns, index=returns.index, name="Monthly Return")
    order_book_df    = pd.DataFrame(order_book_rows)
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
    # --- Derived series ---
    equity      = (1 + strategy_returns).cumprod() * 100_000
    spy_eq      = (1 + spy_returns.reindex(strategy_returns.index).fillna(0)).cumprod() * 100_000
    roll_max    = equity.cummax()
    drawdown    = (equity - roll_max) / roll_max * 100
    roll_sharpe = (
        strategy_returns.rolling(12).mean() /
        strategy_returns.rolling(12).std()
    ) * np.sqrt(12)

    # Monthly returns heatmap data
    df_ret = strategy_returns.to_frame("ret")
    df_ret["year"]  = df_ret.index.year
    df_ret["month"] = df_ret.index.month
    heat_pivot = df_ret.pivot(index="year", columns="month", values="ret") * 100
    heat_pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                          "Jul","Aug","Sep","Oct","Nov","Dec"]

    # Weights area chart data
    all_tickers = sorted({t for wts in weights_history.values() for t in wts})
    wh_df = pd.DataFrame(weights_history).T.fillna(0)[all_tickers] * 100

    # ---- Build figure ----
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
            [{"type": "xy"},     {"type": "xy"}],
            [{"type": "xy"},     {"type": "xy"}],
            [{"type": "xy"},     {"type": "table"}],
        ],
    )

    # ── Panel 1: Equity curve ──────────────────────────────────
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

    # Annotate final values
    fig.add_annotation(
        x=equity.index[-1], y=equity.iloc[-1],
        text=f"  ${float(equity.iloc[-1]):,.0f}",
        showarrow=False, font=dict(color="#4C78A8", size=11), row=1, col=1,
    )
    fig.add_annotation(
        x=spy_eq.index[-1], y=spy_eq.iloc[-1],
        text=f"  ${float(spy_eq.iloc[-1]):,.0f}",
        showarrow=False, font=dict(color="#F58518", size=11), row=1, col=1,
    )

    # ── Panel 2: Drawdown ─────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        fill="tozeroy", fillcolor="rgba(229,115,115,0.25)",
        line=dict(color="#E45756", width=1.5),
        name="Drawdown",
        hovertemplate="%{x|%b %Y}<br>%{y:.1f}%<extra>Drawdown</extra>",
    ), row=1, col=2)

    # ── Panel 3: Monthly returns heatmap ──────────────────────
    zmax = max(abs(heat_pivot.values[~np.isnan(heat_pivot.values)].max()),
               abs(heat_pivot.values[~np.isnan(heat_pivot.values)].min()))
    fig.add_trace(go.Heatmap(
        z=heat_pivot.values,
        x=heat_pivot.columns.tolist(),
        y=[str(y) for y in heat_pivot.index.tolist()],
        colorscale=[
            [0.0,  "#c0392b"],
            [0.5,  "#f7f7f7"],
            [1.0,  "#27ae60"],
        ],
        zmid=0, zmin=-zmax, zmax=zmax,
        text=np.round(heat_pivot.values, 1),
        texttemplate="%{text}",
        textfont=dict(size=9),
        colorbar=dict(len=0.3, y=0.38, thickness=12, title="%"),
        hovertemplate="<b>%{y} %{x}</b><br>%{z:.2f}%<extra></extra>",
        showscale=True,
        name="Monthly Ret",
    ), row=2, col=1)

    # ── Panel 4: Rolling Sharpe ───────────────────────────────
    fig.add_trace(go.Scatter(
        x=roll_sharpe.index,
        y=roll_sharpe.values,
        line=dict(color="#72B7B2", width=2),
        name="Rolling Sharpe",
        hovertemplate="%{x|%b %Y}<br>Sharpe: %{y:.2f}<extra></extra>",
    ), row=2, col=2)

    # Horizontal reference lines
    fig.add_trace(go.Scatter(
        x=[roll_sharpe.index.min(), roll_sharpe.index.max()],
        y=[0, 0],
        mode="lines",
        line=dict(color="gray", dash="dot", width=1),
        showlegend=False,
        hoverinfo="skip",
    ), row=2, col=2)

    fig.add_trace(go.Scatter(
        x=[roll_sharpe.index.min(), roll_sharpe.index.max()],
        y=[1, 1],
        mode="lines",
        line=dict(color="#27ae60", dash="dash", width=1),
        showlegend=False,
        hoverinfo="skip",
    ), row=2, col=2)

    # ── Panel 5: Portfolio composition stacked area ───────────
    # Group by sector for readability
    sector_wh = pd.DataFrame(index=wh_df.index)
    for sec in sorted(set(UNIVERSE_WITH_SECTORS.values())):
        sec_tickers = [t for t in all_tickers if UNIVERSE_WITH_SECTORS.get(t) == sec and t in wh_df.columns]
        if sec_tickers:
            sector_wh[sec] = wh_df[sec_tickers].sum(axis=1)

    for sec in sector_wh.columns:
        color = SECTOR_COLORS.get(sec, "#aaaaaa")
        # rgba fill
        r,g,b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
        fig.add_trace(go.Scatter(
            x=sector_wh.index, y=sector_wh[sec].values,
            stackgroup="one",
            name=sec,
            line=dict(width=0.5, color=color),
            fillcolor=f"rgba({r},{g},{b},0.7)",
            hovertemplate=f"<b>{sec}</b><br>%{{x|%b %Y}}<br>%{{y:.1f}}%<extra></extra>",
            legendgroup=sec,
        ), row=3, col=1)

    # ── Panel 6: Order Book table ─────────────────────────────
    if not order_book_df.empty:
        ob_display = order_book_df.tail(60).iloc[::-1].reset_index(drop=True)

        action_colors = {
            "BUY":  "#d4efdf", "SELL": "#fadbd8",
            "ADD":  "#d6eaf8", "TRIM": "#fdebd0",
        }
        cell_colors = [
            [action_colors.get(a, "#ffffff") for a in ob_display["Action"]],
        ]
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
                align="center",
                height=28,
            ),
            cells=dict(
                values=[ob_display[c].tolist() for c in ob_display.columns],
                fill_color=fill_colors,
                font=dict(size=10),
                align=["center","center","center","center","right","right","right"],
                height=22,
            ),
        ), row=3, col=2)

    # ── Layout ────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text="<b>Markowitz Momentum Portfolio — Full Dashboard</b>",
            font=dict(size=20),
            x=0.5,
        ),
        height=1300,
        template="plotly_white",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="right", x=1, font=dict(size=10),
        ),
        margin=dict(l=60, r=60, t=80, b=40),
        hovermode="x unified",
    )

    # Axis labels
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1, tickprefix="$", tickformat=",")
    fig.update_yaxes(title_text="Drawdown (%)", row=1, col=2)
    fig.update_yaxes(title_text="Return (%)", row=2, col=1)
    fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=2)
    fig.update_yaxes(title_text="Allocation (%)", row=3, col=1)

    return fig


# ============================================================
#  MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  Monthly Rebalancing — v6: Markowitz + Dashboard")
    print("=" * 60)
    print(f"\n  Universe:    {len(SP500_UNIVERSE)} S&P 500 stocks")
    print(f"  Optimizer:   Markowitz [{OBJECTIVE}], Ledoit-Wolf shrinkage")
    print(f"  Constraints: [{MIN_WEIGHT*100:.0f}%, {MAX_WEIGHT*100:.0f}%] / stock | "
          f"{MAX_SECTOR_W*100:.0f}% / sector\n")

    # ---- SPY benchmark ----
    spy_raw = yf.download("SPY", start=START_DATE, end=END_DATE,
                          interval='1mo', auto_adjust=False, progress=False)
    spy_returns = spy_raw['Adj Close'].pct_change().dropna()
    spy_returns.index = spy_returns.index.to_period('M').to_timestamp('M')

    # ---- Strategy data ----
    print("  Fetching data...")
    ohlcv   = fetch_ohlcv_data(SP500_UNIVERSE, start=START_DATE, end=END_DATE, interval='1mo')
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

    print("  Running Markowitz optimization (~2 min)...\n")
    strategy_returns, order_book_df, weights_history = run_strategy(
        prices, returns, mom_12_1, mom_6_1,
    )

    # ---- VBT backtest ----
    strategy_close = (1 + strategy_returns).cumprod() * 100
    entries = pd.Series(True,  index=strategy_returns.index)
    exits   = pd.Series(False, index=strategy_returns.index)
    exits.iloc[-1] = True

    bt = VBTBacktester(
        close=strategy_close, entries=entries, exits=exits,
        freq='30D', init_cash=100_000, commission=0.001,
    )
    bt.full_analysis(n_mc=1000, n_wf_splits=5, n_trials=1)

    # ---- Save order book ----
    ob_path = REPORTS_DIR / "order_book.csv"
    order_book_df.to_csv(ob_path, index=False)
    print(f"\n  ✅ Order book saved → {ob_path}")
    print(f"     {len(order_book_df)} total events | "
          f"{len(order_book_df[order_book_df.Action=='BUY'])} buys | "
          f"{len(order_book_df[order_book_df.Action=='SELL'])} sells")

    # ---- Print last 15 order book rows ----
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print("\n  📋 Order Book — last 15 events:")
    print(order_book_df.tail(15).to_string(index=False))

    # ---- Build & save dashboard ----
    print("\n  Building dashboard...")
    fig = build_dashboard(strategy_returns, order_book_df, weights_history, spy_returns)
    dash_path = REPORTS_DIR / "portfolio_dashboard.html"
    fig.write_html(str(dash_path), include_plotlyjs="cdn", full_html=True)
    print(f"  ✅ Dashboard saved → {dash_path}")
    print(f"     Open in any browser to explore interactively.\n")


if __name__ == '__main__':
    main()
