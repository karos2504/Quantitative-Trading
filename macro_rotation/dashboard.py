"""
Interactive Dashboard (Visualization)
=======================================
Plotly dashboard with performance panels AND diagnostic panels.

Performance panels:
    1. Equity curve vs benchmark
    2. Underwater drawdown
    3. Monthly returns heatmap
    4. Rolling Sharpe / Sortino

Diagnostic panels:
    5. Weight allocation over time (stacked area)
    6. Macro regime timeline with composite index plots
    7. Rebalance event log (why each rebalance triggered)
    8. Rolling turnover + BTC risk metric
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from macro_rotation.config import MacroRegime, logger, REPORTS_DIR
from macro_rotation.backtester import RebalanceEvent


# ============================================================================
# COLOR SCHEMES
# ============================================================================
COLORS = {
    "strategy":     "#4C78A8",
    "benchmark":    "#F58518",
    "drawdown":     "#E45756",
    "cash":         "#BAB0AC",
    "btc":          "#F7931A",
    "eth":          "#627EEA",
    "gold":         "#FFD700",
    "vnindex":      "#DA251D",
    "positive":     "#27ae60",
    "negative":     "#c0392b",
}

REGIME_COLORS = {
    MacroRegime.RISK_ON_DISINFLATION:  "rgba(39, 174, 96, 0.15)",
    MacroRegime.RISK_ON_INFLATION:     "rgba(241, 196, 15, 0.15)",
    MacroRegime.RISK_OFF_INFLATION:    "rgba(231, 76, 60, 0.15)",
    MacroRegime.RISK_OFF_DISINFLATION: "rgba(52, 152, 219, 0.15)",
}

ASSET_COLORS = {
    "BTC": "#F7931A", "ETH": "#627EEA", "BNB": "#F3BA2F",
    "XRP": "#00AAE4", "LINK": "#2A5ADA", "DOGE": "#C2A633",
    "XAUT": "#FFD700", "VNINDEX": "#DA251D",
}


# ============================================================================
# DASHBOARD BUILDER
# ============================================================================
def build_dashboard(
    results: dict,
    benchmark: dict | None = None,
) -> go.Figure:
    """
    Build the full interactive Plotly dashboard.

    Args:
        results: Output from backtester.run_backtest()
        benchmark: Output from backtester.compute_benchmark()
    """
    equity = results["equity_curve"]
    returns_s = results["returns"]
    weight_df = results["weight_history"]
    cash_s = results["cash_history"]
    events = results["events"]
    macro_df = results.get("macro_df", pd.DataFrame())
    btc_risk = results.get("btc_risk", pd.Series())
    metrics = results["metrics"]
    turnover_s = results.get("turnover_history", pd.Series())
    portfolio_name = results.get("portfolio_name", "Strategy")

    bench_equity = benchmark["equity_curve"] if benchmark else None

    # Create figure with subplots
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            "📈 Equity Curve vs Benchmark",
            "🌊 Underwater Drawdown",
            "📅 Monthly Returns Heatmap (%)",
            "📊 Rolling Sharpe (90-day)",
            "🥧 Portfolio Allocation Over Time",
            "🌍 Macro Regime + Composites",
            "🔄 Rebalance Events + Turnover",
            "₿ BTC Risk Metric + Signal Timeline",
        ),
        row_heights=[0.24, 0.24, 0.26, 0.26],
        vertical_spacing=0.06,
        horizontal_spacing=0.08,
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
    )

    # --- Panel 1: Equity Curve ---
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        name="Strategy", line=dict(color=COLORS["strategy"], width=2.5),
        hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra>Strategy</extra>",
    ), row=1, col=1)

    if bench_equity is not None:
        bench_aligned = bench_equity.reindex(equity.index).ffill()
        fig.add_trace(go.Scatter(
            x=bench_aligned.index, y=bench_aligned.values,
            name="Benchmark", line=dict(color=COLORS["benchmark"], width=1.8, dash="dot"),
            hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra>Benchmark</extra>",
        ), row=1, col=1)

    # Final value annotations
    fig.add_annotation(
        x=equity.index[-1], y=float(equity.iloc[-1]),
        text=f"  ${float(equity.iloc[-1]):,.0f}",
        showarrow=False, font=dict(color=COLORS["strategy"], size=11),
        row=1, col=1,
    )

    # --- Panel 2: Drawdown ---
    roll_max = equity.cummax()
    drawdown = ((equity - roll_max) / roll_max * 100)
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        fill="tozeroy", fillcolor="rgba(229,115,115,0.25)",
        line=dict(color=COLORS["drawdown"], width=1.5), name="Drawdown",
        hovertemplate="%{x|%b %Y}<br>%{y:.1f}%<extra>Drawdown</extra>",
    ), row=1, col=2)

    # --- Panel 3: Monthly Returns Heatmap ---
    active = returns_s[returns_s.index >= returns_s.index[0] + pd.Timedelta(days=252)]
    if len(active) > 0:
        df_ret = active.to_frame("ret")
        df_ret["year"] = df_ret.index.year
        df_ret["month"] = df_ret.index.month

        # Monthly aggregation
        monthly = df_ret.groupby(["year", "month"])["ret"].apply(
            lambda x: float((1 + x).prod() - 1)
        ).reset_index()
        monthly.columns = ["year", "month", "ret"]
        heat_pivot = monthly.pivot(index="year", columns="month", values="ret")
        heat_pivot = heat_pivot * 100

        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        heat_pivot.columns = [month_labels[i - 1] for i in heat_pivot.columns if i <= 12]

        valid = heat_pivot.values[~np.isnan(heat_pivot.values)]
        zmax = max(abs(valid.max()), abs(valid.min())) if len(valid) else 10

        fig.add_trace(go.Heatmap(
            z=heat_pivot.values,
            x=heat_pivot.columns.tolist(),
            y=[str(y) for y in heat_pivot.index.tolist()],
            colorscale=[[0.0, "#c0392b"], [0.5, "#f7f7f7"], [1.0, "#27ae60"]],
            zmid=0, zmin=-zmax, zmax=zmax,
            text=np.round(heat_pivot.values, 1), texttemplate="%{text}",
            textfont=dict(size=8),
            colorbar=dict(len=0.22, y=0.75, thickness=10, title="%"),
            name="Monthly Ret",
        ), row=2, col=1)

    # --- Panel 4: Rolling Sharpe ---
    rolling_ret = active.rolling(90, min_periods=30).mean() * 365
    rolling_vol = active.rolling(90, min_periods=30).std() * np.sqrt(365)
    rolling_sharpe = rolling_ret / rolling_vol
    rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan)

    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index, y=rolling_sharpe.values,
        line=dict(color=COLORS["strategy"], width=1.8), name="Rolling Sharpe",
    ), row=2, col=2)
    fig.add_hline(y=0, line=dict(color="gray", dash="dot", width=1), row=2, col=2)
    fig.add_hline(y=1, line=dict(color=COLORS["positive"], dash="dash", width=1), row=2, col=2)

    # --- Panel 5: Weight Allocation (stacked area) ---
    weight_pct = weight_df * 100
    for asset in weight_pct.columns:
        if weight_pct[asset].sum() > 0:
            color = ASSET_COLORS.get(asset, "#aaaaaa")
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            fig.add_trace(go.Scatter(
                x=weight_pct.index, y=weight_pct[asset].values,
                stackgroup="weights", name=asset,
                line=dict(width=0.5, color=color),
                fillcolor=f"rgba({r},{g},{b},0.7)",
                hovertemplate=f"<b>{asset}</b><br>%{{x|%b %Y}}<br>%{{y:.1f}}%<extra></extra>",
            ), row=3, col=1)

    # Cash weight on top
    cash_pct = cash_s * 100
    fig.add_trace(go.Scatter(
        x=cash_pct.index, y=cash_pct.values,
        stackgroup="weights", name="Cash",
        line=dict(width=0.5, color=COLORS["cash"]),
        fillcolor="rgba(186,176,172,0.5)",
    ), row=3, col=1)

    # --- Panel 6: Macro Regime Timeline ---
    if not macro_df.empty:
        # Plot growth and inflation scores
        for col, color, name in [
            ("growth_score", "#27ae60", "Growth Score"),
            ("inflation_score", "#e74c3c", "Inflation Score"),
        ]:
            if col in macro_df.columns:
                fig.add_trace(go.Scatter(
                    x=macro_df.index, y=macro_df[col].values,
                    line=dict(color=color, width=1.5), name=name,
                    hovertemplate=f"%{{x|%b %Y}}<br>{name}: %{{y:.2f}}<extra></extra>",
                ), row=3, col=2)

        fig.add_hline(y=0.5, line=dict(color="gray", dash="dot", width=1), row=3, col=2)

        # Add regime background bands
        if "regime" in macro_df.columns:
            regime_changes = macro_df["regime"].ne(macro_df["regime"].shift())
            change_dates = macro_df.index[regime_changes]
            for j in range(len(change_dates)):
                start = change_dates[j]
                end = change_dates[j + 1] if j + 1 < len(change_dates) else macro_df.index[-1]
                regime = macro_df.loc[start, "regime"]
                if isinstance(regime, MacroRegime):
                    fig.add_vrect(
                        x0=start, x1=end,
                        fillcolor=REGIME_COLORS.get(regime, "rgba(128,128,128,0.1)"),
                        layer="below", line_width=0,
                        row=3, col=2,
                    )

    # --- Panel 7: Rebalance Events + Turnover ---
    if len(events) > 0:
        event_dates = [e.date for e in events]
        event_turnovers = [e.turnover * 100 for e in events]
        event_colors = [COLORS["positive"] if "Initial" not in e.reason else COLORS["cash"]
                        for e in events]

        fig.add_trace(go.Bar(
            x=event_dates, y=event_turnovers,
            marker_color=event_colors, name="Rebalance Turnover (%)",
            opacity=0.7,
            hovertemplate="%{x|%b %d %Y}<br>Turnover: %{y:.1f}%<extra></extra>",
        ), row=4, col=1)

    # Rolling 30-day turnover
    if not turnover_s.empty:
        rolling_turnover = turnover_s.rolling(30, min_periods=1).sum() * 100
        fig.add_trace(go.Scatter(
            x=rolling_turnover.index, y=rolling_turnover.values,
            line=dict(color=COLORS["strategy"], width=1.5, dash="dot"),
            name="Rolling 30d Turnover",
        ), row=4, col=1)

    # --- Panel 8: BTC Risk Metric ---
    if not btc_risk.empty:
        fig.add_trace(go.Scatter(
            x=btc_risk.index, y=btc_risk.values,
            line=dict(color=COLORS["btc"], width=2), name="BTC Risk",
            hovertemplate="%{x|%b %Y}<br>Risk: %{y:.2f}<extra></extra>",
        ), row=4, col=2)

        # Zone bands
        for y0, y1, color, label in [
            (0.0, 0.3, "rgba(39,174,96,0.15)", "Accumulation"),
            (0.3, 0.5, "rgba(39,174,96,0.08)", "DCA"),
            (0.5, 0.6, "rgba(128,128,128,0.08)", "Neutral"),
            (0.6, 0.8, "rgba(231,76,60,0.08)", "DCE"),
            (0.8, 1.0, "rgba(231,76,60,0.15)", "Peak"),
        ]:
            fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0, row=4, col=2)

    # --- Metrics Summary Annotation ---
    metric_lines = [
        f"<b>{portfolio_name}</b>",
        f"CAGR:         {metrics.get('CAGR (%)', 0):.1f}%",
        f"Sharpe:       {metrics.get('Sharpe', 0):.2f}",
        f"Sortino:      {metrics.get('Sortino', 0):.2f}",
        f"Max DD:       {metrics.get('Max Drawdown (%)', 0):.1f}%",
        f"Rebalances:   {metrics.get('Rebalance Events', 0)}",
        f"Total Fees:   ${metrics.get('Total Fees ($)', 0):,.0f}",
        f"Final Value:  ${metrics.get('Final Value ($)', 0):,.0f}",
    ]
    fig.add_annotation(
        x=0.99, y=0.02, xref="paper", yref="paper",
        text="<br>".join(metric_lines),
        align="left", showarrow=False,
        font=dict(size=9, family="monospace"),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#cccccc", borderwidth=1,
        xanchor="right", yanchor="bottom",
    )

    # --- Layout ---
    fig.update_layout(
        title=dict(
            text=f"<b>{portfolio_name} — Dynamic Rebalancing Dashboard</b>",
            font=dict(size=18), x=0.5,
        ),
        height=1600, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, font=dict(size=9)),
        margin=dict(l=60, r=60, t=80, b=40),
        hovermode="x unified",
    )

    # Axis labels
    fig.update_yaxes(title_text="Portfolio ($)", row=1, col=1, tickprefix="$", tickformat=",")
    fig.update_yaxes(title_text="Drawdown (%)", row=1, col=2)
    fig.update_yaxes(title_text="Return (%)", row=2, col=1)
    fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=2)
    fig.update_yaxes(title_text="Allocation (%)", row=3, col=1)
    fig.update_yaxes(title_text="Score (0–1)", row=3, col=2)
    fig.update_yaxes(title_text="Turnover (%)", row=4, col=1)
    fig.update_yaxes(title_text="BTC Risk (0–1)", row=4, col=2, range=[0, 1.05])

    return fig


def save_dashboard(fig: go.Figure, name: str) -> Path:
    """Save dashboard as interactive HTML."""
    path = REPORTS_DIR / f"{name.replace(' ', '_').lower()}_dashboard.html"
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    logger.info(f"  ✅ Dashboard saved → {path}")
    return path


def save_event_log(events: list[RebalanceEvent], name: str) -> Path:
    """Save rebalance event log as CSV for audit."""
    records = []
    for e in events:
        records.append({
            "Date": e.date.strftime("%Y-%m-%d"),
            "Reason": e.reason,
            "Regime": e.regime.value,
            "BTC_Risk": round(e.btc_risk, 3),
            "Sentiment": e.sentiment,
            "Live_Signal": e.live_signal,
            "Turnover_%": round(e.turnover * 100, 2),
            "Fees_USD": round(e.total_fees_usd, 2),
            "Cash_%": round(e.cash_weight * 100, 2),
            "Portfolio_Value": round(e.portfolio_value, 2),
            "Trades": str(e.trades),
        })
    df = pd.DataFrame(records)
    path = REPORTS_DIR / f"{name.replace(' ', '_').lower()}_event_log.csv"
    df.to_csv(path, index=False)
    logger.info(f"  ✅ Event log saved → {path}")
    return path
