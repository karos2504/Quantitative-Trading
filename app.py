"""
Macro Rotation Dashboard — Interactive Backtesting UI
=====================================================
A premium, interactive Streamlit application to run backtests, 
analyze regimes, and visualize portfolio performance.
"""

import sys
from pathlib import Path
import datetime as dt

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Ensure project root is on path for absolute imports
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import Macro Rotation Engine
from macro_rotation.config import SystemConfig, CONFIG, logger, MacroRegime
from macro_rotation.portfolios import CryptoGoldRotation, CoreAssetMacroRotation
from macro_rotation.data_loader import load_all_data
from macro_rotation.backtester import run_backtest, compute_benchmark
from macro_rotation.dashboard import build_dashboard, save_event_log

# =============================================================================
# STREAMLIT SETUP
# =============================================================================
st.set_page_config(
    page_title="Macro Rotation Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium Styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="stSidebar"] {
        background-color: #f1f3f6;
    }
    .status-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# CACHED DATA LOADING
# =============================================================================
@st.cache_data(show_spinner="📥 Fetching market and macro data...")
def fetch_data(fred_api_key: str, vnindex_csv_path: str, backtest_start: str):
    """
    Cached wrapper for load_all_data.
    Only re-runs if critical config parameters change.
    """
    # Create a temporary config for data loading
    temp_config = SystemConfig(
        backtest_start=backtest_start,
        fred_api_key=fred_api_key,
        vnindex_csv_path=vnindex_csv_path,
    )
    data = load_all_data(temp_config)
    return data

@st.cache_data(show_spinner="⚙️ Processing backtest engine...")
def process_portfolio(
    portfolio_type: str, 
    _data: dict, 
    cash_yield_pct: float,
    start_str: str,
    end_str: str
):
    """
    Run the backtest engine for the selected portfolio.
    _data is prefixed with underscore to exclude it from streamlit's hashing 
    (we hash the other parameters instead).
    """
    # Instantiate portfolio
    if portfolio_type == "Crypto + Gold Rotation":
        portfolio = CryptoGoldRotation()
    else:
        portfolio = CoreAssetMacroRotation()
    
    # Configure run
    config = SystemConfig(
        backtest_start=start_str,
        backtest_end=end_str,
        cash_yield_apy=cash_yield_pct / 100.0
    )
    
    # Align data to start date and end date
    prices = _data["prices"][
        (_data["prices"].index >= pd.Timestamp(start_str)) & 
        (_data["prices"].index <= pd.Timestamp(end_str))
    ]
    
    # Execute Backtest
    results = run_backtest(
        portfolio=portfolio,
        prices=prices,
        fred_df=_data["fred"],
        proxy_prices=_data["proxy_prices"],
        sector_prices=_data["sector_prices"],
        config=config
    )
    
    # Benchmark
    bench_ticker = portfolio.get_benchmark_ticker()
    benchmark = compute_benchmark(prices, bench_ticker, config)
    
    return results, benchmark

# =============================================================================
# SIDEBAR — CONFIGURATION
# =============================================================================
with st.sidebar:
    st.header("📊 Engine Control")
    
    portfolio_choice = st.selectbox(
        "Portfolio Strategy",
        options=["Crypto + Gold Rotation", "Core Asset Macro-Rotation"],
        help="Select the quantitative strategy to backtest."
    )
    
    st.divider()
    
    st.subheader("🛠 Parameters")
    
    start_date = st.date_input(
        "Backtest Start",
        value=dt.date(2021, 1, 1),
        min_value=dt.date(2015, 1, 1),
        max_value=dt.date.today(),
    )

    end_date = st.date_input(
        "Backtest End",
        value=dt.date.today(),
        min_value=dt.date(2015, 1, 1),
        max_value=dt.date.today(),
    )
    
    cash_yield_pct = st.slider(
        "Cash Yield APY (%)",
        min_value=0.0,
        max_value=10.0,
        value=5.25,
        step=0.25,
        help="Annual interest rate earned on uninvested cash."
    )
    
    # Safely get default FRED key from secrets if available
    try:
        default_fred_key = st.secrets.get("FRED_API_KEY", "")
    except Exception:
        default_fred_key = ""

    fred_api_key = st.text_input(
        "FRED API Key",
        value=default_fred_key,
        type="password",
        help="Optional: Get a free key from fred.stlouisfed.org to enable live macro data."
    )

    vnindex_csv = st.text_input(
        "VNINDEX CSV Path",
        placeholder="e.g., data/vnindex.csv",
        help="Optional: Path to local CSV for more accurate VNINDEX data."
    )

    st.divider()
    
    run_button = st.button("🚀 Run Backtest", type="primary", use_container_width=True)
    
    st.info("System Status: Ready")

# =============================================================================
# MAIN INTERFACE
# =============================================================================
st.title("Macro Rotation Engine")
st.caption(f"Interactive Quantitative Research Dashboard — {dt.datetime.now().strftime('%Y-%m-%d')}")

# Initial state
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None
    st.session_state.benchmark_results = None

if run_button:
    try:
        # 1. Load Data
        data = fetch_data(
            fred_api_key=fred_api_key,
            vnindex_csv_path=vnindex_csv,
            backtest_start=start_date.strftime("%Y-%m-%d")
        )
        
        # 2. Run Backtest
        results, benchmark = process_portfolio(
            portfolio_type=portfolio_choice,
            _data=data,
            cash_yield_pct=cash_yield_pct,
            start_str=start_date.strftime("%Y-%m-%d"),
            end_str=end_date.strftime("%Y-%m-%d")
        )
        
        st.session_state.backtest_results = results
        st.session_state.benchmark_results = benchmark
        st.success(f"✅ Backtest completed for {portfolio_choice}")
        
    except Exception as e:
        st.error(f"❌ Error during backtest: {str(e)}")
        st.exception(e)

# =============================================================================
# RESULTS DISPLAY
# =============================================================================
if st.session_state.backtest_results:
    results = st.session_state.backtest_results
    benchmark = st.session_state.benchmark_results
    metrics = results["metrics"]
    
    # --- Top KPIs ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("CAGR (%)", f"{metrics['CAGR (%)']}%", delta=f"{(metrics['CAGR (%)'] - benchmark['metrics']['CAGR (%)']):.1f}% vs Bench")
    with c2:
        st.metric("Sharpe Ratio", f"{metrics['Sharpe']}")
    with c3:
        st.metric("Max Drawdown", f"{metrics['Max Drawdown (%)']}%", delta=f"{(metrics['Max Drawdown (%)'] - benchmark['metrics']['Max Drawdown (%)']):.1f}%", delta_color="inverse")
    with c4:
        st.metric("Final Value", f"${metrics['Final Value ($)']:,.0f}")

    # --- MAIN DASHBOARD ---
    with st.spinner("Rendering visualization..."):
        fig = build_dashboard(results, benchmark)
        # Update layout for Streamlit container width
        fig.update_layout(height=1600, width=None)
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    # --- EXPERT LOGS ---
    with st.expander("📝 View Detailed Event Log (Audit Trail)"):
        events = results["events"]
        records = []
        for e in events:
            records.append({
                "Date": e.date.date(),
                "Reason": e.reason,
                "Regime": e.regime.value,
                "BTC Risk": f"{e.btc_risk:.2f}",
                "Sentiment": e.sentiment,
                "Live Signal": e.live_signal,
                "Turnover": f"{e.turnover*100:.1f}%",
                "Fees": f"${e.total_fees_usd:.0f}",
                "Portfolio Val": f"${e.portfolio_value:,.0f}"
            })
        event_df = pd.DataFrame(records)
        st.dataframe(event_df, use_container_width=True)

else:
    # No results yet — show splash
    st.container()
    st.markdown("""
        ### Welcome to the Macro Rotation Dashboard
        
        Click **Run Backtest** in the sidebar to start a simulation. 
        
        This engine models a multi-asset portfolio with:
        * **Trend-Following Signals**: Based on Momentum-Persistence filters.
        * **Macro Overlay**: Dynamic allocation based on Growth vs. Inflation regimes.
        * **Risk Management**: BTC volatility metrics and sentiment-aware sizing.
        * **Realistic Execution**: Marginal trading costs, APY on idle cash, and settlement lags.
    """)
    
    # Show pre-generated reports if available
    reports_dir = Path("macro_rotation/reports")
    if reports_dir.exists():
        html_files = list(reports_dir.glob("*_dashboard.html"))
        if html_files:
            st.divider()
            st.subheader("📂 Pre-generated Reports")
            for f in html_files:
                if st.button(f"View {f.name}"):
                    import streamlit.components.v1 as components
                    components.html(f.read_text(encoding="utf-8"), height=1700, scrolling=True)

