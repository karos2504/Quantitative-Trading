import streamlit as st
import os
import sys
import subprocess
from pathlib import Path

st.set_page_config(page_title="Quant Control Center", page_icon="📈", layout="wide")

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.py"
REPORTS_DIR = PROJECT_ROOT / "strategies" / "reports"

STRATEGIES = {
    "Renko + MACD": "strategies/renko_macd.py",
    "Renko Hybrid MACD/OBV": "strategies/renko_macd_obv.py",
    "Resistance Breakout": "strategies/resistance_breakout.py",
    "Markowitz Portfolio Rebalance": "strategies/rebalance_portfolio.py",
}

def load_settings():
    if not SETTINGS_PATH.exists(): return {}
    with open(SETTINGS_PATH, "r") as f:
        content = f.read()
    scope = {}
    exec(content, scope)
    return scope

def save_settings(tickers_str, cash, commission, data_days, interval, train_m, test_m):
    tickers_list = [t.strip() for t in tickers_str.split(",") if t.strip()]
    formatted_tickers = '["' + '", "'.join(tickers_list) + '"]'
    
    new_content = f"""TICKERS      = {formatted_tickers}
CASH         = {cash}
COMMISSION   = {commission}
TARGET_RISK  = 500          # $ risk per trade for vol-targeted sizing
MIN_TRADES   = 10           # fallback threshold

# Walk-forward windows (months)
WF_TRAIN_MONTHS = {train_m}
WF_TEST_MONTHS  = {test_m}

# General Engine Settings
DATA_DAYS  = {data_days}
INTERVAL   = '{interval}'
"""
    with open(SETTINGS_PATH, "w") as f:
        f.write(new_content)

st.title("📈 Quantitative Research Control Center")
st.markdown("A unified dashboard for globally configuring, executing, and analyzing walk-forward backtests.")

# --- SIDEBAR: GLOBAL CONFIGURATION ---
st.sidebar.header("⚙️ Global Configuration")
current_settings = load_settings()

tickers_val = ", ".join(current_settings.get("TICKERS", []))
cash_val = current_settings.get("CASH", 100000)
comm_val = current_settings.get("COMMISSION", 0.001)
data_days_val = current_settings.get("DATA_DAYS", 1095)
interval_val = current_settings.get("INTERVAL", "1h")
train_m_val = current_settings.get("WF_TRAIN_MONTHS", 6)
test_m_val = current_settings.get("WF_TEST_MONTHS", 2)

with st.sidebar.form("config_form"):
    st.subheader("Capital & Friction")
    new_cash = st.number_input("Initial Cash ($)", value=int(cash_val), step=10000)
    new_comm = st.number_input("Commission (Ratio)", value=float(comm_val), format="%.4f")
    
    st.subheader("Data Universe")
    new_tickers = st.text_area("Tickers (comma separated)", value=tickers_val, height=100)
    new_data_days = st.number_input("Data Days", value=int(data_days_val), step=365)
    
    opts = ["1h", "1d", "1mo"]
    new_interval = st.selectbox("Interval", opts, index=opts.index(interval_val) if interval_val in opts else 0)
    
    st.subheader("Walk-Forward Engine")
    new_train_m = st.number_input("Train Window (Months)", value=int(train_m_val), step=1)
    new_test_m = st.number_input("Test Window (Months)", value=int(test_m_val), step=1)
    
    saved = st.form_submit_button("Update Pipeline Settings")
    if saved:
        save_settings(new_tickers, new_cash, new_comm, new_data_days, new_interval, new_train_m, new_test_m)
        st.success("Global config synchronized!")

# --- MAIN TABS ---
tab_exec, tab_results = st.tabs(["🚀 Execution Engine", "📊 Integrated Results"])

with tab_exec:
    st.header("Launch Backtest")
    st.markdown("Select a strategy from the generalized architecture to run an exhaustive Walk-Forward optimization.")
    
    selected_strategy_name = st.selectbox("Select Strategy Blueprint", list(STRATEGIES.keys()))
    
    if st.button("▶️ Launch Algorithm", type="primary"):
        st.info(f"Running `{STRATEGIES[selected_strategy_name]}` using current global parameters...")
        
        script_path = str(PROJECT_ROOT / STRATEGIES[selected_strategy_name])
        
        terminal_container = st.empty()
        log_output = ""
        
        with st.spinner("Algorithm is computing... monitoring matrix operations."):
            try:
                process = subprocess.Popen(
                    [sys.executable, script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    cwd=str(PROJECT_ROOT)
                )
                
                for line in iter(process.stdout.readline, ""):
                    log_output += line
                    display_text = "\\n".join(log_output.splitlines()[-50:])
                    terminal_container.code(display_text, language="bash")
                
                process.stdout.close()
                return_code = process.wait()
                
                if return_code == 0:
                    st.success("Optimization Concluded Successfully!")
                else:
                    st.error(f"Execution failed with return code {return_code}.")
            except Exception as e:
                st.error(f"Failed to launch process: {e}")

with tab_results:
    st.header("Strategy Reports")
    import streamlit.components.v1 as components
    
    reports_data = []
    
    if REPORTS_DIR.exists():
        md_files = list(REPORTS_DIR.glob("*_report.md"))
        for file in md_files:
            filename = file.name
            if filename == "comparison_report.md":
                reports_data.append({"strategy": "Overview", "ticker": "ALL", "path": file, "type": "markdown"})
                continue
            
            base = filename.replace("_report.md", "")
            parts = base.rsplit("_", 1)
            if len(parts) == 2:
                reports_data.append({"strategy": parts[0].replace("_", " ").title(), "ticker": parts[1].upper(), "path": file, "type": "markdown"})
            else:
                reports_data.append({"strategy": base.title(), "ticker": "UNKNOWN", "path": file, "type": "markdown"})

        html_files = list(REPORTS_DIR.glob("*.html"))
        for file in html_files:
            if file.name == "portfolio_dashboard.html":
                reports_data.append({"strategy": "Portfolio Dashboard", "ticker": "ALL", "path": file, "type": "html"})

    if not reports_data:
        st.info("No reports found yet. Run an algorithm from the Execution Engine to generate reports.")
    else:
        strategies_list = sorted(list(set(d["strategy"] for d in reports_data)))
        for top_item in ["Portfolio Dashboard", "Overview"]:
            if top_item in strategies_list:
                strategies_list.remove(top_item)
                strategies_list.insert(0, top_item)

        col1, col2 = st.columns(2)
        with col1:
            sel_strat = st.selectbox("Select Report Category", strategies_list, key="rep_strat")
        
        filtered = [d for d in reports_data if d["strategy"] == sel_strat]
        tickers_list = sorted(list(set(d["ticker"] for d in filtered)))
        
        with col2:
            if len(tickers_list) > 1 or (len(tickers_list) == 1 and tickers_list[0] != "ALL"):
                sel_tick = st.selectbox("Select Ticker Universe", tickers_list, key="rep_tick")
            else:
                sel_tick = tickers_list[0]
                
        selected_report = next((d for d in filtered if d["ticker"] == sel_tick), None)
        st.divider()
        
        if selected_report and selected_report["path"].exists():
            path = selected_report["path"]
            if selected_report["type"] == "markdown":
                with open(path, "r", encoding="utf-8") as f:
                    st.markdown(f.read())
            elif selected_report["type"] == "html":
                with open(path, "r", encoding="utf-8") as f:
                    components.html(f.read(), height=1000, scrolling=True)
            else:
                st.error("Unknown report type.")
