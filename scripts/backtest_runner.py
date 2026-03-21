import subprocess
import os
import re
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
STRATEGIES_DIR = BASE_DIR / "strategies"
REPORTS_DIR = STRATEGIES_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

STRATEGIES = [
    "renko_macd.py",
    "renko_obv.py",
    "resistance_breakout.py"
]

def run_strategy(strategy_file):
    strategy_path = STRATEGIES_DIR / strategy_file
    strategy_name = strategy_file.replace(".py", "")
    log_file = REPORTS_DIR / f"{strategy_name}_backtest.log"
    
    print(f"Running {strategy_name}...")
    
    # Run the strategy and capture output
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            ["python3", str(strategy_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(BASE_DIR)
        )
        
        for line in process.stdout:
            print(line, end="")
            f.write(line)
            
    process.wait()
    print(f"Finished {strategy_name}. Log saved to {log_file}")
    return log_file

def parse_final_results(log_file):
    with open(log_file, "r") as f:
        lines = f.readlines()
        
    start_line = -1
    for i, line in enumerate(lines):
        if "FINAL BACKTEST RESULTS" in line:
            # Table header starts 2 lines after this header
            start_line = i + 2
            break
            
    if start_line == -1:
        return None
        
    table_lines = []
    for line in lines[start_line:]:
        if line.strip() == "" or "─" in line or "==" in line:
            if table_lines: # Already found the table
                break
            continue
        table_lines.append(line)
        
    if not table_lines:
        return None
        
    # First line is header
    header_str = table_lines[0].strip()
    # Replace known multi-word headers with single tokens
    header_str = header_str.replace("Return [%]", "Return_%")
    header_str = header_str.replace("Sharpe Ratio", "Sharpe_Ratio")
    header_str = header_str.replace("Max Drawdown [%]", "Max_DD_%")
    header_str = header_str.replace("# Trades", "Num_Trades")
    header_str = header_str.replace("Win Rate [%]", "Win_Rate_%")
    new_header = header_str.split()
            
    data = []
    for line in table_lines[1:]:
        parts = line.strip().split()
        if len(parts) >= len(new_header) + 1: # +1 for Ticker
            data.append(parts[:len(new_header)+1])
            
    df = pd.DataFrame(data, columns=["Ticker"] + new_header)
    return df

def generate_comparison_report(results_dict):
    report_file = REPORTS_DIR / "comparison_report.md"
    
    with open(report_file, "w") as f:
        f.write("# Unified Strategy Comparison Report\n\n")
        f.write(f"Generated on: {pd.Timestamp.now()}\n\n")
        
        for name, df in results_dict.items():
            if df is not None:
                f.write(f"## Strategy: {name}\n\n")
                f.write(df.to_markdown(index=False))
                f.write("\n\n")
            else:
                f.write(f"## Strategy: {name}\n\n")
                f.write("Failed to parse results.\n\n")
                
    print(f"Comparison report generated at {report_file}")

def main(force_run=False):
    results = {}
    for strategy in STRATEGIES:
        strategy_name_raw = strategy.replace(".py", "")
        log_file = REPORTS_DIR / f"{strategy_name_raw}_backtest.log"
        
        if force_run or not log_file.exists():
            log_file = run_strategy(strategy)
        else:
            print(f"Using existing log for {strategy_name_raw}")
            
        df = parse_final_results(log_file)
        strategy_name = strategy_name_raw.replace("_", " ").title()
        results[strategy_name] = df
        
    generate_comparison_report(results)

if __name__ == "__main__":
    main(force_run=True)
