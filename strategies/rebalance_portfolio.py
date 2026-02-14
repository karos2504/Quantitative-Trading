"""
Monthly Portfolio Rebalancing Strategy

Picks top performers each month and drops the worst, comparing
the strategy's cumulative returns against the DJI benchmark.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt

from utils.data import fetch_ohlcv_data
from utils.kpi import (
    cagr_from_prices,
    cagr_from_returns,
    sharpe_ratio,
    max_drawdown,
)

# ----------------------------- CONFIG ----------------------------- #
TICKERS = [
    "MMM", "AXP", "T", "BA", "CAT", "CSCO", "KO", "XOM", "GE", "GS", "HD",
    "IBM", "INTC", "JNJ", "JPM", "MCD", "MRK", "MSFT", "NKE", "PFE", "PG",
    "TRV", "UNH", "VZ", "V", "WMT", "DIS",
]
START_DATE = dt.datetime.today() - dt.timedelta(days=365 * 10)
END_DATE = dt.datetime.today()
INTERVAL = '1mo'
PERIODS_PER_YEAR = 12
RISK_FREE_RATE = 0.025


# ---------------------- MONTHLY RETURNS ---------------------- #
def calculate_monthly_returns(data):
    """Calculate monthly returns for each ticker."""
    returns = pd.DataFrame()
    for ticker, df in data.items():
        returns[ticker] = df['Adj Close'].pct_change()
    return returns.dropna()


# ---------------------- STRATEGY LOGIC ---------------------- #
def run_portfolio_strategy(returns_df, portfolio_size=6, drop_count=3):
    """
    Monthly rebalancing: pick top `portfolio_size` performers,
    drop `drop_count` worst each month.
    """
    portfolio = []
    monthly_returns = [0]

    for i in range(len(returns_df)):
        if portfolio:
            monthly_returns.append(returns_df[portfolio].iloc[i].mean())

            worst = returns_df[portfolio].iloc[i].nsmallest(drop_count).index.tolist()
            portfolio = [s for s in portfolio if s not in worst]

        needed = portfolio_size - len(portfolio)
        top = returns_df.iloc[i].nlargest(needed).index.tolist()
        portfolio += top

    return pd.DataFrame(monthly_returns, columns=["Monthly Return"]).iloc[1:]


# ---------------------- PLOT ---------------------- #
def plot_strategy_vs_benchmark(strategy_returns, benchmark_returns, title):
    """Plot cumulative returns: strategy vs benchmark."""
    plt.figure(figsize=(12, 6))
    plt.plot((1 + strategy_returns).cumprod().reset_index(drop=True),
             label="Strategy", linewidth=2)
    plt.plot((1 + benchmark_returns).cumprod().reset_index(drop=True),
             label="Benchmark (DJI)", linewidth=2, linestyle='--')
    plt.title(title)
    plt.xlabel("Months")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# ----------------------------- MAIN ----------------------------- #
def main():
    ohlcv_data = fetch_ohlcv_data(
        TICKERS, start=START_DATE, end=END_DATE, interval=INTERVAL
    )
    returns_df = calculate_monthly_returns(ohlcv_data)
    strategy_result = run_portfolio_strategy(returns_df, portfolio_size=6, drop_count=3)

    # Benchmark
    dji = yf.download("^DJI", start=START_DATE, end=END_DATE,
                       interval=INTERVAL, auto_adjust=False, progress=False)
    dji['Monthly Return'] = dji['Adj Close'].pct_change().dropna()

    # Strategy KPIs
    print("\n--- Strategy KPIs ---")
    s_ret = strategy_result["Monthly Return"]
    print(f"CAGR: {cagr_from_returns(s_ret, PERIODS_PER_YEAR):.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio(s_ret, RISK_FREE_RATE, PERIODS_PER_YEAR):.4f}")
    print(f"Max Drawdown: {max_drawdown(s_ret):.4f}")

    # Benchmark KPIs
    print("\n--- DJI Benchmark KPIs ---")
    print(f"CAGR: {cagr_from_prices(dji, PERIODS_PER_YEAR):.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio(dji['Monthly Return'], RISK_FREE_RATE, PERIODS_PER_YEAR):.4f}")
    print(f"Max Drawdown: {max_drawdown(dji['Monthly Return']):.4f}")

    plot_strategy_vs_benchmark(s_ret, dji["Monthly Return"],
                                "Cumulative Returns: Strategy vs DJI")


if __name__ == '__main__':
    main()
