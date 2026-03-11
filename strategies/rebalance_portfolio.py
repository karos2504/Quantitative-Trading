"""
Monthly Portfolio Rebalancing Strategy

Picks top performers each month and drops the worst, comparing
the strategy's cumulative returns against the DJI benchmark.
Includes advanced backtesting: Monte Carlo, Walk-Forward, Stress Testing,
and Deflated Sharpe Ratio via vectorbt.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt

from utils.data import fetch_ohlcv_data
from utils.backtesting import VBTBacktester

# ----------------------------- CONFIG ----------------------------- #
TICKERS = [
    "MMM", "AXP", "T", "BA", "CAT", "CSCO", "KO", "XOM", "GE", "GS", "HD",
    "IBM", "INTC", "JNJ", "JPM", "MCD", "MRK", "MSFT", "NKE", "PFE", "PG",
    "TRV", "UNH", "VZ", "V", "WMT", "DIS",
]
START_DATE = dt.datetime.today() - dt.timedelta(days=365 * 10)
END_DATE = dt.datetime.today()
INTERVAL = '1mo'


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

    Returns:
        pd.Series: monthly strategy returns (aligned to returns_df index).
    """
    portfolio = []
    monthly_returns = []

    for i in range(len(returns_df)):
        if portfolio:
            monthly_returns.append(returns_df[portfolio].iloc[i].mean())
            worst = returns_df[portfolio].iloc[i].nsmallest(drop_count).index.tolist()
            portfolio = [s for s in portfolio if s not in worst]
        else:
            monthly_returns.append(0.0)

        needed = portfolio_size - len(portfolio)
        top = returns_df.iloc[i].nlargest(needed).index.tolist()
        portfolio += top

    return pd.Series(monthly_returns, index=returns_df.index, name="Monthly Return")


def generate_signals(strategy_returns):
    """
    Convert strategy returns into vectorbt-compatible entry/exit signals.
    Entry when return > 0, exit when return <= 0.
    """
    entries = strategy_returns > 0
    exits = strategy_returns <= 0
    return entries, exits


# ----------------------------- MAIN ----------------------------- #
def main():
    print("=" * 60)
    print("  Monthly Portfolio Rebalancing — Advanced Backtest")
    print("=" * 60)

    # Fetch data
    ohlcv_data = fetch_ohlcv_data(
        TICKERS, start=START_DATE, end=END_DATE, interval=INTERVAL
    )
    returns_df = calculate_monthly_returns(ohlcv_data)
    strategy_returns = run_portfolio_strategy(returns_df, portfolio_size=6, drop_count=3)

    # Benchmark
    dji = yf.download("^DJI", start=START_DATE, end=END_DATE,
                       interval=INTERVAL, auto_adjust=False, progress=False)
    dji_close = dji['Adj Close'].squeeze()

    # Build synthetic close from strategy returns
    strategy_close = (1 + strategy_returns).cumprod() * 100  # Start at $100
    entries, exits = generate_signals(strategy_returns)

    # --- Advanced Backtest ---
    bt = VBTBacktester(
        close=strategy_close,
        entries=entries,
        exits=exits,
        freq='30D',  # Monthly
        init_cash=100_000,
        commission=0.001,
    )

    bt.full_analysis(
        n_mc=1000,
        n_wf_splits=5,
        n_trials=1,
    )

    # --- ML/DL/RL Signal Enhancement ---
    from utils.ml_signals import run_ml_comparison
    synth_df = pd.DataFrame({
        'Open': strategy_close,
        'High': strategy_close * 1.01,
        'Low': strategy_close * 0.99,
        'Close': strategy_close,
        'Volume': np.ones(len(strategy_close)) * 1e6,
    }, index=strategy_close.index)
    run_ml_comparison(synth_df, entries, exits, 'Portfolio', freq='30D')


if __name__ == '__main__':
    main()
