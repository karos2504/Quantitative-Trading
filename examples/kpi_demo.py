"""
KPI Demo

Demonstrates all KPI functions from portfolio_construction.kpi using sample stock data.
Replaces the 4 separate KPI scripts that were in the old project.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_ingestion.data import fetch_ohlcv_data
from portfolio_construction.kpi import (
    cagr_from_prices,
    volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown_from_prices,
    calmar_ratio,
)

TICKERS = ['AAPL', 'MSFT', 'GOOG']
PERIOD = '1y'
INTERVAL = '1d'
PERIODS_PER_YEAR = 252
RISK_FREE_RATE = 0.03


def main():
    data = fetch_ohlcv_data(TICKERS, period=PERIOD, interval=INTERVAL)

    for ticker in TICKERS:
        if ticker not in data:
            print(f"No data for {ticker}")
            continue

        df = data[ticker]
        returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()

        cagr = cagr_from_prices(df, PERIODS_PER_YEAR)
        vol = volatility(returns, PERIODS_PER_YEAR)
        sharpe = sharpe_ratio(returns, RISK_FREE_RATE, PERIODS_PER_YEAR)
        sortino = sortino_ratio(returns, RISK_FREE_RATE, PERIODS_PER_YEAR)
        mdd = max_drawdown_from_prices(df)

        # Calmar uses returns-based CAGR internally
        calmar = calmar_ratio(returns, PERIODS_PER_YEAR)

        print(f"\n--- {ticker} ---")
        print(f"  CAGR:          {cagr:.4f}")
        print(f"  Volatility:    {vol:.4f}")
        print(f"  Sharpe Ratio:  {sharpe:.4f}")
        print(f"  Sortino Ratio: {sortino:.4f}")
        print(f"  Max Drawdown:  {mdd:.4f}")
        print(f"  Calmar Ratio:  {calmar:.4f}")


if __name__ == '__main__':
    main()
