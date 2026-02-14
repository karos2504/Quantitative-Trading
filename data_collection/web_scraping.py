"""
Financial Data Scraping

Fetches balance sheets, income statements, cash flows, and key stats
for a list of tickers using yfinance.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from utils.data import fetch_financial_data


TICKERS = ["AAPL", "MSFT"]


def main():
    financials = {}
    for ticker in TICKERS:
        print(f"Processing {ticker}...")
        data = fetch_financial_data(ticker)
        if data:
            financials[ticker] = data

    df_balance = pd.concat(
        {t: d['balance_sheet'] for t, d in financials.items()}, axis=1
    )
    df_income = pd.concat(
        {t: d['income_statement'] for t, d in financials.items()}, axis=1
    )
    df_cashflow = pd.concat(
        {t: d['cash_flow'] for t, d in financials.items()}, axis=1
    )
    df_stats = pd.DataFrame(
        {t: d['key_stats'] for t, d in financials.items()}
    ).T

    print("\nBalance Sheets:")
    print(df_balance)
    print("\nIncome Statements:")
    print(df_income)
    print("\nCash Flows:")
    print(df_cashflow)
    print("\nKey Statistics:")
    print(df_stats)


if __name__ == '__main__':
    main()
