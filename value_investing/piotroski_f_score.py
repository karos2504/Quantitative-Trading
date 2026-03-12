"""
Piotroski F-Score

Calculates the 9-point Piotroski F-Score for fundamental stock screening
across profitability, leverage/liquidity, and operating efficiency.
"""

import yfinance as yf
import pandas as pd
import numpy as np


TICKERS = [
    "MMM", "AXP", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO", "DIS", "XOM",
    "GE", "GS", "HD", "IBM", "INTC", "JNJ", "JPM", "MCD", "MRK",
    "MSFT", "NKE", "PFE", "PG", "TRV", "UNH", "VZ", "V", "WMT",
]

# Financial institutions excluded (ratios unreliable for F-Score)
FINANCIAL_TICKERS = {"AXP", "GS", "JPM", "TRV", "V"}

STATS_MAP = {
    "NetIncome": ["Net Income"],
    "TotAssets": ["Total Assets"],
    "CashFlowOps": ["Cash Flow From Operating Activities",
                     "Total Cash From Operating Activities",
                     "Operating Cash Flow"],
    "LTDebt": ["Long Term Debt",
               "Long Term Debt And Capital Lease Obligation",
               "Non Current Debt"],
    "CurrAssets": ["Current Assets", "Total Current Assets",
                    "Total Assets Net Of Current Liabilities"],
    "CurrLiab": ["Current Liabilities", "Total Current Liabilities",
                  "Current Liabilities And Debt"],
    "CommStock": ["Ordinary Shares Number", "Share Issued", "Common Stock"],
    "TotRevenue": ["Total Revenue"],
    "GrossProfit": ["Gross Profit"],
}


def fetch_financial_data(ticker):
    """Fetch and align the 9 required Piotroski metrics for 2 most recent years."""
    stock = yf.Ticker(ticker)
    try:
        bs = stock.balance_sheet
        is_ = stock.financials
        cf = stock.cashflow
    except Exception:
        return None

    combined = pd.concat([is_, bs, cf], axis=0)
    required = {}

    for label, fields in STATS_MAP.items():
        series = pd.Series([], dtype=float)
        for field in fields:
            if field in combined.index:
                series = combined.loc[field]
                break
        if series.empty:
            return None

        values = series.head(2)
        if len(values) < 2:
            values = values.reindex(
                values.index.union(pd.Index(['Year_1_Placeholder']))
            ).head(2)
        required[label] = values.values

    df = pd.DataFrame(required).T
    df.columns = ['Year_0', 'Year_1']
    return df


def piotroski_f_score(all_data):
    """Calculate Piotroski F-Score (0-9) for each stock."""
    scores = {}

    for ticker, data in all_data.items():
        if ticker in FINANCIAL_TICKERS:
            continue
        try:
            cy, py = data['Year_0'], data['Year_1']

            def val(x):
                return float(x if not pd.isna(x) else 0.0)

            def safe_div(n, d):
                n, d = val(n), val(d)
                return n / d if d != 0 else 0.0

            # Ratios
            roa_cy = safe_div(cy["NetIncome"], cy["TotAssets"])
            cfo_roa = safe_div(cy["CashFlowOps"], cy["TotAssets"])
            cr_cy = safe_div(cy["CurrAssets"], cy["CurrLiab"])
            gm_cy = safe_div(cy["GrossProfit"], cy["TotRevenue"])
            ato_cy = safe_div(cy["TotRevenue"], cy["TotAssets"])

            roa_py = safe_div(py["NetIncome"], py["TotAssets"])
            cr_py = safe_div(py["CurrAssets"], py["CurrLiab"])
            gm_py = safe_div(py["GrossProfit"], py["TotRevenue"])
            ato_py = safe_div(py["TotRevenue"], py["TotAssets"])

            # Profitability (4 pts)
            f1 = int(roa_cy > 0)
            f2 = int(val(cy["CashFlowOps"]) > 0)
            f3 = int(roa_cy > roa_py)
            f4 = int(cfo_roa > roa_cy)

            # Leverage / Liquidity (3 pts)
            lev_cy = safe_div(cy["LTDebt"], cy["TotAssets"])
            lev_py = safe_div(py["LTDebt"], py["TotAssets"])
            # F5: Score 1 if the leverage ratio fell or if it remained zero
            f5 = int(lev_cy < lev_py or (lev_cy == 0 and lev_py == 0))
            f6 = int(cr_cy > cr_py)
            f7 = int(val(cy["CommStock"]) <= val(py["CommStock"]))

            # Operating Efficiency (2 pts)
            f8 = int(gm_cy > gm_py)
            f9 = int(ato_cy > ato_py)

            scores[ticker] = sum([f1, f2, f3, f4, f5, f6, f7, f8, f9])
        except Exception:
            pass

    return pd.Series(scores, name="F-Score").sort_values(ascending=False)


def main():
    print("Fetching financial data...")
    all_data = {}
    for ticker in TICKERS:
        df = fetch_financial_data(ticker)
        if df is not None:
            all_data[ticker] = df

    f_scores = piotroski_f_score(all_data)
    print("\nTop Piotroski F-Score Stocks:")
    print(f_scores.head(10))


if __name__ == '__main__':
    main()
