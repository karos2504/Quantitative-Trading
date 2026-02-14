"""
Magic Formula Investing

Implements Joel Greenblatt's Magic Formula, combining Earnings Yield
and Return on Capital to rank stocks.  Optionally includes Dividend Yield.
"""

import yfinance as yf
import pandas as pd


TICKERS = [
    "AAPL", "MSFT", "V", "JNJ", "WMT", "JPM", "PG", "MA", "NVDA", "UNH",
    "HD", "VZ", "MRK", "KO", "PEP", "XOM", "CVX", "ABBV", "PFE", "T",
    "INTC", "CSCO", "BA", "MCD", "NKE",
]


def fetch_magic_formula_data(ticker):
    """Fetch financial data needed for Magic Formula ranking."""
    t = yf.Ticker(ticker)
    try:
        info = t.info
    except Exception:
        return None

    bs = t.balance_sheet
    inc = t.financials
    cf = t.cashflow

    def safe(df, key):
        try:
            return df.loc[key].dropna().values[0]
        except Exception:
            return None

    return {
        "EBIT": safe(inc, "EBIT"),
        "MarketCap": info.get("marketCap"),
        "CashFlowOps": safe(cf, "Total Cash From Operating Activities"),
        "Capex": safe(cf, "Capital Expenditures"),
        "CurrAsset": safe(bs, "Total Current Assets"),
        "CurrLiab": safe(bs, "Total Current Liabilities"),
        "PPE": safe(bs, "Property Plant And Equipment Net"),
        "BookValue": safe(bs, "Total Stockholder Equity"),
        "TotDebt": safe(bs, "Long Term Debt") or 0,
        "PrefStock": safe(bs, "Preferred Stock") or 0,
        "MinInterest": 0,
        "DivYield": info.get("dividendYield"),
    }


def main():
    # Fetch data
    financials = {}
    for ticker in TICKERS:
        data = fetch_magic_formula_data(ticker)
        if data:
            financials[ticker] = data

    df = pd.DataFrame.from_dict(financials, orient='index')
    df = df.apply(pd.to_numeric, errors='coerce')

    # Total Enterprise Value
    df["TEV"] = (
        df["MarketCap"].fillna(0)
        + df["TotDebt"].fillna(0)
        + df["PrefStock"].fillna(0)
        + df["MinInterest"].fillna(0)
        - (df["CurrAsset"].fillna(0) - df["CurrLiab"].fillna(0))
    )

    df["EarningYield"] = df["EBIT"] / df["TEV"]
    df["FCFYield"] = (df["CashFlowOps"] - df["Capex"]) / df["MarketCap"]

    # Invested Capital with floor to avoid division by zero
    ic = df["PPE"].fillna(0) + df["CurrAsset"].fillna(0) - df["CurrLiab"].fillna(0)
    df["ROC"] = df["EBIT"] / ic.apply(lambda x: max(x, 1000))

    df = df.dropna(subset=["EarningYield", "ROC", "DivYield"])

    # Rankings
    df["Rank_EY"] = df["EarningYield"].rank(ascending=False)
    df["Rank_ROC"] = df["ROC"].rank(ascending=False)
    df["CombRank"] = df["Rank_EY"] + df["Rank_ROC"]
    df["MagicFormulaRank"] = df["CombRank"].rank()

    # Combined with Dividend Yield
    df["Rank_Div"] = df["DivYield"].rank(ascending=False)
    df["CombinedRank"] = (df["Rank_EY"] + df["Rank_ROC"] + df["Rank_Div"]).rank()

    pd.set_option('display.float_format', lambda x: f'{x:,.4f}')

    print("\n=== Value Stocks — Magic Formula ===")
    print(df.sort_values("MagicFormulaRank")[["EarningYield", "ROC", "MagicFormulaRank"]].head(10))

    print("\n=== Highest Dividend Yield ===")
    print(df.sort_values("DivYield", ascending=False)["DivYield"].head(10))

    print("\n=== Magic Formula + Dividend Yield ===")
    print(df.sort_values("CombinedRank")[["EarningYield", "ROC", "DivYield", "CombinedRank"]].head(10))


if __name__ == '__main__':
    main()
