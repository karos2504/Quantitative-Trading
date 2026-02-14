"""
Renko + OBV Strategy

Uses Renko brick patterns combined with OBV slope for entry/exit signals.
Backtests on 5-minute intraday data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import copy
import yfinance as yf

from indicators.renko import convert_to_renko
from indicators.obv import calculate_obv
from indicators.slope import calculate_slope
from utils.kpi import cagr_from_returns, sharpe_ratio, max_drawdown, volatility

# ----------------------------- CONFIG ----------------------------- #
TICKERS = ["MSFT", "AAPL", "GOOGL", "META", "AMZN", "INTC",
           "CSCO", "VZ", "IBM", "TSLA", "AMD"]
PERIODS_PER_YEAR = 252 * 78
RISK_FREE_RATE = 0.025


# ---------------------- STRATEGY LOGIC ---------------------- #
def run_renko_obv_strategy(ohlc_dict):
    """
    Renko + OBV strategy:
    - Buy when bar_num >= 2 and OBV slope > 30°
    - Sell when bar_num <= -2 and OBV slope < -30°
    - Exit when trend weakens
    """
    ohlc = copy.deepcopy(ohlc_dict)
    tickers_signal = {}
    tickers_ret = {}
    ohlc_merged = {}

    # 1. Merge Renko + OBV indicators
    for ticker in ohlc:
        print(f"📊 Processing {ticker}")
        df = ohlc[ticker]

        # Renko
        renko = convert_to_renko(df)
        df['Date'] = df.index
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

        if not renko.empty:
            renko['date'] = pd.to_datetime(renko['date']).dt.tz_localize(None)
            renko.rename(columns={'date': 'Date'}, inplace=True)
            merged = df.merge(renko[['Date', 'bar_num']], how='outer', on='Date')
        else:
            merged = df.copy()
            merged['bar_num'] = np.nan

        merged['bar_num'] = merged['bar_num'].ffill()

        # OBV + slope
        merged = calculate_obv(merged)
        merged['obv_slope'] = calculate_slope(merged['OBV'], 5)

        merged.dropna(subset=['bar_num'], inplace=True)
        ohlc_merged[ticker] = merged
        tickers_signal[ticker] = ""
        tickers_ret[ticker] = []

    # 2. Backtest
    for ticker in ohlc_merged:
        df = ohlc_merged[ticker]
        for i in range(len(df)):
            if i == 0:
                tickers_ret[ticker].append(0)
                continue

            row = df.iloc[i]
            prev_close = df['Adj Close'].iloc[i - 1]
            current_close = row['Adj Close']

            if tickers_signal[ticker] == "":
                tickers_ret[ticker].append(0)
                if row['bar_num'] >= 2 and row['obv_slope'] > 30:
                    tickers_signal[ticker] = "Buy"
                elif row['bar_num'] <= -2 and row['obv_slope'] < -30:
                    tickers_signal[ticker] = "Sell"

            elif tickers_signal[ticker] == "Buy":
                tickers_ret[ticker].append((current_close / prev_close) - 1)
                if row['bar_num'] <= -2 and row['obv_slope'] < -30:
                    tickers_signal[ticker] = "Sell"
                elif row['bar_num'] < 2:
                    tickers_signal[ticker] = ""

            elif tickers_signal[ticker] == "Sell":
                tickers_ret[ticker].append((prev_close / current_close) - 1)
                if row['bar_num'] >= 2 and row['obv_slope'] > 30:
                    tickers_signal[ticker] = "Buy"
                elif row['bar_num'] > -2:
                    tickers_signal[ticker] = ""

        ohlc_merged[ticker]['ret'] = np.array(tickers_ret[ticker])

    return ohlc_merged


# ----------------------------- MAIN ----------------------------- #
def main():
    print("--- Downloading 5-minute data ---")
    ohlc_intraday = {}
    for ticker in TICKERS:
        try:
            data = yf.download(ticker, interval='5m', period='60d',
                               progress=False, auto_adjust=True)
            data.columns = ["Open", "High", "Low", "Adj Close", "Volume"]
            data['Close'] = data['Adj Close']
            data.dropna(inplace=True)
            ohlc_intraday[ticker] = data
            print(f"✅ {ticker}: {len(data)} rows")
        except Exception as e:
            print(f"❌ {ticker}: {e}")

    tickers = list(ohlc_intraday.keys())
    if not tickers:
        raise ValueError("No data downloaded.")

    print("\nRunning Renko + OBV strategy...")
    results = run_renko_obv_strategy(ohlc_intraday)

    # Portfolio KPIs
    strategy_df = pd.DataFrame({t: results[t]['ret'] for t in tickers})
    strategy_df['ret'] = strategy_df.mean(axis=1)

    print("\n--- 🎯 Overall Strategy KPIs ---")
    print(f"CAGR: {cagr_from_returns(strategy_df['ret'], PERIODS_PER_YEAR) * 100:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio(strategy_df['ret'], RISK_FREE_RATE, PERIODS_PER_YEAR):.2f}")
    print(f"Max Drawdown: {max_drawdown(strategy_df['ret']) * 100:.2f}%")

    # Individual KPIs
    kpi = {}
    for t in tickers:
        r = results[t]['ret']
        kpi[t] = {
            'Return': cagr_from_returns(r, PERIODS_PER_YEAR),
            'Sharpe Ratio': sharpe_ratio(r, RISK_FREE_RATE, PERIODS_PER_YEAR),
            'Max Drawdown': max_drawdown(r),
        }

    print("\n--- 📈 Individual Stock KPIs ---")
    print(pd.DataFrame(kpi).T)


if __name__ == '__main__':
    main()
