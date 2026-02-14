"""
Renko + MACD Strategy

Uses Renko brick patterns combined with MACD/Signal line slopes for
entry/exit signals.  Backtests on 5-minute intraday data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import copy
import yfinance as yf

from indicators.renko import convert_to_renko
from indicators.macd import calculate_macd
from indicators.slope import calculate_slope
from utils.kpi import cagr_from_returns, sharpe_ratio, max_drawdown

# ----------------------------- CONFIG ----------------------------- #
TICKERS = ["MSFT", "AAPL", "GOOGL", "META", "AMZN", "INTC",
           "CSCO", "VZ", "IBM", "TSLA", "AMD"]
PERIODS_PER_YEAR = 252 * 78
RISK_FREE_RATE = 0.025


# ---------------------- STRATEGY LOGIC ---------------------- #
def run_renko_macd_strategy(ohlc_dict):
    """
    Renko + MACD strategy:
    - Buy when bar_num >= 2, MACD > Signal, and MACD slope > Signal slope
    - Sell when bar_num <= -2, MACD < Signal, and MACD slope < Signal slope
    """
    ohlc = copy.deepcopy(ohlc_dict)
    tickers_signal = {}
    tickers_ret = {}
    ohlc_merged = {}

    # 1. Merge Renko + MACD indicators
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

        # MACD
        merged = calculate_macd(merged, fast=12, slow=26, signal=9)
        merged.dropna(subset=['MACD', 'Signal', 'bar_num'], inplace=True)

        # Slopes
        if len(merged) >= 5:
            merged['macd_slope'] = calculate_slope(merged['MACD'], 5)
            merged['signal_slope'] = calculate_slope(merged['Signal'], 5)
        else:
            merged['macd_slope'] = np.nan
            merged['signal_slope'] = np.nan
            merged.dropna(inplace=True)

        ohlc_merged[ticker] = merged
        tickers_signal[ticker] = ""
        tickers_ret[ticker] = []

    # 2. Backtest
    for ticker in ohlc_merged:
        df = ohlc_merged[ticker].reset_index(drop=True)
        if df.empty or len(df) <= 1:
            print(f"Skipping {ticker}: insufficient data")
            continue

        for i in range(len(df)):
            if i == 0:
                tickers_ret[ticker].append(0)
                continue

            row = df.iloc[i]
            prev_close = df['Adj Close'].iloc[i - 1]
            current_close = row['Adj Close']

            if tickers_signal[ticker] == "":
                tickers_ret[ticker].append(0)
                if (row['bar_num'] >= 2 and row['MACD'] > row['Signal']
                        and row['macd_slope'] > row['signal_slope']):
                    tickers_signal[ticker] = "Buy"
                elif (row['bar_num'] <= -2 and row['MACD'] < row['Signal']
                      and row['macd_slope'] < row['signal_slope']):
                    tickers_signal[ticker] = "Sell"

            elif tickers_signal[ticker] == "Buy":
                tickers_ret[ticker].append((current_close / prev_close) - 1)
                if (row['bar_num'] <= -2 and row['MACD'] < row['Signal']
                        and row['macd_slope'] < row['signal_slope']):
                    tickers_signal[ticker] = "Sell"
                elif (row['MACD'] < row['Signal']
                      and row['macd_slope'] < row['signal_slope']):
                    tickers_signal[ticker] = ""

            elif tickers_signal[ticker] == "Sell":
                tickers_ret[ticker].append((prev_close / current_close) - 1)
                if (row['bar_num'] >= 2 and row['MACD'] > row['Signal']
                        and row['macd_slope'] > row['signal_slope']):
                    tickers_signal[ticker] = "Buy"
                elif (row['MACD'] > row['Signal']
                      and row['macd_slope'] > row['signal_slope']):
                    tickers_signal[ticker] = ""

        if tickers_ret[ticker]:
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

    print("\nRunning Renko + MACD strategy...")
    results = run_renko_macd_strategy(ohlc_intraday)

    # Portfolio KPIs
    strategy_df = pd.DataFrame()
    for t in tickers:
        if 'ret' in results[t].columns:
            strategy_df[t] = results[t]['ret']

    if not strategy_df.empty:
        strategy_df['ret'] = strategy_df.mean(axis=1)

        print("\n--- 🎯 Overall Strategy KPIs (Renko + MACD) ---")
        print(f"CAGR: {cagr_from_returns(strategy_df['ret'], PERIODS_PER_YEAR) * 100:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio(strategy_df['ret'], RISK_FREE_RATE, PERIODS_PER_YEAR):.2f}")
        print(f"Max Drawdown: {max_drawdown(strategy_df['ret']) * 100:.2f}%")

        # Individual KPIs
        kpi = {}
        for t in tickers:
            if 'ret' in results[t].columns and not results[t]['ret'].empty:
                r = results[t]['ret']
                kpi[t] = {
                    'Return': cagr_from_returns(r, PERIODS_PER_YEAR),
                    'Sharpe Ratio': sharpe_ratio(r, RISK_FREE_RATE, PERIODS_PER_YEAR),
                    'Max Drawdown': max_drawdown(r),
                }

        print("\n--- 📈 Individual Stock KPIs (Renko + MACD) ---")
        print(pd.DataFrame(kpi).T)
    else:
        print("\n⚠️ Not enough data for backtesting.")


if __name__ == '__main__':
    main()
