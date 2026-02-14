"""
Intraday Resistance Breakout Strategy (yfinance)

Uses rolling high/low breakouts with volume confirmation and ATR-based
stop-losses.  Calculates per-stock and equal-weighted portfolio KPIs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import datetime as dt
import copy
import matplotlib.pyplot as plt

from utils.data import fetch_ohlcv_data
from utils.kpi import cagr_from_returns, sharpe_ratio, max_drawdown, volatility
from indicators.atr import calculate_atr

# ----------------------------- CONFIG ----------------------------- #
TICKERS = ["MSFT", "AAPL", "GOOGL", "META", "AMZN", "INTC",
           "CSCO", "VZ", "IBM", "TSLA", "AMD"]
START_DATE = dt.datetime.today() - dt.timedelta(days=59)
END_DATE = dt.datetime.today()
INTERVAL = '5m'
PERIODS_PER_YEAR = 252 * 78
RISK_FREE_RATE = 0.025


# ---------------------- STRATEGY LOGIC ---------------------- #
def run_breakout_strategy(ohlc_dict, atr_period=20, roll_period=20, vol_factor=1.5):
    """
    Breakout strategy: enters on price/volume breakout, exits on ATR stop-loss
    or reversal signal.
    """
    ohlc = copy.deepcopy(ohlc_dict)
    tickers_ret = {}

    for ticker in ohlc:
        df = ohlc[ticker]
        df = calculate_atr(df, atr_period)
        df['roll_max_cp'] = df['High'].rolling(roll_period).max().shift(1)
        df['roll_min_cp'] = df['Low'].rolling(roll_period).min().shift(1)
        df['roll_max_vol'] = df['Volume'].rolling(roll_period).max().shift(1)
        df.dropna(inplace=True)
        ohlc[ticker] = df
        tickers_ret[ticker] = [0] * len(df)

    for ticker in ohlc:
        df = ohlc[ticker]
        signal = ""

        for i in range(1, len(df)):
            close = df['Close'].iloc[i].item()
            prev_close = df['Close'].iloc[i - 1].item()
            high = df['High'].iloc[i].item()
            low = df['Low'].iloc[i].item()
            vol = df['Volume'].iloc[i].item()
            prev_atr = df['ATR'].iloc[i - 1].item()
            r_max = df['roll_max_cp'].iloc[i].item()
            r_min = df['roll_min_cp'].iloc[i].item()
            r_max_vol = df['roll_max_vol'].iloc[i].item()

            if signal == "":
                tickers_ret[ticker][i] = 0
                if high >= r_max and vol > vol_factor * r_max_vol:
                    signal = "Buy"
                elif low <= r_min and vol > vol_factor * r_max_vol:
                    signal = "Sell"

            elif signal == "Buy":
                stop = prev_close - prev_atr
                if low < stop:
                    signal = ""
                    tickers_ret[ticker][i] = (stop / prev_close) - 1
                elif low <= r_min and vol > vol_factor * r_max_vol:
                    signal = "Sell"
                    tickers_ret[ticker][i] = (close / prev_close) - 1
                else:
                    tickers_ret[ticker][i] = (close / prev_close) - 1

            elif signal == "Sell":
                stop = prev_close + prev_atr
                if high > stop:
                    signal = ""
                    tickers_ret[ticker][i] = (prev_close / stop) - 1
                elif high >= r_max and vol > vol_factor * r_max_vol:
                    signal = "Buy"
                    tickers_ret[ticker][i] = (prev_close / close) - 1
                else:
                    tickers_ret[ticker][i] = (prev_close / close) - 1

        ohlc[ticker]['ret'] = np.array(tickers_ret[ticker])
    return ohlc


# ----------------------------- MAIN ----------------------------- #
def main():
    print("Fetching intraday OHLCV data...")
    ohlcv = fetch_ohlcv_data(
        TICKERS, start=START_DATE, end=END_DATE, interval=INTERVAL, auto_adjust=True
    )
    # Filter trading hours
    for t in ohlcv:
        ohlcv[t] = ohlcv[t][['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        ohlcv[t] = ohlcv[t].between_time('09:35', '16:00')

    tickers = list(ohlcv.keys())
    if not tickers:
        print("No data fetched. Exiting.")
        return

    print("Running breakout strategy...")
    bt = run_breakout_strategy(ohlcv)

    strategy_df = pd.DataFrame({t: bt[t]['ret'] for t in bt})
    strategy_df['ret'] = strategy_df.mean(axis=1)

    print("\n" + "=" * 30)
    print("--- Overall Strategy KPIs ---")
    print("=" * 30)
    print(f"CAGR: {cagr_from_returns(strategy_df['ret'], PERIODS_PER_YEAR):.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio(strategy_df['ret'], RISK_FREE_RATE, PERIODS_PER_YEAR):.4f}")
    print(f"Max Drawdown: {max_drawdown(strategy_df['ret']):.4f}")

    plt.figure(figsize=(12, 6))
    (1 + strategy_df['ret']).cumprod().plot(title="Cumulative Returns: Intraday Breakout")
    plt.xlabel("5-Minute Periods")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n--- Individual Stock KPIs ---")
    kpi = {}
    for t in tickers:
        r = bt[t]['ret']
        kpi[t] = [
            cagr_from_returns(r, PERIODS_PER_YEAR),
            sharpe_ratio(r, RISK_FREE_RATE, PERIODS_PER_YEAR),
            max_drawdown(r),
        ]
    kpi_df = pd.DataFrame.from_dict(kpi, orient='index',
                                     columns=["CAGR", "Sharpe Ratio", "Max Drawdown"])
    print(kpi_df.to_markdown(floatfmt=".4f"))


if __name__ == '__main__':
    main()
