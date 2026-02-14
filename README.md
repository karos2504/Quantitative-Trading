# Quantitative Trading

A modular Python toolkit for quantitative trading research, covering technical indicators, backtesting strategies, KPI analysis, sentiment analysis, and value investing.

## Project Structure

```
├── utils/              # Shared data fetching and KPI calculations
├── indicators/         # Technical indicators (ATR, ADX, RSI, MACD, OBV, BB, Renko, Slope)
├── strategies/         # Backtesting strategies
│   ├── rebalance_portfolio.py       # Monthly portfolio rebalancing
│   ├── resistance_breakout.py       # Intraday resistance breakout (yfinance)
│   ├── resistance_breakout_alpha_vantage.py  # Same strategy via Alpha Vantage
│   ├── renko_obv.py                 # Renko + OBV strategy
│   └── renko_macd.py                # Renko + MACD strategy
├── sentiment/          # Sentiment analysis (VADER, TextBlob, Naive Bayes, NLP)
├── value_investing/    # Fundamental analysis (Magic Formula, Piotroski F-Score)
├── data_collection/    # Financial data scraping
├── examples/           # Demo scripts
├── data/               # Model artifacts and datasets
└── assets/             # Chart images
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run any module from the project root:

```bash
# KPI demo
python -m examples.kpi_demo

# Backtest a strategy
python -m strategies.rebalance_portfolio
python -m strategies.resistance_breakout
python -m strategies.renko_obv

# Technical indicators
python -m indicators.macd
python -m indicators.rsi

# Sentiment analysis
python -m sentiment.crude_oil_scraper
python -m sentiment.vader_textblob

# Value investing
python -m value_investing.magic_formula
python -m value_investing.piotroski_f_score
```

## Key Modules

| Module | Description |
|--------|-------------|
| `utils.data` | Shared `fetch_ohlcv_data()` for Yahoo Finance |
| `utils.kpi` | CAGR, Sharpe, Sortino, volatility, max drawdown, Calmar |
| `indicators` | 8 technical indicators as importable functions |
| `strategies` | 5 backtesting strategies with full KPI reporting |
