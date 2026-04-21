# Quantitative Trading Research & Macro-Rotation Toolkit

A production-grade modular Python ecosystem for quantitative research, transitioning from high-frequency intraday signals to global macro-economic rebalancing. This toolkit covers technical indicators, event-driven backtesting, institutional portfolio optimization, and cross-strategy correlation analysis.

## 🚀 Key Features

- **Macro Rotation Engine**: Dynamic 4-quadrant regime classification based on Growth vs. Inflation (FRED data).
- **Automated Data Pipelines**:
  - **VN-Index**: Native integration with `vnstock` (VCI/TCBS sources) for seamless Vietnamese market data.
  - **Global Macro**: Automated `fredapi` integration for M2 liquidity, Credit Spreads, and VIX.
  - **Crypto/Equities**: High-fidelity `yfinance` caching.
- **Institutional Portfolio Optimization**: Mean-Variance Optimization (MVO) with Ledoit-Wolf shrinkage and Point-In-Time (PiT) universe management to eliminate survivorship bias.
- **Interactive Dashboard**: Full-featured Streamlit UI (`app.py`) for parameter sensitivity testing and visual backtesting.
- **Strategy Correlation Engine**: Unified daily returns alignment for detecting alpha overlap across disparate systems (Macro vs. Intraday).

## 📂 Project Structure

```
├── macro_rotation/     # CORE: Global Macro-Rotation System
│   ├── run.py          # CLI Entry Point
│   ├── backtester.py   # Event-driven backtest loop
│   ├── signal_engine.py# Technical & persistence logic
│   └── macro_regime.py # Regime classification (FRED dependent)
├── strategies/         # Strategy Library
│   ├── rebalance_portfolio.py# Institutional MVO Adaptive Engine
│   ├── renko_macd.py   # Trend-following Renko logic
│   └── resistance_breakout.py# Intraday volatility breakouts
├── data_ingestion/     # High-fidelity data stores & PiT management
├── indicators/         # Standardized technical primitives
├── app.py              # Interactive Streamlit Web Portal
└── strategy_correlation_engine.py # Cross-system orthogonality check
```

## 🛠 Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Configuration**:
   Create a `.env` file in the root directory:
   ```env
   FRED_API_KEY=your_fred_api_key_here
   ```

## 📈 Usage

### 1. Interactive Dashboard (Recommended)
Launch the Streamlit portal to visualize regimes and backtest results:
```bash
streamlit run app.py
```

### 2. Macro Rotation CLI
Run specific portfolio rotations via terminal:
```bash
# Core Asset Rotation with custom dates
python3 -m macro_rotation.run --portfolio core3 --start 2021-01-01 --end 2024-01-01
```

### 3. Multi-Strategy Correlation
Audit all active strategies for overlap and orthogonality:
```bash
python3 strategy_correlation_engine.py
```

## ⚖️ Compliance & Integrity

Every strategy in this repository is audited against the **Triple-Bias Standard**:
1. **Survivorship Bias**: Handled via `PointInTimeUniverse` for equities and fixed major universes for crypto.
2. **Lookahead Bias**: Strict `shift(1)` enforcement on all signal lookups and indicator pre-computations.
3. **Transaction Costs**: Realistic commission modeling (10-25 bps) and slippage approximations applied to every trade.

---
*Built for advanced quantitative research and institutional-grade portfolio construction.*
