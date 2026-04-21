"""
Macro Rotation Portfolio Rebalancing System
============================================
Multi-portfolio dynamic rebalancing with macro regime awareness,
quantitative signal generation, and risk management overlays.

Modules:
    config          - System configuration, FRED series map, enums
    portfolios      - Portfolio definitions (Crypto+Gold, Core Asset Macro)
    signal_engine   - Vectorized indicator pre-computation + signal classification
    macro_regime    - Macro composite construction + regime classification
    allocation      - Dynamic weight computation + marginal rebalancing
    risk_overlay    - BTC risk metric, sentiment persistence, opportunity zones
    backtester      - Event-driven backtest loop + VBT KPI wrapper
    data_loader     - Unified data fetcher (yfinance, FRED, CSV)
    dashboard       - Plotly interactive dashboard with diagnostic panels
    run             - Main orchestrator
"""
