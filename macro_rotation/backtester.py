"""
Backtesting Engine (Module 6)
==============================
Event-driven backtest loop with drift-based rebalancing.
VBT is used ONLY for final KPI computation and reporting (Option B).

Architecture:
    1. Pre-compute all signals and macro regimes (once)
    2. Event-driven loop: check rebalance triggers at each date
    3. On trigger: compute weights, apply risk overlays, compute fees
    4. Otherwise: drift weights with price action + cash yield
    5. At end: pass returns series to VBTBacktester for KPI/reporting

Key features:
    - Marginal rebalancing (trade deltas only, fees on traded notional)
    - 5.25% APY idle cash yield (configurable, compounded daily)
    - Asymmetric VNINDEX fees (0.15% buy, 0.25% sell)
    - VNINDEX T+2 settlement lag
    - Persistence-filtered signal tiers
    - Checkpoint/resume for long backtests
"""

import pickle
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from macro_rotation.config import (
    SignalState, SignalTier, MacroRegime, SentimentRegime,
    RebalanceMode, SIGNAL_TO_TIER,
    SystemConfig, CONFIG, CHECKPOINT_DIR, logger,
)
from macro_rotation.portfolios import AbstractPortfolio
from macro_rotation.signal_engine import (
    precompute_all_signals, classify_all_signals_at_date,
    compute_30d_performance, PersistenceState, get_signal_tier,
)
from macro_rotation.macro_regime import precompute_macro_regimes
from macro_rotation.allocation import (
    compute_final_weights, get_cash_weight,
    compute_trade_deltas, compute_total_fees, compute_turnover,
    drift_weights, should_rebalance, TradeOrder,
)
from macro_rotation.risk_overlay import (
    compute_btc_risk_metric, precompute_sentiment_series,
    SentimentPersistenceState, _make_default_sentiment_state,
    _classify_raw_sentiment,
    apply_risk_overlay, OpportunityZone,
    rank_assets_by_performance, SignalType,
)
from macro_rotation.quant_utils import (
    calculate_ewma_stats, calculate_kelly_fraction
)
from macro_rotation.execution import (
    plan_almgren_chriss_trajectory, compute_slippage_cost
)


# ============================================================================
# REBALANCE EVENT LOG
# ============================================================================
@dataclass
class RebalanceEvent:
    """Record of a single rebalance event for diagnostics."""
    date: pd.Timestamp
    reason: str
    regime: MacroRegime
    signals: dict[str, str]     # asset → SignalState.value
    prev_weights: dict[str, float]
    new_weights: dict[str, float]
    trades: list[dict]          # Simplified trade records
    total_fees_usd: float
    turnover: float
    btc_risk: float
    sentiment: str
    live_signal: str
    cash_weight: float
    portfolio_value: float


# ============================================================================
# BACKTEST STATE (for checkpointing)
# ============================================================================
@dataclass
class BacktestState:
    """Full state of the backtest at a given date — serializable for checkpointing."""
    date_idx: int
    weights: dict[str, float]
    cash_weight: float
    portfolio_value: float
    signal_persistence: dict[str, PersistenceState]
    sentiment_state: SentimentPersistenceState
    prev_regime: MacroRegime
    prev_tiers: dict[str, SignalTier]
    daily_returns: list[float]
    events: list[RebalanceEvent]
    weight_history: list[dict]
    cash_history: list[float]
    value_history: list[float]
    turnover_history: list[float]


# ============================================================================
# MAIN BACKTEST ENGINE
# ============================================================================
def precompute_backtest_data(
    prices: pd.DataFrame,
    fred_df: pd.DataFrame,
    proxy_prices: pd.DataFrame,
    sector_prices: pd.DataFrame,
    config: SystemConfig = CONFIG,
) -> dict:
    """Pre-compute signals, macro regimes, and risk metrics for a backtest."""
    logger.info("\n  Phase 1: Pre-computing signals...")
    indicator_dfs = precompute_all_signals(prices)

    logger.info("  Phase 1: Pre-computing macro regimes...")
    macro_df = precompute_macro_regimes(fred_df, proxy_prices, sector_prices, config)

    logger.info("  Phase 1: Computing BTC risk metric...")
    btc_risk_series = pd.Series(0.5, index=prices.index, name="btc_risk")
    if "BTC" in prices.columns:
        btc_risk_series = compute_btc_risk_metric(prices["BTC"], window=config.btc_risk_window)
        btc_risk_series = btc_risk_series.reindex(prices.index).ffill().fillna(0.5)

    logger.info("  Phase 1: Computing sentiment indicators...")
    sentiment_df = pd.DataFrame()
    if "BTC" in prices.columns:
        sentiment_df = precompute_sentiment_series(prices["BTC"], btc_risk_series, config)

    return {
        "indicator_dfs": indicator_dfs,
        "macro_df": macro_df,
        "btc_risk_series": btc_risk_series,
        "sentiment_df": sentiment_df,
    }


def run_backtest(
    portfolio: AbstractPortfolio,
    prices: pd.DataFrame,
    fred_df: pd.DataFrame,
    proxy_prices: pd.DataFrame,
    sector_prices: pd.DataFrame,
    volumes: pd.DataFrame = None,
    config: SystemConfig = CONFIG,
) -> dict:
    """Standard wrapper for run_backtest_with_data."""
    precomputed = precompute_backtest_data(
        prices=prices,
        fred_df=fred_df,
        proxy_prices=proxy_prices,
        sector_prices=sector_prices,
        config=config
    )
    
    return run_backtest_with_data(
        portfolio=portfolio,
        prices=prices,
        volumes=volumes,
        precomputed=precomputed,
        config=config
    )


def run_backtest_with_data(
    portfolio: AbstractPortfolio,
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    precomputed: dict,
    config: SystemConfig,
    warmup_days: int = 252,
) -> dict:
    """
    Run the event-driven backtest loop using pre-computed signal/macro data.
    Allows for efficient grid-searching of execution/allocation parameters.
    """
    portfolio_name = portfolio.get_name().replace("/", "_").replace(" ", "_")
    logger.info(f"\n{'='*62}")
    logger.info(f"  Backtest: {portfolio_name}")
    logger.info(f"  Config: {config}") # Useful for WFO logging
    
    indicator_dfs = precomputed["indicator_dfs"]
    macro_df = precomputed["macro_df"]
    btc_risk_series = precomputed["btc_risk_series"]
    sentiment_df = precomputed["sentiment_df"]

    # -----------------------------------------------------------------------
    # Phase 1b: TRIAL-SPECIFIC STATS (Covariance, EWMA)
    # -----------------------------------------------------------------------
    logger.info("  Phase 1b: Pre-computing rolling portfolio stats (Covariance, EWMA)...")
    daily_rets = prices.pct_change().fillna(0)
    rolling_cov = daily_rets.rolling(window=126).cov()
    
    kelly_stats = {}
    for col in daily_rets.columns:
        ewm = daily_rets[col].ewm(span=config.kelly_ewma_span)
        kelly_stats[col] = {
            "mu": ewm.mean(),
            "var": ewm.var()
        }

    # -----------------------------------------------------------------------
    # Phase 2: EVENT-DRIVEN BACKTEST LOOP
    # -----------------------------------------------------------------------
    logger.info("\n  Phase 2: Running event-driven backtest loop...")

    dates = prices.index
    n_dates = len(dates)

    # State initialization
    weights: dict[str, float] = {}
    cash_weight = 1.0
    portfolio_value = config.initial_capital
    daily_cash_rate = config.daily_cash_rate

    # Persistence states
    signal_persistence: dict[str, PersistenceState] = {}
    sentiment_state = _make_default_sentiment_state()
    prev_regime = MacroRegime.RISK_ON_DISINFLATION
    
    # Track the confirmed regime
    regime_persistence = PersistenceState(
        confirmed=MacroRegime.RISK_ON_DISINFLATION, 
        candidate=MacroRegime.RISK_ON_DISINFLATION, 
        candidate_bars=0
    )
    
    prev_tiers: dict[str, SignalTier] = {}
    
    last_rebalance_idx = -999

    # Output accumulators
    daily_returns: list[float] = []
    events: list[RebalanceEvent] = []
    weight_history: list[dict] = []
    cash_history: list[float] = []
    value_history: list[float] = []
    turnover_history: list[float] = []
    
    # Institutional Execution Buffer
    # Dict mapping asset -> list of weight deltas to execute on subsequent days
    execution_buffer: dict[str, list[float]] = {}
    locked_assets: set[str] = set()

    # Warm-up period (need enough data for indicators)
    warmup = warmup_days

    start_idx = 0
    if config.resume_from_checkpoint:
        state = _load_checkpoint(portfolio_name)
        if state is not None:
            # Reconstruct state from checkpoint
            weights = state.weights
            cash_weight = state.cash_weight
            portfolio_value = state.portfolio_value
            signal_persistence = state.signal_persistence
            sentiment_state = state.sentiment_state
            prev_regime = state.prev_regime
            prev_tiers = state.prev_tiers
            daily_returns = state.daily_returns
            events = state.events
            weight_history = state.weight_history
            cash_history = state.cash_history
            value_history = state.value_history
            turnover_history = state.turnover_history
            start_idx = state.date_idx + 1
            logger.info(f"  ✅ Resumed from checkpoint at {dates[state.date_idx].date()}")

    for i in range(start_idx, n_dates):
        date = dates[i]

        # --- Skip warm-up ---
        if i < warmup:
            daily_returns.append(0.0)
            weight_history.append({})
            cash_history.append(1.0)
            value_history.append(portfolio_value)
            turnover_history.append(0.0)
            continue

        # --- Daily returns from previous weights ---
        if i > warmup and weights:
            day_returns = {}
            for asset in weights:
                if asset in prices.columns:
                    prev_price = prices[asset].iloc[i - 1]
                    curr_price = prices[asset].iloc[i]
                    if prev_price > 0 and not pd.isna(prev_price) and not pd.isna(curr_price):
                        day_returns[asset] = curr_price / prev_price - 1
                    else:
                        day_returns[asset] = 0.0

            # Portfolio return = weighted sum of asset returns + cash yield
            port_ret = sum(
                weights.get(a, 0.0) * day_returns.get(a, 0.0) for a in weights
            ) + cash_weight * daily_cash_rate

            portfolio_value *= (1 + port_ret)
            daily_returns.append(port_ret)

            # Drift weights (every day)
            weights, cash_weight = drift_weights(
                weights, day_returns, cash_weight, daily_cash_rate
            )
        else:
            daily_returns.append(0.0)

        # --- Get current signals (lookup, not computation) ---
        raw_signals = classify_all_signals_at_date(indicator_dfs, date)

        # Apply signal persistence filter
        current_tiers: dict[str, SignalTier] = {}
        persisted_signals: dict[str, SignalState] = {}

        for asset, raw_signal in raw_signals.items():
            raw_tier = get_signal_tier(raw_signal)

            if asset not in signal_persistence:
                signal_persistence[asset] = PersistenceState(raw_tier, raw_tier, 0)
                
            from macro_rotation.config import ASSET_CLASSES, AssetClass
            a_class = ASSET_CLASSES.get(asset, AssetClass.CRYPTO)
            bars_req = config.equity_persistence_bars if a_class == AssetClass.EQUITY else config.crypto_persistence_bars

            signal_persistence[asset] = signal_persistence[asset].update(
                raw_tier, bars_req
            )
            current_tiers[asset] = signal_persistence[asset].confirmed

            # Use raw signal's modifier but confirmed tier
            persisted_signals[asset] = raw_signal

        # --- Get current macro regime ---
        raw_regime = prev_regime  # Default to previous
        if not macro_df.empty and date in macro_df.index:
            regime_val = macro_df.loc[date, "regime"]
            if isinstance(regime_val, MacroRegime):
                raw_regime = regime_val
        elif not macro_df.empty:
            valid_macro = macro_df.index[macro_df.index <= date]
            if len(valid_macro) > 0:
                nearest = valid_macro[-1]
                regime_val = macro_df.loc[nearest, "regime"]
                if isinstance(regime_val, MacroRegime):
                    raw_regime = regime_val

        # Require 3 days to confirm macro regime change
        regime_persistence = regime_persistence.update(raw_regime, 3)
        current_regime = regime_persistence.confirmed

        # --- Update sentiment (with persistence filter) ---
        if not sentiment_df.empty and date in sentiment_df.index:
            row = sentiment_df.loc[date]
            raw_sentiment = _classify_raw_sentiment(
                rsi=float(row.get("rsi", 50)),
                volatility_rank=float(row.get("vol_rank", 0.5)),
                drawdown_pct=float(row.get("drawdown", 0.0)),
                btc_risk=float(row.get("btc_risk", 0.5)),
            )
            sentiment_state = sentiment_state.update(
                raw_sentiment, config.sentiment_persistence_bars
            )

        btc_risk_val = float(btc_risk_series.iloc[i]) if i < len(btc_risk_series) else 0.5

        # --- Check rebalance trigger ---
        trigger, reason = should_rebalance(
            prev_tiers, current_tiers, prev_regime, current_regime
        )
        
        # Enforce minimum rebalance interval (unless it's an initial allocation or a regime change)
        is_regime_change = (current_regime != prev_regime)
        if trigger and not is_regime_change and (i - last_rebalance_idx) < config.min_rebalance_interval_days:
            trigger = False
            reason = "Cooldown"

        if trigger or (i == warmup):  # Force initial allocation
            if i == warmup:
                reason = "Initial allocation"
                
            last_rebalance_idx = i

            # Compute 30-day performance for momentum ranking
            perf_30d = compute_30d_performance(prices, date)
            
            # Pass volumes for the day if available
            day_volumes = volumes.loc[date] if volumes is not None and date in volumes.index else pd.Series(dtype=float)

            # --- INSTITUTIONAL STATS ---
            # Get rolling covariance for current date
            current_cov = pd.DataFrame()
            if not rolling_cov.empty:
                try:
                    current_stats = rolling_cov.loc[date]
                    if isinstance(current_stats, pd.DataFrame):
                        current_cov = current_stats
                except KeyError:
                    pass
            
            # Get Kelly mu/var for current date
            current_kelly = {}
            for asset, stats_data in kelly_stats.items():
                if date in stats_data["mu"].index:
                    mu = stats_data["mu"].loc[date]
                    var = stats_data["var"].loc[date]
                    # Compute Kelly fraction
                    f = calculate_kelly_fraction(mu, var, daily_cash_rate, config.kelly_max_fraction)
                    current_kelly[asset] = f
            
            stats_context = {
                "cov_matrix": current_cov,
                "kelly_fractions": current_kelly
            }

            # Compute target weights (passes stats for RP/Kelly)
            new_weights = compute_final_weights(
                portfolio, persisted_signals, current_regime, perf_30d,
                current_date=date, volumes=day_volumes, stats=stats_context
            )

            # Apply risk overlay
            new_cash = get_cash_weight(new_weights)
            new_weights, new_cash, live_signal = apply_risk_overlay(
                new_weights, new_cash,
                btc_risk_val, sentiment_state.confirmed,
                current_regime,
            )

            # Compute trades and fees
            orders = compute_trade_deltas(
                weights, new_weights,
                mode=config.rebalance_mode,
                portfolio_value=portfolio_value,
                config=config,
            )
            total_fees = compute_total_fees(orders)
            turnover = compute_turnover(weights, new_weights)

            # Deduct fees from portfolio value
            portfolio_value -= total_fees

            # Log event
            events.append(RebalanceEvent(
                date=date,
                reason=reason,
                regime=current_regime,
                signals={a: s.value for a, s in persisted_signals.items()},
                prev_weights=dict(weights),
                new_weights=dict(new_weights),
                trades=[{"asset": o.asset, "action": o.action,
                         "delta": round(o.delta_weight * 100, 2),
                         "fee": round(o.fee_usd, 2)} for o in orders if o.action != "HOLD"],
                total_fees_usd=total_fees,
                turnover=turnover,
                btc_risk=btc_risk_val,
                sentiment=sentiment_state.confirmed.value,
                live_signal=live_signal,
                cash_weight=new_cash,
                portfolio_value=portfolio_value,
            ))

            # Update State via Execution Buffer (Plan Trajectory)
            # Instead of immediate update: weights = new_weights
            for order in orders:
                if order.asset in locked_assets:
                    continue # Already in flight
                
                # Plan Almgren-Chriss trajectory
                # Get ADV and daily volatility for the impact model
                adv = day_volumes.get(order.asset, 1e9) # Fallback to $1B if missing
                if pd.isna(adv) or adv <= 0:
                    adv = 1e9
                    
                # Use a expanded window for vol to avoid NaNs at segment starts
                vol_series = daily_rets[order.asset].rolling(20).std()
                vol = float(vol_series.loc[date]) if order.asset in daily_rets.columns and not pd.isna(vol_series.loc[date]) else 0.02
                
                # Double check vol isn't zero or NaN
                if vol <= 0 or pd.isna(vol):
                    vol = 0.02

                trajectory = plan_almgren_chriss_trajectory(
                    total_delta=order.delta_weight,
                    daily_vol=vol,
                    adv_usd=adv,
                    portfolio_value=portfolio_value,
                    lambda_risk_aversion=config.lambda_risk_aversion,
                    max_days=config.execution_max_days
                )
                
                if trajectory:
                    execution_buffer[order.asset] = trajectory
                    locked_assets.add(order.asset)

        # --- INSTITUTIONAL EXECUTION: Daily Buffer Processing ---
        daily_slippage = 0.0
        finished_assets = []
        
        for asset, trajectory in execution_buffer.items():
            if not trajectory:
                finished_assets.append(asset)
                continue
                
            # Execute one slice
            slice_delta = trajectory.pop(0)
            
            # Update actual weights
            old_w = weights.get(asset, 0.0)
            weights[asset] = old_w + slice_delta
            cash_weight -= slice_delta # Cash offsets the weight change
            
            # Apply Slippage/Impact Cost
            adv = day_volumes.get(asset, 1e9)
            vol = float(daily_rets[asset].rolling(20).std().loc[date]) if asset in daily_rets.columns else 0.02
            slippage_usd = compute_slippage_cost(
                traded_usd=abs(slice_delta) * portfolio_value,
                adv_usd=adv,
                daily_vol=vol
            )
            daily_slippage += slippage_usd
            
            if not trajectory:
                finished_assets.append(asset)
                
        # Clean up finished trajectories
        for asset in finished_assets:
            execution_buffer.pop(asset, None)
            locked_assets.discard(asset)
            
        portfolio_value -= daily_slippage

        # Record history
        weight_history.append(dict(weights))
        cash_history.append(cash_weight)
        value_history.append(portfolio_value)
        turnover_history.append(turnover if trigger else 0.0)

        # Update state for next iteration
        prev_tiers = dict(current_tiers)
        prev_regime = current_regime

        # --- Checkpoint ---
        if config.checkpoint_interval_days > 0 and i % config.checkpoint_interval_days == 0 and i > warmup:
            _save_checkpoint(BacktestState(
                date_idx=i, weights=weights, cash_weight=cash_weight,
                portfolio_value=portfolio_value,
                signal_persistence=signal_persistence,
                sentiment_state=sentiment_state,
                prev_regime=prev_regime, prev_tiers=prev_tiers,
                daily_returns=daily_returns, events=events,
                weight_history=weight_history,
                cash_history=cash_history,
                value_history=value_history,
                turnover_history=turnover_history,
            ), portfolio_name)

    # -----------------------------------------------------------------------
    # Phase 3: BUILD OUTPUT
    # -----------------------------------------------------------------------
    logger.info(f"\n  Phase 3: Building results...")

    returns_series = pd.Series(daily_returns, index=dates, name="returns")
    equity_series = pd.Series(value_history, index=dates, name="equity")
    cash_series = pd.Series(cash_history, index=dates, name="cash_weight")
    turnover_series = pd.Series(turnover_history, index=dates, name="turnover")

    weight_df = pd.DataFrame(weight_history, index=dates).fillna(0.0)

    # Performance metrics
    active_returns = returns_series.iloc[warmup:]
    total_return = float(equity_series.iloc[-1] / config.initial_capital - 1)
    n_years = len(active_returns) / 365
    cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    ann_vol = float(active_returns.std() * np.sqrt(365))
    sharpe = (float(active_returns.mean()) * 365 - config.cash_yield_apy) / ann_vol if ann_vol > 0 else 0
    max_dd = float(((equity_series / equity_series.cummax()) - 1).min())

    # Sortino
    downside = active_returns[active_returns < 0]
    downside_vol = float(np.sqrt((downside ** 2).mean())) * np.sqrt(365) if len(downside) > 0 else 1e-6
    sortino = (float(active_returns.mean()) * 365 - config.cash_yield_apy) / downside_vol

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    metrics = {
        "Total Return (%)": round(total_return * 100, 2),
        "CAGR (%)": round(cagr * 100, 2),
        "Ann. Volatility (%)": round(ann_vol * 100, 2),
        "Sharpe": round(sharpe, 3),
        "Sortino": round(sortino, 3),
        "Calmar": round(calmar, 3),
        "Max Drawdown (%)": round(max_dd * 100, 2),
        "Rebalance Events": len(events),
        "Avg Turnover": round(turnover_series[turnover_series > 0].mean() * 100, 2) if (turnover_series > 0).any() else 0,
        "Total Fees ($)": round(sum(e.total_fees_usd for e in events), 2),
        "Final Value ($)": round(portfolio_value, 2),
    }

    logger.info(f"\n  📊 Results — {portfolio_name}")
    logger.info(f"  {'='*50}")
    for k, v in metrics.items():
        logger.info(f"     {k:<25} {v:>12}")

    return {
        "returns": returns_series,
        "equity_curve": equity_series,
        "events": events,
        "weight_history": weight_df,
        "cash_history": cash_series,
        "turnover_history": turnover_series,
        "macro_df": macro_df,
        "btc_risk": btc_risk_series,
        "metrics": metrics,
        "config": config,
        "portfolio_name": portfolio_name,
    }


# ============================================================================
# BENCHMARK COMPARISON
# ============================================================================
def compute_benchmark(
    prices: pd.DataFrame,
    benchmark_ticker: str,
    config: SystemConfig = CONFIG,
) -> dict:
    """Compute buy-and-hold benchmark returns and equity curve."""
    if benchmark_ticker not in prices.columns:
        # Try the yfinance ticker mapping
        from macro_rotation.config import ASSET_TICKERS
        reverse_map = {v: k for k, v in ASSET_TICKERS.items()}
        asset_key = reverse_map.get(benchmark_ticker, benchmark_ticker)
        if asset_key in prices.columns:
            benchmark_ticker = asset_key

    if benchmark_ticker not in prices.columns:
        logger.warning(f"  ⚠️ Benchmark {benchmark_ticker} not in price data")
        return {}

    bench_prices = prices[benchmark_ticker].dropna()
    bench_returns = bench_prices.pct_change().fillna(0)
    bench_equity = config.initial_capital * (1 + bench_returns).cumprod()

    total_ret = float(bench_equity.iloc[-1] / config.initial_capital - 1)
    n_years = len(bench_returns) / 365
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
    max_dd = float(((bench_equity / bench_equity.cummax()) - 1).min())

    return {
        "returns": bench_returns,
        "equity_curve": bench_equity,
        "metrics": {
            "Total Return (%)": round(total_ret * 100, 2),
            "CAGR (%)": round(cagr * 100, 2),
            "Max Drawdown (%)": round(max_dd * 100, 2),
        },
    }


# ============================================================================
# CHECKPOINTING
# ============================================================================
def _save_checkpoint(state: BacktestState, portfolio_name: str) -> None:
    path = CHECKPOINT_DIR / f"{portfolio_name.replace(' ', '_')}_checkpoint.pkl"
    with open(path, "wb") as f:
        pickle.dump(state, f)


def _load_checkpoint(portfolio_name: str) -> BacktestState | None:
    path = CHECKPOINT_DIR / f"{portfolio_name.replace(' ', '_')}_checkpoint.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None
