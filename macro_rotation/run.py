"""
Main Orchestrator (run.py)
===========================
Entry point for the Macro Rotation Portfolio Rebalancing System.
Runs both portfolios, compares to benchmarks, generates dashboards.

Usage:
    python -m macro_rotation.run
    python -m macro_rotation.run --portfolio crypto
    python -m macro_rotation.run --portfolio core3
"""

import sys
import argparse
import datetime as dt
from pathlib import Path
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macro_rotation.config import SystemConfig, CONFIG, logger, REPORTS_DIR
from macro_rotation.portfolios import CryptoGoldRotation, CoreAssetMacroRotation, NaiveBitcoinGoldPortfolio
from macro_rotation.data_loader import load_all_data
from macro_rotation.backtester import run_backtest, compute_benchmark
from macro_rotation.dashboard import build_dashboard, save_dashboard, save_event_log


def run_portfolio(portfolio, data: dict, config: SystemConfig) -> dict:
    """Run a single portfolio backtest end-to-end."""
    results = run_backtest(
        portfolio=portfolio,
        prices=data["prices"],
        fred_df=data["fred"],
        proxy_prices=data["proxy_prices"],
        sector_prices=data["sector_prices"],
        config=config,
    )

    # Benchmark comparison
    bench_ticker = portfolio.get_benchmark_ticker()
    benchmark = compute_benchmark(data["prices"], bench_ticker, config)
    if benchmark:
        logger.info(f"\n  📈 Benchmark ({bench_ticker}) B&H:")
        for k, v in benchmark.get("metrics", {}).items():
            logger.info(f"     {k:<25} {v:>12}")

    # Build and save dashboard
    clean_name = portfolio.get_name().replace("/", "_").replace(" ", "_").lower()
    fig = build_dashboard(results, benchmark)
    save_dashboard(fig, clean_name)

    # Save event log
    save_event_log(results["events"], clean_name)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Macro Rotation Portfolio Rebalancing System"
    )
    parser.add_argument(
        "--portfolio", type=str, default="both",
        choices=["crypto", "core3", "naive", "all", "both"],
        help="Which portfolio to run (default: both)",
    )
    parser.add_argument(
        "--oos", action="store_true",
        help="Run walk-forward validation split (IS: 2021-2023, OOS: 2024+)",
    )
    parser.add_argument(
        "--cash-yield", type=float, default=None,
        help="Override cash yield APY (e.g., 0.0525 for 5.25%%)",
    )
    parser.add_argument(
        "--vnindex-csv", type=str, default="",
        help="Path to VNINDEX historical CSV (recommended over VNM ETF proxy)",
    )
    parser.add_argument(
        "--fred-key", type=str, default="",
        help="FRED API key for macro data (from https://fred.stlouisfed.org/docs/api/api_key.html)",
    )
    parser.add_argument(
        "--start", type=str, default="2021-01-01",
        help="Backtest start date (default: 2021-01-01)",
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="Backtest end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--wfo", action="store_true",
        help="Run Walk-Forward Optimization (WFO) with DSR gatekeeping",
    )

    args = parser.parse_args()

    # Build config with overrides
    overrides = {}
    if args.cash_yield is not None:
        overrides["cash_yield_apy"] = args.cash_yield
    if args.vnindex_csv:
        overrides["vnindex_csv_path"] = args.vnindex_csv
    if args.fred_key:
        overrides["fred_api_key"] = args.fred_key
    if args.start:
        overrides["backtest_start"] = args.start
        overrides["crypto_backtest_start"] = args.start
    if getattr(args, "end", None):
        overrides["backtest_end"] = args.end

    config = SystemConfig(**{**{f.name: getattr(CONFIG, f.name) for f in CONFIG.__dataclass_fields__.values()}, **overrides})

    # Header
    logger.info("=" * 62)
    logger.info("  Macro Rotation Portfolio Rebalancing System")
    logger.info("=" * 62)
    logger.info(f"  Start Date:    {config.backtest_start}")
    logger.info(f"  Cash Yield:    {config.cash_yield_apy * 100:.2f}% APY")
    logger.info(f"  Rebalance:     {config.rebalance_mode.value}")
    logger.info(f"  VNINDEX:       {'CSV: ' + config.vnindex_csv_path if config.vnindex_csv_path else 'vnstock API'}")
    logger.info(f"  FRED API:      {'✅ Key provided' if config.fred_api_key else '⚠️ No key (macro fallback)'}\n")

    # Load data
    logger.info("  Loading data...")
    data = load_all_data(config)

    # Print proxy warnings
    for warning in data["metadata"].get("proxy_warnings", []):
        logger.warning(f"  ⚠️ {warning}")

    results_all = {}

    # Calculate common start date to ensure fair comparison
    valid_starts = []
    if args.portfolio in ("crypto", "both") and "BTC" in data["prices"].columns:
        valid_starts.append(data["prices"]["BTC"].dropna().index[0])
    if args.portfolio in ("core3", "both") and "VNINDEX" in data["prices"].columns:
        valid_starts.append(data["prices"]["VNINDEX"].dropna().index[0])
    
    if valid_starts:
        common_start = max(valid_starts)
        # Ensure it respects the configured backtest start
        config_start = pd.Timestamp(config.backtest_start)
        common_start = max(common_start, config_start)
        
        logger.info(f"  📅 Aligning backtest start to common data availability: {common_start.date()}")
        
        # Trim data dictionaries to ensure identical backtest windows
        data["prices"] = data["prices"][data["prices"].index >= common_start]
        data["volumes"] = data["volumes"][data["volumes"].index >= common_start]

    # Handle explicit end date slicing
    if getattr(config, "backtest_end", None):
        end_dt = pd.Timestamp(config.backtest_end)
        logger.info(f"  📅 Slicing backtest end to: {end_dt.date()}")
        data["prices"] = data["prices"][data["prices"].index <= end_dt]
        data["volumes"] = data["volumes"][data["volumes"].index <= end_dt]

    # --- WALK-FORWARD OPTIMIZATION MODE ---
    if args.wfo:
        from macro_rotation.walk_forward_optimizer import run_wfo_session
        
        logger.info("\n" + "=" * 62)
        logger.info("  🧪 Institutional WFO Session")
        logger.info("=" * 62)
        
        # Agreed Institutional Grid
        param_grid = {
            "kelly_ewma_span": [30, 60, 90],
            "min_rebalance_interval_days": [3, 5, 10],
            "execution_max_days": [1, 3, 5]
        }
        
        portfolio = CryptoGoldRotation() if args.portfolio != "core3" else CoreAssetMacroRotation()
        wfo_res = run_wfo_session(portfolio, data, param_grid, config)
        
        # Store for comparison
        results_all["WFO_OOS"] = {"metrics": wfo_res.metrics}
        
        # Save WFO history
        portfolio_slug = portfolio.get_name().lower().replace(" ", "_").replace("/", "_")
        wfo_path = REPORTS_DIR / f"wfo_param_history_{portfolio_slug}.csv"
        wfo_res.param_history.to_csv(wfo_path, index=False)
        logger.info(f"  ✅ WFO parameter history saved to: {wfo_path}")

    # --- STANDARD BACKTEST MODE ---
    if not args.wfo:
        # Run portfolios
        if args.portfolio in ("crypto", "both", "all"):
            logger.info("\n" + "=" * 62)
            crypto_portfolio = CryptoGoldRotation()
            results_all["crypto"] = run_portfolio(crypto_portfolio, data, config)

        if args.portfolio in ("core3", "both", "all"):
            logger.info("\n" + "=" * 62)
            core3_portfolio = CoreAssetMacroRotation()
            results_all["core3"] = run_portfolio(core3_portfolio, data, config)

        if args.portfolio in ("naive", "all"):
            logger.info("\n" + "=" * 62)
            naive_portfolio = NaiveBitcoinGoldPortfolio()
            results_all["naive"] = run_portfolio(naive_portfolio, data, config)

        if args.oos:
            logger.info("\n" + "=" * 62)
            logger.info("  🧪 Walk-Forward / OOS Validation (2024+)")
            logger.info("=" * 62)
            oos_data = data.copy()
            oos_start = pd.Timestamp("2024-01-01")
            oos_data["prices"] = oos_data["prices"][oos_data["prices"].index >= oos_start]
            oos_data["volumes"] = oos_data["volumes"][oos_data["volumes"].index >= oos_start]
            
            if not oos_data["prices"].empty:
                logger.info(f"  Running OOS Test for Crypto Strategy...")
                results_all["crypto_OOS"] = run_portfolio(CryptoGoldRotation(), oos_data, config)
                logger.info(f"  Running OOS Test for Naive Baseline...")
                results_all["naive_OOS"] = run_portfolio(NaiveBitcoinGoldPortfolio(), oos_data, config)

    # Comparison summary
    if len(results_all) > 1:
        logger.info("\n" + "=" * 62)
        logger.info("  📊 Strategy Summary Comparison")
        logger.info("=" * 62)
        header = f"  {'Metric':<25}"
        for name in results_all:
            header += f" {name:>15}"
        logger.info(header)
        logger.info(f"  {'-' * 25}" + f" {'-' * 15}" * len(results_all))

        all_metrics = set()
        for r in results_all.values():
            all_metrics.update(r["metrics"].keys())

        for metric in sorted(all_metrics):
            row = f"  {metric:<25}"
            for name, r in results_all.items():
                val = r["metrics"].get(metric, "—")
                row += f" {str(val):>15}"
            logger.info(row)

    logger.info(f"\n  ✅ Reports saved to: {REPORTS_DIR}")
    logger.info("  Done.\n")


if __name__ == "__main__":
    main()
