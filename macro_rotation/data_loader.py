"""
Data Loader (Data Ingestion Layer)
===================================
Unified data fetcher for crypto, gold, equities, and macro proxies.
Includes GoldProxy splice, FRED API with lag-aware preparation, and CSV override.
"""

import datetime as dt
import pickle
import time
from pathlib import Path
from typing import Optional
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import yfinance as yf

from macro_rotation.config import (
    ASSET_TICKERS, FRED_SERIES, MACRO_WARMUP_YEARS,
    SystemConfig, CONFIG, CACHE_DIR, logger,
)

# Maximum retry attempts for yfinance downloads (handles rate limiting)
_YF_MAX_RETRIES = 3
_YF_RETRY_BACKOFF = 4  # seconds, doubles each retry


# ============================================================================
# CACHING HELPERS
# ============================================================================
def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.parquet"


def _load_cached(name: str, ttl_days: int = 7) -> pd.DataFrame | None:
    """Load a cached Parquet file if it exists and is fresh."""
    path = _cache_path(name)
    if not path.exists():
        return None
    age = dt.datetime.now() - dt.datetime.fromtimestamp(path.stat().st_mtime)
    if age.days > ttl_days:
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _save_cache(name: str, df: pd.DataFrame) -> None:
    df.to_parquet(_cache_path(name))


# ============================================================================
# YFINANCE DATA FETCHER
# ============================================================================
def fetch_yfinance_prices(
    tickers: list[str],
    start: str = "2018-01-01",
    end: str | None = None,
    interval: str = "1d",
    cache_ttl: int = 7,
) -> pd.DataFrame:
    """
    Fetch daily close prices for a list of tickers via yfinance.
    Returns a DataFrame with columns = tickers, index = DatetimeIndex.
    Results are cached as Parquet with configurable TTL.
    """
    if end is None:
        end = dt.datetime.today().strftime("%Y-%m-%d")

    cache_key = f"yf_{'_'.join(sorted(tickers))}_{interval}_{start}_{end}"
    cached = _load_cached(cache_key, ttl_days=cache_ttl)
    if cached is not None:
        logger.info(f"  📂 Loaded cached price data ({len(cached)} rows)")
        return cached

    logger.info(f"  📥 Downloading {len(tickers)} tickers from yfinance...")

    # Retry with exponential backoff to handle yfinance rate limits
    raw = None
    for attempt in range(_YF_MAX_RETRIES):
        try:
            raw = yf.download(
                tickers, start=start, end=end, interval=interval,
                auto_adjust=True, progress=False,
            )
            if raw is not None and not raw.empty:
                break
        except Exception as e:
            wait = _YF_RETRY_BACKOFF * (2 ** attempt)
            logger.warning(f"  ⚠️ yfinance attempt {attempt+1}/{_YF_MAX_RETRIES} failed: {e}")
            if attempt < _YF_MAX_RETRIES - 1:
                logger.info(f"     Retrying in {wait}s...")
                time.sleep(wait)

    if raw is None or raw.empty:
        raise ValueError(f"yfinance returned no data for {tickers} after {_YF_MAX_RETRIES} attempts")

    # Extract close prices
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers[:1]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])

    prices = prices.ffill().dropna(how="all")
    _save_cache(cache_key, prices)
    logger.info(f"  ✅ Fetched {len(prices)} rows for {list(prices.columns)}")
    return prices


# ============================================================================
# GOLD PROXY  (PAXG 2018+ spliced with GLD pre-2018)
# ============================================================================
def build_gold_proxy(
    start: str = "2015-01-01",
    end: str | None = None,
) -> pd.Series:
    """
    Build a gold price series by splicing GLD ETF (pre-2018) with PAXG-USD (2018+).
    Adjustment is returns-based: GLD is rescaled so its last price matches PAXG's first.
    """
    if end is None:
        end = dt.datetime.today().strftime("%Y-%m-%d")

    cache_key = f"gold_proxy_{start}_{end}"
    cached = _load_cached(cache_key)
    if cached is not None:
        return cached.squeeze()

    # Fetch both series
    gld = fetch_yfinance_prices(["GLD"], start=start, end=end).squeeze()
    paxg = fetch_yfinance_prices(["PAXG-USD"], start="2019-09-01", end=end).squeeze()

    if paxg.empty:
        logger.warning("  ⚠️ PAXG-USD unavailable, using GLD only for gold proxy")
        result = gld.rename("XAUT")
        _save_cache(cache_key, result.to_frame())
        return result

    # Returns-based splice: adjust GLD so last GLD price = first PAXG price.
    # Uses .asof() to handle calendar mismatch (PAXG trades 24/7 including
    # weekends; GLD trades NYSE hours only). If PAXG starts on a Saturday,
    # .asof() correctly finds the last Friday GLD price.
    # NOTE: GLD represents 1/10th troy oz, PAXG represents 1 troy oz.
    # The ~10x adjustment ratio is arithmetically correct (the splice
    # normalizes levels), but the level difference is not a data error.
    splice_date = paxg.index[0]
    gld_before = gld[gld.index < splice_date]

    if len(gld_before) > 0:
        # .asof() finds the last valid GLD price on or before splice_date
        gld_at_splice = gld.asof(splice_date)
        if pd.isna(gld_at_splice):
            gld_at_splice = gld_before.iloc[-1]
        paxg_at_splice = paxg.iloc[0]
        adjustment_ratio = paxg_at_splice / gld_at_splice
        gld_adjusted = gld_before * adjustment_ratio
        result = pd.concat([gld_adjusted, paxg[paxg.index >= splice_date]])
    else:
        result = paxg

    result = result[~result.index.duplicated(keep="last")]
    result.name = "XAUT"
    _save_cache(cache_key, result.to_frame())
    return result


# ============================================================================
# VNINDEX DATA (CSV recommended, VNM ETF fallback)
# ============================================================================
def load_vnindex_prices(
    config: SystemConfig = CONFIG,
    start: str = "2018-01-01",
    end: str | None = None,
) -> pd.Series:
    """
    Load VNINDEX prices primarily from vnstock API, using CSV just as a fallback.
    """
    # 1. API Flow via vnstock
    logger.info("  📥 Fetching VNINDEX via vnstock API (VCI source)...")
    try:
        from vnstock import Vnstock
        if not end:
            end = dt.datetime.now().strftime("%Y-%m-%d")
        
        quote = Vnstock().stock(symbol='VNINDEX', source='VCI').quote
        df = quote.history(start=start, end=end)
        
        if df is not None and len(df) > 0:
            df["time"] = pd.to_datetime(df["time"])
            df = df.sort_values("time").set_index("time")
            series = df["close"].astype(float)
            series.name = "VNINDEX"
            return series
    except Exception as e:
        logger.warning(f"  ⚠️ Failed to fetch VNINDEX from vnstock: {e}")

    # 2. CSV Fallback
    if config.vnindex_csv_path and Path(config.vnindex_csv_path).exists():
        logger.info(f"  📂 Fallback to VNINDEX CSV: {config.vnindex_csv_path}")
        df = pd.read_csv(config.vnindex_csv_path, parse_dates=["Date"])
        df = df.sort_values("Date").set_index("Date")

        if "Close" in df.columns:
            series = df["Close"]
        elif "Price" in df.columns:
            series = df["Price"].str.replace(",", "").astype(float)
        else:
            raise ValueError(f"VNINDEX CSV must have 'Close' or 'Price' column")

        series.name = "VNINDEX"
        return series
        
    raise RuntimeError("CRITICAL: Failed to load VNINDEX from both vnstock and CSV. Aborting.")


# ============================================================================
# FRED DATA with lag-aware preparation
# ============================================================================
def prepare_fred_series(
    series_key: str,
    start: str = "2010-01-01",
    end: str | None = None,
    config: SystemConfig = CONFIG,
) -> pd.Series:
    """
    Download a FRED series, apply lag shift in native frequency BEFORE
    resampling, then forward-fill to daily.

    This is the SINGLE code path for all FRED series to ensure
    lag enforcement is never skipped.

    Steps:
        1. Download raw series from FRED (or cache)
        2. Shift by lag_weeks in WEEKLY frequency
        3. Forward-fill to daily
    """
    if series_key not in FRED_SERIES:
        raise ValueError(f"Unknown FRED series key: {series_key}")

    spec = FRED_SERIES[series_key]
    series_id = spec["id"]
    lag_weeks = spec["lag_weeks"]
    native_freq = spec["freq"]

    cache_key = f"fred_{series_id}"
    cached = _load_cached(cache_key, ttl_days=config.fred_cache_ttl_days)

    if cached is not None:
        raw = cached.squeeze()
    elif config.fred_api_key:
        try:
            from fredapi import Fred
            fred = Fred(api_key=config.fred_api_key)
            raw = fred.get_series(series_id, observation_start=start, observation_end=end)
            raw.name = series_key
            _save_cache(cache_key, raw.to_frame())
        except Exception as e:
            logger.warning(f"  ⚠️ FRED API failed for {series_id}: {e}")
            return pd.Series(dtype=float, name=series_key)
    else:
        logger.warning(
            f"  ⚠️ No FRED API key. Series {series_id} unavailable.\n"
            f"     Set config.fred_api_key or provide CSV data."
        )
        return pd.Series(dtype=float, name=series_key)

    if raw.empty:
        return pd.Series(dtype=float, name=series_key)

    # Step 1: Resample to weekly BEFORE lag shift (ensures uniform shift semantics)
    if native_freq == "M":
        # Monthly → weekly via forward-fill, then shift in weekly units
        raw_weekly = raw.resample("W").ffill()
    else:
        raw_weekly = raw.resample("W").last().ffill()

    # Step 2: Apply lag shift in WEEKLY frequency
    if lag_weeks > 0:
        raw_weekly = raw_weekly.shift(lag_weeks)

    # Step 3: Forward-fill to daily
    daily = raw_weekly.resample("D").ffill()
    daily.name = series_key
    return daily.dropna()


def load_all_fred_series(
    config: SystemConfig = CONFIG,
    start: str = "2010-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Load all configured FRED series, lag-adjusted and daily-aligned."""
    series_dict = {}
    for key in FRED_SERIES:
        s = prepare_fred_series(key, start=start, end=end, config=config)
        if not s.empty:
            series_dict[key] = s

    if not series_dict:
        logger.warning("  ⚠️ No FRED data loaded. Macro regime will use fallback.")
        return pd.DataFrame()

    df = pd.DataFrame(series_dict)
    df = df.ffill().dropna(how="all")
    logger.info(f"  ✅ FRED data: {len(df)} rows, {list(df.columns)}")
    return df


# ============================================================================
# UNIFIED DATA LOADER
# ============================================================================
def load_all_data(
    config: SystemConfig = CONFIG,
) -> dict:
    """
    Master data loader.  Returns a dict with keys:
        prices  : pd.DataFrame — daily close prices, columns = asset keys
        volumes : pd.DataFrame — daily volumes (crypto only, for volume filter)
        fred    : pd.DataFrame — lag-adjusted daily FRED series
        metadata: dict         — proxy warnings, data ranges, etc.
    """
    start = config.backtest_start
    end = dt.datetime.today().strftime("%Y-%m-%d")
    metadata = {"proxy_warnings": [], "common_start": start, "common_end": end}

    # Compute macro warmup start: N years before backtest_start
    macro_start_dt = dt.datetime.strptime(start, "%Y-%m-%d") - relativedelta(years=MACRO_WARMUP_YEARS)
    macro_start = macro_start_dt.strftime("%Y-%m-%d")

    # --- Crypto prices ---
    crypto_tickers_yf = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "LINK-USD", "DOGE-USD"]
    crypto_names = ["BTC", "ETH", "BNB", "XRP", "LINK", "DOGE"]
    crypto_prices = fetch_yfinance_prices(crypto_tickers_yf, start=start, end=end)
    crypto_prices.columns = crypto_names

    # --- Crypto volumes (for minimum volume filter) ---
    logger.info("  📥 Fetching crypto volumes...")
    vol_raw = yf.download(
        crypto_tickers_yf, start=start, end=end,
        auto_adjust=True, progress=False,
    )
    if isinstance(vol_raw.columns, pd.MultiIndex) and "Volume" in vol_raw.columns.get_level_values(0):
        crypto_volumes = vol_raw["Volume"]
        crypto_volumes.columns = crypto_names
    else:
        crypto_volumes = pd.DataFrame(index=crypto_prices.index, columns=crypto_names).fillna(0)

    # --- Gold proxy ---
    gold = build_gold_proxy(start="2015-01-01", end=end)

    # --- VNINDEX ---
    vnindex = load_vnindex_prices(config, start=start, end=end)
    # Ensure correct name for mapping
    if vnindex.name != "VNINDEX":
        vnindex.name = "VNINDEX"

    # --- Combine all prices ---
    all_prices = crypto_prices.copy()
    all_prices["XAUT"] = gold.reindex(crypto_prices.index).ffill()
    all_prices["VNINDEX"] = vnindex.reindex(crypto_prices.index).ffill()
    all_prices = all_prices.dropna(how="all").ffill()

    # --- Macro data from FRED ---
    # Download from macro_start (5 years before backtest) for valid percentile ranks
    fred_data = load_all_fred_series(config, start=macro_start, end=end)

    # --- Sector ETFs for expansion tracker ---
    # Also downloaded from macro_start for warm-up
    sector_etfs = [
        "XLK", "XLV", "XLF", "XLY", "XLP", "XLI",
        "XLC", "XLE", "XLU", "XLRE", "XLB", "GDX",
    ]
    try:
        sector_prices = fetch_yfinance_prices(sector_etfs, start=macro_start, end=end)
    except Exception:
        sector_prices = pd.DataFrame()
        logger.warning("  ⚠️ Sector ETF data unavailable for expansion tracker")

    # --- Market proxies (VIX, DXY, copper, etc.) ---
    # Also from macro_start for valid FCI/GEI percentile rank windows
    market_proxies = ["^VIX", "DX-Y.NYB", "HG=F", "GC=F", "HYG", "LQD"]
    try:
        proxy_prices = fetch_yfinance_prices(market_proxies, start=macro_start, end=end)
        proxy_prices.columns = ["VIX", "DXY", "COPPER", "GOLD_FUTURES", "HYG", "LQD"]
    except Exception:
        proxy_prices = pd.DataFrame()
        logger.warning("  ⚠️ Market proxy data partially unavailable")

    logger.info(f"\n  📊 Data Summary:")
    logger.info(f"     Assets:       {list(all_prices.columns)}")
    logger.info(f"     Date range:   {all_prices.index[0].date()} → {all_prices.index[-1].date()}")
    logger.info(f"     Rows:         {len(all_prices)}")

    return {
        "prices": all_prices,
        "volumes": crypto_volumes,
        "fred": fred_data,
        "sector_prices": sector_prices,
        "proxy_prices": proxy_prices,
        "metadata": metadata,
    }
