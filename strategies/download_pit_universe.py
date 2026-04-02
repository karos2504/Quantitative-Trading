import sys
import os
import pickle
import datetime as dt
import logging
from pathlib import Path

# Fix path to allow importing from parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategies.pit_universe import PointInTimeUniverse
from data_ingestion.data_store import update_universe_data, DATA_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

CACHE_FILE = DATA_DIR / "pit_cache.pkl"

def download_and_cache():
    """Fetches Wikipedia data, downloads OHLCV records, and saves to local cache."""
    logger.info("=" * 62)
    logger.info("  S&P 500 Point-in-Time Universe Downloader")
    logger.info("=" * 62)
    
    # 1. Scrape Wikipedia
    pit_engine = PointInTimeUniverse()
    
    # 2. Build Master Ticker List
    master_tickers = set(pit_engine.current_sp500)
    for t in pit_engine.changes_df['Removed_Ticker']:
        if t != 'nan':
            master_tickers.add(t)
    master_tickers_list = list(master_tickers)
    
    logger.info(f"  Total historical universe size: {len(master_tickers_list)} tickers")
    
    # 3. Download Data (10y windows, 1mo interval)
    start_date = dt.datetime.today() - dt.timedelta(days=365 * 13)
    end_date   = dt.datetime.today()
    
    logger.info(f"  Downloading/Verifying data from {start_date.date()} to {end_date.date()}...")
    update_universe_data(master_tickers_list, start=start_date, end=end_date, interval='1mo')
    
    # 4. Filter for Successfully Downloaded Tickers (Silences warnings for delisted stocks)
    logger.info("  Verifying successful downloads...")
    available_tickers = []
    for t in master_tickers_list:
        if (DATA_DIR / f"{t}_1mo.parquet").exists():
            available_tickers.append(t)
            
    missing_count = len(master_tickers_list) - len(available_tickers)
    logger.info(f"  ✅ Verified {len(available_tickers)} tickers. {missing_count} (delisted/failed) tickers filtered.")
    
    # 5. Save Cache
    logger.info(f"  Saving PiT metadata to {CACHE_FILE.name}...")
    cache_data = {
        'pit_engine': pit_engine,
        'master_tickers_list': available_tickers, # Use ONLY available tickers
        'timestamp': dt.datetime.now()
    }
    
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache_data, f)
        
    logger.info(f"  ✅ Cache updated successfully. You can now run rebalance_portfolio.py instantly.")

if __name__ == '__main__':
    download_and_cache()
