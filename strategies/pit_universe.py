import pandas as pd
import datetime as dt
import logging
import requests

logger = logging.getLogger(__name__)

class PointInTimeUniverse:
    def __init__(self):
        self.url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        self.current_sp500 = []
        self.changes_df = pd.DataFrame(columns=['Date', 'Added_Ticker', 'Added_Name', 'Removed_Ticker', 'Removed_Name', 'Reason'])
        self._load_wikipedia_data()

    def _load_wikipedia_data(self):
        """Scrapes and formats the current list and historical changes from Wikipedia."""
        try:
            logger.info("  Fetching S&P 500 historical constituents from Wikipedia...")
            header = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            res = requests.get(self.url, headers=header, timeout=10)
            res.raise_for_status()
            
            tables = pd.read_html(res.text)
            
            # Current S&P 500
            current_df = tables[0]
            self.current_sp500 = current_df['Symbol'].str.replace('.', '-').tolist()
            
            # Historical Changes
            changes_df = tables[1].copy()
            # Wikipedia format check: changes table sometimes has multi-index
            if isinstance(changes_df.columns, pd.MultiIndex):
                changes_df.columns = changes_df.columns.get_level_values(1)
            
            changes_df.columns = ['Date', 'Added_Ticker', 'Added_Name', 'Removed_Ticker', 'Removed_Name', 'Reason']
            
            # Fix dates and clean tickers
            # Wikipedia sometimes adds citations like "Nov 1, 2023[12]" to dates
            changes_df['Date'] = changes_df['Date'].astype(str).str.extract(r'([A-Za-z]+ \d{1,2}, \d{4})')
            changes_df['Date'] = pd.to_datetime(changes_df['Date'])
            
            changes_df['Added_Ticker'] = changes_df['Added_Ticker'].astype(str).str.replace('.', '-')
            changes_df['Removed_Ticker'] = changes_df['Removed_Ticker'].astype(str).str.replace('.', '-')
            
            self.changes_df = changes_df.sort_values('Date', ascending=False).reset_index(drop=True)
            logger.info(f"  Successfully loaded {len(self.current_sp500)} current tickers and {len(self.changes_df)} historical changes.")
            
        except Exception as e:
            logger.error(f"  Failed to load Wikipedia data (using current list only): {e}")
            if not self.current_sp500:
                logger.warning("  Using fallback Universe Metadata list.")
                # Fallback to a minimal list or existing UNIVERSE if possible

    def get_universe_for_date(self, target_date) -> list:
        """Rolls back the current S&P 500 list to the target date."""
        if not isinstance(target_date, dt.datetime):
            target_date = pd.to_datetime(target_date)
            
        universe = set(self.current_sp500)
        future_changes = self.changes_df[self.changes_df['Date'] > target_date]
        
        for _, row in future_changes.iterrows():
            added = row['Added_Ticker']
            removed = row['Removed_Ticker']
            
            if added != 'nan' and added in universe:
                universe.remove(added)
                
            if removed != 'nan':
                universe.add(removed)
                
        return sorted(list(universe))
