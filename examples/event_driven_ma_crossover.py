"""
Event-Driven Moving Average Crossover Strategy Example
Validates the functionality of the new Event-Driven Backtesting Engine.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting_engine.event_engine import Strategy, SignalEvent, run_event_driven_backtest
from data_ingestion.data_store import update_universe_data, load_universe_data

class MACrossover(Strategy):
    """
    Simple Moving Average Crossover Strategy using an Event-Driven setup.
    """
    def __init__(self, bars, events, short_window=10, long_window=30):
        self.bars = bars
        self.events = events
        self.symbol_list = list(self.bars.symbol_data.keys())
        self.short_window = short_window
        self.long_window = long_window

    def calculate_signals(self, event):
        if event.type == 'MARKET':
            for symbol in self.symbol_list:
                bars = self.bars.get_latest_bars(symbol, N=self.long_window)
                if len(bars) == self.long_window:
                    closes = [b['Close'] for b in bars]
                    short_sma = sum(closes[-self.short_window:]) / self.short_window
                    long_sma = sum(closes) / self.long_window
                    
                    dt = bars[-1]['datetime']
                    
                    if short_sma > long_sma:
                        signal = SignalEvent(symbol, dt, 'LONG')
                        self.events.put(signal)
                    elif short_sma < long_sma:
                        signal = SignalEvent(symbol, dt, 'SHORT')
                        self.events.put(signal)

if __name__ == "__main__":
    print("Preparing data for Event-Driven Backtest...")
    tickers = ['AAPL']
    update_universe_data(tickers, interval='1d', period='2y')
    data_dict = load_universe_data(tickers, interval='1d')
    
    if data_dict:
        # Trim data for a faster validation test
        trimmed_dict = {ticker: df.iloc[-200:] for ticker, df in data_dict.items() if not df.empty}
        print("Starting event loop...")
        returns = run_event_driven_backtest(trimmed_dict, MACrossover)
    else:
        print("Failed to load data for validation.")
