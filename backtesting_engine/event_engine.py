"""
Event-Driven Backtesting Engine (Phase 1)
-----------------------------------------
A modular, queue-based architecture designed for tick-by-tick or
bar-by-bar realism. Facilitates modeling of complex slippage, market
microstructure, and latency—crucial for high-frequency strategies.
"""

import queue
import datetime
import pandas as pd
import numpy as np

# =============================================================================
# EVENTS
# =============================================================================
class Event:
    pass

class MarketEvent(Event):
    def __init__(self):
        self.type = 'MARKET'

class SignalEvent(Event):
    def __init__(self, symbol, datetime_tz, signal_type, strength=1.0):
        self.type = 'SIGNAL'
        self.symbol = symbol
        self.datetime = datetime_tz
        self.signal_type = signal_type  # 'LONG' or 'SHORT'
        self.strength = strength

class OrderEvent(Event):
    def __init__(self, symbol, order_type, quantity, direction):
        self.type = 'ORDER'
        self.symbol = symbol
        self.order_type = order_type    # 'MKT' or 'LMT'
        self.quantity = quantity
        self.direction = direction      # 'BUY' or 'SELL'
        
    def print_order(self):
        print(f"Order: {self.direction} {self.quantity} {self.symbol} {self.order_type}")

class FillEvent(Event):
    def __init__(self, timeindex, symbol, exchange, quantity, direction, fill_cost, commission=None):
        self.type = 'FILL'
        self.timeindex = timeindex
        self.symbol = symbol
        self.exchange = exchange
        self.quantity = quantity
        self.direction = direction
        self.fill_cost = fill_cost
        self.commission = commission if commission is not None else self._calculate_commission()

    def _calculate_commission(self):
        return max(1.0, self.quantity * 0.005) # Interactive Brokers style fallback

# =============================================================================
# DATA HANDLER
# =============================================================================
class DataHandler:
    def get_latest_bars(self, symbol, N=1):
        raise NotImplementedError
    def update_bars(self):
        raise NotImplementedError

class HistoricCSVDataHandler(DataHandler):
    """
    Simulates a live market feed from a pandas DataFrame or dictionary of dataframes.
    """
    def __init__(self, events_queue, data_dict):
        self.events = events_queue
        self.symbol_data = data_dict
        self.symbol_data_generator = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True
        
        for symbol, df in data_dict.items():
            self.symbol_data[symbol] = df.sort_index()
            self.symbol_data_generator[symbol] = self.symbol_data[symbol].iterrows()
            self.latest_symbol_data[symbol] = []

    def _get_new_bar(self, symbol):
        for bar in self.symbol_data_generator[symbol]:
            yield bar

    def get_latest_bars(self, symbol, N=1):
        try:
            return self.latest_symbol_data[symbol][-N:]
        except KeyError:
            return []

    def update_bars(self):
        for symbol in self.symbol_data.keys():
            try:
                bar = next(self.symbol_data_generator[symbol])
                # Convert the tuple (index, pd.Series) into a dict for easy access
                bar_dict = bar[1].to_dict()
                bar_dict['datetime'] = bar[0]
                bar_dict['symbol'] = symbol
                self.latest_symbol_data[symbol].append(bar_dict)
            except StopIteration:
                self.continue_backtest = False
        self.events.put(MarketEvent())

# =============================================================================
# STRATEGY & PORTFOLIO
# =============================================================================
class Strategy:
    def calculate_signals(self, event):
        raise NotImplementedError

class NaivePortfolio:
    """
    Handles PnL, sizing, and positions.
    """
    def __init__(self, bars, events, initial_capital=100000.0):
        self.bars = bars
        self.events = events
        self.symbol_list = list(bars.symbol_data.keys())
        self.initial_capital = initial_capital
        
        self.all_positions = []
        self.current_positions = {symbol: 0 for symbol in self.symbol_list}
        self.all_holdings = []
        self.current_holdings = self._construct_current_holdings()

    def _construct_current_holdings(self):
        d = {symbol: 0.0 for symbol in self.symbol_list}
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return d

    def update_timeindex(self):
        latest_datetime = self.bars.get_latest_bars(self.symbol_list[0])[0]['datetime']
        dp = {symbol: self.current_positions[symbol] for symbol in self.symbol_list}
        dp['datetime'] = latest_datetime
        self.all_positions.append(dp)
        
        dh = {symbol: 0.0 for symbol in self.symbol_list}
        dh['datetime'] = latest_datetime
        dh['cash'] = self.current_holdings['cash']
        dh['commission'] = self.current_holdings['commission']
        dh['total'] = self.current_holdings['cash']

        for symbol in self.symbol_list:
            market_value = self.current_positions[symbol] * self.bars.get_latest_bars(symbol)[0]['Close']
            dh[symbol] = market_value
            dh['total'] += market_value

        self.all_holdings.append(dh)

    def update_signal(self, event):
        if event.type == 'SIGNAL':
            order_type = 'MKT'
            mkt_quantity = 100 # Static sizing for proof of concept
            
            if event.signal_type == 'LONG' and self.current_positions[event.symbol] == 0:
                self.events.put(OrderEvent(event.symbol, order_type, mkt_quantity, 'BUY'))
            elif event.signal_type == 'SHORT' and self.current_positions[event.symbol] > 0:
                self.events.put(OrderEvent(event.symbol, order_type, mkt_quantity, 'SELL'))

    def update_fill(self, event):
        if event.type == 'FILL':
            fill_dir = 1 if event.direction == 'BUY' else -1
            self.current_positions[event.symbol] += fill_dir * event.quantity
            fill_cost = event.fill_cost
            self.current_holdings['commission'] += event.commission
            self.current_holdings['cash'] -= (fill_dir * fill_cost * event.quantity + event.commission)

# =============================================================================
# EXECUTION HANDLER
# =============================================================================
class SimulatedExecutionHandler:
    def __init__(self, events, bars):
        self.events = events
        self.bars = bars

    def execute_order(self, event):
        if event.type == 'ORDER':
            # Get latest price to simulate fill (could add slippage model here)
            fill_price = self.bars.get_latest_bars(event.symbol)[0]['Close']
            
            # Simulated Impact (Simplified Almgren-Chriss logic)
            # In a real engine, we'd grab recent volume
            slippage = fill_price * 0.0005 
            fill_price += slippage if event.direction == 'BUY' else -slippage
            
            fill_event = FillEvent(datetime.datetime.utcnow(), event.symbol,
                                   'ARCA', event.quantity, event.direction, fill_price)
            self.events.put(fill_event)

# =============================================================================
# EVENT LOOP ENGINE
# =============================================================================
def run_event_driven_backtest(data_dict, strategy_cls):
    events = queue.Queue()
    bars = HistoricCSVDataHandler(events, data_dict)
    strategy = strategy_cls(bars, events)
    port = NaivePortfolio(bars, events, initial_capital=100000.0)
    broker = SimulatedExecutionHandler(events, bars)

    print("Starting Event-Driven Engine...")
    
    while True:
        if bars.continue_backtest:
            bars.update_bars()
        else:
            break
            
        while True:
            try:
                event = events.get(False)
            except queue.Empty:
                break
            else:
                if event is not None:
                    if event.type == 'MARKET':
                        strategy.calculate_signals(event)
                        port.update_timeindex()
                    elif event.type == 'SIGNAL':
                        port.update_signal(event)
                    elif event.type == 'ORDER':
                        broker.execute_order(event)
                    elif event.type == 'FILL':
                        port.update_fill(event)

    print("Backtest Complete. Calculating Stats...")
    returns_df = pd.DataFrame(port.all_holdings)
    returns_df.set_index('datetime', inplace=True)
    
    total_return = returns_df['total'].iloc[-1] / returns_df['total'].iloc[0] - 1.0
    print(f"Final Portfolio Value: ${returns_df['total'].iloc[-1]:.2f}")
    print(f"Total Return: {total_return*100:.2f}%")
    return returns_df
