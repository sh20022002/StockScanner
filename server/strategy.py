import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from deap import base, creator, tools, algorithms
from concurrent.futures import ThreadPoolExecutor
import scraping  # Assuming this is your module for fetching stock data

class Signal:
    def __init__(self, symbol, signal_type):
        self.symbol = symbol
        self.signal_type = signal_type # 'buy' or 'sell'
        self.signal_time = scraping.get_exchange_time()

    def __str__(self):
        return f"Signal({self.symbol}, {self.signal_type}, {self.signal_time})"

class SignalStack:
    def __init__(self):
        self.signals = []

    def push(self, signal):
        self.signals.append(signal)
        self.remove_irrelevant_signals()

    def pop(self):
        if not self.is_empty():
            return self.signals.pop()
        return None

    def peek(self):
        if not self.is_empty():
            return self.signals[-1]
        return None

    def is_empty(self):
        return len(self.signals) == 0

    def remove_irrelevant_signals(self):
        """
        Remove signals that are not relevant anymore.
        For example, you could define irrelevance by a time threshold or
        when a new signal of the opposite type is issued for the same symbol.
        """
        # Remove signals older than a certain threshold (e.g., 1 hour)
        current_time = scraping.get_exchange_time()
        self.signals = [signal for signal in self.signals if current_time - signal.signal_time <= timedelta(hours=1)]

        # Alternatively, remove all but the latest signal for each symbol
        latest_signals = {}
        for signal in self.signals:
            latest_signals[signal.symbol] = signal
        self.signals = list(latest_signals.values())

    def __str__(self):
        return "\n".join(str(signal) for signal in self.signals)

class Strategy:
    def __init__(self, avg_price, signal_stack, top_percent_from_portfolio=0.05, loss_percent=0.05, profit_percent=0.2, **kwargs):
        self.avg_price = avg_price
        self.signal_stack = signal_stack
        self.top_percent_from_portfolio = top_percent_from_portfolio
        self.loss_percent = loss_percent
        self.profit_percent = profit_percent
        self.stoploss = self.avg_price * (1 - self.loss_percent)
        self.stopprofit = self.avg_price * (1 + self.profit_percent)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        return "\n".join(f'{key}: {value}' for key, value in self.__dict__.items())

    def simulate_trading(self, symbol, cash=10000, commission=0.01):
        df = scraping.get_stock_data(symbol)  # Fetch data using scraping module

        # Ensure necessary columns exist in the DataFrame
        required_columns = ['Close', 'SMA150', 'EMA20', 'MACD', 'Signal']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in the data for {symbol}.")

        # Implement multiple strategies and choose the best one
        best_strategy, final_cash = self.simulate_multiple_strategies(df, symbol, cash, commission)
        print(f"Best Strategy: {best_strategy} | Final Cash: {final_cash}")
        return final_cash

    def moving_average_crossover(self, df, symbol):
        short_ma = df['SMA150']
        long_ma = df['EMA20']

        buy_signals = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
        sell_signals = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))

        self.add_signals_to_stack(buy_signals, sell_signals, symbol)
        return buy_signals, sell_signals

    def macd_strategy(self, df, symbol):
        macd_line = df['MACD']
        signal_line = df['Signal']

        buy_signals = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        sell_signals = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

        self.add_signals_to_stack(buy_signals, sell_signals, symbol)
        return buy_signals, sell_signals

    def add_signals_to_stack(self, buy_signals, sell_signals, symbol):
        latest_buy_signal = buy_signals.index[-1] if buy_signals.iloc[-1] else None
        latest_sell_signal = sell_signals.index[-1] if sell_signals.iloc[-1] else None

        if latest_buy_signal:
            buy_signal = Signal(symbol=symbol, signal_type='buy', signal_time=latest_buy_signal)
            self.signal_stack.push(buy_signal)

        if latest_sell_signal:
            sell_signal = Signal(symbol=symbol, signal_type='sell', signal_time=latest_sell_signal)
            self.signal_stack.push(sell_signal)

    def simulate_multiple_strategies(self, df, symbol, cash, commission):
        strategies = [
            self.moving_average_crossover,
            self.macd_strategy,
            # Add more strategies if needed
        ]

        best_cash = -np.inf
        best_strategy = None

        for strategy_func in strategies:
            buy_signals, sell_signals = strategy_func(df, symbol)
            cash_after_strategy = self.simulate_trading_with_signals(df['Close'].values, buy_signals, sell_signals, symbol, cash, commission)

            if cash_after_strategy > best_cash:
                best_cash = cash_after_strategy
                best_strategy = strategy_func.__name__

        return best_strategy, best_cash

    def simulate_trading_with_signals(self, prices, buy_signals, sell_signals, symbol, cash=10000, commission=0.01):
        in_position = False

        for i in range(1, len(prices)):
            if buy_signals[i] and not in_position:
                in_position = True
                cash -= prices[i] * (1 + commission)
            elif sell_signals[i] and in_position:
                in_position = False
                cash += prices[i] * (1 - commission)

        return cash

def get_current_signals(symbol, strategy_name, signal_stack, **strategy_params):
    df = scraping.get_stock_data(symbol)  # Fetch the latest stock data

    # Ensure necessary columns exist in the DataFrame
    required_columns = ['Close', 'SMA150', 'EMA20', 'MACD', 'Signal']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the data for {symbol}.")

    strategy = Strategy(avg_price=df['Close'].iloc[-1], signal_stack=signal_stack, **strategy_params)

    if hasattr(strategy, strategy_name):
        strategy_func = getattr(strategy, strategy_name)
        buy_signals, sell_signals = strategy_func(df, symbol)

        # Check the latest signal
        latest_buy_signal = buy_signals.iloc[-1]
        latest_sell_signal = sell_signals.iloc[-1]

        return latest_buy_signal, latest_sell_signal
    else:
        raise ValueError(f"Strategy {strategy_name} is not implemented in the Strategy class.")

# Example usage
