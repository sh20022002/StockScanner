import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from deap import base, creator, tools, algorithms
from concurrent.futures import ThreadPoolExecutor
import scraping  # Assuming this is your module for fetching stock data
from sklearn.model_selection import ParameterGrid

class Signal:
    """
    Represents a trading signal.
    
    Attributes:
        symbol (str): The stock symbol.
        signal_type (str): The type of signal ('buy' or 'sell').
        signal_time (datetime): The time the signal was generated.
    """
    def __init__(self, symbol, signal_type):
        self.symbol = symbol
        self.signal_type = signal_type  # 'buy' or 'sell'
        self.signal_time = scraping.get_exchange_time()

    def __str__(self):
        return f"Signal({self.symbol}, {self.signal_type}, {self.signal_time})"

class SignalStack:
    """
    Represents a stack of trading signals.
    
    Attributes:
        signals (list): A list of Signal objects.
    """
    def __init__(self):
        self.signals = []

    def push(self, signal):
        """
        Pushes a new signal onto the stack and removes irrelevant signals.
        
        Args:
            signal (Signal): The signal to be added.
        """
        self.signals.append(signal)
        self.remove_irrelevant_signals()

    def pop(self):
        """
        Pops the most recent signal from the stack.
        
        Returns:
            Signal: The most recent signal, or None if the stack is empty.
        """
        if not self.is_empty():
            return self.signals.pop()
        return None

    def peek(self):
        """
        Peeks at the most recent signal without removing it.
        
        Returns:
            Signal: The most recent signal, or None if the stack is empty.
        """
        if not self.is_empty():
            return self.signals[-1]
        return None

    def is_empty(self):
        """
        Checks if the stack is empty.
        
        Returns:
            bool: True if the stack is empty, False otherwise.
        """
        return len(self.signals) == 0

    def remove_irrelevant_signals(self):
        """
        Removes signals that are not relevant anymore.
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


def backtest_strategy(df, strategy_func, initial_cash=10000, commission=0):
    """
    Backtests a trading strategy on historical data.
    
    Args:
        df (pd.DataFrame): The historical price data.
        strategy_func (function): The trading strategy function.
        initial_cash (float): The initial amount of cash.
        commission (float): The commission per trade.
        
    Returns:
        float: The final amount of cash after backtesting.
    """
    cash = initial_cash
    position = 0
    for index, row in df.iterrows():
        buy_signals, sell_signals = strategy_func(df.loc[:index], 'AAPL')
        if buy_signals.iloc[-1]:
            position = cash / row['Close']
            cash = 0
        elif sell_signals.iloc[-1] and position > 0:
            cash = position * row['Close']
            position = 0
    final_cash = cash + position * df.iloc[-1]['Close']
    return final_cash


def optimize_strategy(df, strategy_func, param_grid):
    """
    Optimizes a trading strategy using a grid search.
    
    Args:
        df (pd.DataFrame): The historical price data.
        strategy_func (function): The trading strategy function.
        param_grid (dict): The grid of parameters to search.
        
    Returns:
        tuple: The best parameters and the best performance.
    """
    best_params = None
    best_performance = -float('inf')
    for params in ParameterGrid(param_grid):
        performance = backtest_strategy(df, lambda df, symbol: strategy_func(df, symbol, **params))
        if performance > best_performance:
            best_performance = performance
            best_params = params
    return best_params, best_performance

class Strategy:
    """
    Represents a trading strategy.
    
    Attributes:
        name (str): The name of the strategy.
        avg_price (float): The average price of the asset.
        risk_tolerance (int): The risk tolerance level (0-100).
        top_percent_from_portfolio (float): The percentage of the asset's value in the portfolio.
        risk_reward_ratio (float): The risk-reward ratio of the strategy.
        max_drawdown (float): The maximum drawdown of the strategy.
        loss_percent (float): The percentage of loss to trigger a stop loss.
        profit_percent (float): The percentage of profit to trigger a stop profit.
        stoploss (float): The stop loss price.
        stopprofit (float): The stop profit price.
        strategy_func (function): The strategy function.
        avg_roi (float): The average return on investment.
        avg_holding_frame (float): The average holding frame.
    """
    def __init__(self, name, avg_price, risk_tolerance, top_percent_from_portfolio, **kwargs):
        self.name = name
        self.avg_price = avg_price
        self.risk_tolerance = risk_tolerance  # 0 - 100
        self.top_percent_from_portfolio = top_percent_from_portfolio
        self.risk_reward_ratio = 0
        self.max_drawdown = 0
        self.loss_percent = kwargs.get('loss_percent', 0.1)
        self.profit_percent = kwargs.get('profit_percent', 0.1)
        self.stoploss = self.avg_price * (1 - self.loss_percent)
        self.stopprofit = self.avg_price * (1 + self.profit_percent)
        self.strategy_func, self.avg_roi, self.avg_holding_frame = self.get_strategy_func()

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        return "\n".join(f'{key}: {value}' for key, value in self.__dict__.items())

    def get_strategy_func(self):
        """
        Gets the strategy function.
        
        Returns:
            function: The strategy function.
        """
        pass

    def macd(self, df):
        """
        Calculates the Moving Average Convergence Divergence (MACD) signals.
        
        Args:
            df (pd.DataFrame): The historical price data.
            
        Returns:
            tuple: The buy and sell signals.
        """
        df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        buy_signals = (df['MACD'] > df['Signal']) & (df['MACD'].shift(1) <= df['Signal'].shift(1))
        sell_signals = (df['MACD'] < df['Signal']) & (df['MACD'].shift(1) >= df['Signal'].shift(1))
        return buy_signals, sell_signals

    def rsi(self, df, period=14):
        """
        Calculates the Relative Strength Index (RSI) signals.
        
        Args:
            df (pd.DataFrame): The historical price data.
            period (int): The period for calculating RSI.
            
        Returns:
            tuple: The buy and sell signals.
        """
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        buy_signals = (df['RSI'] < 30) & (df['RSI'].shift(1) >= 30)
        sell_signals = (df['RSI'] > 70) & (df['RSI'].shift(1) <= 70)
        return buy_signals, sell_signals

    def ma(self, df, short_window=40, long_window=100):
        """
        Calculates the Moving Average (MA) signals.
        
        Args:
            df (pd.DataFrame): The historical price data.
            short_window (int): The short window period for MA.
            long_window (int): The long window period for MA.
            
        Returns:
            tuple: The buy and sell signals.
        """
        df['Short_MA'] = df['Close'].rolling(window=short_window).mean()
        df['Long_MA'] = df['Close'].rolling(window=long_window).mean()
        buy_signals = (df['Short_MA'] > df['Long_MA']) & (df['Short_MA'].shift(1) <= df['Long_MA'].shift(1))
        sell_signals = (df['Short_MA'] < df['Long_MA']) & (df['Short_MA'].shift(1) >= df['Long_MA'].shift(1))
        return buy_signals, sell_signals

    def bollinger_bands(self, df, window=20, num_std_dev=2):
        """
        Calculates the Bollinger Bands signals.
        
        Args:
            df (pd.DataFrame): The historical price data.
            window (int): The window period for calculating the moving average.
            num_std_dev (int): The number of standard deviations for the bands.
            
        Returns:
            tuple: The buy and sell signals.
        """
        df['MA20'] = df['Close'].rolling(window=window).mean()
        df['STD20'] = df['Close'].rolling(window=window).std()
        df['Upper_Band'] = df['MA20'] + (df['STD20'] * num_std_dev)
        df['Lower_Band'] = df['MA20'] - (df['STD20'] * num_std_dev)
        buy_signals = (df['Close'] < df['Lower_Band']) & (df['Close'].shift(1) >= df['Lower_Band'].shift(1))
        sell_signals = (df['Close'] > df['Upper_Band']) & (df['Close'].shift(1) <= df['Upper_Band'].shift(1))
        return buy_signals, sell_signals


def evaluate_strategy(individual, df, strategy_func, transaction_cost=0.002, tax_on_profit=0.25):
    """
    Evaluates a trading strategy using a genetic algorithm.
    
    Args:
        individual (list): The list of parameters for the strategy.
        df (pd.DataFrame): The historical price data.
        strategy_func (function): The trading strategy function.
        transaction_cost (float): The transaction cost per trade.
        tax_on_profit (float): The tax on profit.
        
    Returns:
        tuple: The performance and win rate of the strategy.
    """
    params = {key: val for key, val in zip(param_names, individual)}
    performance, risk_metrics = backtest_strategy(df, lambda df, symbol: strategy_func(df, symbol, **params), transaction_cost, tax_on_profit)
    win_rate = risk_metrics['win_rate']
    max_drawdown = risk_metrics['max_drawdown']
    trading_volume = df['Volume'].mean()
    adjusted_drawdown = max_drawdown / trading_volume
    if win_rate < 0.8:
        return -float('inf'),  # Penalize strategies with win rate below 80%
    return performance - adjusted_drawdown, win_rate

def genetic_algorithm(df, strategy_func, param_grid, generations=50, population_size=100, transaction_cost=0.002, tax_on_profit=0.25):
    """
    Optimizes a trading strategy using a genetic algorithm.
    
    Args:
        df (pd.DataFrame): The historical price data.
        strategy_func (function): The trading strategy function.
        param_grid (dict): The grid of parameters to search.
        generations (int): The number of generations for the genetic algorithm.
        population_size (int): The size of the population for the genetic algorithm.
        transaction_cost (float): The transaction cost per trade.
        tax_on_profit (float): The tax on profit.
        
    Returns:
        dict: The best parameters found by the genetic algorithm.
    """
    global param_names
    param_names = list(param_grid.keys())
    param_ranges = list(param_grid.values())

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(param_names))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_strategy, df=df, strategy_func=strategy_func, transaction_cost=transaction_cost, tax_on_profit=tax_on_profit)

    population = toolbox.population(n=population_size)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, verbose=False)

    best_individual = tools.selBest(population, k=1)[0]
    best_params = {key: val for key, val in zip(param_names, best_individual)}
    return best_params


def backtest_strategy(df, strategy_func, transaction_cost=0.002, tax_on_profit=0.25):
    """
    Backtests a trading strategy on historical data.
    
    Args:
        df (pd.DataFrame): The historical price data.
        strategy_func (function): The trading strategy function.
        transaction_cost (float): The transaction cost per trade.
        tax_on_profit (float): The tax on profit.
        
    Returns:
        tuple: The performance and risk metrics of the strategy.
    """
    cash = 10000
    position = 0
    max_drawdown = 0
    peak_value = cash
    win_rate = 0
    total_trades = 0
    winning_trades = 0
    sectors = df['Sector'].unique()
    sector_allocation = {sector: 0 for sector in sectors}

    for index, row in df.iterrows():
        buy_signals, sell_signals = strategy_func(df.loc[:index], 'AAPL')
        if buy_signals.iloc[-1]:
            position = cash / row['Close']
            cash = 0
            total_trades += 1
            sector_allocation[row['Sector']] += position * row['Close']
        elif sell_signals.iloc[-1] and position > 0:
            cash = position * row['Close'] * (1 - transaction_cost)
            profit = cash - 10000
            cash -= profit * tax_on_profit
            position = 0
            if cash > peak_value:
                peak_value = cash
            drawdown = (peak_value - cash) / peak_value
            if drawdown > max_drawdown:
                max_drawdown = drawdown
            if cash > 10000:
                winning_trades += 1

    final_cash = cash + position * df.iloc[-1]['Close']
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    performance = final_cash - 10000
    risk_metrics = {'max_drawdown': max_drawdown, 'win_rate': win_rate}
    return performance, risk_metrics


# Example usage
param_grid = {
    'short_window': [10, 50],
    'long_window': [50, 200],
    'stop_loss_pct': [0.01, 0.1],
    'take_profit_pct': [0.01, 0.1]
}

df = pd.read_csv('your_data.csv')  # Load your data here
best_params = genetic_algorithm(df, Strategy().ma, param_grid)
print("Best Parameters:", best_params)