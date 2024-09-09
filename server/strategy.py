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
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.avg_price = scraping.current_stock_price(self.symbol)
        self.risk_tolerance = self.calculate_risk_score()
        self.top_percent_from_portfolio = 0
        self.risk_reward_ratio = 0
        self.max_drawdown = 0
        self.loss_percent = 5 if self.risk_tolerance < 80 else 10
        self.profit_percent = 0 if self.risk_tolerance < 80 else 5
        self.stoploss = self.avg_price * (1 - (self.loss_percent / 100))
        self.stopprofit = None if self.profit_percent == 0 else self.avg_price * (1 + (self.profit_percent / 100))
        
        

    def __str__(self):
        return "\n".join(f'{key}: {value}' for key, value in self.__dict__.items())

    def get_strategy_func(self, timeframe='1d'):
        """
        Evaluates multiple strategies and returns the one with the best performance.
    
        Args:
            stock (str): The stock symbol for which to fetch the data and evaluate strategies.
        
        Returns:
            tuple: The best strategy function, along with its performance and risk metrics.
        """
        best_strategy = None
        best_performance = float('-inf')  # Initialize with very low performance
        best_risk_metrics = None

        # List of strategy functions to evaluate
        strategy_functions = ['macd', 'rsi', 'ma', 'bollinger_bands']

        # Fetch stock data for backtesting
        try:
            df = scraping.get_stock_data(self.symbol , DAYS=730, interval=timeframe)
            df = df['DF']
        except Exception as e:
            print(f"Error fetching data for {self.symbol}: {e}")
            

        # Iterate over each strategy function
        for func in strategy_functions:
            
            
            try:
                # Backtest each strategy
                performance, risk_metrics = self.backtest_strategy(df, func, self.sector)
                # print(f"Performance: {performance}, Risk Metrics: {risk_metrics}")
                # Compare performance to find the best strategy
                if performance > best_performance:
                    best_performance = performance
                    best_strategy = func
                    best_risk_metrics = risk_metrics

                # print(f"Strategy: {func}, Performance: {performance}, Risk Metrics: {risk_metrics} \n")

            except Exception as e:
                print(f"Error in {func}: {e}")

        # Return the best strategy with its performance and risk metrics
        return best_strategy, best_performance, best_risk_metrics
    
    def backtest_strategy(self, df, strategy_func, my_sector, transaction_cost=0, tax_on_profit=0.25):
        """
        Backtests a trading strategy based on buy and sell signals.
    
        Args:
        df (pd.DataFrame): The historical price data.
        strategy_func (function): The trading strategy function that returns buy/sell signals.
        my_sector (str): The sector to allocate trades.
        transaction_cost (float): The transaction cost per trade.
        tax_on_profit (float): The tax on profit.
        
        Returns:
            tuple: The performance and risk metrics of the strategy.
        """
        # Initial settings
        cash = 100000  # Initial capital
        starting_cash = cash
        position = 0  # Number of shares held
        average_price = 0  # Average price of shares
        max_drawdown = 0
        peak_value = cash
        win_rate = 0
        total_trades = 0
        winning_trades = 0
        sectors = ['Technology', 'Financial Services', 'Healthcare', 'Consumer Cyclical', 
                'Industrials', 'Communication Services', 'Consumer Defensive', 'Energy', 
                'Real Estate', 'Basic Materials', 'Utilities']
        sector_allocation = {sector: 0 for sector in sectors}

        # Get the buy and sell signals from the strategy function
        try:
            if strategy_func == 'macd':
                signals_df = self.macd(df)

            elif strategy_func == 'rsi':
                signals_df = self.rsi(df)

            elif strategy_func == 'ma':
                signals_df = self.ma(df)

            elif strategy_func == 'bollinger_bands':
                signals_df = self.bollinger_bands(df)

            else:
                raise ValueError(f"Unknown strategy function: {strategy_func}")

        except Exception as e:
            print(f"Error in strategy function: {e}")
            return None, None
        # print(signals_df)
        if signals_df is None:
            return None, None
        # Iterate over each row in the DataFrame
        
        for index, row in signals_df.iterrows():
            # Buy logic
        
            if row['Buy_Signal'] and cash > 0:
                position = cash / df.loc[index]['Close']  # Buy as many shares as possible
                  # Set average price
                average_price = df.loc[index]['Close']
                cash = 0  # All cash used
                total_trades += 1
                sector_allocation[my_sector] += position * df.loc[index]['Close']  # Allocate to sector

            # Sell logic
            elif row['Sell_Signal'] and position > 0:
                # Sell all shares
                sell_value = position * df.loc[index]['Close']
                cash = sell_value * (1 - transaction_cost)  # Deduct transaction cost

                # Calculate profit and apply tax
                profit = cash - starting_cash
                cash -= profit * tax_on_profit  # Apply tax on profit
                # Count winning trades
                if cash >= average_price * position:  # If we made a profit
                    winning_trades += 1
                position = 0  # No shares left after selling

                # Update drawdown and peak value
                if cash > peak_value:
                    peak_value = cash
                drawdown = (peak_value - cash) / peak_value
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

                
        # Final calculations
        final_cash = cash + (position * df.iloc[-1]['Close'] if position > 0 else 0)  # Cash value at end
        win_rate = 0 if (winning_trades <= 0) or (total_trades <= 0) else winning_trades / total_trades  # Calculate win rate
        win_rate = round(win_rate * 100, 2)  # Convert to percentage
        performance = final_cash - starting_cash # Total performance (profit/loss)
        timeframe = (df.index[-1] - df.index[0]).days
        roi = ((final_cash - starting_cash) / starting_cash) * 100  # Return on investment


        # Risk metrics to return
        risk_metrics = {
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'time_frame': timeframe,
            'roi': roi
        }

        return performance, risk_metrics

    def calculate_risk_score(self):
        """
        Returns:
            dict: A dictionary containing the individual risk scores and the overall risk score.
                The keys in the dictionary represent the type of risk, and the values represent the calculated risk score.
                The overall risk score is the sum of all individual risk scores.
        Example:
            
                'debt_to_equity_risk': 0.05,
                'beta_risk': 0.1,
                'profit_margin_risk': 0.02,
                'revenue_growth_risk': 0.03,
                'free_cashflow_risk': 0.04,
                'overall_risk_score': 0.14
        
        Calculate a composite risk score for the company based on financial and operational metrics.
        Example risk score calculation based on selected financial metrics.
        """
        weights = {
            'debt_to_equity': 0.25,
            'beta': 0.2,
            'profit_margins': 0.15,
            'revenue_growth': 0.15,
            'free_cashflow': 0.25
        }
        
        # Calculate individual risks based on weighted metrics
                # Calculate individual risks based on weighted metrics
        risk_score = {}

        '''
        not always has all the data
        '''
        # Debt-to-equity risk (normalized to 0-1 range)
        if self.debtToEquity > 100:
            risk_score['debt_to_equity_risk'] = min(self.debtToEquity / 1000, 1) * weights['debt_to_equity']
        else:
            risk_score['debt_to_equity_risk'] = 0
        
        # Beta risk (normalized to 0-1 range)
        risk_score['beta_risk'] = min(self.beta / 2, 1) * weights['beta']
        
        # Profit margin risk (lower profit margins = more risk)
        if self.profitMargins < 0:
            risk_score['profit_margin_risk'] = abs(self.profitMargins) * weights['profit_margins']  # Negative margins = high risk
        else:
            risk_score['profit_margin_risk'] = max(0.1 - self.profitMargins, 0) * weights['profit_margins']  # Margins below 10% add risk

        # Revenue growth risk (negative or low growth = higher risk)
        if self.revenueGrowth < 0:
            risk_score['revenue_growth_risk'] = abs(self.revenueGrowth) * weights['revenue_growth']
        else:
            risk_score['revenue_growth_risk'] = max(0.05 - self.revenueGrowth, 0) * weights['revenue_growth']

        # Free cash flow risk (low cash flow adds risk)
        if self.ebitda != 0:
            free_cashflow_to_ebitda_ratio = self.freeCashflow / self.ebitda
        else:
            free_cashflow_to_ebitda_ratio = 0
        if free_cashflow_to_ebitda_ratio < 0.1:
            risk_score['free_cashflow_risk'] = (0.1 - free_cashflow_to_ebitda_ratio) * weights['free_cashflow']
        else:
            risk_score['free_cashflow_risk'] = 0

        # Calculate overall risk score and scale to 0-100
        overall_risk_score = sum(risk_score.values()) * 100  # Scale to 0-100

        risk_score['overall_risk_score'] = overall_risk_score

        
        return risk_score['overall_risk_score']

    def macd(self, df):

        try:
            buy_signals = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))

            # Sell signal when MACD crosses below the signal line
            sell_signals = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
        except Exception as e:
            return None
    
        
        signals_df = pd.DataFrame({
            'Buy_Signal': buy_signals,
            'Sell_Signal': sell_signals
        }, index=df.index)

        # Filter the DataFrame to include only rows where either the buy signal or sell signal is true
        filtered_signals_df = signals_df[(signals_df['Buy_Signal']) | (signals_df['Sell_Signal'])]

        return filtered_signals_df
        
    def rsi(self, df):
        """
        Calculates the Relative Strength Index (RSI) signals.
        
        Args:
            df (pd.DataFrame): The historical price data.
            period (int): The period for calculating RSI.
            
        Returns:
            tuple: The buy and sell signals.
        """
        buy_signals = (df['RSI'] < 30) & (df['RSI'].shift(1) >= 30)
        sell_signals = (df['RSI'] > 70) & (df['RSI'].shift(1) <= 70)
        signals_df = pd.DataFrame({
            'Buy_Signal': buy_signals,
            'Sell_Signal': sell_signals
        }, index=df.index)

        # Filter the DataFrame to include only rows where either the buy signal or sell signal is true
        filtered_signals_df = signals_df[(signals_df['Buy_Signal']) | (signals_df['Sell_Signal'])]
        
        return filtered_signals_df

    def ma(self, df):
        """
        Calculates the Moving Average (MA) signals.
        
        Args:
            df (pd.DataFrame): The historical price data.
            
        Returns:
            tuple: The buy and sell signals.
        """
        buy_signals = (df['SMA20'] > df['SMA150']) & (df['SMA20'].shift(1) <= df['SMA150'].shift(1))
        sell_signals = (df['SMA20'] < df['SMA150']) & (df['SMA20'].shift(1) >= df['SMA150'].shift(1))
        signals_df = pd.DataFrame({
            'Buy_Signal': buy_signals,
            'Sell_Signal': sell_signals
        }, index=df.index)

        # Filter the DataFrame to include only rows where either the buy signal or sell signal is true
        filtered_signals_df = signals_df[(signals_df['Buy_Signal']) | (signals_df['Sell_Signal'])]

        
        return filtered_signals_df

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
        
        df['STD20'] = df['Close'].rolling(window=window).std()
        df['Upper_Band'] = df['SMA20'] + (df['STD20'] * num_std_dev)
        df['Lower_Band'] = df['SMA20'] - (df['STD20'] * num_std_dev)
        buy_signals = (df['Close'] < df['Lower_Band']) & (df['Close'].shift(1) >= df['Lower_Band'].shift(1))
        sell_signals = (df['Close'] > df['Upper_Band']) & (df['Close'].shift(1) <= df['Upper_Band'].shift(1))
        signals_df = pd.DataFrame({
            'Buy_Signal': buy_signals,
            'Sell_Signal': sell_signals
        }, index=df.index)

        # Filter the DataFrame to include only rows where either the buy signal or sell signal is true
        filtered_signals_df = signals_df[(signals_df['Buy_Signal']) | (signals_df['Sell_Signal'])]
        
        return filtered_signals_df