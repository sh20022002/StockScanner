import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from multiprocessing import Pool
import itertools
import pandas_ta as ta  # Using pandas_ta for technical indicators
import multiprocessing
import yfinance as yf  # Replacing 'scraping' with 'yfinance'

class Signal:
    """
    Represents a trading signal.
    """
    def __init__(self, symbol, signal_type):
        self.symbol = symbol
        self.signal_type = signal_type  # 'buy' or 'sell'
        self.signal_time = datetime.now()  # Using current time from datetime module

    def __str__(self):
        return f"Signal({self.symbol}, {self.signal_type}, {self.signal_time})"

class SignalStack:
    """
    Represents a stack of trading signals.
    """
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
        # Remove signals older than a certain threshold (e.g., 1 hour)
        current_time = datetime.now()
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
    """
    def __init__(self, symbol, **kwargs):
        self.symbol = symbol
        # Set attributes from kwargs dynamically
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Get the average stock price using yfinance
        try:
            ticker = yf.Ticker(self.symbol)
            self.avg_price = ticker.history(period='1d')['Close'].iloc[-1]
        except Exception as e:
            print(f"Error fetching current stock price for {self.symbol}: {e}")
            self.avg_price = None

        # Try to calculate the risk tolerance, default to 1 if it fails
        try:
            self.risk_tolerance = self.calculate_risk_score()
        except Exception as e:
            print(f"Error calculating risk score: {e}")
            self.risk_tolerance = 1

        # Set other attributes
        self.top_percent_from_portfolio = 0
        self.risk_reward_ratio = 0
        self.max_drawdown = 0
        
        # Loss and profit percent logic based on risk tolerance
        self.loss_percent = 5 if self.risk_tolerance < 80 else 7
        self.profit_percent = None if self.risk_tolerance < 80 else 10

    def __str__(self):
        return "\n".join(f'{key}: {value}' for key, value in self.__dict__.items())

    def compute_indicators(self, df):
        """
        Computes all the technical indicators required for the strategies using pandas_ta.
        """
        # Ensure 'Date' is the index if available
        if 'Date' in df.columns:
            df.set_index('Date', inplace=True)

        # MACD
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        # RSI
        df.ta.rsi(length=14, append=True)
        # SMA
        df.ta.sma(length=20, append=True)
        df.ta.sma(length=150, append=True)
        # Bollinger Bands
        df.ta.bbands(length=20, std=2, append=True)
        # EMA
        df.ta.ema(length=12, append=True)
        df.ta.ema(length=26, append=True)
        # Stochastic Oscillator
        df.ta.stoch(length=14, append=True)
        # ATR
        df.ta.atr(length=14, append=True)
        # Donchian Channels
        df.ta.donchian(length=20, append=True)
        # VWAP
        df.ta.vwap(append=True)
        # Ichimoku Cloud
        df.ta.ichimoku(append=True)
        # Parabolic SAR
        df.ta.psar(append=True)

        return df

    def get_strategy_func(self, timeframe='1h', num_threads=5):
        """
        Evaluates multiple strategies concurrently using backtest_strategy and returns the one with the best performance.
        """
        def backtest_strategy_task(strategy_func_name):
            try:
                strategy_func = getattr(self, strategy_func_name)
                performance, risk_metrics = self.backtest_strategy(df.copy(), strategy_func)
                return strategy_func_name, performance, risk_metrics
            except Exception as e:
                print(f"Error in strategy {strategy_func_name}: {e}")
                return None

        best_strategy = None
        best_performance = float('-inf')  # Initialize with very low performance
        best_risk_metrics = None

        # List of strategy functions to evaluate
        strategy_functions = ['macd', 'rsi', 'ma', 'bollinger_bands', 'vwap', 'donchian_channel',
                              'atr_breakout', 'stochastic_oscillator', 'ema_crossover', 'combined']

        # Fetch stock data for backtesting
        try:
            df = yf.download(self.symbol, period='730d', interval=timeframe)
            df = self.compute_indicators(df)
        except Exception as e:
            print(f"Error fetching data for {self.symbol}: {e}")
            return None, None, None

        # Run the strategy backtests concurrently using multiprocessing
        with multiprocessing.Pool(processes=num_threads) as pool:
            results = pool.map(backtest_strategy_task, strategy_functions)

        # Iterate over results
        for result in results:
            if result:
                strategy_func_name, performance, risk_metrics = result

                if performance is None:
                    continue

                # Check if this strategy has the best performance
                if performance > best_performance:
                    best_performance = performance
                    best_strategy = strategy_func_name
                    best_risk_metrics = risk_metrics
                print(f"Strategy: {strategy_func_name}, Performance: {performance}, \nRisk Metrics: {risk_metrics}")

        return best_strategy, best_performance, best_risk_metrics

    def backtest_strategy(self, df, strategy_func, transaction_cost=0.001, slippage=0.0005, stop_loss_percent=5, stop_profit_percent=5):
        """
        Backtests a trading strategy based on buy and sell signals, with optional moving stop-loss and stop-profit logic.
        """
        # Initial settings
        cash = 100000  # Initial capital
        starting_cash = cash
        position = 0  # Number of shares held
        entry_price = 0  # Price at which we entered the position
        max_drawdown = 0
        peak_value = cash
        win_rate = 0
        total_trades = 0
        winning_trades = 0
        returns = []

        # Get the buy and sell signals from the strategy function
        try:
            signals_df = strategy_func(df)
        except Exception as e:
            print(f"Error in strategy function: {e}")
            return None, None

        if signals_df is None:
            return None, None

        # Merge signals with price data
        df = df.join(signals_df[['Buy_Signal', 'Sell_Signal']], how='left').fillna(False)

        # Convert to NumPy arrays for faster computation
        close_prices = df['Close'].values
        buy_signals = df['Buy_Signal'].values
        sell_signals = df['Sell_Signal'].values

        # Backtesting loop using vectorized operations
        for i in range(len(close_prices)):
            current_price = close_prices[i]
            current_price += current_price * slippage  # Adjust for slippage

            # Buy logic
            if buy_signals[i] and cash > 0:
                position = cash / current_price  # Buy as many shares as possible
                entry_price = current_price  # Set entry price
                cash = 0  # All cash used
                total_trades += 1

                # Set stop-loss and stop-profit prices only if the values are provided
                if stop_loss_percent is not None:
                    stop_loss_price = entry_price * (1 - stop_loss_percent / 100)  # Initial stop-loss price
                else:
                    stop_loss_price = None

                if stop_profit_percent is not None:
                    stop_profit_price = entry_price * (1 + stop_profit_percent / 100)  # Initial stop-profit price
                else:
                    stop_profit_price = None

            # Sell logic based on stop-loss, stop-profit, or sell signal
            elif position > 0:
                # Check stop-loss
                if stop_loss_price is not None and current_price <= stop_loss_price:
                    sell_value = position * current_price
                    cash = sell_value * (1 - transaction_cost)
                    profit = cash - starting_cash
                    position = 0
                    returns.append(profit)

                # Check stop-profit
                elif stop_profit_price is not None and current_price >= stop_profit_price:
                    sell_value = position * current_price
                    cash = sell_value * (1 - transaction_cost)
                    profit = cash - starting_cash
                    position = 0
                    winning_trades += 1
                    returns.append(profit)

                # Sell signal
                elif sell_signals[i]:
                    sell_value = position * current_price
                    cash = sell_value * (1 - transaction_cost)
                    profit = cash - starting_cash
                    position = 0
                    if profit > 0:
                        winning_trades += 1
                    returns.append(profit)

                # Update drawdown and peak value
                total_value = cash + position * current_price
                if total_value > peak_value:
                    peak_value = total_value
                drawdown = (peak_value - total_value) / peak_value
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

        # Final calculations
        final_cash = cash + (position * close_prices[-1] if position > 0 else 0)
        win_rate = 0 if total_trades == 0 else (winning_trades / total_trades) * 100
        performance = final_cash - starting_cash
        timeframe = (df.index[-1] - df.index[0]).days
        roi = ((final_cash - starting_cash) / starting_cash) * 100

        # Risk metrics
        returns = pd.Series(returns)
        risk_metrics = self.calculate_risk_metrics(returns)
        risk_metrics.update({
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'time_frame': timeframe,
            'roi': roi
        })

        return performance, risk_metrics

    def calculate_risk_metrics(self, returns):
        """
        Calculates risk metrics such as Sharpe Ratio and standard deviation of returns.
        """
        if len(returns) == 0:
            return {'sharpe_ratio': None, 'std_return': None}

        mean_return = returns.mean()
        std_return = returns.std()
        if std_return != 0:
            sharpe_ratio = mean_return / std_return * np.sqrt(252)  # Annualized Sharpe Ratio
        else:
            sharpe_ratio = 0

        return {'sharpe_ratio': sharpe_ratio, 'std_return': std_return}

    def calculate_risk_score(self):
        """
        Calculate a composite risk score for the company based on financial and operational metrics.
        """
        weights = {
            'debt_to_equity': 0.25,
            'beta': 0.2,
            'profit_margins': 0.15,
            'revenue_growth': 0.15,
            'free_cashflow': 0.25
        }

        # Fetch company financial data using yfinance
        try:
            ticker = yf.Ticker(self.symbol)
            info = ticker.info

            self.debtToEquity = info.get('debtToEquity', 0)
            self.beta = info.get('beta', 1)
            self.profitMargins = info.get('profitMargins', 0)
            self.revenueGrowth = info.get('revenueGrowth', 0)
            self.freeCashflow = info.get('freeCashflow', 0)
            self.ebitda = info.get('ebitda', 1)  # Avoid division by zero

        except Exception as e:
            print(f"Error fetching financial data for {self.symbol}: {e}")
            self.debtToEquity = 0
            self.beta = 1
            self.profitMargins = 0
            self.revenueGrowth = 0
            self.freeCashflow = 0
            self.ebitda = 1

        # Calculate individual risks based on weighted metrics
        risk_score = {}

        # Debt-to-equity risk (normalized to 0-1 range)
        if self.debtToEquity > 100:
            risk_score['debt_to_equity_risk'] = min(self.debtToEquity / 1000, 1) * weights['debt_to_equity']
        else:
            risk_score['debt_to_equity_risk'] = 0

        # Beta risk (normalized to 0-1 range)
        risk_score['beta_risk'] = min(self.beta / 2, 1) * weights['beta']

        # Profit margin risk (lower profit margins = more risk)
        if self.profitMargins < 0:
            risk_score['profit_margin_risk'] = abs(self.profitMargins) * weights['profit_margins']
        else:
            risk_score['profit_margin_risk'] = max(0.1 - self.profitMargins, 0) * weights['profit_margins']

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
        overall_risk_score = sum(risk_score.values()) * 100

        risk_score['overall_risk_score'] = overall_risk_score

        return risk_score['overall_risk_score']

    def detect_signals_multiprocess(self, df, threshold=3):
        """
        Detects buy and sell signals using multiple trading strategies with multiprocessing.
        """
        if df is None or df.empty:
            print(f"No data available.")
            return None

        df = self.compute_indicators(df)

        # Define tasks for multiprocessing (pass the functions)
        tasks = [
            'macd',
            'rsi',
            'ma',
            'bollinger_bands',
            'vwap',
            'donchian_channel',
            'atr_breakout',
            'stochastic_oscillator',
            'ema_crossover'
        ]

        # Run all tasks concurrently using multiprocessing
        with multiprocessing.Pool() as pool:
            results = pool.map(self.run_strategy, [(getattr(self, task), df) for task in tasks])

        # Initialize buy and sell signal counters
        buy_signals = np.zeros(len(df), dtype=int)
        sell_signals = np.zeros(len(df), dtype=int)

        # Combine signals from all strategies
        for strategy_name, result_df in results:
            if result_df is not None:
                buy_signals += result_df['Buy_Signal'].astype(int).values
                sell_signals += result_df['Sell_Signal'].astype(int).values

        # Create final signals based on threshold
        final_buy_signal = buy_signals >= threshold
        final_sell_signal = sell_signals >= threshold

        # Create a DataFrame for combined signals
        combined_signals_df = pd.DataFrame({
            'Buy_Signal': final_buy_signal,
            'Sell_Signal': final_sell_signal
        }, index=df.index)

        return combined_signals_df

    def run_strategy(self, args):
        """
        Helper function to run a strategy in multiprocessing.
        """
        strategy_func, df = args
        strategy_name = strategy_func.__name__
        signals_df = strategy_func(df)
        return strategy_name, signals_df

    # Strategy functions using pandas_ta indicators

    def macd(self, df):
        buy_signals = (df['MACD_12_26_9'] > df['MACDs_12_26_9']) & (df['MACD_12_26_9'].shift(1) <= df['MACDs_12_26_9'].shift(1))
        sell_signals = (df['MACD_12_26_9'] < df['MACDs_12_26_9']) & (df['MACD_12_26_9'].shift(1) >= df['MACDs_12_26_9'].shift(1))
        signals_df = pd.DataFrame({'Buy_Signal': buy_signals, 'Sell_Signal': sell_signals}, index=df.index)
        return signals_df

    def rsi(self, df, period=14, Upper_Band=70, Lower_Band=30):
        buy_signals = (df[f'RSI_{period}'] < Lower_Band) & (df[f'RSI_{period}'].shift(1) >= Lower_Band)
        sell_signals = (df[f'RSI_{period}'] > Upper_Band) & (df[f'RSI_{period}'].shift(1) <= Upper_Band)
        signals_df = pd.DataFrame({'Buy_Signal': buy_signals, 'Sell_Signal': sell_signals}, index=df.index)
        return signals_df

    def ma(self, df):
        buy_signals = (df['SMA_20'] > df['SMA_150']) & (df['SMA_20'].shift(1) <= df['SMA_150'].shift(1))
        sell_signals = (df['SMA_20'] < df['SMA_150']) & (df['SMA_20'].shift(1) >= df['SMA_150'].shift(1))
        signals_df = pd.DataFrame({'Buy_Signal': buy_signals, 'Sell_Signal': sell_signals}, index=df.index)
        return signals_df

    def bollinger_bands(self, df):
        buy_signals = (df['Close'] < df['BBL_20_2.0']) & (df['Close'].shift(1) >= df['BBL_20_2.0'].shift(1))
        sell_signals = (df['Close'] > df['BBU_20_2.0']) & (df['Close'].shift(1) <= df['BBU_20_2.0'].shift(1))
        signals_df = pd.DataFrame({'Buy_Signal': buy_signals, 'Sell_Signal': sell_signals}, index=df.index)
        return signals_df

    def ema_crossover(self, df):
        buy_signals = (df['EMA_12'] > df['EMA_26']) & (df['EMA_12'].shift(1) <= df['EMA_26'].shift(1))
        sell_signals = (df['EMA_12'] < df['EMA_26']) & (df['EMA_12'].shift(1) >= df['EMA_26'].shift(1))
        signals_df = pd.DataFrame({'Buy_Signal': buy_signals, 'Sell_Signal': sell_signals}, index=df.index)
        return signals_df

    def stochastic_oscillator(self, df, overbought=80, oversold=20):
        buy_signals = (df['STOCHk_14_3_3'] < oversold) & (df['STOCHk_14_3_3'].shift(1) >= oversold)
        sell_signals = (df['STOCHk_14_3_3'] > overbought) & (df['STOCHk_14_3_3'].shift(1) <= overbought)
        signals_df = pd.DataFrame({'Buy_Signal': buy_signals, 'Sell_Signal': sell_signals}, index=df.index)
        return signals_df

    def atr_breakout(self, df, multiplier=1.5):
        df['Upper_Breakout'] = df['Close'] + (df['ATRr_14'] * multiplier)
        df['Lower_Breakout'] = df['Close'] - (df['ATRr_14'] * multiplier)
        buy_signals = df['Close'] > df['Upper_Breakout']
        sell_signals = df['Close'] < df['Lower_Breakout']
        signals_df = pd.DataFrame({'Buy_Signal': buy_signals, 'Sell_Signal': sell_signals}, index=df.index)
        return signals_df

    def donchian_channel(self, df):
        buy_signals = df['Close'] > df['DCH_20_0.0'].shift(1)
        sell_signals = df['Close'] < df['DCL_20_0.0'].shift(1)
        signals_df = pd.DataFrame({'Buy_Signal': buy_signals, 'Sell_Signal': sell_signals}, index=df.index)
        return signals_df

    def vwap(self, df):
        buy_signals = (df['Close'] > df['VWAP_D']) & (df['Close'].shift(1) <= df['VWAP_D'].shift(1))
        sell_signals = (df['Close'] < df['VWAP_D']) & (df['Close'].shift(1) >= df['VWAP_D'].shift(1))
        signals_df = pd.DataFrame({'Buy_Signal': buy_signals, 'Sell_Signal': sell_signals}, index=df.index)
        return signals_df

    # Additional strategy methods can be implemented similarly

    # Example of parameter optimization for RSI
    def optimize_rsi_parameters(self, df):
        best_profit = float('-inf')
        best_params = None

        periods = [10, 14, 20]
        lower_bands = [20, 30, 40]
        upper_bands = [60, 70, 80]

        for period, lower_band, upper_band in itertools.product(periods, lower_bands, upper_bands):
            df[f'RSI_{period}'] = df.ta.rsi(length=period)
            signals_df = self.generate_rsi_signals(df, period, lower_band, upper_band)
            performance, _ = self.backtest_strategy(df.copy(), lambda df: signals_df)
            if performance > best_profit:
                best_profit = performance
                best_params = (period, lower_band, upper_band)
        print(f"Best RSI Params: Period={best_params[0]}, Lower Band={best_params[1]}, Upper Band={best_params[2]}, Profit={best_profit}")

    def generate_rsi_signals(self, df, period, lower_band, upper_band):
        rsi_column = f'RSI_{period}'
        buy_signals = (df[rsi_column] < lower_band) & (df[rsi_column].shift(1) >= lower_band)
        sell_signals = (df[rsi_column] > upper_band) & (df[rsi_column].shift(1) <= upper_band)
        signals_df = pd.DataFrame({'Buy_Signal': buy_signals, 'Sell_Signal': sell_signals}, index=df.index)
        return signals_df

# Example usage:
if __name__ == "__main__":
    # Initialize strategy for a given symbol
    symbol = 'AAPL'  # Example symbol
    strategy = Strategy(symbol=symbol)

    # Fetch data
    try:
        df = yf.download(symbol, period='730d', interval='1h')
        df = strategy.compute_indicators(df)
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        df = None

    if df is not None:
        # Optimize RSI parameters
        strategy.optimize_rsi_parameters(df)

        # Get best strategy
        best_strategy, best_performance, best_risk_metrics = strategy.get_strategy_func()
        print(f"Best Strategy: {best_strategy}, Performance: {best_performance}, Risk Metrics: {best_risk_metrics}")

        # Plotting can be added as needed
