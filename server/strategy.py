import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import scraping  # Assuming this is your module for fetching stock data



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
        # Set attributes from kwargs dynamically
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Get the average stock price using scraping function
        self.avg_price = scraping.current_stock_price(self.symbol)

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
    


    def get_strategy_func(self, timeframe='1m', num_threads=5):
        """
        Evaluates multiple strategies concurrently using backtest_strategy and returns the one with the best performance.
        
        Args:
            timeframe (str): Timeframe for fetching stock data (e.g., '1d', '1h').
            num_threads (int): Number of threads to use for concurrent backtesting.
            
        Returns:
            tuple: The best strategy function, along with its performance and risk metrics.
        """

        def backtest_strategy_task(df, signals_df, strategy_func):
            try:
                # Call the existing backtest_strategy method for the current strategy
                performance, risk_metrics = self.backtest_strategy(df, signals_df, 
                                                                stop_loss_percent=self.loss_percent, 
                                                                stop_profit_percent=self.profit_percent)
                return strategy_func, performance, risk_metrics
            except Exception as e:
                print(f"Error in strategy {strategy_func}: {e}")
                return None

        best_strategy = None
        best_performance = float('-inf')  # Initialize with very low performance
        best_risk_metrics = None
        backtest_res = []
        
        # Fetch stock data for backtesting
        try:
            df = scraping.get_stock_data(self.symbol, period='max', interval=timeframe)
            df = df['DF']
            res = self.detect_signals_multithread(df)
        except Exception as e:
            print(f"Error fetching data for {self.symbol}: {e}")
            return None, None, None

# Run the strategy backtests concurrently using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(backtest_strategy_task, df, signals, strategy_name): signals for strategy_name, signals in res.items()}

            # Iterate over completed futures and get results
            for future in as_completed(futures):
                result = future.result()
                if result:
                    strategy_func, performance, risk_metrics = result

                    if performance is None:
                        continue

                    backtest_res.append({'strategy_func': strategy_func, 'performance': performance, 'risk_metrics': risk_metrics})

                    # Check if this strategy has the best performance
                    if performance > best_performance:
                        best_performance = performance
                        best_strategy = strategy_func
                        best_risk_metrics = risk_metrics
                    # print(f"Strategy: {strategy_func}, Performance: {performance}, \n Risk Metrics: {risk_metrics}")
        
        return best_strategy, backtest_res

        


    def backtest_strategy(self,df , signals_df, transaction_cost=0, tax_on_profit=0.25, stop_loss_percent=None, stop_profit_percent=None):
        """
        Backtests a trading strategy based on buy and sell signals, with optional moving stop-loss and stop-profit logic.
        
        Args:
            df (pd.DataFrame): The historical price data.
            strategy_func (function): The trading strategy function that returns buy/sell signals.
            my_sector (str): The sector to allocate trades.
            transaction_cost (float): The transaction cost per trade.
            tax_on_profit (float): The tax on profit.
            stop_loss_percent (float, optional): The percentage for a moving stop-loss trigger. If None, no stop-loss is applied.
            stop_profit_percent (float, optional): The percentage for stop-profit trigger. If None, no stop-profit is applied.
            
        Returns:
            tuple: The performance and risk metrics of the strategy.
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
        highest_price = 0  # Track the highest price since buying for trailing stop-loss
        
        # Get the buy and sell signals from the strategy function
       
        # Iterate over each row in the DataFrame
        for index, row in signals_df.iterrows():
            current_price = df.loc[index]['Close']

            # Buy logic
            if row['Buy_Signal'] and cash > 0:
                position = cash / current_price  # Buy as many shares as possible
                entry_price = current_price  # Set entry price
                cash = 0  # All cash used
                total_trades += 1
                highest_price = current_price  # Start tracking the highest price for trailing stop-loss
                

                # Set stop-loss and stop-profit prices only if the values are provided
                if stop_loss_percent is not None:
                    stop_loss_price = entry_price * (1 - stop_loss_percent / 100)  # Initial stop-loss price
                else:
                    stop_loss_price = None

                if stop_profit_percent is not None:
                    stop_profit_price = entry_price * (1 + stop_profit_percent / 100)  # Initial stop-profit price
                else:
                    stop_profit_price = None

            # Sell logic based on moving stop-loss, stop-profit, or sell signal
            elif position > 0:
                # Update highest price reached if the current price is higher
                if current_price > highest_price:
                    highest_price = current_price

                    # Update the stop-loss price based on the new highest price (moving stop-loss)
                    if stop_loss_percent is not None:
                        stop_loss_price = highest_price * (1 - stop_loss_percent / 100)

                # Check stop-loss logic (if provided)
                if stop_loss_price is not None and current_price <= stop_loss_price:
                    sell_value = position * current_price
                    cash = sell_value * (1 - transaction_cost)  # Deduct transaction cost
                    position = 0  # Exit position
                    
                
                # Check stop-profit logic (if provided)
                elif stop_profit_price is not None and current_price >= stop_profit_price:
                    sell_value = position * current_price
                    cash = sell_value * (1 - transaction_cost)  # Deduct transaction cost
                    position = 0  # Exit position
                    
                
                # Regular sell logic based on sell signal
                elif row['Sell_Signal']:
                    sell_value = position * current_price
                    cash = sell_value * (1 - transaction_cost)  # Deduct transaction cost
                    profit = cash - starting_cash
                    cash -= profit * tax_on_profit  # Apply tax on profit
                    position = 0  # Exit position
                    
                    # Count winning trades
                    if cash >= entry_price * position:  # If we made a profit
                        winning_trades += 1

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
        performance = final_cash - starting_cash  # Total performance (profit/loss)
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

   


    def detect_signals_multithread(self, df, threshold=3):
        """
        Detects buy and sell signals using multiple trading strategies with multithreading.
        Works with both live stock data and historical stock data.
        
        Args:
            df (pd.DataFrame): The stock data for which signals should be detected.
            threshold (int): Minimum number of strategies that must agree to generate a final buy/sell signal.
            
        Returns:
            pd.DataFrame: A DataFrame containing the combined buy and sell signals.
        """
        if df is None or df.empty:
            print(f"No data available.")
            return None

        # Define tasks for multithreading (pass the functions, not the result of calling them)
        tasks = [
            ('macd', self.macd),
            ('rsi', self.rsi),
            ('ma', self.ma),
            ('bollinger_bands', self.bollinger_bands),
            ('vwap', self.vwap),
            ('ichimoku_cloud', self.ichimoku_cloud),
            ('donchian_channel', self.donchian_channel),
            ('atr_breakout', self.atr_breakout),
            ('parabolic_sar', self.parabolic_sar),
            ('stochastic_oscillator', self.stochastic_oscillator),
            ('ema_crossover', self.ema_crossover)
        ]

        # Run all tasks concurrently using multithreading
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(task[1], df): task[0] for task in tasks}
            results = {}

            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    results[task_name] = future.result()
                except Exception as e:
                    print(f"Error in strategy {task_name}: {e}")

        # Initialize buy and sell signal counters
        buy_signals = pd.Series(0, index=df.index)
        sell_signals = pd.Series(0, index=df.index)

        # Combine signals from all strategies
        for strategy_name, result_df in results.items():
            if result_df is not None:
                buy_signals += result_df['Buy_Signal'].astype(int)
                sell_signals += result_df['Sell_Signal'].astype(int)

        # Create final signals based on threshold
        final_buy_signal = buy_signals >= threshold
        final_sell_signal = sell_signals >= threshold

        # Create a DataFrame for combined signals
        combined_signals_df = pd.DataFrame({
            'Buy_Signal': final_buy_signal,
            'Sell_Signal': final_sell_signal
        }, index=df.index)

        results['combined'] = combined_signals_df

        return results


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

        # signals_df = signals_df[(signals_df['Buy_Signal']) | (signals_df['Sell_Signal'])]

        return signals_df

        
    def rsi(self, df, Upper_Band=60, Lower_Band=30):
        """
        Calculates the Relative Strength Index (RSI) signals.
        
        Args:
            df (pd.DataFrame): The historical price data.
            period (int): The period for calculating RSI.
            
        Returns:
            tuple: The buy and sell signals.
        """
        buy_signals = (df['RSI'] < Lower_Band) & (df['RSI'].shift(1) >= Lower_Band)
        sell_signals = (df['RSI'] > Upper_Band) & (df['RSI'].shift(1) <= Upper_Band)
        signals_df = pd.DataFrame({
            'Buy_Signal': buy_signals,
            'Sell_Signal': sell_signals
        }, index=df.index)
        # signals_df = signals_df[(signals_df['Buy_Signal']) | (signals_df['Sell_Signal'])]
        
        return signals_df

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
        # signals_df = signals_df[(signals_df['Buy_Signal']) | (signals_df['Sell_Signal'])]

        return signals_df

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
        # signals_df = signals_df[(signals_df['Buy_Signal']) | (signals_df['Sell_Signal'])]
        
        return signals_df

    def ema_crossover(self, df, short_window=12, long_window=26):
        """
        Calculates the Exponential Moving Average Crossover signals.
        
        Args:
            df (pd.DataFrame): The historical price data.
            short_window (int): The window period for the short-term EMA.
            long_window (int): The window period for the long-term EMA.
            
        Returns:
            pd.DataFrame: The buy and sell signals.
        """
        
        # Calculate short-term and long-term EMAs
        df['EMA_Short'] = df['Close'].ewm(span=short_window, adjust=False).mean()
        df['EMA_Long'] = df['Close'].ewm(span=long_window, adjust=False).mean()

        # Generate buy and sell signals
        buy_signals = (df['EMA_Short'] > df['EMA_Long']) & (df['EMA_Short'].shift(1) <= df['EMA_Long'].shift(1))
        sell_signals = (df['EMA_Short'] < df['EMA_Long']) & (df['EMA_Short'].shift(1) >= df['EMA_Long'].shift(1))

        signals_df = pd.DataFrame({
            'Buy_Signal': buy_signals,
            'Sell_Signal': sell_signals
        }, index=df.index)

        return signals_df

    def stochastic_oscillator(self, df, k_window=14, d_window=3, overbought=80, oversold=20):
        """
        Calculates the Stochastic Oscillator signals.
        
        Args:
            df (pd.DataFrame): The historical price data.
            k_window (int): The window period for %K calculation.
            d_window (int): The window period for %D calculation (signal line).
            overbought (int): The overbought threshold for sell signals.
            oversold (int): The oversold threshold for buy signals.
            
        Returns:
            pd.DataFrame: The buy and sell signals.
        """
        
        # Calculate %K (stochastic)
        df['Low_K'] = df['Low'].rolling(window=k_window).min()
        df['High_K'] = df['High'].rolling(window=k_window).max()
        df['%K'] = (df['Close'] - df['Low_K']) / (df['High_K'] - df['Low_K']) * 100

        # Calculate %D (signal line)
        df['%D'] = df['%K'].rolling(window=d_window).mean()

        # Generate buy and sell signals
        buy_signals = (df['%K'] < oversold) & (df['%K'].shift(1) >= oversold)
        sell_signals = (df['%K'] > overbought) & (df['%K'].shift(1) <= overbought)

        signals_df = pd.DataFrame({
            'Buy_Signal': buy_signals,
            'Sell_Signal': sell_signals
        }, index=df.index)

        return signals_df



    def parabolic_sar(self, df, step=0.02, max_step=0.2):
        """
        Calculates the Parabolic SAR signals using Pandas and NumPy for more efficient computation.
        
        Args:
            df (pd.DataFrame): The historical price data with 'High', 'Low', and 'Close' columns.
            step (float): The acceleration factor step.
            max_step (float): The maximum acceleration factor.
            
        Returns:
            pd.DataFrame: The buy and sell signals.
        """
        # Initialize arrays
        psar = np.zeros(len(df))  # Parabolic SAR array
        trend = np.zeros(len(df))  # Track the trend: 1 for Up, -1 for Down
        af = step  # Initial acceleration factor
        ep = df['High'].iloc[0]  # Starting extreme point (for Uptrend)
        
        # Start with an Uptrend
        psar[0] = df['Low'].iloc[0]
        trend[0] = 1  # 1 for Uptrend
        
        buy_signals = np.zeros(len(df), dtype=bool)
        sell_signals = np.zeros(len(df), dtype=bool)

        for i in range(1, len(df)):
            # Update PSAR based on current trend
            if trend[i - 1] == 1:  # Uptrend
                psar[i] = psar[i - 1] + af * (ep - psar[i - 1])
                if df['Low'].iloc[i] < psar[i]:  # Reversal to downtrend
                    trend[i] = -1  # Switch to downtrend
                    psar[i] = ep  # Set SAR to previous extreme point
                    ep = df['Low'].iloc[i]  # Reset extreme point
                    af = step  # Reset acceleration factor
                    sell_signals[i] = True  # Mark sell signal
                else:
                    if df['High'].iloc[i] > ep:  # New extreme point
                        ep = df['High'].iloc[i]
                        af = min(af + step, max_step)  # Increase acceleration factor
            else:  # Downtrend
                psar[i] = psar[i - 1] + af * (ep - psar[i - 1])
                if df['High'].iloc[i] > psar[i]:  # Reversal to uptrend
                    trend[i] = 1  # Switch to uptrend
                    psar[i] = ep  # Set SAR to previous extreme point
                    ep = df['High'].iloc[i]  # Reset extreme point
                    af = step  # Reset acceleration factor
                    buy_signals[i] = True  # Mark buy signal
                else:
                    if df['Low'].iloc[i] < ep:  # New extreme point
                        ep = df['Low'].iloc[i]
                        af = min(af + step, max_step)  # Increase acceleration factor

        # Create the result DataFrame
        signals_df = pd.DataFrame({
            'Buy_Signal': buy_signals,
            'Sell_Signal': sell_signals
        }, index=df.index)

        return signals_df

    def atr_breakout(self, df, window=14, multiplier=1.5):
        """
        Calculates the ATR breakout signals.
        
        Args:
            df (pd.DataFrame): The historical price data.
            window (int): The window period for ATR calculation.
            multiplier (float): The multiplier for breakout range.
            
        Returns:
            pd.DataFrame: The buy and sell signals.
        """
        # Calculate True Range (TR)
        df['High_Low'] = df['High'] - df['Low']
        df['High_Close'] = (df['High'] - df['Close'].shift(1)).abs()
        df['Low_Close'] = (df['Low'] - df['Close'].shift(1)).abs()
        df['TR'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)

        # Calculate Average True Range (ATR)
        df['ATR'] = df['TR'].rolling(window=window).mean()

        # Calculate breakout levels
        df['Upper_Breakout'] = df['Close'] + (df['ATR'] * multiplier)
        df['Lower_Breakout'] = df['Close'] - (df['ATR'] * multiplier)

        # Generate buy and sell signals
        buy_signals = (df['Close'] > df['Upper_Breakout'])
        sell_signals = (df['Close'] < df['Lower_Breakout'])

        signals_df = pd.DataFrame({
            'Buy_Signal': buy_signals,
            'Sell_Signal': sell_signals
        }, index=df.index)

        return signals_df

    def donchian_channel(self, df, window=20):
        """
        Calculates the Donchian Channel breakout signals.
        
        Args:
            df (pd.DataFrame): The historical price data.
            window (int): The window period for Donchian Channel calculation.
            
        Returns:
            pd.DataFrame: The buy and sell signals.
        """
        # Calculate Donchian Channel
        df['Donchian_High'] = df['High'].rolling(window=window).max()
        df['Donchian_Low'] = df['Low'].rolling(window=window).min()

        # Generate buy and sell signals
        buy_signals = (df['Close'] > df['Donchian_High'].shift(1))
        sell_signals = (df['Close'] < df['Donchian_Low'].shift(1))

        signals_df = pd.DataFrame({
            'Buy_Signal': buy_signals,
            'Sell_Signal': sell_signals
        }, index=df.index)

        return signals_df

    def ichimoku_cloud(self, df, conversion_window=9, base_window=26, leading_span_window=52):
        """
        Calculates the Ichimoku Cloud signals.
        
        Args:
            df (pd.DataFrame): The historical price data.
            conversion_window (int): The window period for the conversion line.
            base_window (int): The window period for the base line.
            leading_span_window (int): The window period for the leading span A.
            
        Returns:
            pd.DataFrame: The buy and sell signals.
        """
        # Calculate Ichimoku components
        df['Conversion_Line'] = (df['High'].rolling(window=conversion_window).max() + df['Low'].rolling(window=conversion_window).min()) / 2
        df['Base_Line'] = (df['High'].rolling(window=base_window).max() + df['Low'].rolling(window=base_window).min()) / 2
        df['Leading_Span_A'] = ((df['Conversion_Line'] + df['Base_Line']) / 2).shift(base_window)
        df['Leading_Span_B'] = ((df['High'].rolling(window=leading_span_window).max() + df['Low'].rolling(window=leading_span_window).min()) / 2).shift(base_window)

        # Buy signals when price crosses above the cloud
        buy_signals = (df['Close'] > df['Leading_Span_A']) & (df['Close'] > df['Leading_Span_B']) & (df['Close'].shift(1) <= df['Leading_Span_A'])

        # Sell signals when price crosses below the cloud
        sell_signals = (df['Close'] < df['Leading_Span_A']) & (df['Close'] < df['Leading_Span_B']) & (df['Close'].shift(1) >= df['Leading_Span_A'])

        signals_df = pd.DataFrame({
            'Buy_Signal': buy_signals,
            'Sell_Signal': sell_signals
        }, index=df.index)

        return signals_df

    def vwap(self, df):
        """
        Calculates VWAP (Volume Weighted Average Price) buy and sell signals.
        
        Args:
            df (pd.DataFrame): The historical price data with volume information.
            
        Returns:
            pd.DataFrame: The buy and sell signals.
        """
        # Calculate VWAP
        df['Cumulative_Price_Vol'] = (df['Close'] * df['Volume']).cumsum()
        df['Cumulative_Vol'] = df['Volume'].cumsum()
        df['VWAP'] = df['Cumulative_Price_Vol'] / df['Cumulative_Vol']

        # Buy and sell signals
        buy_signals = (df['Close'] > df['VWAP']) & (df['Close'].shift(1) <= df['VWAP'].shift(1))
        sell_signals = (df['Close'] < df['VWAP']) & (df['Close'].shift(1) >= df['VWAP'].shift(1))

        signals_df = pd.DataFrame({
            'Buy_Signal': buy_signals,
            'Sell_Signal': sell_signals
        }, index=df.index)

        return signals_df
