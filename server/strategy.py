import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import scraping  # Assuming this is your module for fetching stock data
import plots
import polars as pl




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
    


    def get_strategy_func(self, df: pl.DataFrame, timeframe, num_threads=5):
        """
        Evaluates multiple strategies concurrently using backtest_strategy and returns the one with the best performance.
        
        Args:
            timeframe (str): Timeframe for fetching stock data (e.g., '1d', '1h').
            num_threads (int): Number of threads to use for concurrent backtesting.
            
        Returns:
            tuple: The best strategy function, along with its performance and risk metrics.
        """

        def backtest_strategy_task(df: pl.DataFrame, signals_df: pl.DataFrame, strategy_func: str):
            try:
                # Call the existing backtest_strategy method for the current strategy
                performance, risk_metrics = self.backtest_strategy(df, signals_df, 
                                                                stop_loss_percent=self.loss_percent, 
                                                                stop_profit_percent=self.profit_percent)
                return strategy_func, performance, risk_metrics, signals_df
            except Exception as e:
                print(f"Error in strategy {strategy_func}: {e}")
                return None

        best_strategy = None
        best_performance = float('-inf')  # Initialize with very low performance
        best_risk_metrics = None
        backtest_res = []
      

        try:
            res = self.detect_signals_multithread(df)

        except Exception as e:
            res = None
            print(f'Error {e}')

# Run the strategy backtests concurrently using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(backtest_strategy_task, df, signals, strategy_name): signals for strategy_name, signals in res.items()}

            # Iterate over completed futures and get results
            for future in as_completed(futures):
                result = future.result()
                if result:
                    strategy_func, performance, risk_metrics, signals = result

                    if performance is None:
                        continue
                    
                    # fig = plots.plot_stock(df, self.symbol, df.columns, signals=signals, show='no', interval=timeframe)
                    backtest_res.append({'strategy_func': strategy_func, 'performance': performance, 'risk_metrics': risk_metrics, 'signals': signals})

                    # Check if this strategy has the best performance
                    if performance > best_performance:
                        best_performance = performance
                        best_strategy = strategy_func
                        best_risk_metrics = risk_metrics
                    # print(f"Strategy: {strategy_func}, Performance: {performance}, \n Risk Metrics: {risk_metrics}")
        
        return best_strategy, backtest_res

        


    def backtest_strategy(self, df: pl.DataFrame, signals_df: pl.DataFrame, transaction_cost: float = 0, tax_on_profit: float = 0.25, stop_loss_percent=None, stop_profit_percent=None) -> tuple:
        """
        Backtests a trading strategy based on buy and sell signals, with optional moving stop-loss and stop-profit logic.

        Args:
            df (pl.DataFrame): The historical price data.
            signals_df (pl.DataFrame): The buy and sell signals DataFrame.
            transaction_cost (float): The transaction cost per trade.
            tax_on_profit (float): The tax on profit.
            stop_loss_percent (float, optional): The percentage for a moving stop-loss trigger. If None, no stop-loss is applied.
            stop_profit_percent (float, optional): The percentage for stop-profit trigger. If None, no stop-profit is applied.

        Returns:
            tuple: The performance and risk metrics of the strategy.
        """
        # Initialize variables
        cash = 100000.0  # Initial capital
        starting_cash = cash
        position = 0.0  # Number of shares held
        entry_price = 0.0  # Price at which we entered the position
        max_drawdown = 0.0
        peak_value = cash
        total_trades = 0
        winning_trades = 0
        highest_price = 0.0  # Track the highest price since buying for trailing stop-loss

        # Convert Polars DataFrame to list of dictionaries for efficient iteration
        df_records = df.select(['Close', 'DateTime']).to_dict(as_series=False)
        signals_records = signals_df.select(['Buy_Signal', 'Sell_Signal', 'DateTime']).to_dict(as_series=False)

        # Iterate over each row
        for i in range(len(df_records)):
            current_price = df_records['Close'][i]
            buy_signal = signals_records['Buy_Signal'][i]
            sell_signal = signals_records['Sell_Signal'][i]

            # Buy logic
            if buy_signal and cash > 0:
                position = cash / current_price  # Buy as many shares as possible
                entry_price = current_price  # Set entry price
                cash = 0.0  # All cash used
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
                    position = 0.0  # Exit position
                    total_trades += 1
                    # Determine if the trade was profitable
                    if current_price > entry_price:
                        winning_trades += 1

                # Check stop-profit logic (if provided)
                elif stop_profit_price is not None and current_price >= stop_profit_price:
                    sell_value = position * current_price
                    cash = sell_value * (1 - transaction_cost)  # Deduct transaction cost
                    position = 0.0  # Exit position
                    total_trades += 1
                    # Determine if the trade was profitable
                    if current_price > entry_price:
                        winning_trades += 1

                # Regular sell logic based on sell signal
                elif sell_signal:
                    sell_value = position * current_price
                    cash = sell_value * (1 - transaction_cost)  # Deduct transaction cost
                    position = 0.0  # Exit position
                    total_trades += 1
                    # Determine if the trade was profitable
                    if current_price > entry_price:
                        winning_trades += 1

                # Update drawdown and peak value
                if cash > peak_value:
                    peak_value = cash
                drawdown = (peak_value - cash) / peak_value
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

        # Final calculations
        final_cash = cash + (position * df_records['Close'][-1] if position > 0 else 0.0)  # Cash value at end
        win_rate = 0.0 if (winning_trades == 0 or total_trades == 0) else (winning_trades / total_trades) * 100  # Percentage
        performance = final_cash - starting_cash  # Total profit/loss
        timeframe_days = (df_records['DateTime'][-1] - df_records['DateTime'][0]).days
        roi = ((final_cash - starting_cash) / starting_cash) * 100  # Return on investment

        # Risk metrics to return
        risk_metrics = {
            'max_drawdown': round(max_drawdown * 100, 2),  # Percentage
            'win_rate': round(win_rate, 2),                # Percentage
            'time_frame_days': timeframe_days,
            'roi': round(roi, 2)                           # Percentage
        }

        return performance, risk_metrics



    def calculate_risk_score(self):
        """
        Calculates a composite risk score based on various financial metrics.

        Returns:
            float: The overall risk score scaled to 0-100.
        """
        weights = {
            'debt_to_equity': 0.25,
            'beta': 0.2,
            'profit_margins': 0.15,
            'revenue_growth': 0.15,
            'free_cashflow': 0.25
        }

        risk_score = {}

        # Debt-to-equity risk (normalized to 0-1 range)
        if hasattr(self, 'debtToEquity'):
            if self.debtToEquity > 100:
                risk_score['debt_to_equity_risk'] = min(self.debtToEquity / 1000, 1) * weights['debt_to_equity']
            else:
                risk_score['debt_to_equity_risk'] = 0
        else:
            risk_score['debt_to_equity_risk'] = weights['debt_to_equity']  # Default high risk if data missing

        # Beta risk (normalized to 0-1 range)
        if hasattr(self, 'beta'):
            risk_score['beta_risk'] = min(self.beta / 2, 1) * weights['beta']
        else:
            risk_score['beta_risk'] = weights['beta']  # Default high risk if data missing

        # Profit margin risk (lower profit margins = more risk)
        if hasattr(self, 'profitMargins'):
            if self.profitMargins < 0:
                risk_score['profit_margin_risk'] = abs(self.profitMargins) * weights['profit_margins']  # Negative margins = high risk
            else:
                risk_score['profit_margin_risk'] = max(0.1 - self.profitMargins, 0) * weights['profit_margins']  # Margins below 10% add risk
        else:
            risk_score['profit_margin_risk'] = weights['profit_margins']  # Default high risk if data missing

        # Revenue growth risk (negative or low growth = higher risk)
        if hasattr(self, 'revenueGrowth'):
            if self.revenueGrowth < 0:
                risk_score['revenue_growth_risk'] = abs(self.revenueGrowth) * weights['revenue_growth']
            else:
                risk_score['revenue_growth_risk'] = max(0.05 - self.revenueGrowth, 0) * weights['revenue_growth']
        else:
            risk_score['revenue_growth_risk'] = weights['revenue_growth']  # Default high risk if data missing

        # Free cash flow risk (low cash flow adds risk)
        if hasattr(self, 'ebitda') and hasattr(self, 'freeCashflow'):
            if self.ebitda != 0:
                free_cashflow_to_ebitda_ratio = self.freeCashflow / self.ebitda
            else:
                free_cashflow_to_ebitda_ratio = 0
            if free_cashflow_to_ebitda_ratio < 0.1:
                risk_score['free_cashflow_risk'] = (0.1 - free_cashflow_to_ebitda_ratio) * weights['free_cashflow']
            else:
                risk_score['free_cashflow_risk'] = 0
        else:
            risk_score['free_cashflow_risk'] = weights['free_cashflow']  # Default high risk if data missing

        # Calculate overall risk score and scale to 0-100
        overall_risk_score = sum(risk_score.values()) * 100  # Scale to 0-100

        risk_score['overall_risk_score'] = overall_risk_score

        return risk_score['overall_risk_score']

   


    def detect_signals_multithread(self, df: pd.DataFrame, threshold=2) -> dict:
        """
        Detects buy and sell signals using multiple trading strategies with multithreading.
        Works with both live stock data and historical stock data.

        Args:
            df (pl.DataFrame): The stock data for which signals should be detected.
            threshold (int): Minimum number of strategies that must agree to generate a final buy/sell signal.

        Returns:
            dict: A dictionary of strategy names and their respective signal DataFrames.
        """

        def combine(results):
            # Combine signals from all strategies
            # Initialize buy and sell signal counters
            buy_signals = pl.Series("Buy_Signal", [False] * len(df))
            sell_signals = pl.Series("Sell_Signal", [False] * len(df))

            for strategy_name, result_df in results.items():
                if result_df is not None:
                    # Align signals with the main DataFrame
                    result_df = result_df.join(df.select(['DateTime']), on='DateTime', how='left').fill_null(False)
                    
                    # Aggregate signals
                    buy_signals = buy_signals + result_df['Buy_Signal'].cast(pl.Int32)
                    sell_signals = sell_signals + result_df['Sell_Signal'].cast(pl.Int32)

            # Create final signals based on threshold
            final_buy_signal = buy_signals >= threshold
            final_sell_signal = sell_signals >= threshold

            # Create a DataFrame for combined signals
            combined_signals_df = pl.DataFrame({
                'Buy_Signal': final_buy_signal,
                'Sell_Signal': final_sell_signal,
                'DateTime': df['DateTime']
            })

            return combined_signals_df

        # convert df to polars
        df = pl.from_pandas(df, include_index=True)
       

        if df is None or df.is_empty():
            print("No data available.")
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

        # Run all tasks concurrently using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(task[1], df): task[0] for task in tasks}
            results = {}

            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    results[task_name] = future.result()
                except Exception as e:
                    print(f"Error in strategy {task_name}: {e}")

        return results



    def macd(self, df: pl.DataFrame) -> pl.DataFrame:

        required_columns = ['MACD', 'MACD_Signal']
        if not all(col in df.columns for col in required_columns):
            print("Required columns for MACD strategy are missing.")
            return None

        try:
            buy_signals = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))

            # Sell signal when MACD crosses below the signal line
            sell_signals = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
        except Exception as e:
            return None
    
        
        return generate_signal(sell_signals.to_list(), buy_signals.to_list(), df['Datetime'].to_list())

        
    def rsi(self, df: pl.DataFrame, Upper_Band=60, Lower_Band=30) -> pl.DataFrame:
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

        return generate_signal(sell_signals.to_list(), buy_signals.to_list(), df['Datetime'].to_list())
        
    def ma(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculates the Moving Average (MA) signals.
        
        Args:
            df (pd.DataFrame): The historical price data.
            
        Returns:
            tuple: The buy and sell signals.
        """
        buy_signals = (df['SMA20'] > df['SMA150']) & (df['SMA20'].shift(1) <= df['SMA150'].shift(1))
        sell_signals = (df['SMA20'] < df['SMA150']) & (df['SMA20'].shift(1) >= df['SMA150'].shift(1))
       
        return generate_signal(sell_signals.to_list(), buy_signals.to_list(), df['Datetime'].to_list())

    def bollinger_bands(self, df: pl.DataFrame, window: int = 20, num_std_dev=2) -> pl.DataFrame:
        """
        Calculates the Bollinger Bands signals.
        """

        # Calculate Bollinger Bands
        df = df.select(pl.col('Close').rolling_std(window).alias('STD20'))

        # Step 2: Drop rows where 'SMA20' or 'STD20' have NaN values
        df = df.filter(
            pl.col('SMA20').is_not_null() & pl.col('STD20').is_not_null()
        )

        # Step 3: Create 'Upper_Band' and 'Lower_Band'
        df = df.select([
            (pl.col('SMA20') + (pl.col('STD20') * num_std_dev)).alias('Upper_Band'),
            (pl.col('SMA20') - (pl.col('STD20') * num_std_dev)).alias('Lower_Band')
        ])

        # Step 4: Generate buy and sell signals
        buy_signals = (
            (pl.col('Close') < pl.col('Lower_Band')) &
            (pl.col('Close').shift(1) >= pl.col('Lower_Band').shift(1))
        )

        sell_signals = (
            (pl.col('Close') > pl.col('Upper_Band')) &
            (pl.col('Close').shift(1) <= pl.col('Upper_Band').shift(1))
        )

        return generate_signal(sell_signals.to_list(), buy_signals.to_list(), df['Datetime'].to_list())


    def ema_crossover(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculates the Exponential Moving Average Crossover signals.

        Args:
            df (pd.DataFrame): The historical price data.

        Returns:
            pd.DataFrame: The buy and sell signals.
        """
        df.dropna(subset=['EMA12', 'EMA26'], inplace=True)

        # Generate buy and sell signals
        buy_signals = (df['EMA12'] > df['EMA26']) & (df['EMA12'].shift(1) <= df['EMA26'].shift(1))
        sell_signals = (df['EMA12'] < df['EMA26']) & (df['EMA12'].shift(1) >= df['EMA26'].shift(1))

        return generate_signal(sell_signals.to_list(), buy_signals.to_list(), df['Datetime'].to_list())

    def stochastic_oscillator(self, df: pl.DataFrame, k_window=14, d_window=3, overbought=80, oversold=20) -> pl.DataFrame:
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
        # Parameter validation
        if not isinstance(k_window, int) or k_window <= 0:
            raise ValueError(f"k_window must be a positive integer. Received k_window={k_window}")
        if not isinstance(d_window, int) or d_window <= 0:
            raise ValueError(f"d_window must be a positive integer. Received d_window={d_window}")

        # Ensure required columns exist
        required_columns = ['High', 'Low', 'Close']
        if not all(col in df.columns for col in required_columns):
            print(f"DataFrame must contain {required_columns} columns.")
            return pl.DataFrame(columns=['Buy_Signal', 'Sell_Signal'], index=df['Datetime'])

        # Ensure sufficient data length
        required_length = max(k_window, d_window)
        if len(df) < required_length:
            print(f"Insufficient data for Stochastic Oscillator calculation. Required: {required_length}, Available: {len(df)}")
            return pl.DataFrame(columns=['Buy_Signal', 'Sell_Signal'], index=df['Datetime'])

        # Sort the DataFrame by index to ensure chronological order
        df.sort_index(inplace=True)

        # Calculate %K (stochastic)
        df['Lowest_Low'] = df['Low'].rolling(window=k_window, min_periods=1).min().astype('float32')
        df['Highest_High'] = df['High'].rolling(window=k_window, min_periods=1).max().astype('float32')
        df['Denominator'] = (df['Highest_High'] - df['Lowest_Low']).astype('float32')

        # Avoid division by zero
        df['Denominator'].replace(0, np.nan, inplace=True)
        df['%K'] = (((df['Close'] - df['Lowest_Low']) / df['Denominator']) * 100).astype('float32')
        df['%K'].fillna(0, inplace=True)  # Handle NaN values resulting from division by zero

        # Calculate %D (signal line)
        df['%D'] = df['%K'].rolling(window=d_window, min_periods=1).mean().astype('float32')

        # Generate buy and sell signals
        buy_signals = (df['%K'] < oversold) & (df['%K'].shift(1) >= oversold)
        sell_signals = (df['%K'] > overbought) & (df['%K'].shift(1) <= overbought)

        # Clean up temporary columns
        df.drop(columns=['Lowest_Low', 'Highest_High', 'Denominator', '%K', '%D'], inplace=True)

        return generate_signal(sell_signals.to_list(), buy_signals.to_list(), df['Datetime'].to_list())



    def parabolic_sar(self, df: pl.DataFrame, step=0.02, max_step=0.2) -> pl.DataFrame:
        """
        Calculates the Parabolic SAR signals.

        Args:
            df (pl.DataFrame): The historical price data.
            step (float): The acceleration factor step.
            max_step (float): The maximum acceleration factor.

        Returns:
            pl.DataFrame: The buy and sell signals.
        """
        # Ensure required columns exist
        required_columns = ['High', 'Low', 'Close']
        if not all(col in df.columns for col in required_columns):
            print(f"DataFrame must contain {required_columns} columns for Parabolic SAR calculation.")
            return pl.DataFrame(columns=['Buy_Signal', 'Sell_Signal'], schema=[('Buy_Signal', pl.Boolean), ('Sell_Signal', pl.Boolean), ('DateTime', pl.Datetime)])

        # Initialize variables
        sar = [0.0] * len(df)
        trend = [1] * len(df)  # 1 for uptrend, -1 for downtrend
        ep = df['High'][0]
        af = step

        buy_signals = [False] * len(df)
        sell_signals = [False] * len(df)

        sar[0] = df['Low'][0]

        for i in range(1, len(df)):
            if trend[i-1] == 1:
                sar[i] = sar[i-1] + af * (ep - sar[i-1])
                if df['Low'][i] < sar[i]:
                    trend[i] = -1
                    sar[i] = ep
                    ep = df['Low'][i]
                    af = step
                    sell_signals[i] = True
                else:
                    trend[i] = 1
                    if df['High'][i] > ep:
                        ep = df['High'][i]
                        af = min(af + step, max_step)
            else:
                sar[i] = sar[i-1] + af * (ep - sar[i-1])
                if df['High'][i] > sar[i]:
                    trend[i] = 1
                    sar[i] = ep
                    ep = df['High'][i]
                    af = step
                    buy_signals[i] = True
                else:
                    trend[i] = -1
                    if df['Low'][i] < ep:
                        ep = df['Low'][i]
                        af = min(af + step, max_step)

        # Assign SAR values to DataFrame
        df = df.with_columns([
            pl.Series('SAR', sar).cast(pl.Float32),
            pl.Series('Trend', trend).cast(pl.Int32)
        ])

        # Generate buy and sell signals
        

        return generate_signal(sell_signals.to_list(), buy_signals.to_list(), df['Datetime'].to_list())

    def atr_breakout(self, df: pl.DataFrame, window=14, multiplier=1.2) -> pl.DataFrame:
        """
        Calculates the ATR breakout signals.

        Args:
            df (pl.DataFrame): The historical price data.
            window (int): The window period for ATR calculation.
            multiplier (float): The multiplier for breakout range.

        Returns:
            pl.DataFrame: The buy and sell signals.
        """
        required_columns = ['Close', 'High', 'Low', 'ATR']
        if not all(col in df.columns for col in required_columns):
            print(f"DataFrame must contain {required_columns} columns for ATR Breakout calculation.")
            return pl.DataFrame(columns=['Buy_Signal', 'Sell_Signal'], schema=[('Buy_Signal', pl.Boolean), ('Sell_Signal', pl.Boolean), ('DateTime', pl.Datetime)])

        # Calculate breakout levels
        df = df.with_columns([
            (df['Close'] + (df['ATR'] * multiplier)).alias('Upper_Breakout'),
            (df['Close'] - (df['ATR'] * multiplier)).alias('Lower_Breakout')
        ])

        # Generate buy and sell signals
        buy_signals = (df['Close'] > df['Upper_Breakout'].shift(1))
        sell_signals = (df['Close'] < df['Lower_Breakout'].shift(1))


        return generate_signal(sell_signals.to_list(), buy_signals.to_list(), df['Datetime'].to_list())

    def donchian_channel(self, df: pl.DataFrame, window=20) -> pl.DataFrame:
        """
        Calculates the Donchian Channel breakout signals.

        Args:
            df (pl.DataFrame): The historical price data.
            window (int): The window period for Donchian Channel calculation.

        Returns:
            pl.DataFrame: The buy and sell signals.
        """
        required_columns = ['High', 'Low', 'Close']
        if not all(col in df.columns for col in required_columns):
            print(f"DataFrame must contain {required_columns} columns.")
            return pl.DataFrame(columns=['Buy_Signal', 'Sell_Signal'], schema=[('Buy_Signal', pl.Boolean), ('Sell_Signal', pl.Boolean), ('DateTime', pl.Datetime)])

        # Calculate Donchian Channel
        df = df.with_columns([
            pl.col('High').rolling_max(window=window).alias('Donchian_High'),
            pl.col('Low').rolling_min(window=window).alias('Donchian_Low')
        ])

        # Generate buy and sell signals
        buy_signals = (df['Close'] > df['Donchian_High'].shift(1))
        sell_signals = (df['Close'] < df['Donchian_Low'].shift(1))

        

        return generate_signal(sell_signals.to_list(), buy_signals.to_list(), df['Datetime'].to_list())

    def ichimoku_cloud(self, df: pl.DataFrame, conversion_window=9, base_window=26, leading_span_window=52) -> pl.DataFrame:
        """
        Calculates the Ichimoku Cloud signals.

        Args:
            df (pl.DataFrame): The historical price data.
            conversion_window (int): The window period for the conversion line.
            base_window (int): The window period for the base line.
            leading_span_window (int): The window period for the leading span B.

        Returns:
            pl.DataFrame: The buy and sell signals.
        """
        required_columns = ['High', 'Low', 'Close']
        if not all(col in df.columns for col in required_columns):
            print(f"DataFrame must contain {required_columns} columns.")
            return pl.DataFrame(columns=['Buy_Signal', 'Sell_Signal'], schema=[('Buy_Signal', pl.Boolean), ('Sell_Signal', pl.Boolean), ('DateTime', pl.Datetime)])

        # Calculate Ichimoku components
        df = df.with_columns([
            ((pl.col('High').rolling_max(window=conversion_window) + pl.col('Low').rolling_min(window=conversion_window)) / 2).alias('Conversion_Line'),
            ((pl.col('High').rolling_max(window=base_window) + pl.col('Low').rolling_min(window=base_window)) / 2).alias('Base_Line'),
            (((pl.col('Conversion_Line') + pl.col('Base_Line')) / 2).shift(base_window)).alias('Leading_Span_A'),
            ((pl.col('High').rolling_max(window=leading_span_window) + pl.col('Low').rolling_min(window=leading_span_window)) / 2).shift(base_window).alias('Leading_Span_B')
        ])

        # Generate buy and sell signals
        buy_signals = (df['Close'] > df[['Leading_Span_A', 'Leading_Span_B']].max(axis=1)) & \
                    (df['Close'].shift(1) <= df[['Leading_Span_A', 'Leading_Span_B']].max(axis=1).shift(1))

        sell_signals = (df['Close'] < df[['Leading_Span_A', 'Leading_Span_B']].min(axis=1)) & \
                    (df['Close'].shift(1) >= df[['Leading_Span_A', 'Leading_Span_B']].min(axis=1).shift(1))


        return generate_signal(sell_signals.to_list(), buy_signals.to_list(), df['Datetime'].to_list())

    def vwap(self, df: pl.DataFrame) -> pl.DataFrame:
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

        return generate_signal(sell_signals.to_list(), buy_signals.to_list(), df['Datetime'].to_list())

def generate_signal(sell_signals: list, buy_signals: list, indexs: list) -> pl.DataFrame:
    """
    Generates a Polars DataFrame containing buy and sell signals.

    Args:
        sell_signals (list): Boolean list indicating sell signals.
        buy_signals (list): Boolean list indicating buy signals.
        indexs (list): The index (datetime) for the signals.

    Returns:
        pl.DataFrame: DataFrame with Buy_Signal and Sell_Signal columns.
    """
    # Ensure the lengths of the lists are equal
    if not (len(sell_signals) == len(buy_signals) == len(indexs)):
        raise ValueError("Length of sell_signals, buy_signals, and indexs must be equal.")

    # Create a Polars DataFrame
    signals_df = pl.DataFrame({
        'Buy_Signal': buy_signals,
        'Sell_Signal': sell_signals,
        'DateTime': indexs
    })

    return signals_df
