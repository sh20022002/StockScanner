import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import scraping
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
        self.loss_percent = 3 if self.risk_tolerance < 80 else 7
        self.profit_percent = None if self.risk_tolerance < 80 else 10
        
        

    def __str__(self):
        return "\n".join(f'{key}: {value}' for key, value in self.__dict__.items())
    


    def get_strategy_func(self, df: pl.DataFrame, timeframe, num_threads=20, plot=False):
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
                    
                    fig = plots.plot_stock(df, self.symbol, df.columns, signals=signals, show='no', interval=timeframe) if plot else None
                    backtest_res.append({'strategy_func': strategy_func, 'performance': performance, 'risk_metrics': risk_metrics, 'signals': signals, 'fig': fig})

                    # Check if this strategy has the best performance
                    if performance > best_performance:
                        best_performance = performance
                        best_strategy = strategy_func
                        best_risk_metrics = risk_metrics
                    # print(f"Strategy: {strategy_func}, Performance: {performance}, \n Risk Metrics: {risk_metrics}")
        
        return best_strategy, backtest_res

        
    def backtest_strategy(self, df: pl.DataFrame, signals_df: pl.DataFrame,
                          transaction_cost: float = 0.01,
                          stop_loss_percent: float = None, stop_profit_percent: float = None,
                          leverage: float = 1, trailing_stop: bool = True) -> tuple:
        """
        Vectorised backtest using numpy arrays.
        Supports long/short positions, leverage, trailing stop-loss, and stop-profit.
        ~3× faster than the previous dict-based loop.
        """
        # Extract numpy arrays once — avoids per-row dict lookups
        close = df['Close'].to_numpy()
        dates = df['Datetime'].to_numpy()
        buy   = signals_df['Buy_Signal'].to_numpy()
        sell  = signals_df['Sell_Signal'].to_numpy()

        n           = len(close)
        cash        = 100_000.0
        starting    = cash
        pos         = 0.0          # shares held (positive=long, negative=short)
        pos_type    = 0            # 0=flat, 1=long, -1=short
        entry       = 0.0
        trade_cash  = 0.0
        borrow      = 0.0
        high_water  = 0.0
        low_water   = 0.0
        peak        = cash
        max_dd      = 0.0
        total       = 0
        wins        = 0
        sl          = None
        sp          = None

        for i in range(n):
            p = close[i]

            if pos_type == 0:                          # ── no position open ──
                if buy[i]:
                    trade_cash = cash
                    borrow     = (leverage - 1) * trade_cash
                    pos        = leverage * trade_cash / p
                    entry      = p;  pos_type = 1;  cash = 0.0;  high_water = p
                    sl = p * (1 - stop_loss_percent   / 100) if stop_loss_percent   else None
                    sp = p * (1 + stop_profit_percent / 100) if stop_profit_percent else None
                    total += 1
                elif sell[i]:
                    trade_cash = cash
                    pos        = -(leverage * trade_cash / p)
                    entry      = p;  pos_type = -1;  cash = 0.0;  low_water = p
                    sl = p * (1 + stop_loss_percent   / 100) if stop_loss_percent   else None
                    sp = p * (1 - stop_profit_percent / 100) if stop_profit_percent else None
                    total += 1

            elif pos_type == 1:                        # ── long position ──
                if trailing_stop and p > high_water:
                    high_water = p
                    if stop_loss_percent:
                        sl = high_water * (1 - stop_loss_percent / 100)
                if (sl and p <= sl) or (sp and p >= sp) or sell[i]:
                    cash = (pos * p - borrow) * (1 - transaction_cost)
                    if p > entry: wins += 1
                    pos = 0.0;  pos_type = 0;  sl = None;  sp = None

            elif pos_type == -1:                       # ── short position ──
                if trailing_stop and p < low_water:
                    low_water = p
                    if stop_loss_percent:
                        sl = low_water * (1 + stop_loss_percent / 100)
                if (sl and p >= sl) or (sp and p <= sp) or buy[i]:
                    cash = (trade_cash + leverage * trade_cash - abs(pos) * p) * (1 - transaction_cost)
                    if p < entry: wins += 1
                    pos = 0.0;  pos_type = 0;  sl = None;  sp = None

            # Drawdown tracking
            portfolio = (cash if pos_type == 0
                         else pos * p - borrow if pos_type == 1
                         else trade_cash + leverage * trade_cash - abs(pos) * p)
            if portfolio > peak: peak = portfolio
            dd = (peak - portfolio) / peak if peak > 0 else 0.0
            if dd > max_dd: max_dd = dd

        # Close open position at final price
        if pos_type != 0:
            fp = close[-1]
            if pos_type == 1:
                cash = (pos * fp - borrow) * (1 - transaction_cost)
            else:
                cash = (trade_cash + leverage * trade_cash - abs(pos) * fp) * (1 - transaction_cost)

        win_rate = wins / total * 100 if total else 0.0
        roi      = (cash - starting) / starting * 100
        try:
            days = int((dates[-1] - dates[0]) / np.timedelta64(1, 'D'))
        except Exception:
            days = n

        return cash - starting, {
            'win_rate':        round(win_rate, 2),
            'time_frame_days': days,
            'roi':             round(roi, 2),
            'max_drawdown':    round(max_dd * 100, 2),
        }



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

   


    def detect_signals_multithread(self, df: pl.DataFrame, threshold=2) -> dict:
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
                    result_df = result_df.join(df.select(['Datetime']), on='Datetime', how='left').fill_null(False)
                    
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
                'DateTime': df['Datetime']
            })

            return combined_signals_df

        # convert df to polars
        # df = pl.from_pandas(df, include_index=True)
       

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
            ('ema_crossover', self.ema_crossover),
            ('prev_high_low', self.prev_high_low)
        ]

        # Run all tasks concurrently using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(task[1], df): task[0] for task in tasks}
            results = {}

            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        # Align signals to full df datetime; cast to matching dtype first
                        dt_dtype = df.schema['Datetime']
                        result = (
                            df.select(['Datetime'])
                            .join(
                                result.with_columns(pl.col('Datetime').cast(dt_dtype)),
                                on='Datetime', how='left'
                            )
                            .with_columns([
                                pl.col('Buy_Signal').fill_null(False),
                                pl.col('Sell_Signal').fill_null(False),
                            ])
                        )
                    results[task_name] = result
                except Exception as e:
                    print(f"Error in strategy {task_name}: {e}")

        return results



    def macd(self, df: pl.DataFrame) -> pl.DataFrame:
        """MACD (Moving Average Convergence Divergence)
        What It Is: A trend-following momentum indicator that shows the relationship between two moving averages of a security's price.
        Key Components:
        MACD Line: The difference between the 12-period EMA and 26-period EMA.
        Signal Line: A 9-period EMA of the MACD line.
        Histogram: The difference between the MACD line and the Signal line.
        How It Works:
        A buy signal occurs when the MACD line crosses above the Signal line.
        A sell signal occurs when the MACD line crosses below the Signal line.
        Use Case: Identify momentum shifts, trend direction, and potential entry/exit points.
        """
        required_columns = ['MACD', 'MACD_Signal', 'Datetime']
        if not all(col in df.columns for col in required_columns):
            print("Required columns for MACD strategy are missing.")
            return None

        try:
            # Create buy signals where MACD crosses above the MACD Signal line
            buy_signals = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
            # Create sell signals where MACD crosses below the MACD Signal line
            sell_signals = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))

            # Fill null values with False directly
            buy_signals = buy_signals.fill_null(False)
            sell_signals = sell_signals.fill_null(False)

            # Assuming generate_signal is defined somewhere else that generates the final DataFrame based on these signals
            return generate_signal(sell_signals.to_list(), buy_signals.to_list(), df['Datetime'].to_list())

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

        
    def rsi(self, df: pl.DataFrame, Upper_Band=70, Lower_Band=35) -> pl.DataFrame:
        """
        Calculates the Relative Strength Index (RSI) signals.
        RSI (Relative Strength Index)
        What It Is: A momentum oscillator that measures the speed and change of price movements, ranging from 0 to 100.
        How It Works:
        Overbought Condition: RSI > 70, indicating the asset may be overvalued and due for a correction.
        Oversold Condition: RSI < 30, indicating the asset may be undervalued and due for a bounce.
        Use Case: Identify overbought or oversold conditions to time entries or exits.

        
        Args:
            df (pd.DataFrame): The historical price data.
            period (int): The period for calculating RSI.
            
        Returns:
            tuple: The buy and sell signals.
        """
        buy_signals = (df['RSI'] < Lower_Band) & (df['RSI'].shift(1) >= Lower_Band)
        sell_signals = (df['RSI'] > Upper_Band) & (df['RSI'].shift(1) <= Upper_Band)

        # Fill null values with False directly
        buy_signals = buy_signals.fill_null(False)
        sell_signals = sell_signals.fill_null(False)

        return generate_signal(sell_signals.to_list(), buy_signals.to_list(), df['Datetime'].to_list())
        
    def ma(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculates the Moving Average (MA) signals.
        What It Is: A smoothing indicator that calculates the average price of a security over a specified time period.
        Types:
        Simple Moving Average (SMA): A straight average of prices over a period.
        Exponential Moving Average (EMA): A moving average that gives more weight to recent prices.
        How It Works:
        If the price is above the MA, it indicates a bullish trend.
        If the price is below the MA, it indicates a bearish trend.
        Use Case: Identify the direction of the trend and potential support/resistance levels
        
        Args:
            df (pd.DataFrame): The historical price data.
            
        Returns:
            tuple: The buy and sell signals.
        """
        # deal with a type error recast the column to the dataframe as a float 64
        df = df.with_columns(
            pl.col("SMA150").cast(pl.Float64, strict=False).alias("SMA150"))

        buy_signals = (df['SMA20'] > df['SMA150']) & (df['SMA20'].shift(1) <= df['SMA150'].shift(1))
        sell_signals = (df['SMA20'] < df['SMA150']) & (df['SMA20'].shift(1) >= df['SMA150'].shift(1))

        # Fill null values with False directly
        buy_signals = buy_signals.fill_null(False)
        sell_signals = sell_signals.fill_null(False)

        return generate_signal(sell_signals.to_list(), buy_signals.to_list(), df['Datetime'].to_list())

    def bollinger_bands(self, df: pl.DataFrame, window: int = 20, num_std_dev=3) -> pl.DataFrame:
        """
        Calculates the Bollinger Bands signals.
        Bollinger Bands
        What It Is: A volatility indicator consisting of a middle band (SMA) and two outer bands (standard deviations above and below the SMA).
        How It Works:
        Buy Signal: When the price touches or crosses the lower band and reverses upward.
        Sell Signal: When the price touches or crosses the upper band and reverses downward.
        Breakout: Significant price movement when the bands contract (low volatility).
        Use Case: Identify periods of high/low volatility and potential price reversals or breakouts.
        """

        # Calculate Bollinger Bands
        df = df.with_columns(
            pl.col('Close').rolling_std(window_size=window).alias('STD20')
)

        # Step 2: Drop rows where 'SMA20' or 'STD20' have NaN values
        df = df.filter(
            pl.col('SMA20').is_not_null() & pl.col('STD20').is_not_null()
        )

        # Step 3: Create 'Upper_Band' and 'Lower_Band'
        df = df.with_columns([
            (pl.col('SMA20') + (pl.col('STD20') * num_std_dev)).alias('Upper_Band'),
            (pl.col('SMA20') - (pl.col('STD20') * num_std_dev)).alias('Lower_Band')
        ])
        
            # Step 4: Generate buy and sell signals
        buy_signals = (
            (df['Close'] < df['Lower_Band']) &
            (df['Close'].shift(1) >= df['Lower_Band'].shift(1))
        )

        sell_signals = (
            (df['Close'] > df['Upper_Band']) &
            (df['Close'].shift(1) <= df['Upper_Band'].shift(1))
            )

        # Fill null values with False directly
        buy_signals = buy_signals.fill_null(False)
        sell_signals = sell_signals.fill_null(False)

        return generate_signal(sell_signals.to_list(), buy_signals.to_list(), df['Datetime'].to_list())


    def ema_crossover(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculates the Exponential Moving Average Crossover signals.
        Exponential Moving Average (EMA): A moving average that gives more weight to recent prices.
        How It Works:
        If the price is above the MA, it indicates a bullish trend.
        If the price is below the MA, it indicates a bearish trend.
        Use Case: Identify the direction of the trend and potential support/resistance levels

        Args:
            df (pl.DataFrame): The historical price data.

        Returns:
            pl.DataFrame: The buy and sell signals.
        """
        df = df.filter(pl.col('EMA12').is_not_null() & pl.col('EMA26').is_not_null())

        # Generate buy and sell signals
        buy_signals = (df['EMA12'] > df['EMA26']) & (df['EMA12'].shift(1) <= df['EMA26'].shift(1))
        sell_signals = (df['EMA12'] < df['EMA26']) & (df['EMA12'].shift(1) >= df['EMA26'].shift(1))

        # Fill null values with False directly
        buy_signals = buy_signals.fill_null(False)
        sell_signals = sell_signals.fill_null(False)

        return generate_signal(sell_signals.to_list(), buy_signals.to_list(), df['Datetime'].to_list())

    def stochastic_oscillator(self, df: pl.DataFrame, k_window=12, d_window=3, overbought=80, oversold=30) -> pl.DataFrame:
        """
        Calculates the Stochastic Oscillator signals.
        What It Is: A momentum indicator that compares the closing price of a security to its price range over a specific period.
        How It Works:
        Overbought Condition: When the oscillator is above 80.
        Oversold Condition: When the oscillator is below 20.
        Buy Signal: %K crosses above %D in the oversold zone.
        Sell Signal: %K crosses below %D in the overbought zone.
        Use Case: Identify overbought/oversold conditions and potential reversals.


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
        df.sort('Datetime')

        # Calculate %K (stochastic)
        df = df.with_columns(
            df['Low'].rolling_min(window_size=k_window, min_periods=1).cast(pl.Float32).alias('Lowest_Low'),
            df['High'].rolling_max(window_size=k_window, min_periods=1).cast(pl.Float32).alias('Highest_High'))
        
        df = df.with_columns((
            df['Highest_High'] - df['Lowest_Low']).cast(pl.Float32).alias('Denominator'))

        # Avoid division by zero
        df = df.with_columns(
            pl.when(pl.col("Denominator") == 0)
            .then(None)  # Replace 0 with null
            .otherwise(pl.col("Denominator"))  # Keep other values unchanged
            
            .alias("Denominator"))  # Update the column
        # Handle NaN values resulting from division by zero
        df = df.with_columns((((df['Close'] - df['Lowest_Low'])/ df['Denominator']) * 100).cast(pl.Float32).fill_nan(0).alias('%K'))

        # Calculate %D (signal line)
        df = df.with_columns(df['%K'].rolling_mean(window_size=d_window, min_periods=1).cast(pl.Float32).alias('%D'))

        # Generate buy and sell signals
        buy_signals = (df['%K'] < oversold) & (df['%K'].shift(1) >= oversold)
        sell_signals = (df['%K'] > overbought) & (df['%K'].shift(1) <= overbought)

        # Fill null values with False directly
        buy_signals = buy_signals.fill_null(False)
        sell_signals = sell_signals.fill_null(False)

        # Clean up temporary columns
        df.drop('Lowest_Low', 'Highest_High', 'Denominator', '%K', '%D')

        return generate_signal(sell_signals.to_list(), buy_signals.to_list(), df['Datetime'].to_list())



    def parabolic_sar(self, df: pl.DataFrame, step=0.02, max_step=0.2) -> pl.DataFrame:
        """
        Calculates the Parabolic SAR signals.
        Parabolic SAR (Stop and Reverse)
        What It Is: A trend-following indicator that places points above or below the price, depending on the trend.
        How It Works:
        Buy Signal: When the dots move below the price.
        Sell Signal: When the dots move above the price.
        Use Case: Identify trends and reversal points. It is also used for trailing stop-losses.


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
        

        return generate_signal(sell_signals, buy_signals, df['Datetime'].to_list())

    def atr_breakout(self, df: pl.DataFrame, window=14, multiplier=1.2) -> pl.DataFrame:
        """
        Calculates the ATR breakout signals.
        What It Is: A volatility indicator that measures the average range of price movement over a period.
        How It Works:
        Use ATR to set dynamic stop-loss and take-profit levels.
        Breakout Strategy: Buy when the price moves above a predefined level based on the ATR, and sell when it moves below a similar level.
        Use Case: Trade based on volatility and manage risk effectively.

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

        # Fill null values with False directly
        buy_signals = buy_signals.fill_null(False)
        sell_signals = sell_signals.fill_null(False)

        return generate_signal(sell_signals.to_list(), buy_signals.to_list(), df['Datetime'].to_list())

    def donchian_channel(self, df: pl.DataFrame, window=20) -> pl.DataFrame:
        """
        Calculates the Donchian Channel breakout signals.
        What It Is: A volatility indicator that plots the highest high and lowest low over a specific period.
        How It Works:
        Buy Signal: When the price breaks above the upper band.
        Sell Signal: When the price breaks below the lower band.
        Use Case: Identify breakouts and trends.

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
            df['High'].rolling_max(window_size=window).alias('Donchian_High'),
            df['Low'].rolling_min(window_size=window).alias('Donchian_Low')
        ])

        # Generate buy and sell signals
        buy_signals = (df['Close'] > df['Donchian_High'].shift(1))
        sell_signals = (df['Close'] < df['Donchian_Low'].shift(1))

        # Fill null values with False directly
        buy_signals = buy_signals.fill_null(False)
        sell_signals = sell_signals.fill_null(False)
 
        return generate_signal(sell_signals.to_list(), buy_signals.to_list(), df['Datetime'].to_list())

    def ichimoku_cloud(self, df: pl.DataFrame, conversion_window=9, base_window=26, leading_span_window=52) -> pl.DataFrame:
        """
        Calculates the Ichimoku Cloud signals.
        What It Is: A comprehensive indicator providing trend direction, momentum, and support/resistance levels.
        Key Components:
        Tenkan-Sen (Conversion Line) and Kijun-Sen (Base Line): Act like fast and slow moving averages.
        Senkou Span A/B (Cloud): The shaded area between them represents support/resistance and trend direction.
        Chikou Span (Lagging Line): Represents past price action.
        How It Works:
        Buy Signal: When the price is above the cloud and the Conversion Line crosses above the Base Line.
        Sell Signal: When the price is below the cloud and the Conversion Line crosses below the Base Line.
        Use Case: Identify trends, reversals, and support/resistance levels.

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
        df = df.with_columns(
            (df['High'].rolling_max(window_size=conversion_window) + df['Low'].rolling_min(window_size=conversion_window) / 2).alias('Conversion_Line'),
            (df['High'].rolling_max(window_size=base_window) + df['Low'].rolling_min(window_size=base_window) / 2).alias('Base_Line'))

        df = df.with_columns(((df['Conversion_Line'] + df['Base_Line']) / 2).shift(base_window).alias('Leading_Span_A'),
            ((df['High'].rolling_max(window_size=leading_span_window) + df['Low'].rolling_min(window_size=leading_span_window)) / 2).shift(base_window).alias('Leading_Span_B'))

        # Generate buy and sell signals and fill null values
        df = df.with_columns(
            ((df["Close"] > pl.max_horizontal(["Leading_Span_A", "Leading_Span_B"])) &
            (df["Close"].shift(1) <= pl.max_horizontal(["Leading_Span_A", "Leading_Span_B"])).shift(1))
        .fill_null(False).alias('buy_signals'))

        df = df.with_columns(
            ((df["Close"] < pl.min_horizontal(["Leading_Span_A", "Leading_Span_B"])) &
            (df["Close"].shift(1) >= pl.min_horizontal(["Leading_Span_A", "Leading_Span_B"])).shift(1))
        .fill_null(False).alias('sell_signals'))

        return generate_signal(df['sell_signals'].to_list(), df['buy_signals'].to_list(), df['Datetime'].to_list())
    
    def prev_high_low(self, df: pl.DataFrame, N: int = 390) -> pl.DataFrame:
        """
        Generate buy/sell signals when the current Close price breaks above or below
        the previous N-bar High/Low (on a minute timeframe).
        
        By default, N=390 represents the approximate number of minutes in a standard 
        U.S. trading day (9:30 AM – 4:00 PM ET). At each row, this function:
        • Looks back at the last N rows (excluding the current row).
        • Finds the minimum Low and maximum High of those N rows.
        • Triggers a buy signal if the current Close >= that maximum High.
        • Triggers a sell signal if the current Close <= that minimum Low.
        
        Args:
            df (pl.DataFrame):
                A Polars DataFrame containing at least the following columns:
                'Datetime', 'High', 'Low', and 'Close'.
            N (int, optional):
                The number of previous rows to consider when determining
                the rolling minimum and maximum. Defaults to 390.
        
        Returns:
            pl.DataFrame:
                A Polars DataFrame of signals, specifically created via 
                the helper function generate_signal(). The resulting DataFrame 
                has 'Buy_Signal' and 'Sell_Signal' columns aligned by 'Datetime'.
        
        Usage Example:
            >>> # Suppose you have a Polars DataFrame df with minute bars
            >>> # for a single trading day.  This function will mark each row
            >>> # where Close crosses above or below the prior day’s range.
            >>> signals_df = prev_high_low(df)
            >>> # signals_df will contain boolean columns 'Buy_Signal' and 'Sell_Signal',
            >>> # as well as 'Datetime'.

        """
        df = df.with_columns([
            pl.col("Low").shift(1).rolling_min(window_size=N).alias("prev_n_min"),
            pl.col("High").shift(1).rolling_max(window_size=N).alias("prev_n_max")
        ])
        
        buy_signals = df["Close"] >= df["prev_n_max"]
        sell_signals = df["Close"] <= df["prev_n_min"]

        # Fill null values with False directly
        buy_signals = buy_signals.fill_null(False)
        sell_signals = sell_signals.fill_null(False)

        return generate_signal(
            sell_signals.to_list(), 
            buy_signals.to_list(), 
            df["Datetime"].to_list()
        )

    def vwap(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculates VWAP (Volume Weighted Average Price) buy and sell signals.
        What It Is: A trading benchmark that calculates the average price of a security based on both price and volume.
        How It Works:
        If the price is above VWAP, it indicates a bullish trend.
        If the price is below VWAP, it indicates a bearish trend.
        Use Case: Commonly used by institutional traders to ensure they buy/sell at favorable prices relative to the average market price
        
        Args:
            df (pd.DataFrame): The historical price data with volume information.
            
        Returns:
            pd.DataFrame: The buy and sell signals.
        """
        # Calculate VWAP
        df = df.with_columns((df['Close'] * df['Volume']).cum_sum().alias('Cumulative_Price_Vol'),
            df['Volume'].cum_sum().alias('Cumulative_Vol'))

        df = df.with_columns((df['Cumulative_Price_Vol'] / df['Cumulative_Vol']).alias('VWAP'))

        # Buy and sell signals
        # 0.98 an
        buy_signals = (df['Close'] > df['VWAP']*0.98) & (df['Close'].shift(1) <= df['VWAP'].shift(1))
        sell_signals = (df['Close'] < df['VWAP']*0.90) & (df['Close'].shift(1) >= df['VWAP'].shift(1))

        # Fill null values with False directly
        buy_signals = buy_signals.fill_null(False)
        sell_signals = sell_signals.fill_null(False)

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
        'Datetime': indexs
    })

    return signals_df

   

def what_is_signal(best, backtest_res, n):
    num_of_sell = 0
    num_of_buy = 0
    roi_sum = 0.0

    # Process all results first
    for res in backtest_res:
        sig = res['signals']

        # Extract the last n signals
        buy_signals = sig['Buy_Signal'][-n:] if len(sig['Buy_Signal']) >= n else sig['Buy_Signal']
        sell_signals = sig['Sell_Signal'][-n:] if len(sig['Sell_Signal']) >= n else sig['Sell_Signal']

        # Check if there's any True in the last n buy/sell signals
        buy_signal_window = any(buy_signals)
        sell_signal_window = any(sell_signals)

        # If no signals in this n-length window, continue
        if not buy_signal_window and not sell_signal_window:
            continue

        # Add ROI to sum
        roi_sum += res['risk_metrics']['roi']

        # Increment counts
        # If both or none are True, you could define a rule. Here we assume it can't be both buy and sell.
        # If it can be both, you might need additional logic.
        if buy_signal_window and not sell_signal_window:
            num_of_buy += 1
        elif sell_signal_window and not buy_signal_window:
            num_of_sell += 1
        else:
            # If both are True, decide what to do. Here we'll skip.
            # Alternatively, you could choose a priority:
            # num_of_buy += 1  # If we want to favor buy in a tie
            continue

    # After processing all results
    total_signals = num_of_buy + num_of_sell
    if total_signals == 0:
        # No signals at all
        return None

    # Compute average ROI
    average_roi = roi_sum / total_signals

    # Determine final signal
    if average_roi > 0:
        if num_of_buy > num_of_sell:
            return True  # More buy signals and positive ROI
        else:
            return False  # More sell signals or equal number of signals, with positive ROI

    # If average ROI is not positive, return None
    return None

def find_local_extremes(prices):# test
    """Find local highs and lows in the price data.
    Returns two lists of tuples: (index, price) for highs and lows."""
    highs = []
    lows = []
    for i in range(1, len(prices)-1):
        if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
            highs.append((i, prices[i]))
        elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
            lows.append((i, prices[i]))
    return highs, lows

def fit_line(points):# test
    """Fit a line to a set of points using linear regression.
    Points: list of (x, y) tuples.
    Returns slope (m), intercept (c)."""
    if len(points) == 0:
        # No points to fit a line
        return 0, 0
    if len(points) == 1:
        # Only one point, slope=0, line at that point's price
        return 0, points[0][1]
    x_vals = np.array([p[0] for p in points])
    y_vals = np.array([p[1] for p in points])
    m, c = np.polyfit(x_vals, y_vals, 1)  # 1st degree polynomial (linear)
    return m, c

def get_trend_line_value(m, c, x):# test
    """Get the y-value (price) of the line at index x."""
    return m * x + c

def determine_signal_from_trend_lines_polars(df: pl.DataFrame, upper_threshold_percent=30.0 , lower_threshold_percent=20.0):# test
    """
    Determine a buy/sell/none signal based on trend lines from a Polars DataFrame.
    The DataFrame must have: 'datetime', 'open', 'high', 'low', 'close'.

    Parameters:
    - df: A Polars DataFrame with OHLC data.
    - upper_threshold_percent: The percentage threshold for sell signals (distance to upper trend line).
    - lower_threshold_percent: The percentage threshold for buy signals (distance to lower trend line).

    Returns: "Buy", "Sell", or None.
    """
    # Extract close prices as a list
    prices = df["close"].to_list()

    # 1. Identify highs and lows (local extremes)
    highs, lows = find_local_extremes(prices)

    # If we do not have any highs or lows, return None
    if not highs or not lows:
        return None

    # 2. Fit trend lines for highs and lows
    m_up, c_up = fit_line(highs)
    m_down, c_down = fit_line(lows)

    # 3. Calculate current distances
    current_index = len(prices) - 1
    current_price = prices[-1]
    upper_line_price = get_trend_line_value(m_up, c_up, current_index)
    lower_line_price = get_trend_line_value(m_down, c_down, current_index)

    if current_price == 0:
        return None

    dist_to_upper = ((upper_line_price - current_price) / current_price) * 100.0
    dist_to_lower = ((current_price - lower_line_price) / current_price) * 100.0

    # 4. Determine signals
    # If the current price is near the upper trend line (within upper_threshold_percent), consider Sell.
    if dist_to_upper >= 0 and dist_to_upper <= upper_threshold_percent:
        return "Sell"

    # If the current price is near the lower trend line (within lower_threshold_percent), consider Buy.
    if dist_to_lower >= 0 and dist_to_lower <= lower_threshold_percent:
        return "Buy"

    # Otherwise, no signal.
    return None