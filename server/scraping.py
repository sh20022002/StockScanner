'''all web scraping functions'''
from datetime import timedelta, datetime, time
import pandas as pd
import yfinance as yf
import pytz, os
import requests
import pandas_ta as ta

exchange_api_key = os.getenv('EXCHANGE_API_KEY')

def current_stock_price(symbol):
    '''
    Get the current stock price for a given symbol.

    Parameters:
    - symbol (str): The stock symbol.

    Returns:
    - float: The current stock price.
    '''
    df = yf.Ticker(symbol).history(period='1h')
    return df['Close'].iloc[-1]


def get_stock_data(
    stock,
    return_flags={
        'DF': True,
        'INDICATORS': True,
        'MAX_KEY': False,
        'SUMMARY': False,
        'DIVID': False,
        'INFO': False
    },
    DAYS=365,
    interval='1h',
    period=None
):
    """
    Get historical stock data for a given stock symbol.

    Parameters:
    - stock (str): The stock symbol.
    - return_flags (dict): A dictionary specifying which return values to include.
        - DF (bool): Include the DataFrame of historical stock data. Default is True.
        - MAX_KEY (bool): Include the maximum recommendation key. Default is False.
        - SUMMARY (bool): Include the long business summary. Default is False.
        - DIVID (bool): Include the last dividend information. Default is False.
        - INFO (bool): Include the stock information. Default is False.
    - DAYS (int): The number of days of historical data to retrieve. Default is 365.
    - interval (str): Valid intervals: ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
    - period (str): Valid periods: ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']

    Returns:
        dict: A dictionary containing the requested return values.
            - DF (DataFrame): A pandas DataFrame containing the historical stock data.
            - MAX_KEY (str): The maximum recommendation key.
            - SUMMARY (str): The long business summary.
            - DIVID (str): The last dividend information.
            - INFO (dict): The stock information.
    """
    # Adjust DAYS based on interval limitations
    if interval == '1m' and DAYS > 7:
        DAYS = 7
    elif interval == '2m' and DAYS > 60:
        DAYS = 60
    elif interval == '1h' and DAYS > 729:
        DAYS = 729

    # Fetch current date
    end_date = get_exchange_time()

    # Calculate start date
    start_date = end_date - timedelta(days=DAYS)

    # Initialize return values dictionary
    return_values = {}

    try:
        stock_ticker = yf.Ticker(stock)
    except Exception as e:
        print(f"Error fetching data for {stock}: {e}")
        return return_values

    # Fetch additional info if requested
    if return_flags.get('SUMMARY', False) or return_flags.get('MAX_KEY', False) or return_flags.get('DIVID', False) or return_flags.get('INFO', False):
        try:
            info = stock_ticker.info
        except Exception as e:
            print(f"Error fetching info for {stock}: {e}")
            info = {}

    # Fetch summary
    if return_flags.get('SUMMARY', False):
        summary = info.get('longBusinessSummary', 'Summary not available.')
        return_values['SUMMARY'] = summary

    # Fetch max recommendation key
    if return_flags.get('MAX_KEY', False):
        max_key = info.get('recommendationKey', 'No recommendation key available.')
        return_values['MAX_KEY'] = max_key

    # Fetch dividend information
    if return_flags.get('DIVID', False):
        last_dividend_date_timestamp = info.get('lastDividendDate')
        if last_dividend_date_timestamp:
            last_dividend_date = datetime.fromtimestamp(last_dividend_date_timestamp)
            last_dividend_value = info.get('lastDividendValue', 0)
            divid = f"{last_dividend_date.strftime('%Y-%m-%d')} : {last_dividend_value} $"
        else:
            divid = 'No Dividend'
        return_values['DIVID'] = divid

    # Fetch stock info
    if return_flags.get('INFO', False):
        return_values['INFO'] = info

    # Fetch historical data
    if return_flags.get('DF', False):
        try:
            if period is None:
                df = stock_ticker.history(start=start_date, end=end_date, interval=interval)
            else:
                df = stock_ticker.history(period=period, interval=interval)
        except Exception as e:
            print(f"Error fetching historical data for {stock}: {e}")
            return return_values

        # Ensure the DataFrame is not empty
        if df.empty:
            print(f"No data fetched for {stock}.")
            return return_values

        # Rename index if necessary
        if df.index.name != 'Datetime':
            df.index.name = 'Datetime'

        # Drop unnecessary columns if they exist
        columns_to_drop = [col for col in ['Dividends', 'Stock Splits'] if col in df.columns]
        if columns_to_drop:
            df.drop(columns=columns_to_drop, inplace=True)

        # Ensure required columns are present
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing columns in data: {missing_columns}")
            return return_values

        # Select required columns
        df = df[required_columns]

        df.index = df.index.tz_localize(None)

        # # Handle duplicate indices
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep='first')]

        # Ensure the index is sorted
        df.sort_index(inplace=True)
        if return_flags.get('INDICATORS', False):
       
            # Compute technical indicators
            try:
                df['SMA20'] = ta.sma(df['Close'], length=20)
                df['SMA50'] = ta.sma(df['Close'], length=50)
                df['SMA100'] = ta.sma(df['Close'], length=100)
                df['SMA150'] = ta.sma(df['Close'], length=150)
                df['SMA200'] = ta.sma(df['Close'], length=200)
                df['EMA20'] = ta.ema(df['Close'], length=20)
                df['EMA12'] = ta.ema(df['Close'], length=12)
                df['EMA26'] = ta.ema(df['Close'], length=26)
                df['RSI'] = ta.rsi(df['Close'], length=14)
                macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
                df['MACD'] = macd['MACD_12_26_9']
                df['MACD_Signal'] = macd['MACDs_12_26_9']
                df['MACD_Hist'] = macd['MACDh_12_26_9']
                df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3, smooth_k=3)
                df['STOCH_%K'] = stoch['STOCHk_14_3_3']
                df['STOCH_%D'] = stoch['STOCHd_14_3_3']
                df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
            except Exception as e:
                print(f"Error computing technical indicators for {stock}: {e}")
                return return_values

        # Return the DataFrame
        return_values['DF'] = df

    return return_values

def get_tickers():
    '''
    Get a list of stock tickers from Wikipedia.

    Returns:
    - ndarray: A numpy array containing the stock tickers.
    '''
    tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    return tickers[0].values

def get_stocks():
    tickers = get_tickers()
    names = []
    symbols = []
    for ticker in tickers:
        names.append(ticker[1])
        symbols.append(ticker[0])
    return names, symbols

def get_exchange_time():
    '''
    Get the current time in New York.

    Returns:
    - datetime: The current time in New York.
    '''
    ny_timezone = pytz.timezone('America/New_York')
    ny_time = datetime.now(ny_timezone)
    return ny_time

def get_exchange_rate(from_currency, to_currency):
    '''
    Get the exchange rate between two currencies.

    Parameters:
    - from_currency (str): The currency to convert from.
    - to_currency (str): The currency to convert to.

    Returns:
    - float: The exchange rate.
    '''
    url = f'https://v6.exchangerate-api.com/v6/{api_keys.exchange_api_key}/latest/{from_currency}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        exchange_rate = data['conversion_rates'][to_currency]
        return exchange_rate
    else:
        return EOFError

def is_nyse_open():
    '''
    Check if the NYSE is currently open for trading.

    Returns:
    - bool: True if the NYSE is open, False otherwise.

    '''
    current_time = get_exchange_time()
    nyse_open_time = time(9, 30)
    nyse_close_time = time(16, 0)

    nyse_holidays = [
        datetime(current_time.year, 1, 1),  # New Year's Day
        next_weekday(datetime(current_time.year, 1, 15), 0),  # Martin Luther King Jr. Day (3rd Monday in January)
        next_weekday(datetime(current_time.year, 2, 15), 0),  # Presidents' Day (3rd Monday in February)
        easter_monday(current_time.year) - timedelta(days=3),  # Good Friday (date varies)
        next_weekday(datetime(current_time.year, 5, 25), 0),  # Memorial Day (last Monday in May)
        datetime(current_time.year, 7, 4),  # Independence Day
        next_weekday(datetime(current_time.year, 9, 1), 0),  # Labor Day (1st Monday in September)
        next_weekday(datetime(current_time.year, 11, 22), 3),  # Thanksgiving Day (4th Thursday in November)
        datetime(current_time.year, 12, 25)  # Christmas Day
    ]

    special_dates = [
        datetime(current_time.year, 12, 24),
        datetime(current_time.year, 12, 25),
        datetime(current_time.year, 2, 19)
    ]

    all_holidays = nyse_holidays + special_dates

    if 0 <= current_time.weekday() <= 4:  # Check if the current date is a weekend
        if any(holiday.date() != current_time.date() for holiday in all_holidays):  # Check if the current date is a holiday
            if nyse_open_time <= current_time.time() <= nyse_close_time:  # Check if the current time is within trading hours
                return True
    
    
    return False

def next_weekday(d, weekday):
    """Return the next date with the given weekday (0=Monday, 6=Sunday)."""
    days_ahead = weekday - d.weekday()
    if days_ahead <= 0:  # Target day already happened this week
        days_ahead += 7
    return d + timedelta(days_ahead)

def easter_monday(year):
    """Returns Easter Monday for a given year."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    easter_sunday = datetime(year, month, day)
    return easter_sunday + timedelta(days=1)



    def next_weekday(d, weekday):
        """Return the next date with the given weekday (0=Monday, 6=Sunday)."""
        days_ahead = weekday - d.weekday()
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        return d + timedelta(days_ahead)

    def easter_monday(year):
        """Returns Easter Monday for a given year."""
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        easter_sunday = datetime(year, month, day)
        return easter_sunday + timedelta(days=1)


if __name__ == '__main__':
    print(get_stock_data('AAPL')['DF'])