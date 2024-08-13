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

def get_stock_data(stock, DAYS=365, interval='1h'):
    '''
    Get historical stock data for a given stock symbol.

    Parameters:
    - stock (str): The stock symbol.
    - DAYS (int): The number of days of historical data to retrieve. Default is 365.
    - interval (str): Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]

    Returns:
    - DataFrame: A pandas DataFrame containing the historical stock data.
    '''
    end_date = get_exchange_time()

    start_date = end_date - timedelta(DAYS)  # days before the end date
    stock_ticker = yf.Ticker(stock)
    df = stock_ticker.history(start=start_date, end=end_date, interval=interval)
    if interval == '1h':
        df = df.reset_index('Datetime')
    else:
        df.reset_index(inplace=True)
        df.rename(columns={'Date': 'Datetime'}, inplace=True)
        
    df['SMA20'] = ta.sma(df['Close'], length=20)
    df['SMA50'] = ta.sma(df['Close'], length=50)
    df['SMA100'] = ta.sma(df['Close'], length=100)
    df['SMA150'] = ta.sma(df['Close'], length=150)
    df['RSI'] = ta.ema(df['Close'], length=20)
    df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['MCAD'] = ta.macd(df['Close'])['MACD_12_26_9']

    return df

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

    if current_time.weekday() >= 5:  # Check if the current date is a weekend
        return False

    if any(holiday.date() == current_time.date() for holiday in all_holidays):  # Check if the current date is a holiday
        return False

    if nyse_open_time <= current_time.time() <= nyse_close_time:  # Check if the current time is within trading hours
        return True
    else:
        return False

