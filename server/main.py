import scraping, database, avg
import time
from datetime import datetime
import strategy #, plots, signal
import polars as pl
import polars.selectors as cs
# import model_rgb 


def chack_data(data):
    fields = {}
    for key, value in data['INFO'].items():
        if key in fields:
            fields[key] = fields[key] + 1  # Increment the value if the key exists
        else:
            fields[key] = 1  # Initialize the key with 1 if it doesn't exist
    for key, value in fields.items():
        print(f"{key}-->{value}")

        
def run_trading_while_market_is_open(fivem=300):
    """
    Runs the trading strategy while the NYSE is open.

    Args:
        strategy (Strategy): The strategy object to use for trading.
        signal_stack (SignalStack): The stack to store buy/sell signals.
        recmondation (object): The recommendation object with buy/sell lists.
        fivem (int): Time in seconds to wait between checks (default is 300 seconds, or 5 minutes).

    """
     
    # signal_stack = signal.SignalStack()
    timeframe = '1d' # '1d', '1h', '1m'

    while True: #scraping.is_nyse_open():  

        for symbol in ['SPY']: #, 'AMZN', 'META', 'MSFT', 'NVDA', 'PYPL', 'BAC', 'CSCO', 'GOOG', 'COST', 'MS', 'UPST', 'TSM', 'ANF', 'IBM', 'PANW', 'HOOD']:

            # symbol = symbol[0]
            # Index(['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry',
            #    'Headquarters Location', 'Date added', 'CIK', 'Founded']
            try:
                data = scraping.get_stock_data(symbol , interval=timeframe, period='max', return_flags={
                                                                    'DF': True,
                                                                    'INDICATORS': True,
                                                                    'MAX_KEY': False,
                                                                    'SUMMERY': False,
                                                                    'DIVD': False,
                                                                    'INFO': True
                                                                    } )
                stock = strategy.Strategy( **data['INFO'])
                df = data['DF']
            except Exception as e:
                print(f"Error: {e}")
                continue

            
            df = pl.from_pandas(df, include_index=True)
            # avg.find_avg(df)
            best, backtest_res = stock.get_strategy_func(df, timeframe=timeframe)
            
            # print(symbol, timeframe)
            for res in backtest_res:
                if res == 'risk_metrics':
                    print(res)
            # if strategy.what_is_signal(best, backtest_res, 4): # returns true for buy and false for sale else None
            #     print(f'buy {symbol}')
            
        break
        
            
     
        
                

    
if __name__ == "__main__":
    run_trading_while_market_is_open()