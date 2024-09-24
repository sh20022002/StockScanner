import scraping, database
import time
from datetime import datetime
import strategy, plots, signal


def chack_data():
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
     
    signal_stack = signal.SignalStack()
    timeframe = ['1d', '1h', '1m']

    while True: #scraping.is_nyse_open():  

        for symbol in scraping.get_tickers():

            symbol = symbol[0]
            # Index(['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry',
            #    'Headquarters Location', 'Date added', 'CIK', 'Founded']
            try:
                data = scraping.get_stock_data(symbol , interval=timeframe[2], period='max', return_flags={
                                                                    'DF': True,
                                                                    'INDICATORS': True,
                                                                    'MAX_KEY': False,
                                                                    'SUMMERY': False,
                                                                    'DIVD': False,
                                                                    'INFO': True
                                                                    } )
            except Exception as e:
                print(f"Error: {e}")
                continue

            stock = strategy.Strategy( **data['INFO'])
            df = data['DF']
            print(stock.detect_signals_multithread(df))
            # best, backtest_res = stock.get_strategy_func(df, timeframe=timeframe[2])
            
            # for res in backtest_res:
            #     if res['strategy_func'] == best:
            #         plots.plot_stock(df, stock.symbol, df.columns, signals=res['signals']).show()
            #     print(res['performance'], res['risk_metrics'])
                
            
            
            break
     
        break
                

    
if __name__ == "__main__":
    run_trading_while_market_is_open()