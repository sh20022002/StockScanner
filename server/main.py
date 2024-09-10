import scraping, database
import time
from datetime import datetime
import strategy


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
     
    signal_stack = strategy.SignalStack()
    timeframe = ['1d', '1h', '1m']

    while scraping.is_nyse_open():  

        for symbol in scraping.get_tickers():

            symbol = symbol[0]
            # Index(['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry',
            #    'Headquarters Location', 'Date added', 'CIK', 'Founded']
            
            data = scraping.get_stock_data('NVDA', interval=timeframe[0], DAYS=730, return_flags={
                                                                'DF': False,
                                                                'MAX_KEY': False,
                                                                'SUMMERY': False,
                                                                'DIVD': False,
                                                                'INFO': True
                                                                } )
                
            stock = strategy.Strategy( **data['INFO'])
            # Assuming 'df' is your DataFrame and 'strategy_func' is the strategy function
            best_strategy, best_performance, best_risk_metrics = stock.get_strategy_func(timeframe=timeframe[0])

            print(f"best_strategy - {best_strategy}, best_performance - {best_performance}, \n best_risk_metrics - {best_risk_metrics}")


            break
            # print(df)
            # for date, (buy_signal, sell_signal) in zip(data['DF'].index, zip(buy_signals, sell_signals)):
            #     if buy_signal or sell_signal:
            #         print(f"Date: {date}, Buy Signal: {buy_signal}, Sell Signal: {sell_signal}")
            # best_strategy, best_performance, best_risk_metrics = stock.get_strategy_func()
            # print(f"Best Strategy: {best_strategy}, Best Performance: {best_performance}, Best Risk Metrics: {best_risk_metrics}")
            
        break
                
        #     best_strategy_name = strategy.simulate_multiple_strategies(df, company["symbol"], cash=1000, commission=0.1)

        #     buy_signal, sell_signal = get_current_signals(symbol=company["symbol"], strategy_name=best_strategy_name, signal_stack=signal_stack, **best_strategy_params)

        # print(f"Buy Signal: {buy_signal}, Sell Signal: {sell_signal}")
        # print("Current Signal Stack:")
        # print(signal_stack)

        # signal_stack.remove_irrelevant_signals()

        # # Check open positions or perform other account maintenance
        # # check_open_positions()  # Implement this function as needed

        # # Analyze and find the best strategy for each company
        #   # Assuming this fetches a list of company symbols

        # # Wait before checking again
        # time.sleep(fivem)




    
if __name__ == "__main__":
    run_trading_while_market_is_open()