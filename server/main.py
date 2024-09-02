import scraping, database
import time
from datetime import datetime


import strategy
class Compeny:
    def __init__(self, compeny_name, symbol, GICS_Sector, GICS_Sub_Industry, Location, CIK, Founded):
        self.compeny_name = compeny_name
        self.symbol = symbol
        self.GICS_Sector = GICS_Sector
        self.GICS_Sub_Industry = GICS_Sub_Industry
        self.Location = Location
        self.CIK = CIK
        self.Founded = Founded
        
def run_trading_while_market_is_open(fivem=300):
    """
    Runs the trading strategy while the NYSE is open.

    Args:
        strategy (Strategy): The strategy object to use for trading.
        signal_stack (SignalStack): The stack to store buy/sell signals.
        recmondation (object): The recommendation object with buy/sell lists.
        fivem (int): Time in seconds to wait between checks (default is 300 seconds, or 5 minutes).

    """
    initialize()

    while scraping.is_nyse_open():
        # Reset recommendations and stack
        signal_stack = SignalStack()

        # Get the best strategy and its parameters
        
        for company in database.get_compenies():
            df = scraping.get_stock_data(company["symbol"], interval="1d", period="1y")

            best_strategy_name = strategy.simulate_multiple_strategies(df, company["symbol"], cash=1000, commission=0.1)

            buy_signal, sell_signal = get_current_signals(symbol=company["symbol"], strategy_name=best_strategy_name, signal_stack=signal_stack, **best_strategy_params)

        print(f"Buy Signal: {buy_signal}, Sell Signal: {sell_signal}")
        print("Current Signal Stack:")
        print(signal_stack)

        signal_stack.remove_irrelevant_signals()

        # Check open positions or perform other account maintenance
        # check_open_positions()  # Implement this function as needed

        # Analyze and find the best strategy for each company
          # Assuming this fetches a list of company symbols

        # Wait before checking again
        time.sleep(fivem)



def initialize():

    """

    Initialize the database.


    This function initializes the database by saving the S&P 500 companies and their information.

    It returns a list of the saved companies.

    """

    # initialize the database

    sp500_compenies = scraping.get_tickers()


    # Index(['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry',


        #    'Headquarters Location', 'Date added', 'CIK', 'Founded']

    for i in range(len(sp500_compenies[0])):

        # compenies.append(sp500_compenies[1][i])

        compeny1 = Compeny(compeny_name=sp500_compenies[i][1], symbol=sp500_compenies[i][0],
                                    GICS_Sector=sp500_compenies[3][i],
                                    GICS_Sub_Industry=sp500_compenies[4][i],
                                    Location=sp500_compenies[5][i],
                                    CIK=sp500_compenies[6][i],
                                    Founded=sp500_compenies[7][i])
        
        # d.append(compeny1)
        database.save_compeny(compeny1)

if __name__ == "__main__":
    run_trading_while_market_is_open()