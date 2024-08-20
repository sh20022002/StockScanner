import scraping, database
import time

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
        

def main():

    """

    The main function of the trade bot.


    This function initializes the database, checks for open positions, and looks for trades.

    It runs in a loop while the NYSE is open, with a delay of one hour between iterations.

    """

    # initialize the database

    initialize()

    fivem = 5 * 60

    recmondation =  strategy.RecomendAction

    while scraping.is_nyse_open():

        chack_open_positions()

        recmondation.reset()

        strategy.adx_rsi()

        for symbol in recmondation.buy:

            strategy.anlayze(symbol)

        for symbol in recmondation.sell:

            strategy.anlayze(symbol)
        
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

        compenies.append(sp500_compenies[1][i])

        compeny1 = compeny.Compeny(compeny_name=sp500_compenies[i][1], symbol=sp500_compenies[i][0],
                                    GICS_Sector=sp500_compenies[3][i],
                                    GICS_Sub_Industry=sp500_compenies[4][i],
                                    Location=sp500_compenies[5][i],
                                    CIK=sp500_compenies[6][i],
                                    Founded=sp500_compenies[7][i])
        

        database.save_compeny(compeny1)

if __name__ == "__main__":
    main()