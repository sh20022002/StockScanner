import scraping
from deap import base, creator, tools, algorithms
import random
import numpy as np
import database
import pandas as pd




class RecomendAction:


    sell = []

    buy = []


    def reset(self):

        # This function resets the class-level array

        self.buy = []

        self.sell = []



    def add_to_buy(self, symbol):

        # This function adds an element to the class-level array

        self.buy.append(symbol)

    def add_to_sell(self, symbol):

        # This function adds an element to the class-level array

        self.sell.append(symbol)

    def get_buy(self):

        # This function returns the current state of the array
        return self.buy

    def get_sell(self): 

        # This function returns the current state of the array
        return self.sell

class strategy:
    """
    A class representing a trading strategy.

    Attributes:

    - avg_price (float): The average price of the asset.

    - top_percent_from_portfolio (float): The top percentage of the portfolio to consider for trading.

    - loss_percent (float): The percentage below the average price to set as the stop loss.

    - profit_percent (float): The percentage above the average price to set as the stop profit.

    Methods:

    - __init__(avg_price, top_percent_from_portfolio, loss_percent, profit_percent, **kwargs): Initializes the strategy object.

    - __str__(): Prints the attributes of the strategy object.

    - optimize(): Optimizes the strategy using a genetic algorithm.

    - adx_rsi(): Looks for potential trades based on ADX and RSI indicators.

    """
    
    def __init__(self, avg_price, top_percent_from_portfolio=0.05, loss_percent=0.05, profit_percent=0.2, **kwargs):

        self.avg_price = avg_price
        self.top_percent_from_portfolio = top_percent_from_portfolio
        self.loss_percent = loss_percent
        self.profit_percent = profit_percent
        # Set stoploss and stopprofit based on avg_price and the respective percents

        self.stoploss = self.avg_price * (1 - self.loss_percent)
        self.stopprofit = self.avg_price * (1 + self.profit_percent)

        # Process additional keyword arguments if necessary

        for key, value in kwargs.items():

            setattr(self, key, value)

    def __str__(self):

        """

        Prints the attributes of the strategy object.

        """

        for key, value in self.__dict__.items():

            if key and value:

                print(f'{key}: {value}')


    def simulate_trading(self, prices, symbol, cash=10000, commission=0.01):

        """

        Simulates trading based on given prices and strategy parameters.


        Args:

            prices (list): List of prices for the given symbol.

            symbol (str): Symbol for which trading is being simulated.

            cash (float, optional): Initial cash amount. Defaults to 10000.

            commission (float, optional): Commission rate for each trade. Defaults to 0.01.


        Returns:

            float: Remaining cash after simulating the trading strategy.

        """

        prices = np.array(prices)
        positions = np.where((prices <= self.stoploss) | (prices >= self.stopprofit), 1, 0)
        cash += np.sum(prices[positions == 1])
        minimize_positions = np.where((prices / cash) >= self.top_percent_from_portfolio, 1, 0)
        cash -= np.sum(prices[minimize_positions == 1] * ((prices[minimize_positions == 1] / cash) - self.top_percent_from_portfolio))
        self.adx_rsi(backtest=True, symbol=symbol)
        return cash

    def optimize(self):

        """
        Optimizes the strategy using a genetic algorithm.

        """
        pass




    def adx_rsi(self, backtest=False, symbol=None):
        """
        Calculates the ADX-RSI strategy for a given company.

        Args:
            backtest (bool): Flag indicating whether to perform a backtest or not.
            symbol (str): Symbol of the company to calculate the strategy for.

        Returns:
            RecommendAction: An instance of the RecommendAction class containing the recommended actions.

        """
        actions = RecommendAction()
        with ThreadPoolExecutor() as executor:
            companies = list(database.get_companies())
            results = executor.map(lambda company: self.check_adx_rsi(company, backtest, symbol), companies)
            for result in results:
                if result:
                    if result[0] == 'buy':
                        actions.add_to_buy(result[1])
                    elif result[0] == 'sell':
                        actions.add_to_sell(result[1])

    def check_adx_rsi(self, company, backtest, symbol):
        if backtest:
            if company.symbol == symbol and symbol is not None:
                backtest_days = 30
                lookback_days = 14
                days = backtest_days + lookback_days
                df = company.get_df(Days=days)
        else:
            df = company.get_df()
        if 30 <= df['ADX'].iloc[-1] <= 50:
            if 30 <= df['RSI'].iloc[-1] <= 70:
                interval, state, prediction, probability = company.probability_of_returns('1h')
                Dinterval, Dstate, Dprediction, Dprobability = company.probability_of_returns('1h')
                if state == 'positive' and Dstate == 'positive':
                    if probability > 0.6 and Dprobability > 0.6:
                        if company.symbol not in actions.get_buy():
                            return ('buy', company.symbol)
            elif df['RSI'].iloc[-1] > 70:
                if company.symbol not in actions.get_sell():
                    return ('sell', company.symbol)
        return None
                    
# Define the evaluation function
def evaluate(strategy, prices):
    return strategy.simulate_trading(prices)

def anlayze(symbol):

    # This function analyzes a symbol and takes appropriate actions
    pass



PARAMETER_RANGES = {
    'top_percent_from_portfolio': (0.01, 0.2),
    'loss_percent': (0.01, 0.2),
    'profit_percent': (0.05, 1.5),
}

# Define the fitness function
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Register the genetic algorithm operators
toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)

def create_individual():
    return [random.uniform(*PARAMETER_RANGES[param]) for param in PARAMETER_RANGES]

toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def evaluate_individual(individual, prices):
    params = {param: value for param, value in zip(PARAMETER_RANGES.keys(), individual)}
    strategy = Strategy(avg_price=100, **params)
    final_cash = strategy.simulate_trading(prices, symbol="AAPL")
    return final_cash,  # Return a tuple

toolbox.register("evaluate", evaluate_individual, prices=[90, 95, 105, 110, 115, 120, 125, 130, 135, 140])

def optimize_strategy_ga(prices):
    # Initialize the population
    population = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda x: sum(x) / len(x))
    stats.register("min", min)
    stats.register("max", max)

    # Run the genetic algorithm
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=stats, halloffame=hof, verbose=True)
    
    # Return the best individual
    return hof[0]