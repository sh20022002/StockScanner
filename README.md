# SmarTraid
# making Algo Trading more accessible.

__trading system__

I created a system with a bot and a paper trading system that uses a microservices approuch, that bot contiosly chack s&p500 compenys and open postions for capitalazing on market flactuations of stocks using a models to predict prices and the probability of the prediction and uses different quant trading methods to find each stock its best strategy useing genetic algoritham optimazition the data is gatherd by scraping the models and costumer data are stored in a database servies and all system actions are rund in the server.


the system is seprated to 3 serviecies a database server the server and the client side.

1. the database is a mongo db instance build with:

    - stocks collection

    - models collection
    
2. the server is writen in python with:
    - scrapping module with scrapping methods for fetching compeny data, historical stock data, exchange rate and more..
  
    - the compeny moduale with the compeny class and methods.
  
    - the database with the database methods in pymongo.
  
    - main with the bot funcinalties thats runs in peralel with the server.
  
    - plots with the plotly stock ploting.
  
    - run uses subprocess to run the server and the bot stock scanner in peralel.
  
    - strategy with the strategies that are the prerequisites for the scanner conditions and Genetic Algoritham for optimazining the strategys per stock and adjast the parameters.
  
    - predictions fetchs the traind prediction models and makes a predictions for day and hour intervals for stock price change and probability useing Markov Model and Reggrasion Model.
  
    - tranining module trains the models for each intervals and for Markov Model also per stock symbol and if the stock is traind trains it up to date.
        
    - app with options to plot all stocks and all open postions and more.
