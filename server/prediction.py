'''uses the model to predict stock vulnerability'''
import os
import scraping
import training, database
import numpy as np
from hmmlearn import hmm
import pickle
import pandas as pd


def stock_and_tecnical(stock, interval='1h'):
    '''
    Retrieves stock data and adds technical indicators to the dataframe.

    Parameters:
    - stock (str): The stock symbol.
    - interval (str): The time interval for the stock data. Default is '1h'.

    Returns:
    - df (pandas.DataFrame): The dataframe containing stock data with added technical indicators.
    '''
    df = scraping.get_stock_data(stock, interval=interval, DAYS=365)
    return df



def predict_next_state_and_probabilities( current_return, stock, interval):
    '''
    Predicts the next state and probabilities of a stock return using a trained model.

    Parameters:
    - path_to_model (str): The path to the trained model file.
    - current_return (list): The current return value as a list.

    Returns:
    - None

    '''
    model = database.get_hmm_model(stock, interval)
    last_updat = model['last_update']
    today = scraping.get_exchange_time()
    time = today - last_updat
    if time.days > 1:
        training.train_hmm_to_date()
    model = pickle.load(model['model'])
    current_return = np.array(current_return).reshape(-1, 1)
        
    state_probs = model.predict_proba(current_return)
    next_state = np.argmax(state_probs)
    next_state_probs = state_probs[0]
    states = ['negative', 'neutral', 'positive']
    # print(f"Predicted state for the next hour: {states[next_state]}")
    state = states[next_state]
    probability = next_state_probs[next_state]
    # print(f"Probability of negative return: {next_state_probs[0]:.2f}")
    # print(f"Probability of neutral return: {next_state_probs[1]:.2f}")
    # print(f"Probability of positive return: {next_state_probs[2]:.2f}")
    return(state, probability)


def predict_next_close(stock, df):
    """
    Predicts the next closing price for a given stock using a trained model.

    Args:
        stock (str): The stock symbol or identifier.
        df (pandas.DataFrame): The input data for prediction.

    Returns:
        float: The predicted closing price.

    Raises:
        FileNotFoundError: If the model file is not found.

    """
    X_train, X_test, y_train, y_test = training.pipline(stock)

    training.train_p(X_train, X_test, y_train, y_test, stock)

    current = None
    last_row = df.iloc[-1]
    last_row = last_row.drop('Datetime')
    
    df = pd.DataFrame(last_row).T
    
    prediction = model.predict(df) 
    return prediction

def probability_of_returns(self, interval):
    """
    Calculate the probability of future stock returns using the HMM model.
    """
    # needs a function to refit the hmm model
    df = self.get_df(interval=interval)
    df = add_all(df)
    current_return = df['Close'][0] - df['Close'][1]
    hmm = database.get_hmm_model(self.symbol, interval=interval)
    if(hmm == None):
        model = train_hmm(self.symbol, df)
    else:
        model = hmm
    state, probability = predict_next_state_and_probabilities(current_return, self.symbol)

    prediction = predict_next_close(self.symbol, self.get_df(interval=interval))    
    return interval, state, prediction, probability
