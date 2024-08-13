from hmmlearn import hmm
import pickle, os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from scraping import get_stock_data, get_exchange_time
import os



def train_hmm(stock ,df, interval):
    """
    Trains a Hidden Markov Model (HMM) using Gaussian emission distribution on the given stock data.

    Args:
        stock (str): The name of the stock.
        df (pandas.DataFrame): The dataframe containing the stock data.

    Returns:
        str: The name of the saved HMM model file.
    """
    file_system = os.getenv('file_system') # if uses file system true if mongodb false

    returns = calculate_hourly_returns(df['Close'])
    n_states = 3
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000)
    model.fit(returns)
    pickled_model = pickle.dump(model)
    database.save_hmm_model(stock, interval, pickled_model, get_exchange_time())
    return True


def train_hmm_to_date(stock, last_update, interval):
    """
    Trains a Hidden Markov Model (HMM) using Gaussian emission distribution on the given stock data up to a specific date.

    Args:
        stock (str): The name of the stock.
        last_update (datetime): The date up to which the model should be trained.

    Returns:
        None
    """
    
    today = get_exchange_time()
    difference = today - last_update
    days = difference.days
    df = get_stock_data(stock, DAYS=days, interval=interval)
    returns = calculate_hourly_returns(df['Close'])
    
    model = database.get_hmm_model(stock, interval)
    model = pickle.load(model['model']) 
    
    model.fit(returns)

    pickled_model = pickle.dump(model)
    database.update_hmm_model(stock, interval, pickled_model, today)
   
    print('model saved.')


def pipline(stock, interval):
    """
    Creates a data pipeline for training a regression model on stock data.

    Args:
        stock (str): The name of the stock.

    Returns:
        tuple: A tuple containing the training and testing data.
    """
    df = get_stock_data(stock, interval=interval, DAYS=365)
    df['Future_Close'] = df['Close'].shift(-1)
    df = df.dropna()
    x = df[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA150', 'EMA', 'ADX', 'KLASS_VOL', 'RSI']]
    y = df['Future_Close']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
    return X_train, X_test, y_train, y_test


def train_p(X_train, X_test, y_train, y_test, stock, interval):
    """
    Trains a regression model using the given training data and evaluates its performance on the testing data.

    Args:
        X_train (pandas.DataFrame): The features of the training data.
        X_test (pandas.DataFrame): The features of the testing data.
        y_train (pandas.Series): The target variable of the training data.
        y_test (pandas.Series): The target variable of the testing data.
        stock (str): The name of the stock.

    Returns:
        None
    """
    model = database.get_model(interval)
            # If model loading failed or didn't exist, create a new one
    model = pickle.load(model['model'])
    # If model loading failed or didn't exist, create a new one
    if model is None:
        model = RandomForestRegressor(random_state=42)

    # Train (or refit) the model
    model.fit(X_train, y_train)
    # print('Model training complete.')
    pickled_model = pickle.dump(model)
    database.update_model(stock, interval, pickled_model, get_exchange_time())
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    # print(f'Mean Squared Error: {mse}')
    rmse = np.sqrt(mse)
    # print(f'Root Mean Squared Error: {rmse}')
    r2 = r2_score(y_test, y_pred)
    # print(f'R-squared: {r2}')
    