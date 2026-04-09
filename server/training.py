"""Model training — HMM (market regime) + RandomForest (next-close prediction)."""
import pickle
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from scraping import get_stock_data, get_exchange_time
import database


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _calculate_returns(series: pd.Series) -> np.ndarray:
    """Log returns reshaped for HMM fitting."""
    returns = np.log(series / series.shift(1)).dropna()
    return returns.values.reshape(-1, 1)


def _get_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select the feature columns used by the regression model."""
    cols = ['Open', 'High', 'Low', 'Close', 'Volume',
            'SMA150', 'EMA20', 'RSI', 'ATR', 'MACD']
    available = [c for c in cols if c in df.columns]
    return df[available]


# ---------------------------------------------------------------------------
# HMM  (market-regime model)
# ---------------------------------------------------------------------------

def train_hmm(stock: str, df: pd.DataFrame, interval: str) -> bool:
    """
    Fit a 3-state Gaussian HMM on log returns and persist it to the database.

    States map to: negative / neutral / positive market regimes.
    """
    try:
        from hmmlearn import hmm
    except ImportError:
        print('hmmlearn not installed — skipping HMM training.')
        return False

    returns = _calculate_returns(df['Close'])
    if len(returns) < 30:
        print(f'Not enough data to train HMM for {stock}.')
        return False

    model = hmm.GaussianHMM(n_components=3, covariance_type='diag', n_iter=1000)
    model.fit(returns)

    pickled = pickle.dumps(model)   # bytes — was incorrectly pickle.dump(model) before
    database.save_hmm_model(stock, interval, pickled, get_exchange_time())
    return True


def train_hmm_to_date(stock: str, last_update, interval: str) -> bool:
    """Re-fit existing HMM with data since last_update."""
    try:
        from hmmlearn import hmm
    except ImportError:
        return False

    today = get_exchange_time()
    days  = (today - last_update).days
    if days < 1:
        return True  # nothing to do

    data = get_stock_data(stock, DAYS=days, interval=interval,
                          return_flags={'DF': True, 'INDICATORS': True})
    df = data.get('DF')
    if df is None or df.empty:
        return False

    returns  = _calculate_returns(df['Close'])
    record   = database.get_hmm_model(stock, interval)
    if record is None:
        return train_hmm(stock, df, interval)

    model = pickle.loads(record['model'])   # loads from bytes
    model.fit(returns)

    pickled = pickle.dumps(model)
    database.update_hmm_model(stock, interval, pickled, today)
    return True


# ---------------------------------------------------------------------------
# RandomForest  (next-close regression)
# ---------------------------------------------------------------------------

def pipeline(stock: str, interval: str) -> tuple:
    """Fetch data and split into train/test for regression."""
    data = get_stock_data(stock, interval=interval, DAYS=365,
                          return_flags={'DF': True, 'INDICATORS': True})
    df = data.get('DF')
    if df is None or df.empty:
        raise ValueError(f'No data for {stock}')

    df = df.copy()
    df['Future_Close'] = df['Close'].shift(-1)
    df = df.dropna()

    X = _get_features(df)
    y = df['Future_Close']
    return train_test_split(X, y, test_size=0.15, random_state=42)


def train_model(stock: str, interval: str) -> dict:
    """
    Train (or retrain) a RandomForest regressor for next-close prediction.
    Returns performance metrics.
    """
    X_train, X_test, y_train, y_test = pipeline(stock, interval)

    record = database.get_model(interval)
    if record is not None:
        model = pickle.loads(record['model'])
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse  = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2   = float(r2_score(y_test, y_pred))

    pickled = pickle.dumps(model)
    database.update_model(stock, interval, pickled, get_exchange_time())

    return {'rmse': round(rmse, 4), 'r2': round(r2, 4)}
