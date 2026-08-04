"""
Model training — HMM (market regime) + RandomForest (next-close prediction).

Optional subsystem: requires MongoDB and is not used by the scanner or the
dashboard. See the caveat in predict_next_close about what these numbers mean.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import database
from scraping import get_exchange_time, get_stock_data

FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume',
                   'SMA150', 'EMA20', 'RSI', 'ATR', 'MACD']

# Fraction of the (chronological) history used for fitting.
TRAIN_FRACTION = 0.85


def _calculate_returns(series: pd.Series) -> np.ndarray:
    """Log returns, reshaped for HMM fitting."""
    returns = np.log(series / series.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    return returns.values.reshape(-1, 1)


def _get_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select the feature columns used by the regression model."""
    return df[[c for c in FEATURE_COLUMNS if c in df.columns]]


def _fetch(stock: str, interval: str, days: int = 365) -> pd.DataFrame | None:
    data = get_stock_data(stock, DAYS=days, interval=interval,
                          return_flags={'DF': True, 'INDICATORS': True})
    df = data.get('DF')
    return df if df is not None and not df.empty else None


# ---------------------------------------------------------------------------
# HMM  (market-regime model)
# ---------------------------------------------------------------------------

def state_labels(model) -> dict:
    """
    Map each HMM state index to 'negative' / 'neutral' / 'positive'.

    Gaussian HMM state indices are arbitrary — the fitting procedure does not
    order them. Reading `states[argmax]` off a fixed list, as this used to,
    assigned labels essentially at random. Rank the states by their mean return
    instead.
    """
    names = ['negative', 'neutral', 'positive']
    means = np.asarray(model.means_).ravel()
    order = np.argsort(means)                       # ascending mean return
    n = len(order)
    if n == len(names):
        return {int(state): names[rank] for rank, state in enumerate(order)}
    # Fall back to lowest/highest for an unexpected component count.
    return {int(state): (names[0] if rank == 0
                         else names[-1] if rank == n - 1
                         else names[1])
            for rank, state in enumerate(order)}


def train_hmm(stock: str, df: pd.DataFrame, interval: str) -> bool:
    """
    Fit a 3-state Gaussian HMM on log returns and persist it.

    States map to negative / neutral / positive regimes via state_labels().
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

    model = hmm.GaussianHMM(n_components=3, covariance_type='diag',
                            n_iter=1000, random_state=42)
    model.fit(returns)

    database.save_hmm_model(stock, interval, model, get_exchange_time())
    return True


def train_hmm_to_date(stock: str, last_update, interval: str) -> bool:
    """
    Re-fit the HMM for a symbol using data since last_update.

    hmmlearn has no partial_fit, so this refits from a fresh window rather than
    pretending to update incrementally.
    """
    today = get_exchange_time()
    try:
        days = (today - last_update).days
    except TypeError:
        days = 365
    if days < 1:
        return True

    # Always refit on a meaningful window, not just the few new bars.
    df = _fetch(stock, interval, days=max(days, 365))
    if df is None:
        return False
    return train_hmm(stock, df, interval)


# ---------------------------------------------------------------------------
# RandomForest  (next-close regression)
# ---------------------------------------------------------------------------

def pipeline(stock: str, interval: str) -> tuple:
    """
    Fetch data and split it CHRONOLOGICALLY into train/test.

    sklearn's train_test_split shuffles by default, which on a time series puts
    future bars in the training set and leaks the answer into the model. Every R²
    this used to report was inflated by that leak.
    """
    df = _fetch(stock, interval, days=365)
    if df is None:
        raise ValueError(f'No data for {stock}')

    df = df.copy()
    df['Future_Close'] = df['Close'].shift(-1)
    df = df.dropna()
    if len(df) < 30:
        raise ValueError(f'Not enough data for {stock}: {len(df)} rows')

    X = _get_features(df)
    y = df['Future_Close']

    split = int(len(df) * TRAIN_FRACTION)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def train_model(stock: str, interval: str) -> dict:
    """
    Train a RandomForest next-close regressor for one symbol and persist it.

    Fits a fresh model each time. Calling .fit() on a loaded RandomForest — as
    this used to — replaces every tree rather than extending it, so the old
    "retrain the master model" path silently threw away all prior training and
    left a model fitted on whichever symbol happened to be requested last.

    Returns:
        {'rmse': float, 'r2': float, 'naive_rmse': float}

    Read naive_rmse before believing r2. It is the error from simply predicting
    "tomorrow's close equals today's close". On daily equity data that baseline
    is very hard to beat, and an r2 near 1.0 here reflects price levels being
    autocorrelated, not any forecasting skill.
    """
    X_train, X_test, y_train, y_test = pipeline(stock, interval)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = float(r2_score(y_test, y_pred))

    # Persistence baseline: predict today's close for tomorrow.
    naive_rmse = float(np.sqrt(mean_squared_error(y_test, X_test['Close'])))

    database.save_model(stock, interval, model, get_exchange_time())

    return {'rmse': round(rmse, 4), 'r2': round(r2, 4),
            'naive_rmse': round(naive_rmse, 4)}
