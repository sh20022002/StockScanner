"""Use trained models to predict next close price and market regime."""
import pickle

import numpy as np

import scraping
import training
import database


# ---------------------------------------------------------------------------
# HMM — market regime
# ---------------------------------------------------------------------------

_STATES = ['negative', 'neutral', 'positive']


def predict_regime(symbol: str, interval: str) -> tuple[str, float]:
    """
    Return the predicted next market regime and its probability.

    Returns:
        (state, probability) e.g. ('positive', 0.82)
    """
    record = database.get_hmm_model(symbol, interval)
    if record is None:
        # No model yet — fetch data and train on the fly
        data = scraping.get_stock_data(symbol, interval=interval, DAYS=365,
                                       return_flags={'DF': True, 'INDICATORS': True})
        df = data.get('DF')
        if df is None or df.empty:
            return 'neutral', 0.0
        training.train_hmm(symbol, df, interval)
        record = database.get_hmm_model(symbol, interval)
        if record is None:
            return 'neutral', 0.0

    # Retrain if stale (>1 day old)
    last_update = record.get('last_update')
    if last_update is not None:
        today = scraping.get_exchange_time()
        if hasattr(today, 'replace'):
            today = today.replace(tzinfo=None)
        if hasattr(last_update, 'replace'):
            last_update = last_update.replace(tzinfo=None)
        if (today - last_update).days > 1:
            training.train_hmm_to_date(symbol, last_update, interval)
            record = database.get_hmm_model(symbol, interval)

    model = pickle.loads(record['model'])

    # Get the most recent return as input
    data = scraping.get_stock_data(symbol, interval=interval, DAYS=30,
                                   return_flags={'DF': True, 'INDICATORS': False})
    df = data.get('DF')
    if df is None or len(df) < 2:
        return 'neutral', 0.0

    current_return = float(np.log(df['Close'].iloc[-1] / df['Close'].iloc[-2]))
    obs = np.array([[current_return]])

    state_probs  = model.predict_proba(obs)[0]   # shape (n_states,)
    next_state   = int(np.argmax(state_probs))
    return _STATES[next_state], float(state_probs[next_state])


# ---------------------------------------------------------------------------
# RandomForest — next close price
# ---------------------------------------------------------------------------

def predict_next_close(symbol: str, interval: str = '1d') -> float | None:
    """
    Predict the next closing price for symbol using the trained RF model.

    Trains the model first if it doesn't exist.
    Returns predicted price or None on failure.
    """
    record = database.get_model(interval)
    if record is None:
        try:
            training.train_model(symbol, interval)
            record = database.get_model(interval)
        except Exception as e:
            print(f'predict_next_close: training failed for {symbol}: {e}')
            return None

    model = pickle.loads(record['model'])

    data = scraping.get_stock_data(symbol, interval=interval, DAYS=60,
                                   return_flags={'DF': True, 'INDICATORS': True})
    df = data.get('DF')
    if df is None or df.empty:
        return None

    feature_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume',
                                 'SMA150', 'EMA20', 'RSI', 'ATR', 'MACD']
                    if c in df.columns]
    last_row = df[feature_cols].iloc[[-1]]   # 2-D DataFrame, not Series

    try:
        prediction = model.predict(last_row)
        return float(prediction[0])
    except Exception as e:
        print(f'predict_next_close: inference failed for {symbol}: {e}')
        return None


# ---------------------------------------------------------------------------
# Combined signal
# ---------------------------------------------------------------------------

def full_prediction(symbol: str, interval: str = '1d') -> dict:
    """
    Return both regime and price prediction for a symbol.

    Returns:
        {
            'interval':    '1d',
            'regime':      'positive',
            'probability': 0.82,
            'next_close':  185.40,
        }
    """
    regime, prob = predict_regime(symbol, interval)
    next_close   = predict_next_close(symbol, interval)

    return {
        'interval':    interval,
        'regime':      regime,
        'probability': round(prob, 4),
        'next_close':  round(next_close, 2) if next_close is not None else None,
    }
