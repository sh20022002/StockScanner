"""
Use trained models to predict market regime and next close price.

Optional subsystem: requires MongoDB (see .env.example) and is not wired into
the scanner or the dashboard.
"""
import numpy as np

import database
import scraping
import training

# How many recent returns to feed the HMM when inferring the current regime.
_REGIME_WINDOW = 60


def predict_regime(symbol: str, interval: str) -> tuple[str, float]:
    """
    Return the current market regime and its posterior probability.

    Returns:
        (state, probability) e.g. ('positive', 0.82)
    """
    record = database.get_hmm_model(symbol, interval)
    if record is None:
        df = _history(symbol, interval, days=365)
        if df is None:
            return 'neutral', 0.0
        training.train_hmm(symbol, df, interval)
        record = database.get_hmm_model(symbol, interval)
        if record is None:
            return 'neutral', 0.0

    # Refit if stale.
    last_update = record.get('last_update')
    if last_update is not None:
        today = _naive(scraping.get_exchange_time())
        try:
            if (today - _naive(last_update)).days > 1:
                training.train_hmm_to_date(symbol, _naive(last_update), interval)
                record = database.get_hmm_model(symbol, interval) or record
        except TypeError:
            pass

    model = database.load_model(record)
    if model is None:
        return 'neutral', 0.0

    df = _history(symbol, interval, days=180, indicators=False)
    if df is None or len(df) < 3:
        return 'neutral', 0.0

    # Feed the recent return SEQUENCE. Scoring a single observation, as this used
    # to, throws away the state history that makes an HMM an HMM.
    returns = np.log(df['Close'] / df['Close'].shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return 'neutral', 0.0
    obs = returns.values[-_REGIME_WINDOW:].reshape(-1, 1)

    try:
        state_probs = model.predict_proba(obs)[-1]
    except Exception as e:
        print(f'predict_regime: inference failed for {symbol}: {e}')
        return 'neutral', 0.0

    state = int(np.argmax(state_probs))
    # State indices are arbitrary until ranked by mean return.
    label = training.state_labels(model).get(state, 'neutral')
    return label, float(state_probs[state])


def predict_next_close(symbol: str, interval: str = '1d') -> float | None:
    """
    Predict the next closing price for a symbol.

    Trains the model first if there isn't one for this symbol and interval.
    Returns the predicted price, or None on failure.

    Treat the output with suspicion. The features are raw price levels, so the
    model largely learns "tomorrow ≈ today"; compare train_model()'s naive_rmse
    against its rmse before reading any skill into it.
    """
    record = database.get_model(symbol, interval)
    if record is None:
        try:
            training.train_model(symbol, interval)
            record = database.get_model(symbol, interval)
        except Exception as e:
            print(f'predict_next_close: training failed for {symbol}: {e}')
            return None
    if record is None:
        return None

    try:
        model = database.load_model(record)
    except Exception as e:
        print(f'predict_next_close: could not load model for {symbol}: {e}')
        return None
    if model is None:
        return None

    df = _history(symbol, interval, days=120)
    if df is None:
        return None

    feature_cols = [c for c in training.FEATURE_COLUMNS if c in df.columns]
    last_row = df[feature_cols].iloc[[-1]]      # 2-D, as sklearn expects

    try:
        return float(model.predict(last_row)[0])
    except Exception as e:
        print(f'predict_next_close: inference failed for {symbol}: {e}')
        return None


def full_prediction(symbol: str, interval: str = '1d') -> dict:
    """
    Return both regime and price prediction for a symbol.

    Returns:
        {'interval', 'regime', 'probability', 'next_close'}
    """
    regime, prob = predict_regime(symbol, interval)
    next_close   = predict_next_close(symbol, interval)

    return {
        'interval':    interval,
        'regime':      regime,
        'probability': round(prob, 4),
        'next_close':  round(next_close, 2) if next_close is not None else None,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _history(symbol: str, interval: str, days: int, indicators: bool = True):
    data = scraping.get_stock_data(
        symbol, interval=interval, DAYS=days,
        return_flags={'DF': True, 'INDICATORS': indicators})
    df = data.get('DF')
    return df if df is not None and not df.empty else None


def _naive(dt):
    """Drop tzinfo so naive and aware timestamps can be subtracted."""
    return dt.replace(tzinfo=None) if getattr(dt, 'tzinfo', None) else dt
