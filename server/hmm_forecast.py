"""
Inline HMM regime detection + forward price projection for the chart.

Unlike server/prediction.py (which requires MongoDB and a persisted,
periodically-refit model), this fits a fresh Gaussian HMM on whatever close
series the caller already has in hand and throws the model away afterwards —
the same "compute it fresh per request, no database" design server/strategy.py
and the RL pipeline already use for everything else in this dashboard.

Read the projection for what it actually is, not more: an HMM assumes the
future behaves like a mixture of the regimes it just fit on the past. It is
not a price prediction in the sense of "the stock will be here" — it is the
expected value of a stochastic process *if* the fitted regime dynamics hold,
with a band widening over the horizon to make the growing uncertainty visible
rather than implicit.
"""
import logging

import numpy as np

# A 3-state HMM on a few hundred noisy daily returns routinely doesn't fully
# converge within n_iter — expected, not a sign anything is wrong — but
# hmmlearn logs it at WARNING on every single fit, which would spam the
# server log on every chart load. ERROR+ only; real failures still surface
# (fit_and_project catches exceptions and returns None on those regardless).
logging.getLogger('hmmlearn').setLevel(logging.ERROR)

N_STATES = 3
MIN_OBSERVATIONS = 60          # below this, fitting is noise, not signal
STATE_LABELS = ('negative', 'neutral', 'positive')

# ~80% central interval on the projected cumulative log-return at each step.
_Z_80 = 1.2816


def _state_label_map(means: np.ndarray) -> dict:
    """
    Map each HMM state index to 'negative' / 'neutral' / 'positive'.

    State indices are arbitrary — hmmlearn does not order them — so states
    are ranked by mean return instead. Mirrors training.state_labels, kept
    separate rather than imported so this module has no dependency on the
    MongoDB-backed training/database modules.
    """
    order = np.argsort(means)
    n = len(order)
    if n == len(STATE_LABELS):
        return {int(state): STATE_LABELS[rank] for rank, state in enumerate(order)}
    return {int(state): (STATE_LABELS[0] if rank == 0
                         else STATE_LABELS[-1] if rank == n - 1
                         else STATE_LABELS[1])
            for rank, state in enumerate(order)}


def fit_and_project(close: np.ndarray, horizon_bars: int = 20) -> dict | None:
    """
    Fit a 3-state Gaussian HMM on log returns and project `horizon_bars`
    bars of expected price forward.

    Returns None if hmmlearn isn't installed or there's too little history
    to fit anything meaningful — callers should treat that as "unavailable",
    not an error; this is a nice-to-have overlay, not core to the chart.

    Returns:
        {
            'current_state': 'negative' | 'neutral' | 'positive',
            'state_probs': {'negative': .., 'neutral': .., 'positive': ..},
            'projection': [{'step': 1, 'price': .., 'lower': .., 'upper': ..}, ...],
        }
    """
    try:
        from hmmlearn import hmm
    except ImportError:
        return None

    close = np.asarray(close, dtype=float)
    returns = np.diff(np.log(close))
    returns = returns[np.isfinite(returns)]
    if len(returns) < MIN_OBSERVATIONS:
        return None

    obs = returns.reshape(-1, 1)
    model = hmm.GaussianHMM(n_components=N_STATES, covariance_type='diag',
                            n_iter=1000, random_state=42)
    try:
        model.fit(obs)
        state_probs_now = model.predict_proba(obs)[-1]
    except Exception:
        return None

    means = np.asarray(model.means_).ravel()
    variances = np.asarray(model.covars_).reshape(N_STATES, -1)[:, 0]
    variances = np.clip(variances, 0.0, None)   # a covariance is never negative; guard drift
    label_of = _state_label_map(means)

    last_price = float(close[-1])
    probs = state_probs_now.copy()
    transmat = model.transmat_
    price = last_price
    cum_var = 0.0
    projection = []
    for step in range(1, horizon_bars + 1):
        probs = probs @ transmat
        exp_return = float(np.dot(probs, means))
        exp_var = float(np.dot(probs, variances))
        price *= np.exp(exp_return)
        cum_var += exp_var
        band = _Z_80 * np.sqrt(cum_var)
        projection.append({
            'step':  step,
            'price': round(price, 4),
            'lower': round(price * np.exp(-band), 4),
            'upper': round(price * np.exp(band), 4),
        })

    current_state = int(np.argmax(state_probs_now))
    return {
        'current_state': label_of.get(current_state, 'neutral'),
        'state_probs': {label_of.get(i, str(i)): round(float(p), 4)
                        for i, p in enumerate(state_probs_now)},
        'projection': projection,
    }
