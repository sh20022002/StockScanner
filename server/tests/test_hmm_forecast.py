"""
Tests for the inline HMM regime/projection helper (server/hmm_forecast.py).

Run with: pytest server/tests -v
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import hmm_forecast


def _make_close(n=200, seed=0, drift=0.001, vol=0.01):
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(loc=drift, scale=vol, size=n)
    return 100.0 * np.exp(np.cumsum(log_returns))


class TestFitAndProject:
    def test_too_little_history_returns_none(self):
        close = _make_close(n=30)   # fewer than MIN_OBSERVATIONS returns
        assert hmm_forecast.fit_and_project(close) is None

    def test_hmmlearn_unavailable_returns_none(self, monkeypatch):
        monkeypatch.setitem(sys.modules, 'hmmlearn', None)
        close = _make_close(n=200)
        assert hmm_forecast.fit_and_project(close) is None

    def test_happy_path_shape(self):
        close = _make_close(n=200)
        out = hmm_forecast.fit_and_project(close, horizon_bars=15)
        assert out is not None
        assert out['current_state'] in hmm_forecast.STATE_LABELS
        assert set(out['state_probs']) == set(hmm_forecast.STATE_LABELS)
        assert abs(sum(out['state_probs'].values()) - 1.0) < 1e-6
        assert len(out['projection']) == 15
        for i, p in enumerate(out['projection'], start=1):
            assert p['step'] == i
            assert p['lower'] <= p['price'] <= p['upper']

    def test_projection_continues_from_last_price(self):
        close = _make_close(n=200)
        out = hmm_forecast.fit_and_project(close, horizon_bars=5)
        last_price = float(close[-1])
        first_step = out['projection'][0]
        # One step out shouldn't have drifted wildly from the last real price.
        assert first_step['price'] == pytest.approx(last_price, rel=0.5)

    def test_band_widens_over_the_horizon(self):
        close = _make_close(n=200)
        out = hmm_forecast.fit_and_project(close, horizon_bars=20)
        first, last = out['projection'][0], out['projection'][-1]
        first_spread = first['upper'] - first['lower']
        last_spread = last['upper'] - last['lower']
        assert last_spread > first_spread

    def test_default_horizon_is_twenty_steps(self):
        close = _make_close(n=200)
        out = hmm_forecast.fit_and_project(close)
        assert len(out['projection']) == 20

    def test_flat_state_probs_sum_to_one_at_every_step_is_not_required_but_current_is(self):
        # Sanity check on the entry point specifically, not an internal detail.
        close = _make_close(n=200, drift=-0.002)   # downward-drifting series
        out = hmm_forecast.fit_and_project(close)
        assert out['current_state'] in hmm_forecast.STATE_LABELS
