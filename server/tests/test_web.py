"""
Tests for the pure logic behind the web app's HTTP layer.

FastAPI's TestClient needs an httpx test-client package this environment
doesn't have installed, so these test the underlying functions directly rather
than pulling in a new dependency for one feature's worth of endpoint tests.

Run with: pytest server/tests -v
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from web.app import call_performance


class TestCallPerformance:
    def test_buy_that_went_up_is_a_correct_call(self):
        price_change, call_return = call_performance(
            signal_price=100.0, current_price=110.0, direction='BUY')
        assert price_change == 10.0
        assert call_return == 10.0          # positive: the BUY call was right

    def test_buy_that_went_down_is_a_wrong_call(self):
        price_change, call_return = call_performance(
            signal_price=100.0, current_price=90.0, direction='BUY')
        assert price_change == -10.0
        assert call_return == -10.0         # negative: the BUY call was wrong

    def test_sell_that_went_down_is_a_correct_call(self):
        """The sign flip is the whole point of this function."""
        price_change, call_return = call_performance(
            signal_price=100.0, current_price=90.0, direction='SELL')
        assert price_change == -10.0
        assert call_return == 10.0          # positive: the SELL call was right

    def test_sell_that_went_up_is_a_wrong_call(self):
        price_change, call_return = call_performance(
            signal_price=100.0, current_price=110.0, direction='SELL')
        assert price_change == 10.0
        assert call_return == -10.0         # negative: the SELL call was wrong

    def test_unchanged_price_is_zero_either_direction(self):
        assert call_performance(100.0, 100.0, 'BUY') == (0.0, 0.0)
        assert call_performance(100.0, 100.0, 'SELL') == (0.0, 0.0)

    def test_missing_current_price_returns_none(self):
        assert call_performance(100.0, None, 'BUY') == (None, None)

    def test_missing_signal_price_returns_none(self):
        assert call_performance(None, 100.0, 'BUY') == (None, None)

    def test_zero_current_price_treated_as_missing(self):
        # current_stock_price() returns None on failure, never 0 — but a
        # falsy 0.0 must not slip through and divide/produce a bogus 0% move.
        assert call_performance(100.0, 0.0, 'BUY') == (None, None)
