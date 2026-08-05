"""
Tests for the pure logic behind the web app's HTTP layer.

FastAPI's TestClient needs an httpx test-client package this environment
doesn't have installed, so these test the underlying functions directly rather
than pulling in a new dependency for one feature's worth of endpoint tests.

Run with: pytest server/tests -v
"""
import base64
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from web.app import call_performance, sector_summary
from web.auth import verify_basic_auth


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


class TestSectorSummary:
    SECTOR_MAP = {'AAPL': 'Technology', 'MSFT': 'Technology', 'JPM': 'Financial Services'}

    def _sig(self, symbol, direction, time='2026-08-05 00:00:00', excess_roi=0.0):
        return {'symbol': symbol, 'direction': direction, 'time': time, 'excess_roi': excess_roi}

    def test_groups_by_sector_and_counts_directions(self):
        signals = [
            self._sig('AAPL', 'BUY'), self._sig('MSFT', 'SELL'), self._sig('JPM', 'BUY'),
        ]
        rows = sector_summary(signals, self.SECTOR_MAP, '2026-08-05')
        by_sector = {r['sector']: r for r in rows}
        assert by_sector['Technology']['total'] == 2
        assert by_sector['Technology']['buys'] == 1
        assert by_sector['Technology']['sells'] == 1
        assert by_sector['Financial Services']['total'] == 1
        assert by_sector['Financial Services']['buys'] == 1

    def test_excludes_signals_from_other_days(self):
        signals = [
            self._sig('AAPL', 'BUY', time='2026-08-05 00:00:00'),
            self._sig('MSFT', 'BUY', time='2026-08-04 00:00:00'),
        ]
        rows = sector_summary(signals, self.SECTOR_MAP, '2026-08-05')
        assert sum(r['total'] for r in rows) == 1

    def test_symbol_missing_from_sector_map_becomes_unknown_not_dropped(self):
        signals = [self._sig('ZZZZ', 'BUY')]
        rows = sector_summary(signals, self.SECTOR_MAP, '2026-08-05')
        assert len(rows) == 1
        assert rows[0]['sector'] == 'Unknown'
        assert rows[0]['total'] == 1

    def test_avg_excess_is_the_mean_within_the_sector(self):
        signals = [
            self._sig('AAPL', 'BUY', excess_roi=4.0),
            self._sig('MSFT', 'BUY', excess_roi=-2.0),
        ]
        rows = sector_summary(signals, self.SECTOR_MAP, '2026-08-05')
        assert rows[0]['sector'] == 'Technology'
        assert rows[0]['avg_excess'] == 1.0

    def test_beat_bench_counts_positive_excess_only(self):
        signals = [
            self._sig('AAPL', 'BUY', excess_roi=4.0),
            self._sig('MSFT', 'BUY', excess_roi=-2.0),
            self._sig('JPM', 'BUY', excess_roi=0.0),
        ]
        rows = sector_summary(signals, self.SECTOR_MAP, '2026-08-05')
        by_sector = {r['sector']: r for r in rows}
        assert by_sector['Technology']['beat_bench'] == 1

    def test_missing_excess_roi_treated_as_zero(self):
        signals = [{'symbol': 'AAPL', 'direction': 'BUY', 'time': '2026-08-05 00:00:00'}]
        rows = sector_summary(signals, self.SECTOR_MAP, '2026-08-05')
        assert rows[0]['avg_excess'] == 0.0

    def test_sorted_by_total_descending(self):
        signals = [
            self._sig('AAPL', 'BUY'), self._sig('MSFT', 'BUY'), self._sig('JPM', 'BUY'),
        ]
        rows = sector_summary(signals, self.SECTOR_MAP, '2026-08-05')
        assert rows[0]['sector'] == 'Technology'    # 2 signals, ahead of Financial Services' 1

    def test_no_signals_today_returns_empty(self):
        assert sector_summary([], self.SECTOR_MAP, '2026-08-05') == []


def _basic_header(user, password):
    token = base64.b64encode(f'{user}:{password}'.encode()).decode()
    return f'Basic {token}'


class TestVerifyBasicAuth:
    def test_correct_credentials_pass(self):
        header = _basic_header('alice', 'hunter2')
        assert verify_basic_auth(header, 'alice', 'hunter2') is True

    def test_wrong_password_fails(self):
        header = _basic_header('alice', 'wrong')
        assert verify_basic_auth(header, 'alice', 'hunter2') is False

    def test_wrong_username_fails(self):
        header = _basic_header('mallory', 'hunter2')
        assert verify_basic_auth(header, 'alice', 'hunter2') is False

    def test_missing_header_fails(self):
        assert verify_basic_auth(None, 'alice', 'hunter2') is False

    def test_non_basic_scheme_fails(self):
        assert verify_basic_auth('Bearer sometoken', 'alice', 'hunter2') is False

    def test_malformed_base64_fails_closed_not_open(self):
        assert verify_basic_auth('Basic not-valid-base64!!!', 'alice', 'hunter2') is False

    def test_missing_colon_separator_fails(self):
        token = base64.b64encode(b'no-colon-here').decode()
        assert verify_basic_auth(f'Basic {token}', 'alice', 'hunter2') is False

    def test_empty_password_in_header_fails_against_real_password(self):
        header = _basic_header('alice', '')
        assert verify_basic_auth(header, 'alice', 'hunter2') is False

    def test_username_with_embedded_colon_is_not_confused_with_password(self):
        # 'partition' on the first colon: "a:b:c" -> user="a", password="b:c".
        token = base64.b64encode(b'alice:pass:word').decode()
        assert verify_basic_auth(f'Basic {token}', 'alice', 'pass:word') is True
