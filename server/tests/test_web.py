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
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import scanner
import signal_log
import web.app as webapp
from web.app import (call_performance, sector_summary, _most_recent_signal_day,
                     _project_future_times, _should_scan_this_iteration,
                     get_signals_summary)
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

    def test_symbols_lists_the_distinct_symbols_in_each_sector(self):
        signals = [
            self._sig('AAPL', 'BUY'), self._sig('MSFT', 'SELL'), self._sig('JPM', 'BUY'),
        ]
        rows = sector_summary(signals, self.SECTOR_MAP, '2026-08-05')
        by_sector = {r['sector']: r for r in rows}
        assert set(by_sector['Technology']['symbols']) == {'AAPL', 'MSFT'}
        assert by_sector['Financial Services']['symbols'] == ['JPM']

    def test_symbols_deduplicated_when_a_symbol_fires_twice(self):
        signals = [self._sig('AAPL', 'BUY'), self._sig('AAPL', 'SELL')]
        rows = sector_summary(signals, self.SECTOR_MAP, '2026-08-05')
        assert rows[0]['symbols'] == ['AAPL']

    def test_no_signals_today_returns_empty(self):
        assert sector_summary([], self.SECTOR_MAP, '2026-08-05') == []


class TestGetSignalsSummary:
    # Real incident: a long_term_value signal has 'excess_roi' present but
    # explicitly None (it's a pass/fail fundamentals screen, no ROI figures
    # — see web.app._lt_row_to_signal). `.get('excess_roi', 0)` only falls
    # back on a MISSING key, not an explicit None, so the first long-term
    # signal in the log crashed this endpoint's sum() with a TypeError.
    def _sig(self, symbol='AAPL', direction='BUY', excess_roi=0.0):
        return {'symbol': symbol, 'direction': direction, 'excess_roi': excess_roi}

    def test_long_term_value_signal_with_none_excess_roi_does_not_crash(self):
        signals = [
            self._sig('AAPL', excess_roi=4.0),
            self._sig('MU', excess_roi=None),   # long_term_value shape
        ]
        with patch.object(signal_log, 'load_signals', return_value=signals):
            result = get_signals_summary()
        assert result['total'] == 2
        assert result['avg_excess'] == 2.0   # None treated as 0, not skipped
        assert result['excess_hist'] == [4.0, 0]

    def test_all_none_excess_roi_averages_to_zero(self):
        signals = [self._sig('AAPL', excess_roi=None), self._sig('MU', excess_roi=None)]
        with patch.object(signal_log, 'load_signals', return_value=signals):
            result = get_signals_summary()
        assert result['avg_excess'] == 0.0

    def test_empty_signals_returns_zeroed_summary(self):
        with patch.object(signal_log, 'load_signals', return_value=[]):
            result = get_signals_summary()
        assert result == {'total': 0, 'buys': 0, 'sells': 0, 'beat_bench': 0,
                          'avg_excess': 0, 'excess_hist': []}


class TestMostRecentSignalDay:
    def test_no_signals_returns_none(self):
        assert _most_recent_signal_day([]) is None

    def test_signals_missing_time_ignored(self):
        assert _most_recent_signal_day([{'symbol': 'AAPL'}]) is None

    def test_single_day_returned(self):
        signals = [{'time': '2026-08-05 00:00:00'}]
        assert _most_recent_signal_day(signals) == '2026-08-05'

    def test_picks_the_latest_of_several_days(self):
        signals = [
            {'time': '2026-08-01 00:00:00'},
            {'time': '2026-08-05 00:00:00'},
            {'time': '2026-08-03 00:00:00'},
        ]
        assert _most_recent_signal_day(signals) == '2026-08-05'


class TestProjectFutureTimes:
    def test_too_few_timestamps_returns_empty(self):
        assert _project_future_times([datetime(2026, 1, 1)], 10) == []

    def test_zero_or_negative_horizon_returns_empty(self):
        times = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(5)]
        assert _project_future_times(times, 0) == []
        assert _project_future_times(times, -3) == []

    def test_returns_one_timestamp_per_horizon_step(self):
        times = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(5)]
        out = _project_future_times(times, 7)
        assert len(out) == 7

    def test_daily_spacing_steps_by_one_day(self):
        times = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(10)]
        out = _project_future_times(times, 3)
        assert out[1] - out[0] == 86400
        assert out[2] - out[1] == 86400

    def test_first_step_follows_last_known_timestamp(self):
        times = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(10)]
        last_ts = int(times[-1].replace(tzinfo=timezone.utc).timestamp())
        out = _project_future_times(times, 1)
        assert out[0] == last_ts + 86400

    def test_median_gap_is_not_skewed_by_one_weekend_jump(self):
        # 9 daily gaps + one 3-day (weekend) gap -> median stays 1 day.
        times = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(10)]
        times.append(times[-1] + timedelta(days=3))
        out = _project_future_times(times, 2)
        assert out[1] - out[0] == 86400

    def test_timestamps_strictly_increasing(self):
        times = [datetime(2026, 1, 1) + timedelta(hours=i) for i in range(20)]
        out = _project_future_times(times, 10)
        assert all(b > a for a, b in zip(out, out[1:]))


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


class TestShouldScanThisIteration:
    def test_forced_first_scan_ignores_market_hours(self):
        with patch.object(scanner, 'should_scan_now', return_value=False) as mock:
            assert _should_scan_this_iteration(force_first=True, timeframe='1h') is True
        # Short-circuited by `or` — force_first=True must never even need to
        # ask should_scan_now, not just happen to override its answer.
        mock.assert_not_called()

    def test_non_first_scan_respects_market_hours_open(self):
        with patch.object(scanner, 'should_scan_now', return_value=True):
            assert _should_scan_this_iteration(force_first=False, timeframe='1h') is True

    def test_non_first_scan_respects_market_hours_closed(self):
        with patch.object(scanner, 'should_scan_now', return_value=False):
            assert _should_scan_this_iteration(force_first=False, timeframe='1h') is False

    def test_daily_timeframe_always_scans_regardless_of_force_first(self):
        # should_scan_now already returns True unconditionally for non-intraday
        # timeframes — force_first shouldn't change that, just add to it.
        assert _should_scan_this_iteration(force_first=False, timeframe='1d') is True


class TestLtRowToSignal:
    def test_sets_buy_direction_and_daily_timeframe(self):
        row = {'symbol': 'AAPL', 'strategy': 'long_term_value', 'horizon': 'long_term', 'score': 50.0}
        sig = webapp._lt_row_to_signal(row)
        assert sig['direction'] == 'BUY'
        assert sig['timeframe'] == '1d'
        assert sig['symbol'] == 'AAPL'
        assert sig['strategy'] == 'long_term_value'

    def test_roi_fields_are_none_not_faked_zero(self):
        sig = webapp._lt_row_to_signal({'symbol': 'AAPL'})
        for key in ('roi', 'benchmark_roi', 'excess_roi', 'win_rate', 'trades', 'avg_roi', 'avg_win_rate'):
            assert sig[key] is None


class TestRunScanCycle:
    def setup_method(self):
        self._orig_symbols = webapp.state.symbols
        self._orig_timeframe = webapp.state.timeframe
        webapp.state.symbols = ['AAPL', 'MSFT']
        webapp.state.timeframe = '1d'

    def teardown_method(self):
        webapp.state.symbols = self._orig_symbols
        webapp.state.timeframe = self._orig_timeframe

    def test_pushes_technical_and_long_term_signals_incrementally(self):
        technical = {'symbol': 'AAPL', 'direction': 'BUY', 'strategy': 'macd',
                    'time': '2026-08-05 00:00:00'}
        lt_row = {'symbol': 'MSFT', 'time': '2026-08-05 00:00:00', 'price': 100.0,
                 'strategy': 'long_term_value', 'horizon': 'long_term'}

        def fake_scan(symbols, **kwargs):
            kwargs['on_result'](technical, None)   # df unused when rl_live is None
            return [technical], {}

        pushed = []
        with patch.object(webapp, 'rl_live', None), \
             patch.object(webapp.scanner, 'scan', side_effect=fake_scan), \
             patch.object(webapp.long_term_screen, 'screen', return_value=[lt_row]):
            technical_count, lt_count = webapp._run_scan_cycle(pushed.append)

        assert (technical_count, lt_count) == (1, 1)
        assert pushed[0] is technical
        assert pushed[1]['symbol'] == 'MSFT'
        assert pushed[1]['direction'] == 'BUY'
        assert pushed[1]['strategy'] == 'long_term_value'

    def test_on_signal_failure_for_one_long_term_row_does_not_drop_the_rest(self):
        # A real incident: on_signal (signal_log write) failed for one row
        # and the whole scan cycle aborted, losing every row after it and
        # never reaching the last_scan_at/last_scan_count bookkeeping.
        lt_rows = [
            {'symbol': 'AAPL', 'strategy': 'long_term_value', 'time': '2026-08-05 00:00:00'},
            {'symbol': 'MSFT', 'strategy': 'long_term_value', 'time': '2026-08-05 00:00:00'},
            {'symbol': 'GOOG', 'strategy': 'long_term_value', 'time': '2026-08-05 00:00:00'},
        ]
        pushed = []

        def flaky_on_signal(entry):
            if entry['symbol'] == 'MSFT':
                raise OSError('Access is denied')
            pushed.append(entry)

        with patch.object(webapp, 'rl_live', None), \
             patch.object(webapp.scanner, 'scan', return_value=([], {})), \
             patch.object(webapp.long_term_screen, 'screen', return_value=lt_rows):
            technical_count, lt_count = webapp._run_scan_cycle(flaky_on_signal)

        # The cycle completes and reports the full long-term count even
        # though one push failed...
        assert (technical_count, lt_count) == (0, 3)
        # ...and the two rows around the failure still made it through.
        assert [e['symbol'] for e in pushed] == ['AAPL', 'GOOG']

    def test_long_term_screen_reuses_stock_data_on_a_daily_scanner(self):
        webapp.state.timeframe = '1d'
        stock_data = {'AAPL': 'fake-df'}
        captured = {}

        with patch.object(webapp, 'rl_live', None), \
             patch.object(webapp.scanner, 'scan', return_value=([], stock_data)), \
             patch.object(webapp.long_term_screen, 'screen',
                          side_effect=lambda symbols, **kw: captured.update(kw) or []):
            webapp._run_scan_cycle(lambda e: None)

        assert captured['stock_data'] is stock_data

    def test_long_term_screen_does_not_reuse_frames_on_a_non_daily_scanner(self):
        # SMA150 off weekly/monthly bars is a different, wrong indicator —
        # see long_term_screen's module docstring — so it must fetch its own
        # daily data rather than reusing the scanner's own-timeframe frames.
        webapp.state.timeframe = '1wk'
        stock_data = {'AAPL': 'fake-df'}
        captured = {}

        with patch.object(webapp, 'rl_live', None), \
             patch.object(webapp.scanner, 'scan', return_value=([], stock_data)), \
             patch.object(webapp.long_term_screen, 'screen',
                          side_effect=lambda symbols, **kw: captured.update(kw) or []):
            webapp._run_scan_cycle(lambda e: None)

        assert captured['stock_data'] is None
        assert captured['timeframe'] == '1d'

    def test_long_term_screen_failure_does_not_break_the_technical_scan(self):
        technical = {'symbol': 'AAPL', 'direction': 'BUY', 'strategy': 'macd',
                    'time': '2026-08-05 00:00:00'}

        def fake_scan(symbols, **kwargs):
            kwargs['on_result'](technical, None)
            return [technical], {}

        pushed = []
        with patch.object(webapp, 'rl_live', None), \
             patch.object(webapp.scanner, 'scan', side_effect=fake_scan), \
             patch.object(webapp.long_term_screen, 'screen', side_effect=RuntimeError('boom')):
            technical_count, lt_count = webapp._run_scan_cycle(pushed.append)

        assert (technical_count, lt_count) == (1, 0)
        assert pushed == [technical]

    def test_rl_annotates_in_place_per_symbol_before_push(self):
        technical = {'symbol': 'AAPL', 'direction': 'BUY', 'strategy': 'macd',
                    'time': '2026-08-05 00:00:00'}
        df = 'fake-df'

        def fake_scan(symbols, **kwargs):
            kwargs['on_result'](technical, df)
            return [technical], {}

        fake_rl = MagicMock()
        pushed = []
        with patch.object(webapp, 'rl_live', fake_rl), \
             patch.object(webapp.scanner, 'scan', side_effect=fake_scan), \
             patch.object(webapp.long_term_screen, 'screen', return_value=[]):
            webapp._run_scan_cycle(pushed.append)

        fake_rl.annotate.assert_called_once_with([technical], {'AAPL': df})
        assert pushed == [technical]

    def test_rl_annotate_failure_does_not_block_the_push(self):
        technical = {'symbol': 'AAPL', 'direction': 'BUY', 'strategy': 'macd',
                    'time': '2026-08-05 00:00:00'}

        def fake_scan(symbols, **kwargs):
            kwargs['on_result'](technical, None)
            return [technical], {}

        fake_rl = MagicMock()
        fake_rl.annotate.side_effect = RuntimeError('checkpoint corrupt')
        pushed = []
        with patch.object(webapp, 'rl_live', fake_rl), \
             patch.object(webapp.scanner, 'scan', side_effect=fake_scan), \
             patch.object(webapp.long_term_screen, 'screen', return_value=[]):
            technical_count, _ = webapp._run_scan_cycle(pushed.append)

        assert technical_count == 1
        assert pushed == [technical]
