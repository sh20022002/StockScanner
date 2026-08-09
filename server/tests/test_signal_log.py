"""
Tests for signal_log.py's dedupe/persistence logic.
Run with: pytest server/tests -v
"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import signal_log


def _sig(symbol='AAPL', direction='BUY', strategy='macd', time='2026-08-05 00:00:00'):
    return {'symbol': symbol, 'direction': direction, 'strategy': strategy, 'time': time}


class TestDedupeKey:
    def test_same_symbol_direction_strategy_day_is_one_key(self):
        a = _sig()
        b = _sig()
        assert signal_log._dedupe_key(a) == signal_log._dedupe_key(b)

    def test_different_strategy_is_a_different_key(self):
        # A technical BUY and a same-day long-term-value BUY for the same
        # symbol are different information, not the same signal twice.
        technical = _sig(strategy='macd')
        long_term = _sig(strategy='long_term_value')
        assert signal_log._dedupe_key(technical) != signal_log._dedupe_key(long_term)

    def test_different_direction_is_a_different_key(self):
        assert signal_log._dedupe_key(_sig(direction='BUY')) != signal_log._dedupe_key(_sig(direction='SELL'))

    def test_different_day_is_a_different_key(self):
        a = _sig(time='2026-08-05 00:00:00')
        b = _sig(time='2026-08-06 00:00:00')
        assert signal_log._dedupe_key(a) != signal_log._dedupe_key(b)

    def test_time_only_uses_the_date_part(self):
        a = _sig(time='2026-08-05 09:30:00')
        b = _sig(time='2026-08-05 16:00:00')
        assert signal_log._dedupe_key(a) == signal_log._dedupe_key(b)


class TestAddSignals:
    def setup_method(self, monkeypatch=None):
        self._original_load = signal_log.load_signals
        self._original_write = signal_log._write_atomic
        self._store = []
        signal_log.load_signals = lambda: list(self._store)
        signal_log._write_atomic = lambda signals: self._store.__setitem__(slice(None), signals)

    def teardown_method(self):
        signal_log.load_signals = self._original_load
        signal_log._write_atomic = self._original_write

    def test_empty_entries_returns_empty_without_touching_the_log(self):
        assert signal_log.add_signals([]) == []
        assert self._store == []

    def test_new_signal_is_added_and_returned_as_fresh(self):
        fresh = signal_log.add_signals([_sig()])
        assert len(fresh) == 1
        assert self._store == fresh

    def test_duplicate_signal_is_dropped(self):
        signal_log.add_signals([_sig()])
        fresh = signal_log.add_signals([_sig()])
        assert fresh == []
        assert len(self._store) == 1

    def test_same_symbol_different_strategy_both_persist(self):
        signal_log.add_signals([_sig(strategy='macd')])
        fresh = signal_log.add_signals([_sig(strategy='long_term_value')])
        assert len(fresh) == 1
        assert len(self._store) == 2


class TestReplaceWithRetries:
    # A real incident: os.replace failed once with WinError 5 (another
    # process/AV/indexer briefly holding the destination open) and the
    # exception aborted an entire scan cycle's worth of signal writes.

    def test_succeeds_first_try_without_sleeping(self):
        with patch.object(signal_log.os, 'replace') as mock_replace, \
             patch.object(signal_log.time, 'sleep') as mock_sleep:
            signal_log._replace_with_retries('src', 'dst')
        mock_replace.assert_called_once_with('src', 'dst')
        mock_sleep.assert_not_called()

    def test_recovers_after_a_transient_failure(self):
        with patch.object(signal_log.os, 'replace',
                          side_effect=[OSError('Access is denied'), None]) as mock_replace, \
             patch.object(signal_log.time, 'sleep') as mock_sleep:
            signal_log._replace_with_retries('src', 'dst')
        assert mock_replace.call_count == 2
        mock_sleep.assert_called_once()

    def test_raises_after_exhausting_retries(self):
        with patch.object(signal_log.os, 'replace', side_effect=OSError('Access is denied')), \
             patch.object(signal_log.time, 'sleep'):
            try:
                signal_log._replace_with_retries('src', 'dst', retries=3)
                assert False, 'expected OSError to propagate'
            except OSError:
                pass
