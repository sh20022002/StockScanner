"""
Tests for the long-term value screen (long_term_screen.py) and its
fundamentals fetch (scraping.get_fundamentals).
Run with: pytest server/tests -v
"""
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import scraping
from long_term_screen import _clamp01, _score, screen

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_frame(close: float, sma150: float, n: int = 5,
              sma100: float | None = None, sma200: float | None = None) -> pl.DataFrame:
    """A minimal OHLCV+SMA frame — only the columns screen() reads."""
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    cols = {
        'Datetime': dates,
        'Close':    [close] * n,
        'SMA150':   [sma150] * n,
    }
    if sma100 is not None:
        cols['SMA100'] = [sma100] * n
    if sma200 is not None:
        cols['SMA200'] = [sma200] * n
    return pl.DataFrame(cols)


# ---------------------------------------------------------------------------
# _score / _clamp01
# ---------------------------------------------------------------------------

class TestClamp01:
    def test_clamps_below_zero(self):
        assert _clamp01(-5.0) == 0.0

    def test_clamps_above_one(self):
        assert _clamp01(5.0) == 1.0

    def test_passes_through_midrange(self):
        assert _clamp01(0.4) == 0.4


class TestScore:
    def test_best_case_scores_near_100(self):
        # trend at the top of the band, near-zero P/E, strong earnings yield.
        s = _score(trend_ratio=1.5, pe=0.01, earnings_yield=0.20,
                   min_trend_ratio=1.0, max_trend_ratio=1.5, max_pe=35.0)
        assert s == pytest.approx(100.0, abs=0.5)

    def test_worst_case_scores_near_zero(self):
        # trend at the bottom of the band, P/E at the ceiling, no yield.
        s = _score(trend_ratio=1.0, pe=35.0, earnings_yield=0.0,
                   min_trend_ratio=1.0, max_trend_ratio=1.5, max_pe=35.0)
        assert s == pytest.approx(0.0, abs=0.5)

    def test_higher_trend_ratio_scores_higher(self):
        low = _score(1.1, 20.0, 0.04, 1.0, 1.5, 35.0)
        high = _score(1.4, 20.0, 0.04, 1.0, 1.5, 35.0)
        assert high > low

    def test_lower_pe_scores_higher(self):
        expensive = _score(1.2, 30.0, 0.04, 1.0, 1.5, 35.0)
        cheap = _score(1.2, 10.0, 0.04, 1.0, 1.5, 35.0)
        assert cheap > expensive

    def test_out_of_band_inputs_do_not_exceed_100(self):
        # pe below zero / yield far above the 8% cap would blow past their
        # sub-score's share without the clamp.
        s = _score(trend_ratio=2.0, pe=-10.0, earnings_yield=5.0,
                   min_trend_ratio=1.0, max_trend_ratio=1.5, max_pe=35.0)
        assert s <= 100.0


# ---------------------------------------------------------------------------
# screen()
# ---------------------------------------------------------------------------

class TestScreen:
    def test_symbol_passing_every_bar_is_included(self):
        stock_data = {'AAPL': make_frame(close=150.0, sma150=120.0)}
        with patch.object(scraping, 'get_fundamentals',
                          return_value={'AAPL': {'trailing_pe': 20.0, 'trailing_eps': 6.0}}):
            rows = screen(['AAPL'], stock_data=stock_data)

        assert len(rows) == 1
        r = rows[0]
        assert r['symbol'] == 'AAPL'
        assert r['price'] == 150.0
        assert r['sma150'] == 120.0
        assert r['trend_ratio'] == pytest.approx(1.25, abs=0.001)
        assert r['pe_ratio'] == 20.0
        assert r['eps'] == 6.0
        assert r['earnings_yield'] == pytest.approx(4.0, abs=0.01)   # 6/150 = 4%
        assert r['strategy'] == 'long_term_value'
        assert r['horizon'] == 'long_term'
        assert 0 <= r['score'] <= 100

    def test_trend_ratio_below_floor_excluded(self):
        # price below its SMA150 -> not an uptrend by this screen's definition.
        stock_data = {'WEAK': make_frame(close=90.0, sma150=120.0)}
        with patch.object(scraping, 'get_fundamentals',
                          return_value={'WEAK': {'trailing_pe': 10.0, 'trailing_eps': 5.0}}) as mock_fund:
            rows = screen(['WEAK'], stock_data=stock_data)
        assert rows == []
        # trend filter runs before the (network) fundamentals call.
        mock_fund.assert_not_called()

    def test_trend_ratio_above_ceiling_excluded(self):
        # overextended: more than 50% above its SMA150.
        stock_data = {'HOT': make_frame(close=200.0, sma150=100.0)}
        with patch.object(scraping, 'get_fundamentals', return_value={}) as mock_fund:
            rows = screen(['HOT'], stock_data=stock_data)
        assert rows == []
        mock_fund.assert_not_called()

    def test_missing_sma150_column_excluded(self):
        df = pl.DataFrame({
            'Datetime': [datetime(2024, 1, 1)],
            'Close':    [150.0],
        })
        with patch.object(scraping, 'get_fundamentals', return_value={}):
            rows = screen(['NOIND'], stock_data={'NOIND': df})
        assert rows == []

    def test_missing_fundamentals_excluded(self):
        stock_data = {'AAPL': make_frame(close=150.0, sma150=120.0)}
        with patch.object(scraping, 'get_fundamentals', return_value={}):
            rows = screen(['AAPL'], stock_data=stock_data)
        assert rows == []

    def test_non_positive_pe_excluded(self):
        stock_data = {'LOSS': make_frame(close=150.0, sma150=120.0)}
        with patch.object(scraping, 'get_fundamentals',
                          return_value={'LOSS': {'trailing_pe': -5.0, 'trailing_eps': -2.0}}):
            rows = screen(['LOSS'], stock_data=stock_data)
        assert rows == []

    def test_non_positive_eps_excluded(self):
        stock_data = {'BREAKEVEN': make_frame(close=150.0, sma150=120.0)}
        with patch.object(scraping, 'get_fundamentals',
                          return_value={'BREAKEVEN': {'trailing_pe': 20.0, 'trailing_eps': 0.0}}):
            rows = screen(['BREAKEVEN'], stock_data=stock_data)
        assert rows == []

    def test_pe_above_max_excluded(self):
        stock_data = {'PRICEY': make_frame(close=150.0, sma150=120.0)}
        with patch.object(scraping, 'get_fundamentals',
                          return_value={'PRICEY': {'trailing_pe': 60.0, 'trailing_eps': 2.5}}):
            rows = screen(['PRICEY'], stock_data=stock_data, max_pe=35.0)
        assert rows == []

    def test_results_sorted_by_score_descending(self):
        stock_data = {
            'CHEAP':  make_frame(close=150.0, sma150=130.0),   # trend_ratio ~1.15
            'PRICEY': make_frame(close=150.0, sma150=130.0),
        }
        fundamentals = {
            'CHEAP':  {'trailing_pe': 8.0,  'trailing_eps': 15.0},   # cheap + strong yield
            'PRICEY': {'trailing_pe': 30.0, 'trailing_eps': 5.0},    # near the P/E ceiling
        }
        with patch.object(scraping, 'get_fundamentals', return_value=fundamentals):
            rows = screen(['CHEAP', 'PRICEY'], stock_data=stock_data)

        assert [r['symbol'] for r in rows] == ['CHEAP', 'PRICEY']
        assert rows[0]['score'] > rows[1]['score']

    def test_empty_stock_data_short_circuits_without_fetching_fundamentals(self):
        with patch.object(scraping, 'get_fundamentals') as mock_fund:
            rows = screen(['AAPL'], stock_data={})
        assert rows == []
        mock_fund.assert_not_called()

    def test_no_stock_data_arg_falls_back_to_batch_download(self):
        stock_data = {'AAPL': make_frame(close=150.0, sma150=120.0)}
        with patch.object(scraping, 'batch_download', return_value=stock_data) as mock_dl, \
             patch.object(scraping, 'get_fundamentals',
                          return_value={'AAPL': {'trailing_pe': 20.0, 'trailing_eps': 6.0}}):
            rows = screen(['AAPL'])
        mock_dl.assert_called_once()
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# SMA100 / SMA200 (informational, non-gating)
# ---------------------------------------------------------------------------

class TestSma100Sma200:
    def test_included_as_context_when_present(self):
        stock_data = {'AAPL': make_frame(close=150.0, sma150=120.0, sma100=140.0, sma200=100.0)}
        with patch.object(scraping, 'get_fundamentals',
                          return_value={'AAPL': {'trailing_pe': 20.0, 'trailing_eps': 6.0}}):
            rows = screen(['AAPL'], stock_data=stock_data)

        assert len(rows) == 1
        r = rows[0]
        assert r['sma100'] == 140.0
        assert r['sma200'] == 100.0
        assert r['sma100_ratio'] == pytest.approx(150.0 / 140.0, abs=0.001)
        assert r['sma200_ratio'] == pytest.approx(150.0 / 100.0, abs=0.001)

    def test_none_when_columns_missing_does_not_exclude_the_symbol(self):
        # make_frame's default omits SMA100/SMA200 entirely -- the symbol
        # must still pass since only SMA150 gates inclusion.
        stock_data = {'AAPL': make_frame(close=150.0, sma150=120.0)}
        with patch.object(scraping, 'get_fundamentals',
                          return_value={'AAPL': {'trailing_pe': 20.0, 'trailing_eps': 6.0}}):
            rows = screen(['AAPL'], stock_data=stock_data)

        assert len(rows) == 1
        assert rows[0]['sma100'] is None
        assert rows[0]['sma200'] is None
        assert rows[0]['sma100_ratio'] is None
        assert rows[0]['sma200_ratio'] is None

    def test_does_not_affect_trend_gating_or_score(self):
        # Same SMA150/trend_ratio/P-E/EPS, only SMA100/SMA200 differ (one
        # frame has them, one doesn't) -- score and pass/fail must match.
        with_context = {'AAPL': make_frame(close=150.0, sma150=120.0, sma100=140.0, sma200=100.0)}
        without_context = {'MSFT': make_frame(close=150.0, sma150=120.0)}
        fundamentals = {
            'AAPL': {'trailing_pe': 20.0, 'trailing_eps': 6.0},
            'MSFT': {'trailing_pe': 20.0, 'trailing_eps': 6.0},
        }
        with patch.object(scraping, 'get_fundamentals', return_value=fundamentals):
            rows_a = screen(['AAPL'], stock_data=with_context)
            rows_b = screen(['MSFT'], stock_data=without_context)

        assert rows_a[0]['score'] == rows_b[0]['score']
        assert rows_a[0]['trend_ratio'] == rows_b[0]['trend_ratio']


# ---------------------------------------------------------------------------
# scraping.get_fundamentals
# ---------------------------------------------------------------------------

class TestGetFundamentals:
    def setup_method(self):
        scraping._fundamentals_cache.clear()

    def _mock_ticker(self, info: dict):
        ticker = MagicMock()
        ticker.info = info
        return ticker

    def test_fetches_pe_and_eps(self):
        with patch.object(scraping.yf, 'Ticker',
                          return_value=self._mock_ticker({'trailingPE': 22.5, 'trailingEps': 6.1,
                                                           'forwardPE': 20.0, 'forwardEps': 6.8})):
            result = scraping.get_fundamentals(['AAPL'], quiet=True)
        assert result['AAPL'] == {
            'trailing_pe': 22.5, 'forward_pe': 20.0,
            'trailing_eps': 6.1, 'forward_eps': 6.8,
        }

    def test_symbol_missing_both_fields_is_omitted(self):
        with patch.object(scraping.yf, 'Ticker', return_value=self._mock_ticker({})):
            result = scraping.get_fundamentals(['SHELL'], quiet=True)
        assert 'SHELL' not in result

    def test_failed_fetch_is_omitted_not_raised(self):
        class _BrokenTicker:
            @property
            def info(self):
                raise RuntimeError('boom')

        with patch.object(scraping.yf, 'Ticker', return_value=_BrokenTicker()):
            result = scraping.get_fundamentals(['BAD'], quiet=True, max_workers=1)
        assert result == {}

    def test_second_call_within_ttl_does_not_refetch(self):
        ticker_calls = []

        def _make_ticker(ysym):
            ticker_calls.append(ysym)
            return self._mock_ticker({'trailingPE': 15.0, 'trailingEps': 3.0})

        with patch.object(scraping.yf, 'Ticker', side_effect=_make_ticker):
            scraping.get_fundamentals(['AAPL'], quiet=True)
            scraping.get_fundamentals(['AAPL'], quiet=True)

        assert len(ticker_calls) == 1

    def test_multiple_symbols_all_returned(self):
        def _make_ticker(ysym):
            return self._mock_ticker({'trailingPE': 15.0, 'trailingEps': 3.0})

        with patch.object(scraping.yf, 'Ticker', side_effect=_make_ticker):
            result = scraping.get_fundamentals(['AAPL', 'MSFT', 'GOOG'], quiet=True)

        assert set(result.keys()) == {'AAPL', 'MSFT', 'GOOG'}
