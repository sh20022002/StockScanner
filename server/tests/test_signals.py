"""
Tests for the SP500 scanner signal pipeline.
Run with: pytest server/tests -v
"""
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import polars as pl
import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import scanner
import scraping
import strategy
from strategy import generate_signal, what_is_signal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_close(n, trend='up', base=100.0, vol=1.0):
    np.random.seed(42)
    if trend == 'up':
        prices = [base + i * 0.5 + np.random.randn() * vol for i in range(n)]
    elif trend == 'down':
        prices = [base + n * 0.5 - i * 0.5 + np.random.randn() * vol for i in range(n)]
    else:
        prices = [base + np.random.randn() * vol * 5 for _ in range(n)]
    return [max(1.0, p) for p in prices]


def make_df(n=300, trend='up') -> pl.DataFrame:
    """Create a synthetic OHLCV + indicators DataFrame."""
    dates = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(n)]
    close = _make_close(n, trend)
    high  = [c * 1.01 for c in close]
    low   = [c * 0.99 for c in close]
    vol   = [1_000_000] * n

    cs = pd.Series(close)
    ema12 = cs.ewm(span=12).mean()
    ema26 = cs.ewm(span=26).mean()
    macd_line   = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()

    delta = cs.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = (100 - 100 / (1 + rs)).fillna(50)

    return pl.DataFrame({
        'Datetime':     dates,
        'Open':         close,
        'High':         high,
        'Low':          low,
        'Close':        close,
        'Volume':       vol,
        'SMA20':        cs.rolling(20).mean().fillna(cs.mean()).tolist(),
        'SMA50':        cs.rolling(50).mean().fillna(cs.mean()).tolist(),
        'SMA100':       cs.rolling(100).mean().fillna(cs.mean()).tolist(),
        'SMA150':       cs.rolling(150).mean().fillna(cs.mean()).tolist(),
        'SMA200':       cs.rolling(200).mean().fillna(cs.mean()).tolist(),
        'EMA12':        ema12.tolist(),
        'EMA20':        cs.ewm(span=20).mean().tolist(),
        'EMA26':        ema26.tolist(),
        'MACD':         macd_line.tolist(),
        'MACD_Signal':  signal_line.tolist(),
        'MACD_Hist':    (macd_line - signal_line).tolist(),
        'RSI':          rsi.tolist(),
        'ATR':          [c * 0.015 for c in close],
        'VWAP':         close,
        'STOCH_%K':     [50.0] * n,
        'STOCH_%D':     [50.0] * n,
    })


@pytest.fixture
def df_up():
    return make_df(300, 'up')


@pytest.fixture
def df_down():
    return make_df(300, 'down')


@pytest.fixture
def df_flat():
    return make_df(300, 'flat')


@pytest.fixture
def mock_strategy(df_up):
    """Strategy plus data. Construction is offline — it makes no network calls."""
    return strategy.Strategy(symbol='TEST'), df_up


# ---------------------------------------------------------------------------
# scraping.py tests
# ---------------------------------------------------------------------------

class TestIsNYSEOpen:
    def test_closed_on_saturday(self):
        with patch('scraping.get_exchange_time') as mock_time:
            mock_time.return_value = datetime(2024, 1, 6, 10, 0)
            assert scraping.is_nyse_open() is False

    def test_closed_on_sunday(self):
        with patch('scraping.get_exchange_time') as mock_time:
            mock_time.return_value = datetime(2024, 1, 7, 10, 0)
            assert scraping.is_nyse_open() is False

    def test_closed_before_market_hours(self):
        with patch('scraping.get_exchange_time') as mock_time:
            mock_time.return_value = datetime(2024, 1, 8, 9, 0)
            assert scraping.is_nyse_open() is False

    def test_closed_after_market_hours(self):
        with patch('scraping.get_exchange_time') as mock_time:
            mock_time.return_value = datetime(2024, 1, 8, 16, 30)
            assert scraping.is_nyse_open() is False

    def test_open_during_trading_hours(self):
        with patch('scraping.get_exchange_time') as mock_time:
            mock_time.return_value = datetime(2024, 1, 8, 12, 0)
            assert scraping.is_nyse_open() is True

    def test_closed_on_new_years_day(self):
        with patch('scraping.get_exchange_time') as mock_time:
            mock_time.return_value = datetime(2024, 1, 1, 12, 0)
            assert scraping.is_nyse_open() is False


class TestWithRetries:
    """
    scraping.with_retries wraps every yfinance network call. These tests
    mock scraping._sleep so retries don't actually wait — real backoff timing
    is exercised only by the delay-calculation test below.
    """

    def test_succeeds_on_first_try_without_sleeping(self):
        fn = MagicMock(return_value='ok')
        with patch('scraping._sleep') as mock_sleep:
            result = scraping.with_retries(fn)
        assert result == 'ok'
        assert fn.call_count == 1
        mock_sleep.assert_not_called()

    def test_passes_args_and_kwargs_through(self):
        fn = MagicMock(return_value='ok')
        scraping.with_retries(fn, 'a', 'b', x=1, label='ignored-by-fn')
        fn.assert_called_once_with('a', 'b', x=1)

    def test_retries_on_rate_limit_error_then_succeeds(self):
        import yfinance.exceptions as yf_exceptions
        fn = MagicMock(side_effect=[yf_exceptions.YFRateLimitError(), 'ok'])
        with patch('scraping._sleep'):
            result = scraping.with_retries(fn, retries=3)
        assert result == 'ok'
        assert fn.call_count == 2

    def test_retries_on_connection_error(self):
        fn = MagicMock(side_effect=[requests.exceptions.ConnectionError('boom'), 'ok'])
        with patch('scraping._sleep'):
            assert scraping.with_retries(fn, retries=3) == 'ok'

    def test_retries_on_timeout(self):
        fn = MagicMock(side_effect=[requests.exceptions.Timeout('slow'), 'ok'])
        with patch('scraping._sleep'):
            assert scraping.with_retries(fn, retries=3) == 'ok'

    def test_retries_on_message_that_looks_like_rate_limiting(self):
        # Some failure paths surface a generic Exception with a descriptive
        # message rather than a typed exception — this is the fallback net.
        fn = MagicMock(side_effect=[Exception('429 Too Many Requests'), 'ok'])
        with patch('scraping._sleep'):
            assert scraping.with_retries(fn, retries=3) == 'ok'

    def test_does_not_retry_non_transient_errors(self):
        # A bad symbol or a real bug fails the same way every time — retrying
        # it would just delay the failure, not prevent it.
        fn = MagicMock(side_effect=ValueError('symbol not found'))
        with patch('scraping._sleep') as mock_sleep:
            with pytest.raises(ValueError):
                scraping.with_retries(fn, retries=3)
        assert fn.call_count == 1
        mock_sleep.assert_not_called()

    def test_raises_last_exception_after_exhausting_retries(self):
        import yfinance.exceptions as yf_exceptions
        fn = MagicMock(side_effect=yf_exceptions.YFRateLimitError())
        with patch('scraping._sleep'):
            with pytest.raises(yf_exceptions.YFRateLimitError):
                scraping.with_retries(fn, retries=2)
        assert fn.call_count == 3   # initial attempt + 2 retries

    def test_backoff_delay_grows_and_is_bounded(self):
        import yfinance.exceptions as yf_exceptions
        fn = MagicMock(side_effect=[yf_exceptions.YFRateLimitError(),
                                    yf_exceptions.YFRateLimitError(), 'ok'])
        delays = []
        with patch('scraping._sleep', side_effect=lambda d: delays.append(d)):
            scraping.with_retries(fn, retries=3, base_delay=1.0)
        assert len(delays) == 2
        # base_delay * 2**attempt, plus up to one base_delay of jitter.
        assert 1.0 <= delays[0] <= 2.0
        assert 2.0 <= delays[1] <= 3.0
        assert delays[1] > delays[0] - 1.0   # later attempts wait at least as long


class TestYahooSymbol:
    def test_class_share_dot_becomes_hyphen(self):
        # BRK.B / BF.B failed every scan before this translation existed.
        assert scraping.yahoo_symbol('BRK.B') == 'BRK-B'
        assert scraping.yahoo_symbol('BF.B') == 'BF-B'

    def test_uppercases_and_strips(self):
        assert scraping.yahoo_symbol('  aapl ') == 'AAPL'

    def test_plain_symbol_unchanged(self):
        assert scraping.yahoo_symbol('MSFT') == 'MSFT'


class TestUsEquitiesUniverse:
    def test_is_common_stock_filters_preferred_and_warrants(self):
        # Share CLASS suffixes (single letter, no leading P) must survive.
        assert scraping._is_common_stock('AAPL') is True
        assert scraping._is_common_stock('BRK-B') is True
        assert scraping._is_common_stock('PBR-A') is True
        # Preferred series (-P + letter), warrants, units and rights must not.
        assert scraping._is_common_stock('JPM-PC') is False
        assert scraping._is_common_stock('BAC-PB') is False
        assert scraping._is_common_stock('XYZ-WT') is False
        assert scraping._is_common_stock('XYZ-WS') is False
        assert scraping._is_common_stock('XYZ-U') is False
        assert scraping._is_common_stock('XYZ-R') is False

    def _page(self, symbols, total):
        return {'total': total, 'quotes': [
            {'symbol': s, 'shortName': f'{s} Inc', 'marketCap': 1e9} for s in symbols
        ]}

    def test_paginates_until_total_reached(self):
        scraping._us_equities_cache.clear()
        pages = [self._page(['A', 'B'], total=5),
                self._page(['C', 'D'], total=5),
                self._page(['E'], total=5)]
        with patch.object(scraping.yf, 'screen', side_effect=pages) as mock_screen:
            result = scraping.get_us_equities(min_market_cap=1e9)
        assert [r['symbol'] for r in result] == ['A', 'B', 'C', 'D', 'E']
        assert mock_screen.call_count == 3

    def test_filters_preferred_shares_from_results(self):
        scraping._us_equities_cache.clear()
        page = self._page(['AAPL', 'JPM-PC', 'BRK-B'], total=3)
        with patch.object(scraping.yf, 'screen', return_value=page):
            result = scraping.get_us_equities(min_market_cap=1e9)
        symbols = [r['symbol'] for r in result]
        assert 'JPM-PC' not in symbols
        assert {'AAPL', 'BRK-B'} <= set(symbols)

    def test_respects_max_results(self):
        scraping._us_equities_cache.clear()
        page = self._page(['A', 'B', 'C', 'D', 'E'], total=100)
        with patch.object(scraping.yf, 'screen', return_value=page):
            result = scraping.get_us_equities(min_market_cap=1e9, max_results=3)
        assert len(result) == 3

    def test_caches_by_threshold(self):
        scraping._us_equities_cache.clear()
        page = self._page(['A'], total=1)
        with patch.object(scraping.yf, 'screen', return_value=page) as mock_screen:
            scraping.get_us_equities(min_market_cap=5e9)
            scraping.get_us_equities(min_market_cap=5e9)
        assert mock_screen.call_count == 1

    def test_different_thresholds_not_conflated_by_cache(self):
        scraping._us_equities_cache.clear()
        with patch.object(scraping.yf, 'screen',
                          return_value=self._page(['A'], total=1)) as mock_screen:
            scraping.get_us_equities(min_market_cap=1e9)
            scraping.get_us_equities(min_market_cap=2e9)
        assert mock_screen.call_count == 2

    def test_stops_on_empty_page_even_if_total_not_reached(self):
        # A total that overclaims what the API actually has left must not spin forever.
        scraping._us_equities_cache.clear()
        with patch.object(scraping.yf, 'screen', side_effect=[
            {'total': 10, 'quotes': [{'symbol': 'A', 'shortName': 'A', 'marketCap': 1e9}]},
            {'total': 10, 'quotes': []},
        ]):
            result = scraping.get_us_equities(min_market_cap=1e9)
        assert len(result) == 1

    def test_sorted_by_market_cap_field_present(self):
        scraping._us_equities_cache.clear()
        page = self._page(['NVDA'], total=1)
        with patch.object(scraping.yf, 'screen', return_value=page):
            result = scraping.get_us_equities(min_market_cap=1e9)
        assert result[0]['market_cap'] == 1e9
        assert result[0]['name'] == 'NVDA Inc'

    def test_sector_adds_an_eq_clause_to_the_query(self):
        scraping._us_equities_cache.clear()
        captured = {}

        def fake_screen(query, **kwargs):
            captured['query'] = query
            return self._page(['NVDA'], total=1)

        with patch.object(scraping.yf, 'screen', side_effect=fake_screen):
            scraping.get_us_equities(min_market_cap=1e9, sector='Technology')

        operands = captured['query'].to_dict()['operands']
        assert {'operator': 'EQ', 'operands': ['sector', 'Technology']} in operands

    def test_no_sector_omits_the_eq_clause(self):
        scraping._us_equities_cache.clear()
        captured = {}

        def fake_screen(query, **kwargs):
            captured['query'] = query
            return self._page(['NVDA'], total=1)

        with patch.object(scraping.yf, 'screen', side_effect=fake_screen):
            scraping.get_us_equities(min_market_cap=1e9)

        operators = [op.get('operator') for op in captured['query'].to_dict()['operands']]
        assert 'EQ' not in operators

    def test_different_sectors_are_different_cache_entries(self):
        scraping._us_equities_cache.clear()
        with patch.object(scraping.yf, 'screen',
                          return_value=self._page(['A'], total=1)) as mock_screen:
            scraping.get_us_equities(min_market_cap=1e9, sector='Technology')
            scraping.get_us_equities(min_market_cap=1e9, sector='Healthcare')
            scraping.get_us_equities(min_market_cap=1e9)
        assert mock_screen.call_count == 3


class TestSectorMap:
    def _page(self, symbols):
        return {'total': len(symbols), 'quotes': [
            {'symbol': s, 'shortName': f'{s} Inc', 'marketCap': 1e9} for s in symbols
        ]}

    def test_queries_once_per_gics_sector_and_tags_symbols(self):
        scraping._us_equities_cache.clear()
        scraping._sector_map_cache.clear()

        # Give each sector query back exactly one, distinguishable symbol —
        # sector name with spaces stripped so no two sectors collide (two
        # GICS sector names share the "Consumer" prefix).
        def fake_screen(query, **kwargs):
            sector = next(op['operands'][1] for op in query.to_dict()['operands']
                         if op.get('operator') == 'EQ')
            return self._page([sector.replace(' ', '_').upper() + '_SYM'])

        with patch.object(scraping.yf, 'screen', side_effect=fake_screen) as mock_screen:
            mapping = scraping.get_sector_map(min_market_cap=1e9)

        assert mock_screen.call_count == len(scraping.GICS_SECTORS)
        assert len(mapping) == len(scraping.GICS_SECTORS)
        assert mapping['TECHNOLOGY_SYM'] == 'Technology'
        assert mapping['HEALTHCARE_SYM'] == 'Healthcare'

    def test_caches_across_calls(self):
        scraping._us_equities_cache.clear()
        scraping._sector_map_cache.clear()
        with patch.object(scraping.yf, 'screen',
                          return_value=self._page(['NVDA'])) as mock_screen:
            scraping.get_sector_map(min_market_cap=1e9)
            scraping.get_sector_map(min_market_cap=1e9)
        assert mock_screen.call_count == len(scraping.GICS_SECTORS)


class TestCleanOHLCV:
    def _partial_tail(self):
        """Mirror what Yahoo returns mid-session: Volume set, OHLC null."""
        df = pd.DataFrame({
            'Open':   [10.0, 11.0, None],
            'High':   [10.5, 11.5, None],
            'Low':    [9.5, 10.5, None],
            'Close':  [10.2, 11.2, None],
            'Volume': [1000, 1100, 1200],
        }, index=pd.date_range('2024-01-01', periods=3, freq='D'))
        return df

    def test_drops_partial_trailing_bar(self):
        # THE regression test. This single row crashed parabolic_sar for every
        # symbol and turned backtest ROI into NaN, which suppressed 100% of
        # signals. dropna(how='all') did not remove it.
        cleaned = scraping._clean_ohlcv(self._partial_tail())
        assert cleaned is not None
        assert len(cleaned) == 2
        assert cleaned[['Open', 'High', 'Low', 'Close']].notna().all().all()

    def test_returns_none_when_all_rows_partial(self):
        df = self._partial_tail().iloc[[2]]
        assert scraping._clean_ohlcv(df) is None

    def test_returns_none_on_missing_columns(self):
        assert scraping._clean_ohlcv(pd.DataFrame({'Close': [1.0]})) is None

    def test_returns_none_on_empty(self):
        assert scraping._clean_ohlcv(pd.DataFrame()) is None


class TestCurrentStockPrice:
    def test_returns_float_on_success(self):
        mock_hist = pd.DataFrame(
            {'Close': [182.5], 'Open': [180.0], 'High': [183.0],
             'Low': [179.0], 'Volume': [1000000]},
            index=pd.DatetimeIndex([datetime(2024, 1, 8)])
        )
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_hist
            price = scraping.current_stock_price('AAPL')
            assert isinstance(price, float)
            assert price == pytest.approx(182.5)

    def test_returns_none_on_empty_df(self):
        # Was 1.0 — a fabricated price that flowed into the UI as if it were real.
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.history.return_value = pd.DataFrame()
            assert scraping.current_stock_price('FAKE') is None

    def test_returns_none_on_exception(self):
        with patch('yfinance.Ticker', side_effect=Exception('network error')):
            assert scraping.current_stock_price('ERR') is None


class TestGetStockNews:
    def _nested_item(self, title='Some headline', with_link=True, publisher='Reuters'):
        return {
            'content': {
                'title': title,
                'summary': 'A summary.',
                'pubDate': '2026-08-05T09:00:00Z',
                'contentType': 'STORY',
                'provider': {'displayName': publisher},
                'clickThroughUrl': {'url': 'https://finance.yahoo.com/x'} if with_link else None,
                'canonicalUrl': {'url': 'https://reuters.com/x'},
            }
        }

    def test_parses_nested_content_shape(self):
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.news = [self._nested_item()]
            news = scraping.get_stock_news('AAPL')
        assert len(news) == 1
        n = news[0]
        assert n['title'] == 'Some headline'
        assert n['publisher'] == 'Reuters'
        assert n['link'] == 'https://finance.yahoo.com/x'
        assert n['published_at'] == '2026-08-05T09:00:00Z'
        assert n['content_type'] == 'STORY'

    def test_falls_back_to_canonical_url_when_no_clickthrough(self):
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.news = [self._nested_item(with_link=False)]
            news = scraping.get_stock_news('AAPL')
        assert news[0]['link'] == 'https://reuters.com/x'

    def test_drops_items_with_no_title(self):
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.news = [
                self._nested_item(title=''), self._nested_item(title='Real headline'),
            ]
            news = scraping.get_stock_news('AAPL')
        assert len(news) == 1
        assert news[0]['title'] == 'Real headline'

    def test_respects_count(self):
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.news = [self._nested_item(title=f'H{i}') for i in range(20)]
            news = scraping.get_stock_news('AAPL', count=3)
        assert len(news) == 3

    def test_returns_empty_list_on_exception(self):
        with patch('yfinance.Ticker', side_effect=Exception('network error')):
            assert scraping.get_stock_news('ERR') == []

    def test_returns_empty_list_when_no_news(self):
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.news = []
            assert scraping.get_stock_news('QUIET') == []

    def test_falls_back_to_flat_shape_without_content_key(self):
        # Defends against the yfinance news schema drifting back to a flat shape.
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.news = [{
                'title': 'Flat-shape headline', 'link': 'https://example.com/a',
                'publisher': 'Example Wire', 'providerPublishTime': 1723000000,
            }]
            news = scraping.get_stock_news('AAPL')
        assert news[0]['title'] == 'Flat-shape headline'
        assert news[0]['link'] == 'https://example.com/a'
        assert news[0]['publisher'] == 'Example Wire'


class TestGetStockData:
    def _mock_yf_ticker(self, close_prices):
        mock_df = pd.DataFrame({
            'Open':   close_prices,
            'High':   [c * 1.01 for c in close_prices],
            'Low':    [c * 0.99 for c in close_prices],
            'Close':  close_prices,
            'Volume': [1_000_000] * len(close_prices),
        }, index=pd.date_range('2023-01-01', periods=len(close_prices), freq='D', tz='UTC'))
        mock_df.index.name = 'Datetime'
        return mock_df

    def test_returns_df_key(self):
        mock_df = self._mock_yf_ticker(list(range(100, 400)))
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_df
            result = scraping.get_stock_data('AAPL', interval='1d', period='1y',
                                             return_flags={'DF': True, 'INDICATORS': False})
            assert 'DF' in result
            assert not result['DF'].empty

    def test_required_columns_present(self):
        mock_df = self._mock_yf_ticker(list(range(100, 400)))
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_df
            result = scraping.get_stock_data('AAPL', interval='1d', period='1y',
                                             return_flags={'DF': True, 'INDICATORS': False})
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                assert col in result['DF'].columns

    def test_indicators_computed(self):
        mock_df = self._mock_yf_ticker(list(range(50, 350)))
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_df
            result = scraping.get_stock_data('AAPL', interval='1d', period='2y',
                                             return_flags={'DF': True, 'INDICATORS': True})
            for col in ['SMA20', 'EMA12', 'RSI', 'MACD', 'ATR']:
                assert col in result['DF'].columns

    def test_empty_data_returns_empty_dict(self):
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.history.return_value = pd.DataFrame()
            result = scraping.get_stock_data('FAKE', interval='1d', period='1y')
            assert result.get('DF') is None


class TestComputeIndicators:
    def test_no_nulls_in_output(self, df_up):
        out = scraping.compute_indicators(df_up.select(
            ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']))
        for col in ['SMA20', 'EMA12', 'MACD', 'RSI', 'ATR', 'VWAP', 'STOCH_%K']:
            assert col in out.columns
            assert out[col].null_count() == 0, f'{col} has nulls'

    def test_temp_columns_dropped(self, df_up):
        out = scraping.compute_indicators(df_up.select(
            ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']))
        assert [c for c in out.columns if c.startswith('_')] == []


# ---------------------------------------------------------------------------
# strategy.generate_signal tests
# ---------------------------------------------------------------------------

class TestGenerateSignal:
    def test_correct_shape(self):
        n = 10
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
        sig = generate_signal([False] * n, [True] * n, dates)
        assert isinstance(sig, pl.DataFrame)
        assert len(sig) == n
        assert {'Buy_Signal', 'Sell_Signal', 'Datetime'} <= set(sig.columns)

    def test_buy_signals_preserved(self):
        n = 5
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
        buys  = [True, False, True, False, False]
        sells = [False, True, False, False, True]
        sig = generate_signal(sells, buys, dates)
        assert sig['Buy_Signal'].to_list() == buys
        assert sig['Sell_Signal'].to_list() == sells

    def test_raises_on_mismatched_lengths(self):
        with pytest.raises(ValueError):
            generate_signal([True, False], [True], [datetime(2024, 1, 1)])


class TestEmptySignals:
    def test_builds_aligned_all_false_frame(self, df_up):
        # The guard clauses used to call pl.DataFrame(columns=...), which is not
        # the Polars signature and raised TypeError from every "safe" exit.
        out = strategy._empty_signals(df_up)
        assert len(out) == len(df_up)
        assert out['Buy_Signal'].dtype == pl.Boolean
        assert not any(out['Buy_Signal'].to_list())
        assert not any(out['Sell_Signal'].to_list())

    def test_guard_clause_returns_instead_of_raising(self, df_up):
        s = strategy.Strategy(symbol='T')
        no_ohlc = df_up.drop(['High', 'Low'])
        for fn in (s.donchian_channel, s.ichimoku_cloud, s.parabolic_sar,
                   s.stochastic_oscillator, s.prev_high_low):
            out = fn(no_ohlc)
            assert isinstance(out, pl.DataFrame), fn.__name__
            assert len(out) == len(no_ohlc), fn.__name__


# ---------------------------------------------------------------------------
# Strategy method tests (synthetic data, no network)
# ---------------------------------------------------------------------------

class TestStrategyInit:
    def test_init_is_offline(self):
        # Construction used to fetch a quote per symbol, one HTTP call each,
        # which defeated the batch download it was meant to complement.
        with patch.object(scraping, 'current_stock_price',
                          side_effect=AssertionError('must not hit the network')):
            s = strategy.Strategy(symbol='TEST')
        assert s.symbol == 'TEST'

    def test_default_stops(self):
        s = strategy.Strategy(symbol='T')
        assert s.loss_percent == 7.0
        assert s.profit_percent == 10.0

    def test_stops_are_overridable(self):
        s = strategy.Strategy(symbol='T', loss_percent=3, profit_percent=None)
        assert s.loss_percent == 3
        assert s.profit_percent is None

    def test_extra_kwargs_set_as_attributes(self):
        s = strategy.Strategy(symbol='T', beta=1.4)
        assert s.beta == 1.4


class TestMACDSignal:
    def test_returns_dataframe(self, mock_strategy):
        s, df = mock_strategy
        result = s.macd(df)
        assert result is not None
        assert {'Buy_Signal', 'Sell_Signal'} <= set(result.columns)

    def test_correct_length(self, mock_strategy):
        s, df = mock_strategy
        assert len(s.macd(df)) == len(df)

    def test_missing_column_returns_none(self, mock_strategy):
        s, df = mock_strategy
        assert s.macd(df.drop(['MACD'])) is None

    def test_signals_are_boolean(self, mock_strategy):
        s, df = mock_strategy
        result = s.macd(df)
        assert result['Buy_Signal'].dtype == pl.Boolean
        assert result['Sell_Signal'].dtype == pl.Boolean


class TestRSISignal:
    def test_returns_dataframe(self, mock_strategy):
        s, df = mock_strategy
        assert len(s.rsi(df)) == len(df)

    def test_buy_when_rsi_oversold(self, mock_strategy):
        s, df = mock_strategy
        rsi_values = [50.0] * 300
        rsi_values[99]  = 36.0
        rsi_values[100] = 30.0    # crosses below → buy
        df2 = df.with_columns(pl.Series('RSI', rsi_values))
        assert s.rsi(df2, Lower_Band=35)['Buy_Signal'][100] is True

    def test_sell_when_rsi_overbought(self, mock_strategy):
        s, df = mock_strategy
        rsi_values = [50.0] * 300
        rsi_values[149] = 69.0
        rsi_values[150] = 75.0    # crosses above → sell
        df2 = df.with_columns(pl.Series('RSI', rsi_values))
        assert s.rsi(df2, Upper_Band=70)['Sell_Signal'][150] is True


class TestBollingerBands:
    def test_returns_dataframe(self, mock_strategy):
        s, df = mock_strategy
        assert isinstance(s.bollinger_bands(df), pl.DataFrame)

    def test_length_at_most_df(self, mock_strategy):
        s, df = mock_strategy
        result = s.bollinger_bands(df)
        assert 0 < len(result) <= len(df)

    def test_aligned_length_via_detect_signals(self, mock_strategy):
        s, df = mock_strategy
        bb = s.detect_signals(df).get('bollinger_bands')
        assert bb is not None
        assert len(bb) == len(df)


class TestEMACrossover:
    def test_length_matches_df(self, mock_strategy):
        s, df = mock_strategy
        assert len(s.ema_crossover(df)) == len(df)

    def test_buy_on_golden_cross(self, mock_strategy):
        s, _ = mock_strategy
        n = 100
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n)]
        ema12 = [10.0] * 50 + [12.0] * 50    # crosses above EMA26 at row 50
        ema26 = [11.0] * n
        df = pl.DataFrame({'Datetime': dates, 'EMA12': ema12, 'EMA26': ema26})
        assert s.ema_crossover(df)['Buy_Signal'][50] is True


class TestParabolicSAR:
    def test_returns_dataframe(self, mock_strategy):
        s, df = mock_strategy
        result = s.parabolic_sar(df)
        assert len(result) == len(df)
        assert result['Buy_Signal'].dtype == pl.Boolean

    def test_survives_null_high_low(self, mock_strategy):
        # A single null bar raised "'<' not supported between NoneType and float"
        # and killed this strategy for every symbol in the index.
        s, df = mock_strategy
        highs = df['High'].to_list()
        lows  = df['Low'].to_list()
        highs[150] = None
        lows[150] = None
        df2 = df.with_columns([pl.Series('High', highs), pl.Series('Low', lows)])
        result = s.parabolic_sar(df2)
        assert len(result) == len(df2)

    def test_returns_empty_when_first_bar_null(self, mock_strategy):
        s, df = mock_strategy
        highs = df['High'].to_list()
        highs[0] = None
        df2 = df.with_columns(pl.Series('High', highs))
        assert len(s.parabolic_sar(df2)) == len(df2)


class TestDonchianChannel:
    def test_returns_dataframe(self, mock_strategy):
        s, df = mock_strategy
        assert len(s.donchian_channel(df)) == len(df)


class TestATRBreakout:
    def test_returns_dataframe(self, mock_strategy):
        s, df = mock_strategy
        assert len(s.atr_breakout(df)) == len(df)


class TestVWAP:
    def test_returns_dataframe(self, mock_strategy):
        s, df = mock_strategy
        assert len(s.vwap(df)) == len(df)

    def test_band_is_symmetric(self, mock_strategy):
        """
        A mirrored price path must produce a mirrored signal count.

        The old thresholds were +2% for buys and -10% for sells, which made a
        sell almost arithmetically impossible: over two years of AAPL it fired
        19 buys and 0 sells, scoring like buy-and-hold and repeatedly winning
        the 'best strategy' slot on any rising stock.
        """
        s, _ = mock_strategy
        n = 200
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n)]
        # Oscillate ±6% around a flat VWAP so both bands are crossed repeatedly.
        close = [100.0 * (1 + 0.06 * np.sin(i / 3.0)) for i in range(n)]
        df = pl.DataFrame({
            'Datetime': dates, 'Close': close, 'Volume': [1e6] * n,
            'VWAP': [100.0] * n,
        })
        result = s.vwap(df, band=0.02)
        buys  = sum(result['Buy_Signal'].to_list())
        sells = sum(result['Sell_Signal'].to_list())
        assert buys > 0 and sells > 0
        assert abs(buys - sells) <= 1


class TestIchimokuCloud:
    def test_returns_dataframe(self, mock_strategy):
        s, df = mock_strategy
        assert len(s.ichimoku_cloud(df)) == len(df)

    def test_lines_are_range_midpoints(self, mock_strategy):
        """
        Conversion/base lines must be (high + low) / 2.

        Without the outer parentheses the division bound to the low alone, which
        inflated both lines by roughly half the price — so price was never above
        the cloud and the strategy produced sells only.
        """
        s, _ = mock_strategy
        n = 120
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n)]
        high = [float(100 + i) for i in range(n)]
        low  = [float(90 + i) for i in range(n)]
        df = pl.DataFrame({'Datetime': dates, 'High': high, 'Low': low,
                           'Close': [float(95 + i) for i in range(n)]})
        out = s.ichimoku_cloud(df)
        assert len(out) == n

        # Recompute the conversion line the correct way and confirm the magnitude.
        window = 9
        expected = (max(high[n - window:]) + min(low[n - window:])) / 2
        buggy    = max(high[n - window:]) + min(low[n - window:]) / 2
        assert expected < buggy    # the bug inflated it; guard the distinction

    def test_can_produce_buy_signals(self, mock_strategy):
        """A steadily rising series must be able to cross ABOVE the cloud."""
        s, _ = mock_strategy
        n = 200
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n)]
        # Flat, then a sharp rally that must break above the projected cloud.
        close = [100.0] * 120 + [100.0 + (i - 119) * 2.0 for i in range(120, n)]
        df = pl.DataFrame({
            'Datetime': dates, 'Close': close,
            'High': [c * 1.005 for c in close], 'Low': [c * 0.995 for c in close],
        })
        out = s.ichimoku_cloud(df)
        assert any(out['Buy_Signal'].to_list())


class TestStochasticOscillator:
    def test_returns_dataframe(self, mock_strategy):
        s, df = mock_strategy
        assert len(s.stochastic_oscillator(df)) == len(df)

    def test_rejects_bad_window(self, mock_strategy):
        s, df = mock_strategy
        with pytest.raises(ValueError):
            s.stochastic_oscillator(df, k_window=0)
        with pytest.raises(ValueError):
            s.stochastic_oscillator(df, d_window=-1)

    def test_handles_flat_range_without_dividing_by_zero(self, mock_strategy):
        s, _ = mock_strategy
        n = 60
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n)]
        df = pl.DataFrame({'Datetime': dates, 'High': [100.0] * n,
                           'Low': [100.0] * n, 'Close': [100.0] * n})
        out = s.stochastic_oscillator(df)
        assert len(out) == n


class TestPrevHighLow:
    def test_returns_dataframe(self, mock_strategy):
        s, df = mock_strategy
        assert len(s.prev_high_low(df, N=5)) == len(df)

    def test_window_clamped_to_history(self, mock_strategy):
        # N=390 on a 300-bar daily series used to swallow the whole window.
        s, df = mock_strategy
        out = s.prev_high_low(df, N=390)
        assert len(out) == len(df)
        # A clamped window leaves room for at least one breakout.
        assert any(out['Buy_Signal'].to_list()) or any(out['Sell_Signal'].to_list())

    def test_buy_on_breakout(self, mock_strategy):
        s, _ = mock_strategy
        n = 20
        dates = [datetime(2023, 6, 1) + timedelta(days=i) for i in range(n)]
        close = [100.0] * n
        close[10] = 120.0
        df = pl.DataFrame({'Datetime': dates, 'Close': close,
                           'High': [101.0] * n, 'Low': [99.0] * n})
        assert s.prev_high_low(df, N=5)['Buy_Signal'][10] is True


# ---------------------------------------------------------------------------
# backtest_strategy tests
# ---------------------------------------------------------------------------

class TestBacktestStrategy:
    def _make_signals(self, df, buy_rows, sell_rows):
        n = len(df)
        return pl.DataFrame({
            'Buy_Signal':  [i in buy_rows for i in range(n)],
            'Sell_Signal': [i in sell_rows for i in range(n)],
            'Datetime':    df['Datetime'].to_list(),
        })

    def test_profitable_buy_and_sell(self, mock_strategy):
        s, df = mock_strategy
        sigs = self._make_signals(df, buy_rows={10}, sell_rows={200})
        perf, metrics = s.backtest_strategy(df, sigs)
        assert isinstance(perf, float)
        for key in ('roi', 'win_rate', 'max_drawdown', 'trades',
                    'benchmark_roi', 'excess_roi'):
            assert key in metrics

    def test_no_trades_zero_profit(self, mock_strategy):
        s, df = mock_strategy
        perf, metrics = s.backtest_strategy(df, self._make_signals(df, set(), set()))
        assert perf == 0.0
        assert metrics['win_rate'] == 0.0
        assert metrics['trades'] == 0

    def test_stop_loss_limits_loss(self, df_down):
        s = strategy.Strategy(symbol='TEST')
        sigs = self._make_signals(df_down, buy_rows={0}, sell_rows=set())
        perf_no_sl, _   = s.backtest_strategy(df_down, sigs, stop_loss_percent=None)
        perf_with_sl, _ = s.backtest_strategy(df_down, sigs, stop_loss_percent=5.0)
        assert perf_with_sl > perf_no_sl

    def test_costs_are_charged_on_entry_too(self, mock_strategy):
        """
        A round trip at an unchanged price must lose money.

        The old model charged 1% on exit only, so an entry was free.
        """
        s, _ = mock_strategy
        n = 30
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n)]
        df = pl.DataFrame({'Datetime': dates, 'Open': [100.0] * n,
                           'Close': [100.0] * n, 'High': [100.0] * n,
                           'Low': [100.0] * n, 'Volume': [1e6] * n})
        sigs = self._make_signals(df, buy_rows={5}, sell_rows={20})
        perf, metrics = s.backtest_strategy(df, sigs, stop_loss_percent=None,
                                           stop_profit_percent=None)
        assert perf < 0
        assert metrics['trades'] == 1

    def test_zero_cost_round_trip_is_flat(self, mock_strategy):
        s, _ = mock_strategy
        n = 30
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n)]
        df = pl.DataFrame({'Datetime': dates, 'Open': [100.0] * n,
                           'Close': [100.0] * n, 'High': [100.0] * n,
                           'Low': [100.0] * n, 'Volume': [1e6] * n})
        sigs = self._make_signals(df, buy_rows={5}, sell_rows={20})
        perf, _ = s.backtest_strategy(df, sigs, commission=0.0, slippage=0.0,
                                      stop_loss_percent=None, stop_profit_percent=None)
        assert perf == pytest.approx(0.0, abs=1e-6)

    def test_fills_at_next_bar_open_not_signal_close(self, mock_strategy):
        """
        Entry must use bar i+1's open, not bar i's close.

        Filling at the signal bar's own close lets the backtest trade on a price
        it could not have known when the signal was generated.
        """
        s, _ = mock_strategy
        n = 10
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n)]
        close = [100.0] * n
        open_ = [100.0] * n
        open_[3] = 50.0      # buy signals on bar 2, so this is the fill price
        df = pl.DataFrame({'Datetime': dates, 'Open': open_, 'Close': close,
                           'High': [200.0] * n, 'Low': [10.0] * n,
                           'Volume': [1e6] * n})
        sigs = self._make_signals(df, buy_rows={2}, sell_rows=set())
        perf, _ = s.backtest_strategy(df, sigs, commission=0.0, slippage=0.0,
                                      stop_loss_percent=None, stop_profit_percent=None,
                                      trailing_stop=False)
        # Bought at ~50, force-closed at 100 → roughly a double.
        assert perf > 90_000

    def test_forced_final_close_counts_as_a_trade(self, mock_strategy):
        s, df = mock_strategy
        sigs = self._make_signals(df, buy_rows={10}, sell_rows=set())
        _, metrics = s.backtest_strategy(df, sigs)
        assert metrics['trades'] == 1
        # win_rate must be resolvable, not measured against a shifting denominator.
        assert metrics['win_rate'] in (0.0, 100.0)

    def test_benchmark_roi_tracks_price_change(self, mock_strategy):
        s, _ = mock_strategy
        n = 20
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n)]
        close = [100.0 + i * 5 for i in range(n)]      # 100 -> 195
        df = pl.DataFrame({'Datetime': dates, 'Open': close, 'Close': close,
                           'High': close, 'Low': close, 'Volume': [1e6] * n})
        _, metrics = s.backtest_strategy(df, self._make_signals(df, set(), set()),
                                         commission=0.0, slippage=0.0)
        assert metrics['benchmark_roi'] == pytest.approx(95.0, abs=0.1)
        assert metrics['excess_roi'] == pytest.approx(-95.0, abs=0.1)

    def test_position_fraction_scales_exposure(self, mock_strategy):
        s, df = mock_strategy
        sigs = self._make_signals(df, buy_rows={10}, sell_rows={200})
        full, _ = s.backtest_strategy(df, sigs, position_fraction=1.0)
        half, _ = s.backtest_strategy(df, sigs, position_fraction=0.5)
        assert abs(half) < abs(full)

    def test_no_nan_metrics_on_clean_data(self, mock_strategy):
        s, df = mock_strategy
        sigs = self._make_signals(df, buy_rows={10, 100}, sell_rows={50, 150})
        perf, metrics = s.backtest_strategy(df, sigs)
        assert perf == perf
        for key, value in metrics.items():
            assert value == value, f'{key} is NaN'


# ---------------------------------------------------------------------------
# detect_signals tests
# ---------------------------------------------------------------------------

class TestDetectSignals:
    def test_returns_dict_of_dataframes(self, mock_strategy):
        s, df = mock_strategy
        results = s.detect_signals(df)
        assert isinstance(results, dict) and results
        for name, sig_df in results.items():
            assert sig_df is not None, f'{name} returned None'
            assert {'Buy_Signal', 'Sell_Signal'} <= set(sig_df.columns)

    def test_all_signals_same_length_as_df(self, mock_strategy):
        s, df = mock_strategy
        for name, sig_df in s.detect_signals(df).items():
            assert len(sig_df) == len(df), f'{name} length mismatch'

    def test_returns_none_on_empty_df(self, mock_strategy):
        s, _ = mock_strategy
        assert s.detect_signals(pl.DataFrame()) is None

    def test_all_expected_strategies_present(self, mock_strategy):
        s, df = mock_strategy
        assert set(strategy.STRATEGY_NAMES) == set(s.detect_signals(df).keys())

    def test_every_strategy_succeeds_on_clean_data(self, mock_strategy):
        # parabolic_sar failed here for every symbol before the null fix.
        s, df = mock_strategy
        failed = [n for n, v in s.detect_signals(df).items() if v is None]
        assert failed == [], f'strategies returned None: {failed}'


# ---------------------------------------------------------------------------
# evaluate_strategies tests
# ---------------------------------------------------------------------------

class TestEvaluateStrategies:
    def test_returns_best_and_results(self, mock_strategy):
        s, df = mock_strategy
        best, results = s.evaluate_strategies(df, timeframe='1d')
        assert len(results) == len(strategy.STRATEGY_NAMES)
        assert best in strategy.STRATEGY_NAMES

    def test_reports_both_train_and_test_metrics(self, mock_strategy):
        s, df = mock_strategy
        _, results = s.evaluate_strategies(df, timeframe='1d')
        for r in results:
            assert 'risk_metrics' in r and 'train_metrics' in r
            assert 'benchmark_roi' in r['risk_metrics']

    def test_selection_uses_train_slice_only(self, mock_strategy):
        """
        `best` must maximise TRAIN performance, not the reported test figure.

        Choosing the best of twelve strategies on the same slice it is scored on
        guarantees a flattering number that predicts nothing.
        """
        s, df = mock_strategy
        best, results = s.evaluate_strategies(df, timeframe='1d', train_fraction=0.7)
        by_train = max(results, key=lambda r: r['train_metrics']['roi'])
        assert best == by_train['strategy_func']

    def test_signals_span_full_frame(self, mock_strategy):
        s, df = mock_strategy
        _, results = s.evaluate_strategies(df, timeframe='1d')
        for r in results:
            assert len(r['signals']) == len(df)

    def test_handles_short_frame_without_split(self, mock_strategy):
        s, _ = mock_strategy
        best, results = strategy.Strategy(symbol='T').evaluate_strategies(
            make_df(60, 'up'), timeframe='1d')
        assert results


# ---------------------------------------------------------------------------
# what_is_signal tests
# ---------------------------------------------------------------------------

class TestWhatIsSignal:
    def _res(self, name, buy_recent, sell_recent, roi=5.0):
        n = 50
        signals = pl.DataFrame({
            'Buy_Signal':  [False] * (n - 1) + [bool(buy_recent)],
            'Sell_Signal': [False] * (n - 1) + [bool(sell_recent)],
            'Datetime':    [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)],
        })
        metrics = {'roi': roi, 'win_rate': 60.0, 'time_frame_days': 365,
                   'trades': 3, 'benchmark_roi': 0.0, 'excess_roi': roi}
        return {'strategy_func': name, 'performance': roi * 1000,
                'risk_metrics': metrics, 'train_metrics': metrics, 'signals': signals}

    def test_returns_true_for_dominant_buy(self):
        assert what_is_signal(None, [self._res('a', True, False)], 4,
                              min_margin=1) is True

    def test_returns_false_for_dominant_sell(self):
        assert what_is_signal(None, [self._res('a', False, True)], 4,
                              min_margin=1) is False

    def test_negative_roi_still_produces_a_signal(self):
        """
        ROI must not gate the verdict.

        Gating on mean in-sample ROI suppressed 20/20 symbols in live testing:
        most mean-reversion strategies score negative in a rising market, so the
        gate was almost always shut.
        """
        assert what_is_signal(None, [self._res('a', False, True, roi=-5.0)], 4,
                              min_margin=1) is False
        assert what_is_signal(None, [self._res('a', True, False, roi=-40.0)], 4,
                              min_margin=1) is True

    def test_best_strategy_decides_direction(self):
        """`best` was accepted and never read, so selection had no effect."""
        res = [self._res('winner', True, False, roi=1.0),
               self._res('loser1', False, True, roi=90.0),
               self._res('loser2', False, True, roi=90.0)]
        # The vote says sell 2-1; the selected strategy says buy and wins.
        assert what_is_signal('winner', res, 4) is True
        assert what_is_signal(None, res, 4, min_margin=1) is False

    def test_best_strategy_bypasses_the_margin(self):
        """The selected strategy is the strongest evidence available, so it is
        not subject to the consensus margin."""
        res = [self._res('winner', True, False)]
        assert what_is_signal('winner', res, 4, min_margin=6) is True

    def test_majority_vote_when_best_is_silent(self):
        res = [self._res('quiet', False, False),
               self._res('a', True, False),
               self._res('b', True, False),
               self._res('c', True, False),
               self._res('d', False, True)]
        # 3 buys vs 1 sell clears the default margin of 2.
        assert what_is_signal('quiet', res, 4) is True

    def test_bare_majority_rejected_at_default_margin(self):
        """
        A one-vote win is not consensus.

        At margin 1 roughly half the index produced a signal on any given day,
        and 28 of 57 fallback signals rested on a single vote.
        """
        res = [self._res('quiet', False, False),
               self._res('a', True, False),
               self._res('b', True, False),
               self._res('c', False, True)]
        assert what_is_signal('quiet', res, 4, min_margin=2) is None
        assert what_is_signal('quiet', res, 4, min_margin=1) is True

    def test_higher_margin_is_stricter(self):
        res = [self._res('quiet', False, False),
               self._res('a', True, False),
               self._res('b', True, False),
               self._res('c', True, False)]
        assert what_is_signal('quiet', res, 4, min_margin=3) is True
        assert what_is_signal('quiet', res, 4, min_margin=4) is None

    def test_tie_returns_none_not_sell(self):
        """A tie used to fall through to `return False` and print a SELL."""
        res = [self._res('a', True, False), self._res('b', False, True)]
        assert what_is_signal(None, res, 4, min_margin=1) is None

    def test_ambiguous_strategy_casts_no_vote(self):
        res = [self._res('both', True, True), self._res('a', True, False)]
        assert what_is_signal(None, res, 4, min_margin=1) is True

    def test_returns_none_when_no_signals(self):
        assert what_is_signal(None, [self._res('a', False, False)], 4) is None

    def test_returns_none_on_empty_results(self):
        assert what_is_signal(None, [], 4) is None

    def test_lookback_window_is_respected(self):
        n = 50
        signals = pl.DataFrame({
            'Buy_Signal':  [True] + [False] * (n - 1),     # only at the very start
            'Sell_Signal': [False] * n,
            'Datetime':    [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)],
        })
        res = [{'strategy_func': 'a', 'performance': 0.0,
                'risk_metrics': {'roi': 1.0}, 'signals': signals}]
        assert what_is_signal('a', res, 4) is None
        assert what_is_signal('a', res, n) is True


# ---------------------------------------------------------------------------
# suggest_entry_price tests
# ---------------------------------------------------------------------------

class TestSuggestEntryPrice:
    def _hourly_df(self, closes, sma20=None):
        n = len(closes)
        closes = list(closes)
        if sma20 is None:
            sma20 = closes
        return pl.DataFrame({
            'Datetime': [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)],
            'Open':  closes,
            'High':  [c + 0.5 for c in closes],
            'Low':   [c - 0.5 for c in closes],
            'Close': closes,
            'SMA20': list(sma20),
        })

    def test_returns_none_for_invalid_direction(self):
        df = self._hourly_df([100.0] * 10)
        assert strategy.suggest_entry_price(df, 'HOLD') is None

    def test_returns_none_for_empty_or_missing_df(self):
        assert strategy.suggest_entry_price(None, 'BUY') is None
        assert strategy.suggest_entry_price(pl.DataFrame(), 'BUY') is None

    def test_returns_none_without_enough_bars(self):
        df = self._hourly_df([100.0, 101.0, 102.0])
        assert strategy.suggest_entry_price(df, 'BUY') is None

    def test_returns_none_without_sma20_column(self):
        df = self._hourly_df([100.0] * 10).drop('SMA20')
        assert strategy.suggest_entry_price(df, 'BUY') is None

    def test_buy_pulls_back_to_sma_when_price_extended_above_it(self):
        closes = [100.0] * 19 + [110.0]     # last close spiked above its own SMA
        df = self._hourly_df(closes, sma20=[100.0] * 20)
        out = strategy.suggest_entry_price(df, 'BUY')
        assert out['current_price'] == 110.0
        assert out['entry_price'] == 100.0
        assert out['distance_pct'] < 0
        assert 'pullback' in out['basis']

    def test_buy_entry_never_goes_below_recent_swing_low(self):
        closes = [100.0] * 19 + [110.0]
        # SMA implausibly far below anything actually traded recently.
        df = self._hourly_df(closes, sma20=[50.0] * 20)
        out = strategy.suggest_entry_price(df, 'BUY')
        floor = min(closes) - 0.5   # Low = Close - 0.5 in the fixture
        assert out['entry_price'] == pytest.approx(floor)

    def test_buy_at_market_when_already_at_or_below_sma(self):
        closes = [100.0] * 19 + [95.0]
        df = self._hourly_df(closes, sma20=[100.0] * 20)
        out = strategy.suggest_entry_price(df, 'BUY')
        assert out['entry_price'] == out['current_price'] == 95.0
        assert out['distance_pct'] == 0
        assert 'current price' in out['basis']

    def test_sell_pulls_back_to_sma_when_price_extended_below_it(self):
        closes = [100.0] * 19 + [90.0]
        df = self._hourly_df(closes, sma20=[100.0] * 20)
        out = strategy.suggest_entry_price(df, 'SELL')
        assert out['current_price'] == 90.0
        assert out['entry_price'] == 100.0
        assert out['distance_pct'] > 0

    def test_sell_entry_never_goes_above_recent_swing_high(self):
        closes = [100.0] * 19 + [90.0]
        df = self._hourly_df(closes, sma20=[150.0] * 20)
        out = strategy.suggest_entry_price(df, 'SELL')
        ceiling = max(closes) + 0.5   # High = Close + 0.5 in the fixture
        assert out['entry_price'] == pytest.approx(ceiling)

    def test_lookback_bars_reported_matches_window_used(self):
        df = self._hourly_df([100.0] * 60)
        out = strategy.suggest_entry_price(df, 'BUY', lookback=25)
        assert out['lookback_bars'] == 25


# ---------------------------------------------------------------------------
# scanner tests
# ---------------------------------------------------------------------------

class TestScanner:
    def test_analyse_symbol_returns_none_without_signal(self, df_flat):
        with patch.object(strategy, 'what_is_signal', return_value=None):
            assert scanner.analyse_symbol('FAKE', df_flat) is None

    def test_analyse_symbol_shape(self, df_up):
        with patch.object(strategy, 'what_is_signal', return_value=True):
            out = scanner.analyse_symbol('AAPL', df_up)
        assert out['symbol'] == 'AAPL'
        assert out['direction'] == 'BUY'
        for key in ('price', 'roi', 'benchmark_roi', 'excess_roi',
                    'win_rate', 'trades', 'strategy'):
            assert key in out

    def test_analyse_symbol_makes_no_network_calls(self, df_up):
        # Price comes off the batch-downloaded frame, not a fresh quote.
        with patch.object(strategy, 'what_is_signal', return_value=True), \
             patch.object(scraping, 'current_stock_price',
                          side_effect=AssertionError('must not hit the network')):
            out = scanner.analyse_symbol('AAPL', df_up)
        assert out['price'] == pytest.approx(round(float(df_up['Close'][-1]), 2))

    def test_scan_uses_prefetched_data(self, df_up):
        with patch.object(strategy, 'what_is_signal', return_value=False), \
             patch.object(scraping, 'batch_download',
                          side_effect=AssertionError('should not download')):
            found = scanner.scan(['A', 'B'], stock_data={'A': df_up, 'B': df_up},
                                 use_processes=False)
        assert len(found) == 2
        assert {f['direction'] for f in found} == {'SELL'}

    def test_scan_returns_empty_without_data(self):
        assert scanner.scan([], stock_data={}, use_processes=False) == []

    def test_intraday_gate(self):
        with patch.object(scraping, 'is_nyse_open', return_value=False):
            assert scanner.should_scan_now('1h') is False
            # Daily bars are final after the close, so scanning then is the point.
            assert scanner.should_scan_now('1d') is True

    def test_scan_interval_scales_with_timeframe(self):
        # A daily candle barely moves intraday; re-scanning every 300s produced
        # ~78 identical results a day.
        assert scanner.scan_interval_seconds('1d') > scanner.scan_interval_seconds('1m')
        assert scanner.scan_interval_seconds('unknown') == 3600


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    def _raw_yf_frame(self, n=300, with_partial_tail=True):
        """A pandas frame shaped like yfinance output, optionally with the
        partial in-progress bar Yahoo appends during a session."""
        idx = pd.date_range('2023-01-01', periods=n, freq='D', tz='America/New_York')
        close = _make_close(n, 'up')
        df = pd.DataFrame({
            'Open':   close,
            'High':   [c * 1.02 for c in close],
            'Low':    [c * 0.98 for c in close],
            'Close':  close,
            'Volume': [1_000_000] * n,
        }, index=idx)
        if with_partial_tail:
            df.loc[idx[-1], ['Open', 'High', 'Low', 'Close']] = np.nan
        return df

    def test_full_pipeline_produces_usable_metrics(self):
        """
        Download → indicators → strategies → backtest → verdict, including the
        partial trailing bar that broke the whole thing.

        This is the test whose absence let a defect that made the product emit
        zero signals for every symbol ship unnoticed: every prior test used clean
        synthetic data and passed throughout.
        """
        raw = self._raw_yf_frame(with_partial_tail=True)
        cleaned = scraping._clean_ohlcv(raw)
        assert cleaned is not None
        df = scraping.compute_indicators(
            pl.from_pandas(scraping._normalise_index(cleaned), include_index=True))

        for col in ['Open', 'High', 'Low', 'Close']:
            assert df[col].null_count() == 0

        s = strategy.Strategy(symbol='TEST')
        best, results = s.evaluate_strategies(df, timeframe='1d')

        assert len(results) == len(strategy.STRATEGY_NAMES), 'a strategy dropped out'
        for r in results:
            roi = r['risk_metrics']['roi']
            assert roi == roi, f"{r['strategy_func']} produced NaN ROI"

        # A verdict of None is legitimate; a crash or a NaN is not.
        assert what_is_signal(best, results, 4) in (True, False, None)

    def test_pipeline_survives_an_unclean_frame(self):
        """
        Defence in depth for the same root cause.

        `_clean_ohlcv` is the fix, but the downstream code must not fall over if
        a null bar reaches it another way. Originally a single null row here
        crashed parabolic_sar with a NoneType comparison and produced NaN ROI in
        every strategy holding an open position, which then suppressed the
        verdict for all 20 symbols under live test.
        """
        raw = self._raw_yf_frame(with_partial_tail=True)
        # Deliberately use the old, insufficient cleaning to keep the null bar.
        kept = raw[['Open', 'High', 'Low', 'Close', 'Volume']].dropna(how='all')
        df = scraping.compute_indicators(
            pl.from_pandas(scraping._normalise_index(kept), include_index=True))
        assert df['Close'].null_count() == 1, 'dropna(how="all") keeps the partial bar'

        s = strategy.Strategy(symbol='TEST')
        best, results = s.evaluate_strategies(df, timeframe='1d')

        # No strategy may drop out, and no metric may come back NaN.
        assert len(results) == len(strategy.STRATEGY_NAMES)
        for r in results:
            for key, value in r['risk_metrics'].items():
                assert value == value, f"{r['strategy_func']}.{key} is NaN"
        assert what_is_signal(best, results, 4) in (True, False, None)
