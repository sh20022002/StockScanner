"""
Tests for the SP500 scanner signal pipeline.
Run with: pytest server/tests/test_signals.py -v
"""
import os, sys
import pytest
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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

    df = pl.DataFrame({
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
    return df


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
    with patch.object(scraping, 'current_stock_price', return_value=150.0):
        s = strategy.Strategy(symbol='TEST')
    return s, df_up


# ---------------------------------------------------------------------------
# scraping.py tests
# ---------------------------------------------------------------------------

class TestIsNYSEOpen:
    def test_closed_on_saturday(self):
        # 2024-01-06 is a Saturday
        with patch('scraping.get_exchange_time') as mock_time:
            mock_time.return_value = datetime(2024, 1, 6, 10, 0)
            assert scraping.is_nyse_open() is False

    def test_closed_on_sunday(self):
        with patch('scraping.get_exchange_time') as mock_time:
            mock_time.return_value = datetime(2024, 1, 7, 10, 0)
            assert scraping.is_nyse_open() is False

    def test_closed_before_market_hours(self):
        # Monday 2024-01-08 at 9:00 AM — before open
        with patch('scraping.get_exchange_time') as mock_time:
            mock_time.return_value = datetime(2024, 1, 8, 9, 0)
            assert scraping.is_nyse_open() is False

    def test_closed_after_market_hours(self):
        # Monday 2024-01-08 at 16:30
        with patch('scraping.get_exchange_time') as mock_time:
            mock_time.return_value = datetime(2024, 1, 8, 16, 30)
            assert scraping.is_nyse_open() is False

    def test_open_during_trading_hours(self):
        # Monday 2024-01-08 at 12:00 — mid-day
        with patch('scraping.get_exchange_time') as mock_time:
            mock_time.return_value = datetime(2024, 1, 8, 12, 0)
            assert scraping.is_nyse_open() is True

    def test_closed_on_new_years_day(self):
        # 2024-01-01 is New Year's Day (Monday)
        with patch('scraping.get_exchange_time') as mock_time:
            mock_time.return_value = datetime(2024, 1, 1, 12, 0)
            assert scraping.is_nyse_open() is False


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

    def test_returns_1_on_empty_df(self):
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.history.return_value = pd.DataFrame()
            price = scraping.current_stock_price('FAKE')
            assert price == 1

    def test_returns_1_on_exception(self):
        with patch('yfinance.Ticker', side_effect=Exception('network error')):
            price = scraping.current_stock_price('ERR')
            assert price == 1


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
        prices = list(range(100, 400))
        mock_df = self._mock_yf_ticker(prices)
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_df
            result = scraping.get_stock_data('AAPL', interval='1d', period='1y',
                                             return_flags={'DF': True, 'INDICATORS': False})
            assert 'DF' in result
            assert not result['DF'].empty

    def test_required_columns_present(self):
        prices = list(range(100, 400))
        mock_df = self._mock_yf_ticker(prices)
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_df
            result = scraping.get_stock_data('AAPL', interval='1d', period='1y',
                                             return_flags={'DF': True, 'INDICATORS': False})
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                assert col in result['DF'].columns

    def test_indicators_computed(self):
        prices = list(range(50, 350))  # 300 rows — enough for SMA200
        mock_df = self._mock_yf_ticker(prices)
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_df
            result = scraping.get_stock_data('AAPL', interval='1d', period='2y',
                                             return_flags={'DF': True, 'INDICATORS': True})
            df = result['DF']
            for col in ['SMA20', 'EMA12', 'RSI', 'MACD', 'ATR']:
                assert col in df.columns

    def test_empty_data_returns_empty_dict(self):
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.history.return_value = pd.DataFrame()
            result = scraping.get_stock_data('FAKE', interval='1d', period='1y')
            assert 'DF' not in result or result.get('DF') is None or (
                hasattr(result.get('DF'), 'empty') and result['DF'].empty
            )


# ---------------------------------------------------------------------------
# strategy.generate_signal tests
# ---------------------------------------------------------------------------

class TestGenerateSignal:
    def test_correct_shape(self):
        n = 10
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
        sig = generate_signal([False]*n, [True]*n, dates)
        assert isinstance(sig, pl.DataFrame)
        assert len(sig) == n
        assert 'Buy_Signal' in sig.columns
        assert 'Sell_Signal' in sig.columns
        assert 'Datetime' in sig.columns

    def test_buy_signals_preserved(self):
        n = 5
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
        buys  = [True, False, True, False, False]
        sells = [False, True, False, False, True]
        sig = generate_signal(sells, buys, dates)
        assert sig['Buy_Signal'].to_list()  == buys
        assert sig['Sell_Signal'].to_list() == sells

    def test_raises_on_mismatched_lengths(self):
        with pytest.raises(ValueError):
            generate_signal([True, False], [True], [datetime(2024, 1, 1)])


# ---------------------------------------------------------------------------
# Strategy method tests (using synthetic data, no network)
# ---------------------------------------------------------------------------

class TestStrategyInit:
    def test_init_with_mock_price(self):
        with patch.object(scraping, 'current_stock_price', return_value=200.0):
            s = strategy.Strategy(symbol='TEST')
        assert s.symbol == 'TEST'
        assert s.avg_price == 200.0
        assert hasattr(s, 'loss_percent')
        assert hasattr(s, 'risk_tolerance')

    def test_loss_percent_low_risk(self):
        with patch.object(scraping, 'current_stock_price', return_value=100.0), \
             patch.object(strategy.Strategy, 'calculate_risk_score', return_value=50):
            s = strategy.Strategy(symbol='T')
        assert s.loss_percent == 3

    def test_loss_percent_high_risk(self):
        # Force risk_tolerance > 80 by patching calculate_risk_score
        with patch.object(scraping, 'current_stock_price', return_value=100.0), \
             patch.object(strategy.Strategy, 'calculate_risk_score', return_value=85):
            s = strategy.Strategy(symbol='T')
        assert s.loss_percent == 7


class TestMACDSignal:
    def test_returns_dataframe(self, mock_strategy):
        s, df = mock_strategy
        result = s.macd(df)
        assert result is not None
        assert isinstance(result, pl.DataFrame)
        assert 'Buy_Signal' in result.columns
        assert 'Sell_Signal' in result.columns

    def test_correct_length(self, mock_strategy):
        s, df = mock_strategy
        result = s.macd(df)
        assert len(result) == len(df)

    def test_missing_column_returns_none(self, mock_strategy):
        s, df = mock_strategy
        df_no_macd = df.drop(['MACD'])
        result = s.macd(df_no_macd)
        assert result is None

    def test_signals_are_boolean(self, mock_strategy):
        s, df = mock_strategy
        result = s.macd(df)
        assert result['Buy_Signal'].dtype == pl.Boolean
        assert result['Sell_Signal'].dtype == pl.Boolean


class TestRSISignal:
    def test_returns_dataframe(self, mock_strategy):
        s, df = mock_strategy
        result = s.rsi(df)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(df)

    def test_buy_when_rsi_oversold(self, mock_strategy):
        s, df = mock_strategy
        # Force RSI to cross below 35 on row 100
        rsi_values = [50.0] * 300
        rsi_values[99] = 36.0  # was above lower band
        rsi_values[100] = 30.0  # crosses below → buy
        df2 = df.with_columns(pl.Series('RSI', rsi_values))
        result = s.rsi(df2, Lower_Band=35)
        assert result['Buy_Signal'][100] is True

    def test_sell_when_rsi_overbought(self, mock_strategy):
        s, df = mock_strategy
        rsi_values = [50.0] * 300
        rsi_values[149] = 69.0
        rsi_values[150] = 75.0  # crosses above upper band → sell
        df2 = df.with_columns(pl.Series('RSI', rsi_values))
        result = s.rsi(df2, Upper_Band=70)
        assert result['Sell_Signal'][150] is True


class TestBollingerBands:
    def test_returns_dataframe(self, mock_strategy):
        s, df = mock_strategy
        result = s.bollinger_bands(df)
        assert isinstance(result, pl.DataFrame)

    def test_length_at_most_df(self, mock_strategy):
        # bollinger_bands filters NaN rows internally; length may be <= df
        s, df = mock_strategy
        result = s.bollinger_bands(df)
        assert len(result) <= len(df)
        assert len(result) > 0

    def test_aligned_length_matches_df_via_multithread(self, mock_strategy):
        # When called through detect_signals_multithread the alignment is applied
        s, df = mock_strategy
        results = s.detect_signals_multithread(df)
        bb = results.get('bollinger_bands')
        assert bb is not None
        assert len(bb) == len(df)


class TestEMACrossover:
    def test_returns_dataframe(self, mock_strategy):
        s, df = mock_strategy
        result = s.ema_crossover(df)
        assert isinstance(result, pl.DataFrame)

    def test_length_matches_df(self, mock_strategy):
        s, df = mock_strategy
        result = s.ema_crossover(df)
        assert len(result) == len(df)

    def test_buy_on_golden_cross(self, mock_strategy):
        s, _ = mock_strategy
        n = 100
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n)]
        # EMA12 crosses above EMA26 at row 50
        ema12 = [10.0] * 50 + [12.0] * 50
        ema26 = [11.0] * 50 + [11.0] * 50

        df = pl.DataFrame({'Datetime': dates, 'EMA12': ema12, 'EMA26': ema26,
                           'Close': [100.0]*n, 'Open': [100.0]*n,
                           'High': [101.0]*n, 'Low': [99.0]*n, 'Volume': [1e6]*n,
                           'SMA20': [100.0]*n, 'SMA150': [100.0]*n,
                           'MACD': [0.0]*n, 'MACD_Signal': [0.0]*n, 'MACD_Hist': [0.0]*n,
                           'RSI': [50.0]*n, 'ATR': [1.5]*n, 'VWAP': [100.0]*n,
                           'STOCH_%K': [50.0]*n, 'STOCH_%D': [50.0]*n,
                           'SMA50': [100.0]*n, 'SMA100': [100.0]*n,
                           'SMA200': [100.0]*n, 'EMA20': [100.0]*n})
        result = s.ema_crossover(df)
        assert result['Buy_Signal'][50] is True


class TestParabolicSAR:
    def test_returns_dataframe(self, mock_strategy):
        s, df = mock_strategy
        result = s.parabolic_sar(df)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(df)

    def test_signals_are_boolean(self, mock_strategy):
        s, df = mock_strategy
        result = s.parabolic_sar(df)
        assert result['Buy_Signal'].dtype == pl.Boolean
        assert result['Sell_Signal'].dtype == pl.Boolean


class TestDonchianChannel:
    def test_returns_dataframe(self, mock_strategy):
        s, df = mock_strategy
        result = s.donchian_channel(df)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(df)


class TestATRBreakout:
    def test_returns_dataframe(self, mock_strategy):
        s, df = mock_strategy
        result = s.atr_breakout(df)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(df)


class TestVWAP:
    def test_returns_dataframe(self, mock_strategy):
        s, df = mock_strategy
        result = s.vwap(df)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(df)


class TestIchimokuCloud:
    def test_returns_dataframe(self, mock_strategy):
        s, df = mock_strategy
        result = s.ichimoku_cloud(df)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(df)


class TestPrevHighLow:
    def test_returns_dataframe(self, mock_strategy):
        s, df = mock_strategy
        result = s.prev_high_low(df, N=5)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(df)

    def test_buy_on_breakout(self, mock_strategy):
        s, _ = mock_strategy
        n = 20
        dates = [datetime(2023, 6, 1) + timedelta(days=i) for i in range(n)]
        # Spike on row 10 breaks above prior highs
        close = [100.0] * n
        high  = [101.0] * n
        low   = [99.0] * n
        close[10] = 120.0  # breakout
        df = pl.DataFrame({'Datetime': dates, 'Close': close,
                           'High': high, 'Low': low,
                           'Open': close, 'Volume': [1e6]*n,
                           'SMA20': [100.0]*n, 'SMA150': [100.0]*n,
                           'MACD': [0.0]*n, 'MACD_Signal': [0.0]*n,
                           'MACD_Hist': [0.0]*n, 'RSI': [50.0]*n,
                           'ATR': [1.5]*n, 'VWAP': [100.0]*n,
                           'STOCH_%K': [50.0]*n, 'STOCH_%D': [50.0]*n,
                           'SMA50': [100.0]*n, 'SMA100': [100.0]*n,
                           'SMA200': [100.0]*n, 'EMA12': [100.0]*n,
                           'EMA20': [100.0]*n, 'EMA26': [100.0]*n})
        result = s.prev_high_low(df, N=5)
        assert result['Buy_Signal'][10] is True


# ---------------------------------------------------------------------------
# backtest_strategy tests
# ---------------------------------------------------------------------------

class TestBacktestStrategy:
    def _make_signals(self, df, buy_rows, sell_rows):
        n = len(df)
        buys  = [i in buy_rows  for i in range(n)]
        sells = [i in sell_rows for i in range(n)]
        return pl.DataFrame({'Buy_Signal': buys, 'Sell_Signal': sells,
                             'Datetime': df['Datetime'].to_list()})

    def test_profitable_buy_and_sell(self, mock_strategy):
        s, df = mock_strategy
        # Buy early, sell after big price rise
        sigs = self._make_signals(df, buy_rows={10}, sell_rows={200})
        perf, metrics = s.backtest_strategy(df, sigs)
        assert isinstance(perf, float)
        assert 'roi' in metrics
        assert 'win_rate' in metrics
        assert 'max_drawdown' in metrics

    def test_no_trades_zero_profit(self, mock_strategy):
        s, df = mock_strategy
        sigs = self._make_signals(df, buy_rows=set(), sell_rows=set())
        perf, metrics = s.backtest_strategy(df, sigs)
        assert perf == 0.0
        assert metrics['win_rate'] == 0.0

    def test_stop_loss_limits_loss(self, mock_strategy, df_down):
        # Buy at top of a downtrend — stop-loss should limit losses
        with patch.object(scraping, 'current_stock_price', return_value=200.0):
            s = strategy.Strategy(symbol='TEST')
        sigs = self._make_signals(df_down, buy_rows={0}, sell_rows=set())

        perf_no_sl, _ = s.backtest_strategy(df_down, sigs, stop_loss_percent=None)
        perf_with_sl, _ = s.backtest_strategy(df_down, sigs, stop_loss_percent=5.0)
        # With stop-loss, loss should be smaller
        assert perf_with_sl > perf_no_sl


# ---------------------------------------------------------------------------
# detect_signals_multithread tests
# ---------------------------------------------------------------------------

class TestDetectSignalsMultithread:
    def test_returns_dict_of_dataframes(self, mock_strategy):
        s, df = mock_strategy
        results = s.detect_signals_multithread(df)
        assert isinstance(results, dict)
        assert len(results) > 0
        for name, sig_df in results.items():
            assert sig_df is not None
            assert isinstance(sig_df, pl.DataFrame)
            assert 'Buy_Signal'  in sig_df.columns
            assert 'Sell_Signal' in sig_df.columns

    def test_all_signals_same_length_as_df(self, mock_strategy):
        s, df = mock_strategy
        results = s.detect_signals_multithread(df)
        for name, sig_df in results.items():
            assert len(sig_df) == len(df), f"{name} length mismatch: {len(sig_df)} != {len(df)}"

    def test_returns_none_on_empty_df(self, mock_strategy):
        s, _ = mock_strategy
        result = s.detect_signals_multithread(pl.DataFrame())
        assert result is None

    def test_all_expected_strategies_present(self, mock_strategy):
        s, df = mock_strategy
        results = s.detect_signals_multithread(df)
        expected = {'macd', 'rsi', 'ma', 'bollinger_bands', 'vwap',
                    'ichimoku_cloud', 'donchian_channel', 'atr_breakout',
                    'parabolic_sar', 'stochastic_oscillator', 'ema_crossover', 'prev_high_low'}
        assert expected == set(results.keys())


# ---------------------------------------------------------------------------
# what_is_signal tests
# ---------------------------------------------------------------------------

class TestWhatIsSignal:
    def _make_backtest_res(self, buy_recent, sell_recent, roi=5.0):
        """Build a mock backtest_res list with signal on the final row."""
        n_total = 50
        # Signal on the very last row — always within any lookback window
        buy_sigs  = [False] * (n_total - 1) + ([True]  if buy_recent  else [False])
        sell_sigs = [False] * (n_total - 1) + ([True]  if sell_recent else [False])
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_total)]
        signals = pl.DataFrame({'Buy_Signal': buy_sigs[:n_total], 'Sell_Signal': sell_sigs[:n_total],
                                'Datetime': dates})
        return [{'strategy_func': 'test', 'performance': roi * 1000,
                 'risk_metrics': {'roi': roi, 'win_rate': 60.0, 'time_frame_days': 365},
                 'signals': signals, 'fig': None}]

    def test_returns_true_for_dominant_buy(self):
        res = self._make_backtest_res(buy_recent=True, sell_recent=False, roi=10.0)
        result = what_is_signal(None, res, 4)
        assert result is True

    def test_returns_false_for_dominant_sell(self):
        # what_is_signal returns False (sell) only when avg_roi > 0 AND sells >= buys
        res = self._make_backtest_res(buy_recent=False, sell_recent=True, roi=5.0)
        result = what_is_signal(None, res, 4)
        assert result is False

    def test_returns_none_when_roi_negative(self):
        # Negative ROI always returns None regardless of signal direction
        res = self._make_backtest_res(buy_recent=False, sell_recent=True, roi=-5.0)
        result = what_is_signal(None, res, 4)
        assert result is None

    def test_returns_none_when_no_signals(self):
        res = self._make_backtest_res(buy_recent=False, sell_recent=False, roi=5.0)
        result = what_is_signal(None, res, 4)
        assert result is None

    def test_returns_none_on_empty_results(self):
        result = what_is_signal(None, [], 4)
        assert result is None


# ---------------------------------------------------------------------------
# main.process_symbol integration test (mocked)
# ---------------------------------------------------------------------------

class TestProcessSymbol:
    def test_skips_on_no_signal(self, capsys):
        import main
        df = make_df(300, 'up')
        with patch.object(strategy.Strategy, 'get_strategy_func', return_value=('macd', [])):
            main._process_symbol('FAKE', df)
        # No signal printed — output should be empty
        assert capsys.readouterr().out == ''

    def test_prints_buy_signal(self, capsys):
        import main
        df = make_df(300, 'up')

        fake_backtest = [{
            'strategy_func': 'macd',
            'performance': 5000.0,
            'risk_metrics': {'roi': 15.0, 'win_rate': 65.0, 'time_frame_days': 300},
            'signals': pl.DataFrame({
                'Buy_Signal':  [True]  + [False] * 299,
                'Sell_Signal': [False] * 300,
                'Datetime':    df['Datetime'].to_list(),
            }),
            'fig': None,
        }]

        with patch.object(strategy.Strategy, 'get_strategy_func',
                          return_value=('macd', fake_backtest)), \
             patch('strategy.what_is_signal', return_value=True), \
             patch('scraping.current_stock_price', return_value=150.0):
            main._process_symbol('AAPL', df)

        captured = capsys.readouterr()
        assert 'BUY' in captured.out
        assert 'AAPL' in captured.out
