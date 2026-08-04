"""
The scan pipeline, shared by the CLI (main.py) and the dashboard (ui.py).

One batch download, then one process pool across symbols. Everything inside a
symbol runs sequentially: the work is pure-Python compute, so the nested
ThreadPoolExecutors this replaces (20 symbol workers, each spawning 20 backtest
workers and 5 signal workers — 500+ threads) only contended for the GIL.
"""
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import polars as pl

import scraping
import strategy

DEFAULT_TIMEFRAME = '1d'
DEFAULT_PERIOD    = '2y'
BATCH_SIZE        = 200
TRAIN_FRACTION    = 0.7

# How many recent bars count as a live signal. 1 means "fired on the most recent
# completed bar", which is also the only set that is still actionable given the
# backtest fills at the next bar's open.
#
# This was 4, a value never calibrated because the pipeline emitted no signals at
# all. Measured over the full index on daily bars:
#     lookback=1, margin=2 ->  58 signals (12% of the index)
#     lookback=4, margin=2 -> 281 signals (56%) — not a screen
# Raise it to catch signals from bars you missed between scans.
DEFAULT_LOOKBACK = 1

# Vote margin the fallback consensus must clear when the selected strategy is
# silent. At 1 (bare majority) half of all fallback signals were 5-4 splits.
DEFAULT_MIN_MARGIN = 2

# Timeframes that only move while the exchange is open.
INTRADAY = {'1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h'}

# Minimum seconds between scans, per timeframe. A daily candle barely moves
# intraday, so the old fixed 300s re-scanned identical data ~78 times a day and
# re-logged the same signals every time.
_SCAN_INTERVALS = {
    '1m': 60, '2m': 120, '5m': 300, '15m': 900, '30m': 1800,
    '60m': 1800, '90m': 1800, '1h': 1800,
    '1d': 3600, '5d': 3600, '1wk': 3600, '1mo': 3600,
}
CLOSED_MARKET_SLEEP = 900


def is_intraday(timeframe: str) -> bool:
    return timeframe in INTRADAY


def scan_interval_seconds(timeframe: str) -> int:
    """Minimum sensible gap between two scans of this timeframe."""
    return _SCAN_INTERVALS.get(timeframe, 3600)


def should_scan_now(timeframe: str) -> bool:
    """
    Whether a scan would see anything new.

    Intraday timeframes are gated on exchange hours — hammering Yahoo overnight
    and at weekends gains nothing and invites the rate limiting behind the
    "yfinance api crashes" note in tasks.txt. Daily and slower timeframes are
    allowed to run after the close, which is when the bar is actually final.
    """
    if not is_intraday(timeframe):
        return True
    return scraping.is_nyse_open()


def analyse_symbol(symbol: str, df: pl.DataFrame,
                   timeframe: str = DEFAULT_TIMEFRAME,
                   lookback: int = DEFAULT_LOOKBACK,
                   train_fraction: float = TRAIN_FRACTION,
                   min_margin: int = DEFAULT_MIN_MARGIN) -> dict | None:
    """
    Run the strategy set over one pre-fetched frame and return a signal, or None.

    Module-level and side-effect free so it can be shipped to a worker process.
    Takes no network calls: the price comes off the frame the batch download
    already produced, rather than a fresh per-symbol quote.

    The reported ROI figures are OUT-OF-SAMPLE — measured on the held-out slice,
    not the slice the winning strategy was chosen on.
    """
    try:
        stock = strategy.Strategy(symbol=symbol)
        best, backtest_res = stock.evaluate_strategies(
            df, timeframe=timeframe, train_fraction=train_fraction)
        if not backtest_res:
            return None

        verdict = strategy.what_is_signal(best, backtest_res, lookback,
                                          min_margin=min_margin)
        if verdict is None:
            return None

        def _mean(key):
            vals = [r['risk_metrics'][key] for r in backtest_res
                    if r.get('risk_metrics') and r['risk_metrics'].get(key) is not None]
            vals = [v for v in vals if v == v]        # drop any NaN
            return round(sum(vals) / len(vals), 2) if vals else 0.0

        chosen = next((r for r in backtest_res if r['strategy_func'] == best), None)
        metrics = (chosen or backtest_res[0])['risk_metrics']

        return {
            'time':          str(df['Datetime'][-1]),
            'symbol':        symbol,
            'direction':     'BUY' if verdict else 'SELL',
            'price':         round(float(df['Close'][-1]), 2),
            'strategy':      best or 'combined',
            'roi':           metrics.get('roi', 0.0),
            'benchmark_roi': metrics.get('benchmark_roi', 0.0),
            'excess_roi':    metrics.get('excess_roi', 0.0),
            'win_rate':      metrics.get('win_rate', 0.0),
            'trades':        metrics.get('trades', 0),
            'avg_roi':       _mean('roi'),
            'avg_win_rate':  _mean('win_rate'),
        }
    except Exception as e:
        print(f'[ERROR] {symbol}: {e}')
        return None


def default_workers() -> int:
    # Leave a core for the UI process and the OS.
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def scan(symbols: list[str],
         timeframe: str = DEFAULT_TIMEFRAME,
         period: str = DEFAULT_PERIOD,
         lookback: int = DEFAULT_LOOKBACK,
         max_workers: int | None = None,
         use_processes: bool = True,
         stock_data: dict | None = None,
         quiet: bool = False,
         min_margin: int = DEFAULT_MIN_MARGIN,
         stop_event=None) -> list[dict]:
    """
    Download every symbol once, analyse them in parallel, return the signals.

    Args:
        symbols: Symbols to scan.
        timeframe: yfinance interval.
        period: How much history to pull.
        lookback: How many recent bars count as a live signal.
        max_workers: Process count. Defaults to cpu_count-1, capped at 8.
        use_processes: Falls back to in-process execution if a pool cannot start
            (some sandboxed and frozen environments cannot spawn children).
        stock_data: Pre-fetched {symbol: frame}, to skip the download.
        stop_event: Optional threading.Event for cooperative cancellation.

    Returns:
        A list of signal dicts, as produced by analyse_symbol.
    """
    if stock_data is None:
        stock_data = scraping.batch_download(
            symbols, period=period, interval=timeframe,
            chunk_size=BATCH_SIZE, quiet=quiet)

    if not stock_data:
        return []

    items = list(stock_data.items())
    workers = max_workers or default_workers()
    signals: list[dict] = []

    def _cancelled() -> bool:
        return stop_event is not None and stop_event.is_set()

    if use_processes and workers > 1 and len(items) > 1:
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = {
                    ex.submit(analyse_symbol, sym, df, timeframe, lookback,
                              TRAIN_FRACTION, min_margin): sym
                    for sym, df in items
                }
                for future in as_completed(futures):
                    if _cancelled():
                        for f in futures:
                            f.cancel()
                        break
                    try:
                        result = future.result()
                    except Exception as e:
                        print(f'[ERROR] {futures[future]}: {e}')
                        continue
                    if result:
                        signals.append(result)
            return signals
        except Exception as e:
            print(f'[scan] process pool unavailable ({e}); running in-process.')
            signals.clear()

    for sym, df in items:
        if _cancelled():
            break
        result = analyse_symbol(sym, df, timeframe, lookback, TRAIN_FRACTION,
                                min_margin)
        if result:
            signals.append(result)

    return signals
