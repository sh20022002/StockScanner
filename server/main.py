"""
Terminal S&P 500 signal scanner.

Usage:
    python server/main.py                    # scan the full S&P 500 on daily bars
    python server/main.py --timeframe 1h     # intraday, gated on exchange hours
    python server/main.py --symbols AAPL,MSFT --once

The ROI shown is measured out-of-sample, and BENCH is buy-and-hold over the same
window. A strategy that does not beat BENCH is not worth trading.
"""
import argparse
import sys
import time
from datetime import datetime

import scanner
import scraping


def _arrows() -> tuple[str, str]:
    """
    Pick BUY/SELL markers the terminal can actually render.

    The default Windows console codepage is cp1252, which cannot encode ▲/▼ and
    raises UnicodeEncodeError mid-scan. Try to switch the stream to UTF-8 first,
    then fall back to ASCII.
    """
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    try:
        '▲▼'.encode(sys.stdout.encoding or 'ascii')
        return '▲ BUY ', '▼ SELL'
    except (UnicodeEncodeError, LookupError):
        return 'BUY   ', 'SELL  '


BUY_MARK, SELL_MARK = _arrows()


def _print_header(n_symbols: int, timeframe: str):
    print(f"\n{'=' * 78}")
    print(f"  SCAN STARTED  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
          f"  ({n_symbols} symbols, {timeframe})")
    print(f"{'=' * 78}")


def _print_signals(signals: list[dict]):
    if not signals:
        print('  No signals this scan.')
        return

    print(f"  {'dir':<6} {'ticker':<7} {'price':>10} {'roi':>9} {'bench':>9}"
          f" {'excess':>9} {'wr':>7} {'trades':>7}  strategy")
    print(f"  {'-' * 88}")
    # Strongest excess return first — that is the only ranking that means anything.
    for s in sorted(signals, key=lambda x: -x['excess_roi']):
        arrow = BUY_MARK if s['direction'] == 'BUY' else SELL_MARK
        print(f"  {arrow:<6} {s['symbol']:<7} {s['price']:>10.2f}"
              f" {s['roi']:>+8.1f}% {s['benchmark_roi']:>+8.1f}%"
              f" {s['excess_roi']:>+8.1f}% {s['win_rate']:>6.1f}%"
              f" {s['trades']:>7}  {s['strategy']}")


def scan_once(symbols: list[str], timeframe: str, period: str, lookback: int,
              workers: int | None, min_margin: int) -> list[dict]:
    _print_header(len(symbols), timeframe)
    t0 = time.time()

    signals = scanner.scan(symbols, timeframe=timeframe, period=period,
                           lookback=lookback, max_workers=workers,
                           min_margin=min_margin)

    print()
    _print_signals(signals)
    print(f"\n  Scan complete in {time.time() - t0:.1f}s"
          f"  ({len(signals)} signal(s) from {len(symbols)} symbols).")
    return signals


def run(timeframe: str, period: str, lookback: int, symbols: list[str] | None,
        once: bool, workers: int | None, min_margin: int):
    if symbols is None:
        print('Fetching S&P 500 symbols...')
        try:
            _, symbols = scraping.get_stocks()
        except Exception as e:
            print(f'Failed to fetch S&P 500 list: {e}')
            return
        print(f'Loaded {len(symbols)} symbols.')

    interval = scanner.scan_interval_seconds(timeframe)

    while True:
        try:
            if not scanner.should_scan_now(timeframe):
                print(f'[{datetime.now():%H:%M:%S}] Exchange closed — '
                      f'sleeping {scanner.CLOSED_MARKET_SLEEP // 60} min.')
                time.sleep(scanner.CLOSED_MARKET_SLEEP)
                continue

            scan_once(symbols, timeframe, period, lookback, workers, min_margin)

            if once:
                return
            print(f'  Next scan in {interval // 60} min.')
            time.sleep(interval)

        except KeyboardInterrupt:
            print('\nScanner stopped.')
            return
        except Exception as e:
            print(f'[SCAN ERROR] {e}')
            if once:
                return
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description='S&P 500 technical signal scanner.')
    parser.add_argument('--timeframe', default=scanner.DEFAULT_TIMEFRAME,
                        help='Bar interval (1d, 1h, 5d, 1wk...).')
    parser.add_argument('--period', default=scanner.DEFAULT_PERIOD,
                        help='History to download (2y, 1y, 6mo...).')
    parser.add_argument('--lookback', type=int, default=scanner.DEFAULT_LOOKBACK,
                        help='How many recent bars count as a live signal.')
    parser.add_argument('--symbols', default=None,
                        help='Comma-separated symbols instead of the full index.')
    parser.add_argument('--workers', type=int, default=None,
                        help='Worker processes (default: cpu_count-1, max 8).')
    parser.add_argument('--min-margin', type=int, default=scanner.DEFAULT_MIN_MARGIN,
                        help='Consensus vote margin required when the selected '
                             'strategy is silent. Raise it for fewer, stronger '
                             'signals; 1 accepts a bare majority.')
    parser.add_argument('--once', action='store_true',
                        help='Run a single scan and exit.')
    args = parser.parse_args()

    symbols = None
    if args.symbols:
        symbols = [scraping.yahoo_symbol(s) for s in args.symbols.split(',') if s.strip()]

    run(args.timeframe, args.period, args.lookback, symbols, args.once,
        args.workers, args.min_margin)


# Required on Windows: worker processes re-import this module under spawn.
if __name__ == '__main__':
    main()
