"""
Terminal US-market signal scanner.

Usage:
    python server/main.py                        # all US common stocks >= $2B market cap
    python server/main.py --min-market-cap 5e9   # >= $5B instead
    python server/main.py --timeframe 1h         # intraday, gated on exchange hours
    python server/main.py --symbols AAPL,MSFT --once

The ROI shown is measured out-of-sample, and BENCH is buy-and-hold over the same
window. A strategy that does not beat BENCH is not worth trading.

The RL policy (server/rl/) is always blended in when a trained checkpoint exists
at server/rl/checkpoints/best.pt — there is no flag to turn it off. With no
checkpoint it's simply absent from the output; see server/rl/train.py.
"""
import argparse
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# rl.live (torch) must import before scraping/strategy (pandas) — on Windows
# this process has been observed to access-violate loading torch's c10.dll if
# pandas' compiled extensions initialise first. See server/rl/features.py.
#
# This does mean every ProcessPoolExecutor worker also imports torch, since
# under Windows spawn each worker re-imports this module as __main__ before
# running scanner.analyse_symbol (pure rule-based backtesting — no worker ever
# touches RL). That's a fixed, one-time-per-scan cost paid at pool startup,
# not per symbol; deferring the import to dodge it would reintroduce the
# import-order bug in the one process that actually needs it.
try:
    import rl.live as rl_live
except Exception as _rl_import_error:
    rl_live = None
    print(f'[main] RL live-inference hook unavailable: {_rl_import_error}')

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


def _print_header(n_symbols: int, timeframe: str, min_market_cap: float | None):
    scope = (f'{n_symbols} symbols >= ${min_market_cap / 1e9:.1f}B mkt cap'
             if min_market_cap else f'{n_symbols} symbols')
    print(f"\n{'=' * 78}")
    print(f"  SCAN STARTED  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
          f"  ({scope}, {timeframe})")
    print(f"{'=' * 78}")


def _print_signals(signals: list[dict]):
    if not signals:
        print('  No signals this scan.')
        return

    has_rl = any('rl_action' in s for s in signals)
    header = (f"  {'dir':<6} {'ticker':<7} {'price':>10} {'roi':>9} {'bench':>9}"
             f" {'excess':>9} {'wr':>7} {'trades':>7}  strategy")
    if has_rl:
        header += '            rl'
    print(header)
    print(f"  {'-' * (88 + (14 if has_rl else 0))}")

    # Strongest excess return first — that is the only ranking that means anything.
    for s in sorted(signals, key=lambda x: -x['excess_roi']):
        arrow = BUY_MARK if s['direction'] == 'BUY' else SELL_MARK
        line = (f"  {arrow:<6} {s['symbol']:<7} {s['price']:>10.2f}"
               f" {s['roi']:>+8.1f}% {s['benchmark_roi']:>+8.1f}%"
               f" {s['excess_roi']:>+8.1f}% {s['win_rate']:>6.1f}%"
               f" {s['trades']:>7}  {s['strategy']}")
        if has_rl and 'rl_action' in s:
            mark = '=' if s.get('rl_agrees') else 'x'
            line += f"   RL {s['rl_action']:<5}{mark}({s['rl_confidence']:.0%})"
        print(line)


def scan_once(symbols: list[str], timeframe: str, period: str, lookback: int,
              workers: int | None, min_margin: int,
              min_market_cap: float | None = None) -> list[dict]:
    _print_header(len(symbols), timeframe, min_market_cap)
    t0 = time.time()

    want_rl = rl_live is not None
    result = scanner.scan(symbols, timeframe=timeframe, period=period,
                          lookback=lookback, max_workers=workers,
                          min_margin=min_margin, return_data=want_rl)
    if want_rl:
        signals, stock_data = result
        signals = rl_live.annotate(signals, stock_data)
    else:
        signals = result

    print()
    _print_signals(signals)
    print(f"\n  Scan complete in {time.time() - t0:.1f}s"
          f"  ({len(signals)} signal(s) from {len(symbols)} symbols).")
    return signals


def run(timeframe: str, period: str, lookback: int, symbols: list[str] | None,
        once: bool, workers: int | None, min_margin: int,
        min_market_cap: float):
    resolved_cap = None
    if symbols is None:
        print(f'Fetching US common stocks >= ${min_market_cap / 1e9:.1f}B market cap...')
        try:
            equities = scraping.get_us_equities(min_market_cap=min_market_cap)
        except Exception as e:
            print(f'Failed to fetch the US equities universe: {e}')
            return
        symbols = [e['symbol'] for e in equities]
        resolved_cap = min_market_cap
        print(f'Loaded {len(symbols)} symbols.')
        if rl_live is not None and not rl_live.model_available():
            print('(No RL checkpoint at server/rl/checkpoints/best.pt yet — '
                  'run server/rl/train.py to enable it. Rule-based signals only for now.)')

    interval = scanner.scan_interval_seconds(timeframe)

    while True:
        try:
            if not scanner.should_scan_now(timeframe):
                print(f'[{datetime.now():%H:%M:%S}] Exchange closed — '
                      f'sleeping {scanner.CLOSED_MARKET_SLEEP // 60} min.')
                time.sleep(scanner.CLOSED_MARKET_SLEEP)
                continue

            scan_once(symbols, timeframe, period, lookback, workers, min_margin,
                     resolved_cap)

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
    parser = argparse.ArgumentParser(description='US-market technical signal scanner.')
    parser.add_argument('--timeframe', default=scanner.DEFAULT_TIMEFRAME,
                        help='Bar interval (1d, 1h, 5d, 1wk...).')
    parser.add_argument('--period', default=scanner.DEFAULT_PERIOD,
                        help='History to download (2y, 1y, 6mo...).')
    parser.add_argument('--lookback', type=int, default=scanner.DEFAULT_LOOKBACK,
                        help='How many recent bars count as a live signal.')
    parser.add_argument('--symbols', default=None,
                        help='Comma-separated symbols instead of the full market.')
    parser.add_argument('--min-market-cap', type=float,
                        default=scraping.DEFAULT_MIN_MARKET_CAP,
                        help='Minimum market cap in dollars for the default '
                             '(non --symbols) universe. Lower values mean many '
                             'more symbols and a longer scan. Default: $2B.')
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
        args.workers, args.min_margin, args.min_market_cap)


# Required on Windows: worker processes re-import this module under spawn.
if __name__ == '__main__':
    main()
