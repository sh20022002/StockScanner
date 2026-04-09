import scraping, strategy
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import polars as pl


SCAN_INTERVAL_SECONDS = 300   # re-scan every 5 minutes
MAX_WORKERS           = 20    # parallel backtest workers (I/O is done in batch)
SIGNAL_LOOKBACK       = 4     # candles to check for a live signal
TIMEFRAME             = '1d'
BATCH_SIZE            = 200   # symbols per yf.download() call


def _print_signal(symbol: str, direction: str, price: float, roi: float, win_rate: float):
    ts    = datetime.now().strftime('%H:%M:%S')
    arrow = '▲ BUY ' if direction == 'buy' else '▼ SELL'
    print(f'[{ts}]  {arrow}  {symbol:<6}  ${price:<9.2f}  ROI: {roi:+.1f}%  WR: {win_rate:.1f}%')


def _process_symbol(symbol: str, df: pl.DataFrame) -> None:
    """Run all strategies + backtest on one pre-fetched DataFrame."""
    try:
        stock = strategy.Strategy(symbol=symbol)
        best, backtest_res = stock.get_strategy_func(df, timeframe=TIMEFRAME, plot=False)
    except Exception as e:
        print(f'[ERROR] {symbol}: {e}')
        return

    if not backtest_res:
        return

    sig = strategy.what_is_signal(best, backtest_res, SIGNAL_LOOKBACK)
    if sig is None:
        return

    rois  = [r['risk_metrics']['roi']      for r in backtest_res if r.get('risk_metrics')]
    wins  = [r['risk_metrics']['win_rate'] for r in backtest_res if r.get('risk_metrics')]
    price = scraping.current_stock_price(symbol)

    direction = 'buy' if sig is True else 'sell'
    _print_signal(symbol, direction, price, sum(rois) / len(rois), sum(wins) / len(wins))


def scan_all(symbols: list[str]) -> None:
    print(f"\n{'='*62}")
    print(f"  SCAN STARTED  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ({len(symbols)} symbols)")
    print(f"{'='*62}")

    # ── One batch download covers all symbols ──────────────────────────
    t0 = time.time()
    stock_data = scraping.batch_download(symbols, period='2y', interval=TIMEFRAME,
                                         chunk_size=BATCH_SIZE)
    print(f'  Fetched {len(stock_data)} stocks in {time.time()-t0:.1f}s')

    if not stock_data:
        print('  No data returned — check internet connection.')
        return

    print(f"  {'time':8} {'dir':7} {'ticker':7} {'price':10} {'roi':10} {'win%'}")
    print(f"  {'-'*58}")

    # ── Run strategy analysis in parallel (CPU-bound, no more I/O) ────
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_process_symbol, sym, df): sym
                   for sym, df in stock_data.items()}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f'[ERROR] {sym}: {e}')

    print(f"\n  Scan complete in {time.time()-t0:.1f}s total."
          f"  Next scan in {SCAN_INTERVAL_SECONDS // 60} min.")


def run():
    print('Fetching S&P 500 symbols...')
    try:
        _, symbols = scraping.get_stocks()
    except Exception as e:
        print(f'Failed to fetch S&P 500 list: {e}')
        return

    print(f'Loaded {len(symbols)} symbols. Starting real-time scanner...')

    while True:
        try:
            scan_all(symbols)
        except KeyboardInterrupt:
            print('\nScanner stopped.')
            break
        except Exception as e:
            print(f'[SCAN ERROR] {e}')

        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == '__main__':
    run()
