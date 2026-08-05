"""
SmarTraid monitor station — FastAPI + Server-Sent Events.

Run with:  python server/run.py
       or:  uvicorn server.web.app:app --reload   (from the repo root)

The scanner runs as a background asyncio task inside this process. Each new
signal is persisted (server/signal_log.py) and pushed to every connected
browser over /api/stream without the browser having to poll or the page having
to reload.
"""
import asyncio
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# rl.live (torch) must import before scraping/strategy (pandas) — on Windows
# this process has been observed to access-violate loading torch's c10.dll if
# pandas' compiled extensions initialise first. Import order is the fix; see
# the longer comment in server/rl/features.py.
try:
    import rl.live as rl_live
except Exception as _rl_import_error:      # torch/RL stack is optional
    rl_live = None
    logging.getLogger('smartraid.web').warning(
        'RL live-inference hook unavailable: %s', _rl_import_error)

import polars as pl

import scanner
import scraping
import signal_log
import strategy

log = logging.getLogger('smartraid.web')

STATIC_DIR = Path(__file__).parent / 'static'

app = FastAPI(title='SmarTraid Monitor Station')


# ---------------------------------------------------------------------------
# JSON safety
# ---------------------------------------------------------------------------

def _clean(obj):
    """
    Replace NaN/Infinity with None recursively.

    Python's json module happily emits the literals NaN/Infinity by default,
    which is not valid JSON and makes JS's JSON.parse throw. Backtests can
    legitimately produce NaN (e.g. a strategy with zero trades), so scrub it
    at the API boundary rather than upstream.
    """
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def json_ok(payload):
    return _clean(payload)


# ---------------------------------------------------------------------------
# Scanner state — one scanner per process, shared by every connected browser
# ---------------------------------------------------------------------------

class ScannerState:
    """
    RL has no on/off switch here on purpose: it is always blended into a scan
    whenever server/rl/checkpoints/best.pt exists. `rl_available` in status()
    reports whether that's currently true; there is no companion flag to
    disable it when it is.
    """
    def __init__(self):
        self.running       = False
        self.symbols: list[str] = []
        self.timeframe      = scanner.DEFAULT_TIMEFRAME
        self.lookback        = scanner.DEFAULT_LOOKBACK
        self.min_margin      = scanner.DEFAULT_MIN_MARGIN
        self.exchange        = 'all'          # key into scraping.EXCHANGE_GROUPS
        self.sector: str | None = None        # None = every sector
        self.min_market_cap  = scraping.DEFAULT_MIN_MARKET_CAP
        self.last_scan_at: str | None = None
        self.last_scan_count = 0
        self.last_error: str | None = None
        self.task: asyncio.Task | None = None
        self.subscribers: set[asyncio.Queue] = set()

    def status(self) -> dict:
        return {
            'running':          self.running,
            'symbols_count':    len(self.symbols),
            'timeframe':        self.timeframe,
            'lookback':         self.lookback,
            'min_margin':       self.min_margin,
            'exchange':         self.exchange,
            'sector':           self.sector,
            'min_market_cap':   self.min_market_cap,
            'rl_available':     rl_live is not None and rl_live.model_available(),
            'scan_interval_s':  scanner.scan_interval_seconds(self.timeframe),
            'market_open':      scraping.is_nyse_open(),
            'last_scan_at':     self.last_scan_at,
            'last_scan_count':  self.last_scan_count,
            'last_error':       self.last_error,
        }


state = ScannerState()


async def broadcast(event: dict):
    dead = []
    for q in state.subscribers:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        state.subscribers.discard(q)


async def _scanner_loop():
    """Mirrors scanner semantics from the old Streamlit thread, as an asyncio task."""
    while state.running:
        try:
            if scanner.should_scan_now(state.timeframe):
                want_data = rl_live is not None
                result = await asyncio.to_thread(
                    scanner.scan, state.symbols,
                    timeframe=state.timeframe, lookback=state.lookback,
                    min_margin=state.min_margin, quiet=True, return_data=want_data,
                )
                if want_data:
                    found, stock_data = result
                    found = await asyncio.to_thread(rl_live.annotate, found, stock_data)
                else:
                    found = result
                fresh = signal_log.add_signals(found)
                state.last_scan_at = datetime.now(timezone.utc).isoformat()
                state.last_scan_count = len(found)
                state.last_error = None
                for entry in fresh:
                    await broadcast({'type': 'signal', 'data': json_ok(entry)})
                await broadcast({'type': 'status', 'data': json_ok(state.status())})
                wait = scanner.scan_interval_seconds(state.timeframe)
            else:
                await broadcast({'type': 'status', 'data': json_ok(state.status())})
                wait = scanner.CLOSED_MARKET_SLEEP
        except Exception as e:
            log.exception('scanner loop error')
            state.last_error = str(e)
            await broadcast({'type': 'status', 'data': json_ok(state.status())})
            wait = scanner.scan_interval_seconds(state.timeframe)

        for _ in range(wait):
            if not state.running:
                return
            await asyncio.sleep(1)


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')


@app.get('/')
def index():
    return FileResponse(STATIC_DIR / 'index.html')


# ---------------------------------------------------------------------------
# Scanner control
# ---------------------------------------------------------------------------

def _resolve_exchanges(exchange: str) -> tuple:
    if exchange not in scraping.EXCHANGE_GROUPS:
        raise HTTPException(400, f'Unknown exchange "{exchange}".')
    return scraping.EXCHANGE_GROUPS[exchange]


def _resolve_sector(sector: str | None) -> str | None:
    """'' (an empty query param/form field) and None both mean 'every sector'."""
    sector = sector or None
    if sector is not None and sector not in scraping.GICS_SECTORS:
        raise HTTPException(400, f'Unknown sector "{sector}".')
    return sector


@app.get('/api/status')
def get_status():
    return json_ok(state.status())


@app.post('/api/scanner/start')
async def start_scanner(body: dict):
    if state.running:
        raise HTTPException(409, 'Scanner already running')

    exchange = body.get('exchange', 'all')
    exchanges = _resolve_exchanges(exchange)
    sector = _resolve_sector(body.get('sector'))

    try:
        min_market_cap = float(body.get('min_market_cap', scraping.DEFAULT_MIN_MARKET_CAP))
    except (TypeError, ValueError):
        raise HTTPException(400, 'min_market_cap must be a number')
    if min_market_cap < 0:
        raise HTTPException(400, 'min_market_cap must not be negative')

    try:
        equities = await asyncio.to_thread(
            scraping.get_us_equities, min_market_cap, exchanges, sector)
    except Exception as e:
        raise HTTPException(502, f'Failed to load the US equities universe: {e}')
    symbols = [e['symbol'] for e in equities]
    if not symbols:
        raise HTTPException(400, 'No symbols matched that exchange/sector/market cap filter.')

    state.symbols        = symbols
    state.timeframe       = body.get('timeframe', scanner.DEFAULT_TIMEFRAME)
    state.lookback         = int(body.get('lookback', scanner.DEFAULT_LOOKBACK))
    state.min_margin       = int(body.get('min_margin', scanner.DEFAULT_MIN_MARGIN))
    state.exchange          = exchange
    state.sector            = sector
    state.min_market_cap   = min_market_cap
    state.running          = True
    state.last_error       = None
    state.task = asyncio.create_task(_scanner_loop())

    await broadcast({'type': 'status', 'data': json_ok(state.status())})
    return json_ok(state.status())


@app.post('/api/scanner/stop')
async def stop_scanner():
    state.running = False
    if state.task:
        state.task.cancel()
        state.task = None
    await broadcast({'type': 'status', 'data': json_ok(state.status())})
    return json_ok(state.status())


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

@app.get('/api/signals')
def get_signals(direction: str | None = None, symbol: str | None = None,
                limit: int = Query(200, ge=1, le=500)):
    signals = signal_log.load_signals()
    if direction and direction != 'All':
        signals = [s for s in signals if s.get('direction') == direction]
    if symbol:
        needle = symbol.upper()
        signals = [s for s in signals if needle in s.get('symbol', '')]
    return json_ok(signals[:limit])


@app.get('/api/signals/summary')
def get_signals_summary():
    signals = signal_log.load_signals()
    buys  = [s for s in signals if s.get('direction') == 'BUY']
    sells = [s for s in signals if s.get('direction') == 'SELL']
    beat  = [s for s in signals if (s.get('excess_roi') or 0) > 0]
    avg_excess = (sum(s.get('excess_roi', 0) for s in signals) / len(signals)) if signals else 0
    hist = [s.get('excess_roi', 0) for s in signals]
    return json_ok({
        'total':       len(signals),
        'buys':        len(buys),
        'sells':       len(sells),
        'beat_bench':  len(beat),
        'avg_excess':  avg_excess,
        'excess_hist': hist,
    })


@app.delete('/api/signals')
def clear_signals():
    signal_log.save_signals([])
    return {'ok': True}


def sector_summary(signals: list[dict], sector_map: dict[str, str], today: str) -> list[dict]:
    """
    Group today's signals by sector, for the "Today's Signals by Sector" panel.

    Pulled out as a pure function so it's testable without spinning up the
    HTTP layer or hitting the sector map's own network call — see
    server/tests/test_web.py. `today` is a 'YYYY-MM-DD' string; signals whose
    time doesn't start with it are excluded. A symbol missing from sector_map
    (below the sector map's own market cap floor, or a delisting/rename
    the screener no longer carries) is grouped under 'Unknown' rather than
    silently dropped — the daily total across sectors should still add up to
    the actual number of signals today.
    """
    todays = [s for s in signals if str(s.get('time', ''))[:10] == today]

    buckets: dict[str, dict] = {}
    for s in todays:
        sector = sector_map.get(s.get('symbol'), 'Unknown')
        b = buckets.setdefault(sector, {'sector': sector, 'total': 0, 'buys': 0,
                                        'sells': 0, 'beat_bench': 0, '_excess_sum': 0.0})
        b['total'] += 1
        if s.get('direction') == 'BUY':
            b['buys'] += 1
        elif s.get('direction') == 'SELL':
            b['sells'] += 1
        excess = s.get('excess_roi') or 0
        b['_excess_sum'] += excess
        if excess > 0:
            b['beat_bench'] += 1

    rows = [{
        'sector':     b['sector'],
        'total':      b['total'],
        'buys':       b['buys'],
        'sells':      b['sells'],
        'beat_bench': b['beat_bench'],
        'avg_excess': round(b['_excess_sum'] / b['total'], 2),
    } for b in buckets.values()]
    rows.sort(key=lambda r: -r['total'])
    return rows


@app.get('/api/signals/sector-summary')
async def get_sector_summary():
    """
    Today's signals grouped by sector.

    "Today" is the NYSE-local date (scraping.get_exchange_time), matching
    what signals are actually keyed on — a daily bar's date, not a wall-clock
    timestamp in the server's or browser's own timezone.
    """
    today = scraping.get_exchange_time().strftime('%Y-%m-%d')
    signals = signal_log.load_signals()
    try:
        sector_map = await asyncio.to_thread(scraping.get_sector_map)
    except Exception as e:
        raise HTTPException(502, f'Failed to load the sector map: {e}')
    return json_ok({'date': today, 'sectors': sector_summary(signals, sector_map, today)})


# ---------------------------------------------------------------------------
# Symbol detail — candles, indicators, backtest table, verdict
# ---------------------------------------------------------------------------

@app.get('/api/universe')
async def get_universe(min_market_cap: float = scraping.DEFAULT_MIN_MARKET_CAP,
                       exchange: str = 'all', sector: str | None = None):
    """
    Symbol picker autocomplete — the same US-equities universe the scanner
    itself uses (same exchange/sector/market-cap filters), so 'pick a symbol
    and Analyse' covers whatever the scan covers. scraping.get_us_equities()
    caches this internally (6h TTL, keyed by all three filters), so repeated
    page loads don't re-hit Yahoo's screener.
    """
    exchanges = _resolve_exchanges(exchange)
    sector = _resolve_sector(sector)
    try:
        equities = await asyncio.to_thread(
            scraping.get_us_equities, min_market_cap, exchanges, sector)
    except Exception as e:
        raise HTTPException(502, f'Failed to load the US equities universe: {e}')
    return json_ok([{'name': e['name'], 'symbol': e['symbol']} for e in equities])


@app.get('/api/universe/count')
async def get_universe_count(min_market_cap: float = scraping.DEFAULT_MIN_MARKET_CAP,
                             exchange: str = 'all', sector: str | None = None):
    """
    Just the size of the universe an exchange/sector/market-cap filter
    resolves to — for the dashboard's slider to show a real number instead of
    a guess as the user drags it. Backed by the same cache as /api/universe,
    so this is cheap for any combination already looked up this session.
    """
    exchanges = _resolve_exchanges(exchange)
    sector = _resolve_sector(sector)
    try:
        equities = await asyncio.to_thread(
            scraping.get_us_equities, min_market_cap, exchanges, sector)
    except Exception as e:
        raise HTTPException(502, f'Failed to load the US equities universe: {e}')
    return {'count': len(equities), 'min_market_cap': min_market_cap}


def _candles(df: pl.DataFrame) -> list[dict]:
    out = []
    for row in df.iter_rows(named=True):
        ts = row['Datetime']
        out.append({
            'time':  int(ts.replace(tzinfo=timezone.utc).timestamp()),
            'open':  row['Open'], 'high': row['High'],
            'low':   row['Low'],  'close': row['Close'],
            'volume': row.get('Volume'),
        })
    return out


def _overlay_series(df: pl.DataFrame, columns: list[str]) -> dict:
    series = {}
    for col in columns:
        if col not in df.columns:
            continue
        pts = []
        for row in df.iter_rows(named=True):
            v = row[col]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            ts = row['Datetime']
            pts.append({'time': int(ts.replace(tzinfo=timezone.utc).timestamp()), 'value': v})
        series[col] = pts
    return series


@app.get('/api/symbol/{symbol}')
async def get_symbol(symbol: str, interval: str = '1d', period: str = '2y',
                     overlays: str = 'SMA20,SMA50'):
    symbol = scraping.yahoo_symbol(symbol)

    def _fetch():
        data = scraping.get_stock_data(
            symbol, interval=interval, period=period,
            return_flags={'DF': True, 'INDICATORS': True,
                          'MAX_KEY': True, 'SUMMARY': True, 'DIVD': True, 'INFO': True})
        return data

    data = await asyncio.to_thread(_fetch)
    if not data or data.get('DF') is None or data['DF'].empty:
        raise HTTPException(404, f'No data for {symbol}')

    df_pl = pl.from_pandas(data['DF'], include_index=True)

    def _analyse():
        s = strategy.Strategy(symbol=symbol)
        best, results = s.evaluate_strategies(df_pl, timeframe=interval)
        return best, results

    best, results = await asyncio.to_thread(_analyse)
    verdict = strategy.what_is_signal(best, results, scanner.DEFAULT_LOOKBACK,
                                      min_margin=scanner.DEFAULT_MIN_MARGIN)

    best_signals = None
    if results:
        best_signals = next((r['signals'] for r in results
                             if r['strategy_func'] == best), results[0]['signals'])

    markers = []
    if best_signals is not None:
        for row in best_signals.iter_rows(named=True):
            if row['Buy_Signal']:
                markers.append({'time': int(row['Datetime'].replace(tzinfo=timezone.utc).timestamp()),
                                'position': 'belowBar', 'color': '#00c853',
                                'shape': 'arrowUp', 'text': 'BUY'})
            elif row['Sell_Signal']:
                markers.append({'time': int(row['Datetime'].replace(tzinfo=timezone.utc).timestamp()),
                                'position': 'aboveBar', 'color': '#ff1744',
                                'shape': 'arrowDown', 'text': 'SELL'})

    strategy_rows = []
    for r in results:
        rm, tm = r.get('risk_metrics', {}), r.get('train_metrics', {})
        strategy_rows.append({
            'strategy':      r['strategy_func'],
            'selected':      r['strategy_func'] == best,
            'train_roi':     tm.get('roi'),
            'roi':           rm.get('roi'),
            'benchmark_roi': rm.get('benchmark_roi'),
            'excess_roi':    rm.get('excess_roi'),
            'win_rate':      rm.get('win_rate'),
            'trades':        rm.get('trades'),
            'max_drawdown':  rm.get('max_drawdown'),
        })
    strategy_rows.sort(key=lambda r: (r['excess_roi'] is None, -(r['excess_roi'] or 0)))

    info = data.get('INFO', {}) or {}
    return json_ok({
        'symbol':   symbol,
        'interval': interval,
        'candles':  _candles(df_pl),
        'overlays': _overlay_series(df_pl, [c.strip() for c in overlays.split(',') if c.strip()]),
        'markers':  markers,
        'verdict':  {True: 'BUY', False: 'SELL', None: None}[verdict],
        'best_strategy': best,
        'strategies':    strategy_rows,
        'meta': {
            'last_close':       float(df_pl['Close'][-1]),
            'recommendation':   data.get('MAX_KEY'),
            'dividend':         data.get('DIVD'),
            'market_cap':       info.get('marketCap'),
            'business_summary': data.get('SUMMARY'),
            'sector':           info.get('sector'),
            'industry':         info.get('industry'),
            'employees':        info.get('fullTimeEmployees'),
            'website':          info.get('website'),
            'pe_ratio':         info.get('trailingPE'),
            'forward_pe_ratio': info.get('forwardPE'),
        },
    })


@app.get('/api/symbol/{symbol}/entry-price')
async def get_symbol_entry_price(symbol: str,
                                 direction: str = Query(..., pattern='^(BUY|SELL)$')):
    """
    A suggested limit price to get into a position, derived from the last
    ~10 days of hourly bars rather than the daily/weekly signal price — see
    strategy.suggest_entry_price for the reasoning.

    Kept as its own endpoint (same reasoning as /news and /signal-history):
    it needs a second, hourly-interval fetch that the main chart doesn't, so
    it shouldn't be able to slow down or break loading the chart itself.
    """
    symbol = scraping.yahoo_symbol(symbol)

    def _fetch():
        return scraping.get_stock_data(
            symbol, interval='1h', DAYS=10,
            return_flags={'DF': True, 'INDICATORS': True})

    data = await asyncio.to_thread(_fetch)
    df = data.get('DF')
    if df is None or df.empty:
        raise HTTPException(404, f'No hourly data for {symbol}')

    df_pl = pl.from_pandas(df, include_index=True)
    suggestion = strategy.suggest_entry_price(df_pl, direction)
    if suggestion is None:
        raise HTTPException(404, f'Not enough recent hourly history for {symbol}')

    return json_ok({'symbol': symbol, 'direction': direction, **suggestion})


@app.get('/api/symbol/{symbol}/news')
async def get_symbol_news(symbol: str, count: int = Query(10, ge=1, le=25)):
    """
    Latest news headlines for a symbol — a separate, slower fetch kept off the
    main /api/symbol/{symbol} response so a news hiccup can't break the chart
    that loads alongside it (same reasoning as /signal-history).
    """
    symbol = scraping.yahoo_symbol(symbol)
    try:
        news = await asyncio.to_thread(scraping.get_stock_news, symbol, count)
    except Exception as e:
        raise HTTPException(502, f'Failed to load news for {symbol}: {e}')
    return json_ok({'symbol': symbol, 'news': news})


def call_performance(signal_price: float | None, current_price: float | None,
                     direction: str | None) -> tuple[float | None, float | None]:
    """
    (price_change_pct, call_return_pct) since one past signal.

    call_return_pct is price_change_pct signed so a positive number always
    means the call was right: a BUY that went up, or a SELL that went down.
    Pulled out as a pure function so it's testable without spinning up the
    HTTP layer — see server/tests/test_web.py.
    """
    if not current_price or not signal_price:
        return None, None
    price_change_pct = (current_price - signal_price) / signal_price * 100
    call_return_pct = price_change_pct if direction == 'BUY' else -price_change_pct
    return price_change_pct, call_return_pct


@app.get('/api/symbol/{symbol}/signal-history')
async def get_symbol_signal_history(symbol: str):
    """
    Every past signal the scanner has recorded for one exact symbol, plus how
    each call actually performed — the real price move since that signal,
    signed so a positive number always means the call was right (a BUY that
    went up, or a SELL that went down).

    This is deliberately not a backtested or compounded equity curve: each
    entry's roi/excess_roi is the out-of-sample backtest metric AS SCORED AT
    THAT SCAN, already shown elsewhere. call_return_pct here is the one number
    on this page that isn't a backtest artifact — it's what the stock actually
    did afterward.
    """
    symbol = scraping.yahoo_symbol(symbol)
    signals = [s for s in signal_log.load_signals() if s.get('symbol') == symbol]
    signals.sort(key=lambda s: str(s.get('time', '')))   # chronological, oldest first

    current_price = await asyncio.to_thread(scraping.current_stock_price, symbol)

    enriched = []
    for s in signals:
        price_change_pct, call_return_pct = call_performance(
            s.get('price'), current_price, s.get('direction'))
        enriched.append({
            **s,
            'price_change_pct': price_change_pct,
            'call_return_pct':  call_return_pct,
        })

    return json_ok({
        'symbol':        symbol,
        'current_price': current_price,
        'signals':       enriched,
    })


# ---------------------------------------------------------------------------
# Live event stream
# ---------------------------------------------------------------------------

@app.get('/api/stream')
async def stream():
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    state.subscribers.add(queue)

    async def gen():
        try:
            yield {'event': 'status', 'data': json.dumps(json_ok(state.status()))}
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                    yield {'event': event['type'], 'data': json.dumps(event['data'])}
                except asyncio.TimeoutError:
                    yield {'event': 'ping', 'data': '{}'}
        finally:
            state.subscribers.discard(queue)

    return EventSourceResponse(gen())
