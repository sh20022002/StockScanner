"""All web scraping and data fetching functions."""
from datetime import timedelta, datetime, time
import random
import re
# datetime.time is already bound to the name `time` above, so the sleep
# function needs its own name rather than `import time`.
from time import sleep as _sleep

import pandas as pd
import polars as pl
import yfinance as yf
import yfinance.exceptions as yf_exceptions
import pytz, os
import requests

exchange_api_key = os.getenv('EXCHANGE_API_KEY')

_REQUIRED = ['Open', 'High', 'Low', 'Close', 'Volume']

# Minimum bars needed before the 150/200-period indicators mean anything.
MIN_BARS = 50

# ---------------------------------------------------------------------------
# Retry with backoff for transient failures (rate limiting, network hiccups)
# ---------------------------------------------------------------------------

# Errors worth retrying: Yahoo rate-limiting us, or the network hiccuping.
# Deliberately NOT retried: a bad/delisted symbol, a schema mismatch, or
# anything else that will fail identically on the next attempt — retrying
# those just burns time before failing anyway.
_RETRYABLE_EXCEPTIONS = (
    yf_exceptions.YFRateLimitError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)
_RETRYABLE_MESSAGE_MARKERS = ('rate limit', 'too many requests', '429')

DEFAULT_RETRIES = 3
DEFAULT_RETRY_BASE_DELAY = 1.5   # seconds; doubles each attempt, plus jitter


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, _RETRYABLE_EXCEPTIONS):
        return True
    msg = str(exc).lower()
    return any(marker in msg for marker in _RETRYABLE_MESSAGE_MARKERS)


def with_retries(fn, *args, retries: int = DEFAULT_RETRIES,
                 base_delay: float = DEFAULT_RETRY_BASE_DELAY,
                 label: str = '', **kwargs):
    """
    Call fn(*args, **kwargs), retrying with exponential backoff + jitter on
    what looks like a transient rate-limit or network failure.

    Re-raises immediately on anything else — a bad symbol or a real error in
    how we're calling the API will fail the same way every time, so retrying
    it just delays the failure instead of preventing it.
    """
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt >= retries or not _is_retryable(e):
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
            tag = f' [{label}]' if label else ''
            # Plain hyphen, not an em-dash: the default Windows console
            # codepage can't encode it and this print would crash mid-retry.
            print(f'[retry]{tag} {type(e).__name__}: {e} - '
                  f'retrying in {delay:.1f}s ({attempt + 1}/{retries})')
            _sleep(delay)
    raise last_exc  # pragma: no cover — loop always returns or raises above


def yahoo_symbol(symbol: str) -> str:
    """
    Translate an exchange ticker into Yahoo Finance's form.

    Wikipedia lists class shares with a dot (BRK.B, BF.B); Yahoo uses a hyphen
    (BRK-B, BF-B). Without this the affected symbols fail every single scan.
    """
    return symbol.strip().upper().replace('.', '-')


# ---------------------------------------------------------------------------
# Indicators — pure Polars, no pandas_ta (9× faster than the old approach)
# ---------------------------------------------------------------------------

def compute_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute all technical indicators using native Polars expressions.
    No pandas_ta, no extra conversions — runs in one pass per group.
    """
    # ── Moving averages ────────────────────────────────────────────────────
    df = df.with_columns([
        pl.col('Close').rolling_mean(window_size=20,  min_periods=1).alias('SMA20'),
        pl.col('Close').rolling_mean(window_size=50,  min_periods=1).alias('SMA50'),
        pl.col('Close').rolling_mean(window_size=100, min_periods=1).alias('SMA100'),
        pl.col('Close').rolling_mean(window_size=150, min_periods=1).alias('SMA150'),
        pl.col('Close').rolling_mean(window_size=200, min_periods=1).alias('SMA200'),
        pl.col('Close').ewm_mean(span=12, min_periods=1).alias('EMA12'),
        pl.col('Close').ewm_mean(span=20, min_periods=1).alias('EMA20'),
        pl.col('Close').ewm_mean(span=26, min_periods=1).alias('EMA26'),
    ])

    # ── MACD (needs EMA12/26 first) ────────────────────────────────────────
    df = df.with_columns(
        (pl.col('EMA12') - pl.col('EMA26')).alias('MACD')
    )
    df = df.with_columns(
        pl.col('MACD').ewm_mean(span=9, min_periods=1).alias('MACD_Signal')
    )
    df = df.with_columns(
        (pl.col('MACD') - pl.col('MACD_Signal')).alias('MACD_Hist')
    )

    # ── RSI ────────────────────────────────────────────────────────────────
    # diff() is null on the first row, which propagated all the way to RSI and
    # left a null the strategies then had to paper over. Seed it at zero and
    # report a flat or empty window as neutral 50 rather than a misleading 0.
    df = df.with_columns(pl.col('Close').diff().fill_null(0.0).alias('_delta'))
    df = df.with_columns([
        pl.col('_delta').clip(lower_bound=0.0).alias('_gain'),
        (-pl.col('_delta')).clip(lower_bound=0.0).alias('_loss'),
    ])
    df = df.with_columns([
        pl.col('_gain').rolling_mean(window_size=14, min_periods=1).alias('_ag'),
        pl.col('_loss').rolling_mean(window_size=14, min_periods=1).alias('_al'),
    ])
    df = df.with_columns(
        pl.when((pl.col('_ag') + pl.col('_al')) <= 0)
          .then(pl.lit(50.0))
          .otherwise(100.0 - 100.0 / (1.0 + pl.col('_ag') / (pl.col('_al') + 1e-10)))
          .alias('RSI')
    )

    # ── ATR ────────────────────────────────────────────────────────────────
    df = df.with_columns([
        (pl.col('High') - pl.col('Low')).alias('_hl'),
        (pl.col('High') - pl.col('Close').shift(1)).abs().alias('_hc'),
        (pl.col('Low')  - pl.col('Close').shift(1)).abs().alias('_lc'),
    ])
    df = df.with_columns(
        pl.max_horizontal(['_hl', '_hc', '_lc']).alias('_tr')
    )
    df = df.with_columns(
        pl.col('_tr').rolling_mean(window_size=14, min_periods=1).alias('ATR')
    )

    # ── Stochastic ─────────────────────────────────────────────────────────
    df = df.with_columns([
        pl.col('Low').rolling_min(window_size=14,  min_periods=1).alias('_ll'),
        pl.col('High').rolling_max(window_size=14, min_periods=1).alias('_hh'),
    ])
    df = df.with_columns(
        ((pl.col('Close') - pl.col('_ll')) /
         (pl.col('_hh') - pl.col('_ll') + 1e-10) * 100.0).alias('STOCH_%K')
    )
    df = df.with_columns(
        pl.col('STOCH_%K').rolling_mean(window_size=3, min_periods=1).alias('STOCH_%D')
    )

    # ── VWAP ───────────────────────────────────────────────────────────────
    df = df.with_columns(
        ((pl.col('High') + pl.col('Low') + pl.col('Close')) / 3.0).alias('_tp')
    )
    df = df.with_columns([
        (pl.col('_tp') * pl.col('Volume')).cum_sum().alias('_ctpv'),
        pl.col('Volume').cum_sum().alias('_cvol'),
    ])
    df = df.with_columns(
        (pl.col('_ctpv') / (pl.col('_cvol') + 1e-10)).alias('VWAP')
    )

    # Drop temp columns
    temps = [c for c in df.columns if c.startswith('_')]
    return df.drop(temps)


# ---------------------------------------------------------------------------
# Batch download — ONE network call for up to chunk_size symbols
# ---------------------------------------------------------------------------

def _normalise_index(df_pd: pd.DataFrame) -> pd.DataFrame:
    """Strip timezone, name the index 'Datetime', de-duplicate and sort."""
    if getattr(df_pd.index, 'tz', None) is not None:
        df_pd.index = df_pd.index.tz_localize(None)
    df_pd.index.name = 'Datetime'
    if df_pd.index.has_duplicates:
        df_pd = df_pd[~df_pd.index.duplicated(keep='first')]
    return df_pd.sort_index()


def _clean_ohlcv(df_pd: pd.DataFrame) -> pd.DataFrame | None:
    """
    Reduce a raw yfinance frame to clean, complete OHLCV rows.

    Yahoo emits a partial row for the session currently in progress: Volume is
    populated but Open/High/Low/Close are NaN. `dropna(how='all')` leaves that
    row in place, which used to crash parabolic_sar and turn every backtest ROI
    into NaN — so drop on the required columns instead.
    """
    if df_pd is None or df_pd.empty:
        return None
    if not all(c in df_pd.columns for c in _REQUIRED):
        return None
    df_pd = df_pd[_REQUIRED].dropna()
    return df_pd if not df_pd.empty else None


def batch_download(
    symbols: list,
    period: str = '2y',
    interval: str = '1d',
    chunk_size: int = 200,
    quiet: bool = False,
) -> dict:
    """
    Download OHLCV data for all symbols in batches using yf.download().
    Returns {symbol: pl.DataFrame} with indicators already computed, keyed by
    the symbol as it was passed in (not the Yahoo-normalised form).

    Replaces 500 individual HTTP calls with ~3 batch calls.
    """
    result: dict = {}
    n_chunks = -(-len(symbols) // chunk_size)

    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i : i + chunk_size]
        # Yahoo ticker -> the symbol the caller asked for (BRK-B -> BRK.B).
        wanted = {yahoo_symbol(s): s for s in chunk}

        try:
            raw = with_retries(
                yf.download, list(wanted),
                period=period, interval=interval, group_by='ticker',
                auto_adjust=True, progress=False, threads=True,
                label=f'batch_download chunk {i // chunk_size + 1}/{n_chunks}',
            )
        except Exception as e:
            print(f'[batch_download] chunk {i//chunk_size + 1} error: {e}')
            continue

        if raw is None or raw.empty:
            continue

        # yfinance returns per-ticker MultiIndex columns for a list of symbols
        # (including a 1-element list) and flat columns for a bare string.
        # Keying off len(chunk) silently returned zero rows for 1-symbol lists.
        is_multi = isinstance(raw.columns, pd.MultiIndex)
        available = set(raw.columns.get_level_values(0)) if is_multi else None

        for ysym, orig in wanted.items():
            try:
                if is_multi:
                    if ysym not in available:
                        continue
                    df_pd = raw[ysym].copy()
                else:
                    df_pd = raw.copy()

                df_pd = _clean_ohlcv(df_pd)
                if df_pd is None:
                    continue
                df_pd = _normalise_index(df_pd)

                df_pl = pl.from_pandas(df_pd, include_index=True)
                if len(df_pl) < MIN_BARS:
                    continue

                result[orig] = compute_indicators(df_pl)

            except Exception as e:
                print(f'[batch_download] {orig}: {e}')
                continue

        if not quiet:
            print(f'  downloaded chunk {i//chunk_size + 1}/{n_chunks}'
                  f' ({len(result)} stocks ready so far)')

    return result


# ---------------------------------------------------------------------------
# Single-stock fetch — kept for the UI analysis tab
# ---------------------------------------------------------------------------

def get_stock_data(
    stock,
    return_flags=None,
    DAYS=365,
    interval='1h',
    period=None,
):
    """
    Get historical stock data for a single symbol (used by the analysis UI tab).
    Uses compute_indicators() instead of pandas_ta.
    """
    if return_flags is None:
        return_flags = {'DF': True, 'INDICATORS': True}

    if interval == '1m'  and DAYS > 7:   DAYS = 7
    elif interval == '2m' and DAYS > 60:  DAYS = 60
    elif interval == '1h' and DAYS > 729: DAYS = 728

    end_date   = get_exchange_time()
    start_date = end_date - timedelta(days=DAYS)
    result     = {}

    try:
        ticker = yf.Ticker(yahoo_symbol(stock))
    except Exception as e:
        print(f'Error creating ticker for {stock}: {e}')
        return result

    # Optional metadata fields
    info = {}
    needs_info = any(return_flags.get(k) for k in ('SUMMARY', 'SUMMERY', 'MAX_KEY', 'DIVD', 'INFO'))
    if needs_info:
        try:
            # ticker.info is a property (the network call happens on access), so
            # it has to go through a lambda rather than being called directly —
            # passing ticker.info itself would already have made the (possibly
            # failing) request before with_retries ever saw it.
            info = with_retries(lambda: ticker.info, label=f'{stock} info')
        except Exception as e:
            print(f'Error fetching info for {stock}: {e}')

    if return_flags.get('SUMMARY') or return_flags.get('SUMMERY'):
        result['SUMMARY'] = info.get('longBusinessSummary', '')
    if return_flags.get('MAX_KEY'):
        result['MAX_KEY'] = info.get('recommendationKey', '')
    if return_flags.get('DIVD'):
        ts = info.get('lastDividendDate')
        if ts:
            d   = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            val = info.get('lastDividendValue', 0)
            result['DIVD'] = f'{d} : {val} $'
        else:
            result['DIVD'] = 'No Dividend'
    if return_flags.get('INFO'):
        result['INFO'] = info

    # Historical OHLCV
    if return_flags.get('DF'):
        try:
            if period is None:
                df = with_retries(ticker.history, start=start_date, end=end_date,
                                  interval=interval, label=f'{stock} history')
            else:
                df = with_retries(ticker.history, period=period, interval=interval,
                                  label=f'{stock} history')
        except Exception as e:
            print(f'Error fetching history for {stock}: {e}')
            return result

        if df.empty:
            print(f'No data for {stock}.')
            return result

        df.drop(columns=[c for c in ['Dividends', 'Stock Splits'] if c in df.columns],
                inplace=True)

        missing = [c for c in _REQUIRED if c not in df.columns]
        if missing:
            print(f'Missing columns for {stock}: {missing}')
            return result

        # Same partial-bar problem as batch_download — drop incomplete rows.
        df = _clean_ohlcv(df)
        if df is None:
            print(f'No complete bars for {stock}.')
            return result
        df = _normalise_index(df)

        if return_flags.get('INDICATORS', True):
            df_pl = pl.from_pandas(df, include_index=True)
            df_pl = compute_indicators(df_pl)
            # Return as pandas for backward-compat with the analysis UI
            result['DF'] = df_pl.to_pandas().set_index('Datetime')
        else:
            result['DF'] = df

    return result


def get_stock_news(symbol: str, count: int = 10) -> list[dict]:
    """
    Latest news headlines for a symbol.

    Only for one-off single-symbol lookups (same rule as current_stock_price):
    do not call this inside a scan loop.

    Returns:
        [{'title', 'summary', 'publisher', 'link', 'published_at', 'content_type'}, ...]
        Newest first, as yfinance returns them. Empty list on any failure —
        news is a nice-to-have, not something that should break the caller.
    """
    try:
        ticker = yf.Ticker(yahoo_symbol(symbol))
        raw = with_retries(lambda: ticker.news, label=f'{symbol} news')
    except Exception as e:
        print(f'Error fetching news for {symbol}: {e}')
        return []

    items = []
    for entry in raw[:count] if raw else []:
        # yfinance nests article fields under 'content' as of the version this
        # was written against; fall back to flat top-level keys in case that
        # shape drifts again, rather than silently returning nothing.
        content = entry.get('content') or entry
        url_obj = content.get('clickThroughUrl') or content.get('canonicalUrl')
        link = url_obj.get('url') if isinstance(url_obj, dict) else None
        link = link or entry.get('link')

        provider = content.get('provider')
        publisher = provider.get('displayName') if isinstance(provider, dict) else None
        publisher = publisher or entry.get('publisher')

        title = content.get('title')
        if not title:
            continue
        items.append({
            'title':        title,
            'summary':      content.get('summary'),
            'publisher':    publisher,
            'link':         link,
            'published_at': content.get('pubDate') or entry.get('providerPublishTime'),
            'content_type': content.get('contentType'),
        })

    return items


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def current_stock_price(symbol: str) -> float | None:
    """
    Latest close for a symbol, or None if it cannot be determined.

    Returns None rather than a placeholder — a fabricated $1.00 price used to
    flow straight into the dashboard as if it were real.

    Only for one-off single-symbol lookups. Do NOT call this inside a scan:
    the batch download already carries the last close.
    """
    try:
        ticker = yf.Ticker(yahoo_symbol(symbol))
        df = with_retries(ticker.history, period='5d', label=f'{symbol} price')
        if df.empty or 'Close' not in df.columns:
            return None
        close = df['Close'].dropna()
        return float(close.iloc[-1]) if not close.empty else None
    except Exception as e:
        print(f'Error getting price for {symbol}: {e}')
        return None


def get_tickers():
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    resp = requests.get(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    return pd.read_html(resp.text)[0].values


def get_stocks():
    """
    Return (names, symbols) for the S&P 500, with symbols already normalised
    to Yahoo's form so downstream callers never have to think about it.

    Superseded as the scanner's default universe by get_us_equities(), which
    covers the whole US market rather than one index. Kept because it's a
    useful, cheap, smaller universe on its own.
    """
    tickers = get_tickers()
    names   = [t[1] for t in tickers]
    symbols = [yahoo_symbol(str(t[0])) for t in tickers]
    return names, symbols


# NASDAQ Global/Global Select (NMS), NYSE (NYQ), NYSE American (ASE) — the
# three exchanges that between them list virtually every US common stock.
US_EQUITY_EXCHANGES = ('NMS', 'NYQ', 'ASE')

# UI-facing "Universe" control -> the exchange(s) it scans. Keys match what
# the web app's exchange-select sends.
EXCHANGE_GROUPS = {
    'all':    US_EQUITY_EXCHANGES,
    'nasdaq': ('NMS',),
    'nyse':   ('NYQ',),
    'amex':   ('ASE',),
}

DEFAULT_MIN_MARKET_CAP = 2_000_000_000   # $2B — mid-cap and up; ~2,100 symbols as of writing

# Yahoo's sector taxonomy (GICS-derived) — every value the "Sector" filter on
# the web app's screener can be set to. Fixed set, not derived from data:
# unlike market cap, Yahoo's screener query needs an exact string match, and
# the 'eq' operator does not do a "list distinct sectors" query for us.
GICS_SECTORS = (
    'Technology', 'Healthcare', 'Financial Services', 'Consumer Cyclical',
    'Industrials', 'Communication Services', 'Consumer Defensive',
    'Energy', 'Utilities', 'Real Estate', 'Basic Materials',
)

# Yahoo's ticker suffix convention: -PA.._PZ marks a preferred share series,
# -WT/-WS a warrant, -U a unit, -R/-RT a right. A share CLASS suffix (BRK-B,
# PBR-A) is a single letter with no leading P, so it survives this filter.
# This is a heuristic — Yahoo's screener has no "is common stock" flag to
# check against directly — but it is the documented convention and matches
# what every other consumer of Yahoo tickers relies on.
_NON_COMMON_SUFFIX = re.compile(r'-(P[A-Z]|W[TS]?|U|R|RT)$')

_us_equities_cache: dict = {}
_US_EQUITIES_CACHE_TTL = timedelta(hours=6)


def _is_common_stock(symbol: str) -> bool:
    return _NON_COMMON_SUFFIX.search(symbol) is None


def get_us_equities(min_market_cap: float = DEFAULT_MIN_MARKET_CAP,
                    exchanges: tuple = US_EQUITY_EXCHANGES,
                    sector: str | None = None,
                    max_results: int | None = None) -> list[dict]:
    """
    All US-exchange-listed common stocks above a market cap floor.

    Uses Yahoo's screener (yfinance's yf.screen/EquityQuery), which filters by
    market cap server-side and returns up to 250 results per request — a
    handful of paginated calls for the whole universe, not one HTTP call per
    symbol. Fetching market cap for thousands of tickers via yf.Ticker(...).info
    one at a time was the anti-pattern this project already removed once
    (Strategy.__init__ used to fetch its own quote per symbol); this is the
    batched equivalent for building the universe itself.

    Args:
        min_market_cap: Floor in dollars. Lower thresholds mean more symbols
            and proportionally more paginated requests plus a longer scan —
            there is no hard ceiling here, only in what a caller then does
            with the result.
        exchanges: Yahoo exchange codes to include.
        sector: Optional exact match against Yahoo's sector taxonomy (see
            GICS_SECTORS) — filtered server-side, same as market cap. None
            means every sector.
        max_results: Optional cap on how many symbols to return (still sorted
            by market cap descending, so this keeps the largest names).

    Returns:
        [{'symbol', 'name', 'market_cap'}, ...] sorted by market cap descending.
    """
    cache_key = (round(min_market_cap), tuple(exchanges), sector, max_results)
    cached = _us_equities_cache.get(cache_key)
    if cached and (datetime.now() - cached[0]) < _US_EQUITIES_CACHE_TTL:
        return cached[1]

    clauses = [
        yf.EquityQuery('is-in', ['exchange', *exchanges]),
        yf.EquityQuery('gt', ['intradaymarketcap', min_market_cap]),
    ]
    if sector:
        clauses.append(yf.EquityQuery('eq', ['sector', sector]))
    query = yf.EquityQuery('and', clauses)

    page_size = 250
    results = []
    seen = set()
    offset = 0
    total = None

    while total is None or offset < total:
        if max_results is not None and len(results) >= max_results:
            break
        page = with_retries(yf.screen, query, offset=offset, size=page_size,
                            sortField='intradaymarketcap', sortAsc=False,
                            label=f'get_us_equities offset={offset}')
        total = page.get('total', 0)
        quotes = page.get('quotes', [])
        if not quotes:
            break

        for q in quotes:
            symbol = q.get('symbol')
            if not symbol or symbol in seen or not _is_common_stock(symbol):
                continue
            seen.add(symbol)
            results.append({
                'symbol':     symbol,
                'name':       q.get('shortName') or q.get('longName') or symbol,
                'market_cap': q.get('marketCap'),
            })
            if max_results is not None and len(results) >= max_results:
                break

        offset += len(quotes)

    _us_equities_cache[cache_key] = (datetime.now(), results)
    return results


_sector_map_cache: dict = {}
_SECTOR_MAP_CACHE_TTL = timedelta(hours=6)

# Coverage floor for the sector map, distinct from DEFAULT_MIN_MARKET_CAP:
# this backs sector *labelling* (e.g. "what sector did today's signals come
# from"), not a scan universe, so it should recognise as many symbols as
# plausible rather than match whatever cap floor one particular scan used.
SECTOR_MAP_MIN_MARKET_CAP = 100_000_000   # $100M — micro-cap and up


def get_sector_map(min_market_cap: float = SECTOR_MAP_MIN_MARKET_CAP,
                   exchanges: tuple = US_EQUITY_EXCHANGES) -> dict[str, str]:
    """
    {symbol: sector} for every US equity above min_market_cap, across all of
    GICS_SECTORS.

    Yahoo's screener quotes do not actually carry a 'sector' field (confirmed
    empirically — every quote comes back with sector=None), so the only way
    to label a symbol's sector from the screener is to already know which
    sector query it matched. This runs one get_us_equities() call per sector
    — still a handful of batched, paginated requests total, not one .info
    call per symbol — and tags every symbol in each result with the sector
    that was queried for it.
    """
    cache_key = (round(min_market_cap), tuple(exchanges))
    cached = _sector_map_cache.get(cache_key)
    if cached and (datetime.now() - cached[0]) < _SECTOR_MAP_CACHE_TTL:
        return cached[1]

    mapping: dict[str, str] = {}
    for sector in GICS_SECTORS:
        equities = get_us_equities(min_market_cap, exchanges=exchanges, sector=sector)
        for e in equities:
            mapping[e['symbol']] = sector

    _sector_map_cache[cache_key] = (datetime.now(), mapping)
    return mapping


def get_exchange_time() -> datetime:
    ny = pytz.timezone('America/New_York')
    return datetime.now(ny)


def get_exchange_rate(from_currency: str, to_currency: str):
    if not exchange_api_key:
        return None
    url = f'https://v6.exchangerate-api.com/v6/{exchange_api_key}/latest/{from_currency}'
    resp = requests.get(url, timeout=30)
    if resp.status_code == 200:
        return resp.json()['conversion_rates'].get(to_currency)
    return None


def is_nyse_open() -> bool:
    now = get_exchange_time()
    nyse_open  = time(9, 30)
    nyse_close = time(16, 0)

    holidays = [
        datetime(now.year, 1, 1),
        next_weekday(datetime(now.year, 1, 15), 0),
        next_weekday(datetime(now.year, 2, 15), 0),
        easter_monday(now.year) - timedelta(days=3),
        next_weekday(datetime(now.year, 5, 25), 0),
        datetime(now.year, 7, 4),
        next_weekday(datetime(now.year, 9, 1), 0),
        next_weekday(datetime(now.year, 11, 22), 3),
        datetime(now.year, 12, 25),
        datetime(now.year, 12, 24),
    ]

    if 0 <= now.weekday() <= 4:
        if not any(h.date() == now.date() for h in holidays):
            if nyse_open <= now.time() <= nyse_close:
                return True
    return False


def next_weekday(d: datetime, weekday: int) -> datetime:
    ahead = weekday - d.weekday()
    if ahead <= 0:
        ahead += 7
    return d + timedelta(ahead)


def easter_monday(year: int) -> datetime:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day   = ((h + l - 7 * m + 114) % 31) + 1
    return datetime(year, month, day) + timedelta(days=1)
