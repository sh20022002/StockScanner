"""All web scraping and data fetching functions."""
from datetime import timedelta, datetime, time
import pandas as pd
import polars as pl
import yfinance as yf
import pytz, os
import requests

exchange_api_key = os.getenv('EXCHANGE_API_KEY')

_REQUIRED = ['Open', 'High', 'Low', 'Close', 'Volume']

# Minimum bars needed before the 150/200-period indicators mean anything.
MIN_BARS = 50


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
            raw = yf.download(
                list(wanted),
                period=period,
                interval=interval,
                group_by='ticker',
                auto_adjust=True,
                progress=False,
                threads=True,
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
            info = ticker.info
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
                df = ticker.history(start=start_date, end=end_date, interval=interval)
            else:
                df = ticker.history(period=period, interval=interval)
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
        df = yf.Ticker(yahoo_symbol(symbol)).history(period='5d')
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
    """
    tickers = get_tickers()
    names   = [t[1] for t in tickers]
    symbols = [yahoo_symbol(str(t[0])) for t in tickers]
    return names, symbols


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
