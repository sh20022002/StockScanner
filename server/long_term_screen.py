"""
Long-term value screen.

Ranks symbols on a "good ratio" between three long-term signals: price
sitting in a healthy uptrend relative to its 150-period moving average, a
reasonable P/E, and real (positive) earnings behind it. This is a genuinely
different methodology from the 12 strategies in strategy.py — see
LONG_TERM_TIMEFRAMES's docstring in scanner.py, which is explicit that the
existing 'horizon' field is a timeframe proxy, not a fundamentals screen.

Kept out of the Strategy/STRATEGY_NAMES machinery on purpose: those 12
strategies each backtest a bar-to-bar crossover signal with a real
train/test P&L. P/E and EPS don't move bar-to-bar, so there is no
meaningful backtest here — this is a pass/fail screen with a ranking score,
run on demand rather than on every scan cycle (see
scraping.get_fundamentals for why it stays off the auto-scan hot path).
"""
import scraping

DEFAULT_MIN_TREND_RATIO = 1.0    # price at or above its SMA150
DEFAULT_MAX_TREND_RATIO = 1.5    # ...but not more than 50% above it (overextended)
DEFAULT_MAX_PE = 35.0


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _score(trend_ratio: float, pe: float, earnings_yield: float,
          min_trend_ratio: float, max_trend_ratio: float, max_pe: float) -> float:
    """
    0-100 ranking score across symbols that already passed every threshold —
    it does not gate inclusion, so a high sub-score can't smuggle in a
    symbol that failed the trend/P/E/EPS bounds themselves.

    Split evenly three ways: trend health, valuation, and earnings yield
    (EPS/price — the inverse of P/E, expressed as a return so "good EPS
    relative to price" scores independently of the P/E band chosen above).
    """
    trend_score = _clamp01((trend_ratio - min_trend_ratio) / (max_trend_ratio - min_trend_ratio)) * 34
    value_score = _clamp01((max_pe - pe) / max_pe) * 33
    # 8%+ earnings yield (roughly a P/E of 12.5 or better) earns full marks.
    yield_score = _clamp01(earnings_yield / 0.08) * 33
    return round(trend_score + value_score + yield_score, 1)


def screen(symbols: list,
          timeframe: str = '1d',
          period: str = '2y',
          min_trend_ratio: float = DEFAULT_MIN_TREND_RATIO,
          max_trend_ratio: float = DEFAULT_MAX_TREND_RATIO,
          max_pe: float = DEFAULT_MAX_PE,
          stock_data: dict | None = None,
          quiet: bool = True) -> list:
    """
    Rank symbols by the long-term value ratio, best first.

    Two-stage on purpose: the trend filter runs first against OHLCV data
    that's already batch-downloaded for free (SMA150 is computed by
    scraping.compute_indicators on every frame already), so the separate,
    per-symbol P/E-and-EPS network call in scraping.get_fundamentals only
    ever runs for the survivors instead of the whole universe.

    Returns a list of dicts (symbol, price, sma150, trend_ratio, pe_ratio,
    eps, earnings_yield, score, strategy, horizon), sorted by score
    descending.
    """
    if stock_data is None:
        stock_data = scraping.batch_download(
            symbols, period=period, interval=timeframe, quiet=quiet)

    trend_pass = {}
    for sym, df in stock_data.items():
        if 'SMA150' not in df.columns or len(df) == 0:
            continue
        close = df['Close'][-1]
        sma150 = df['SMA150'][-1]
        if close is None or sma150 is None or sma150 <= 0:
            continue
        ratio = close / sma150
        if min_trend_ratio <= ratio <= max_trend_ratio:
            trend_pass[sym] = (float(close), float(sma150), float(ratio), str(df['Datetime'][-1]))

    if not trend_pass:
        return []

    fundamentals = scraping.get_fundamentals(list(trend_pass.keys()), quiet=quiet)

    results = []
    for sym, (close, sma150, trend_ratio, time) in trend_pass.items():
        fund = fundamentals.get(sym)
        if not fund:
            continue
        pe, eps = fund.get('trailing_pe'), fund.get('trailing_eps')
        if pe is None or eps is None or pe <= 0 or eps <= 0 or pe > max_pe:
            continue

        earnings_yield = eps / close
        results.append({
            'symbol':         sym,
            'time':           time,
            'price':          round(close, 2),
            'sma150':         round(sma150, 2),
            'trend_ratio':    round(trend_ratio, 3),
            'pe_ratio':       round(pe, 2),
            'eps':            round(eps, 2),
            'earnings_yield': round(earnings_yield * 100, 2),
            'score':          _score(trend_ratio, pe, earnings_yield,
                                     min_trend_ratio, max_trend_ratio, max_pe),
            'strategy':       'long_term_value',
            'horizon':        'long_term',
        })

    results.sort(key=lambda r: -r['score'])
    return results
