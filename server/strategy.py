"""
Signal generation, backtesting and strategy selection.

Twelve independent technical strategies each turn an OHLCV+indicator frame into
Buy_Signal / Sell_Signal columns. `Strategy.evaluate_strategies` backtests all of
them, picks the best on a training slice and reports metrics on a held-out test
slice, and `what_is_signal` reduces the whole set to one verdict.

Read the docstring on `backtest_strategy` before trusting any number it returns.
"""
import numpy as np
import polars as pl


# Signal strategies, in the order they are evaluated.
STRATEGY_NAMES = (
    'macd', 'rsi', 'ma', 'bollinger_bands', 'vwap', 'ichimoku_cloud',
    'donchian_channel', 'atr_breakout', 'parabolic_sar',
    'stochastic_oscillator', 'ema_crossover', 'prev_high_low',
)

# Default cost model, per side, as a fraction of the fill price.
DEFAULT_COMMISSION = 0.0005   # 5 bps
DEFAULT_SLIPPAGE   = 0.0005   # 5 bps


class Strategy:
    """
    Evaluates the strategy set for a single symbol.

    Attributes:
        symbol (str): The stock symbol.
        loss_percent (float): Stop-loss distance, in percent. None disables it.
        profit_percent (float): Stop-profit distance, in percent. None disables it.

    Constructing this is deliberately free of network I/O — the caller already
    holds the price data, and fetching per-symbol quotes here defeated the whole
    point of the batch download.
    """

    def __init__(self, symbol: str, loss_percent: float | None = 7.0,
                 profit_percent: float | None = 10.0, **kwargs):
        self.symbol         = symbol
        self.loss_percent   = loss_percent
        self.profit_percent = profit_percent
        # Any extra fundamentals the caller happens to have (beta, debtToEquity…).
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        return "\n".join(f'{key}: {value}' for key, value in self.__dict__.items())

    # -----------------------------------------------------------------------
    # Orchestration
    # -----------------------------------------------------------------------

    def detect_signals(self, df: pl.DataFrame) -> dict | None:
        """
        Run every strategy and return {strategy_name: signals_df}.

        Each result is re-aligned to the full 'Datetime' axis of df, so every
        returned frame has exactly len(df) rows even for the strategies that
        filter warm-up rows internally.

        Runs sequentially on purpose. These are pure-Python/Polars compute paths,
        so the ThreadPoolExecutor this used to use bought nothing against the GIL
        while nesting pools several layers deep. Parallelism now lives at the
        symbol level, in scanner.py.
        """
        if df is None or df.is_empty():
            print("No data available.")
            return None

        dt_dtype = df.schema['Datetime']
        axis     = df.select(['Datetime'])
        results  = {}

        for name in STRATEGY_NAMES:
            try:
                result = getattr(self, name)(df)
                if result is not None:
                    result = (
                        axis.join(
                            result.with_columns(pl.col('Datetime').cast(dt_dtype)),
                            on='Datetime', how='left'
                        )
                        .with_columns([
                            pl.col('Buy_Signal').fill_null(False),
                            pl.col('Sell_Signal').fill_null(False),
                        ])
                    )
                results[name] = result
            except Exception as e:
                print(f"Error in strategy {name}: {e}")
                results[name] = None

        return results

    def evaluate_strategies(self, df: pl.DataFrame, timeframe: str = '1d',
                            train_fraction: float = 0.7, **backtest_kwargs) -> tuple:
        """
        Backtest every strategy and return (best_strategy_name, results).

        Selection and reporting are deliberately separated:

          * `best` is whichever strategy performed best on the first
            `train_fraction` of the data.
          * `risk_metrics` on each result is measured on the remaining
            out-of-sample slice — that is what callers should display.
          * `train_metrics` is kept alongside so the in-sample/out-of-sample gap
            is visible. A large gap means the strategy is curve-fitted.

        Picking the best of twelve strategies on the same data they are scored on
        guarantees a flattering number and predicts nothing, which is exactly what
        the previous single-slice version did.
        """
        signals = self.detect_signals(df)
        if not signals:
            return None, []

        n     = len(df)
        split = int(n * train_fraction)
        # Both slices need enough bars to open and close a trade.
        if split < 3 or (n - split) < 3:
            split = None

        kwargs = dict(
            stop_loss_percent=self.loss_percent,
            stop_profit_percent=self.profit_percent,
        )
        kwargs.update(backtest_kwargs)

        results    = []
        best_name  = None
        best_train = float('-inf')

        for name, sig in signals.items():
            if sig is None or len(sig) != n:
                continue
            try:
                if split is None:
                    perf, metrics = self.backtest_strategy(df, sig, **kwargs)
                    train_perf, train_metrics = perf, metrics
                else:
                    train_perf, train_metrics = self.backtest_strategy(
                        df[:split], sig[:split], **kwargs)
                    perf, metrics = self.backtest_strategy(
                        df[split:], sig[split:], **kwargs)
            except Exception as e:
                print(f"Error backtesting {name}: {e}")
                continue

            results.append({
                'strategy_func': name,
                'performance':   perf,            # out-of-sample P&L
                'risk_metrics':  metrics,         # out-of-sample
                'train_metrics': train_metrics,   # in-sample, for comparison
                'signals':       sig,
            })

            # Selection uses the training slice only.
            if np.isfinite(train_perf) and train_perf > best_train:
                best_train = train_perf
                best_name  = name

        return best_name, results

    # -----------------------------------------------------------------------
    # Backtest
    # -----------------------------------------------------------------------

    def backtest_strategy(self, df: pl.DataFrame, signals_df: pl.DataFrame,
                          commission: float = DEFAULT_COMMISSION,
                          slippage: float = DEFAULT_SLIPPAGE,
                          stop_loss_percent: float | None = None,
                          stop_profit_percent: float | None = None,
                          leverage: float = 1.0,
                          trailing_stop: bool = True,
                          position_fraction: float = 1.0) -> tuple:
        """
        Backtest one signal series. Returns (pnl, metrics).

        Execution model, and its limits:

          * A signal on bar i is filled at bar i+1's OPEN. Filling at the same
            bar's close — as this used to — lets the backtest trade on a price it
            could not have known yet.
          * `commission` and `slippage` are charged on BOTH sides of every trade,
            against the fill price. The old model charged 1% on exit only.
          * Stops are evaluated against bar i's close and exit at bar i+1's open,
            so an intrabar gap through the stop is not modelled. Real stop fills
            can be worse than this reports.
          * `position_fraction` of available cash is committed per trade.

        metrics contains:
            win_rate, trades, roi, max_drawdown, time_frame_days,
            benchmark_roi (buy-and-hold over the same window) and excess_roi.
            A strategy that does not beat benchmark_roi is not worth running.
        """
        close  = df['Close'].to_numpy().astype(float)
        open_  = df['Open'].to_numpy().astype(float)
        dates  = df['Datetime'].to_numpy()
        buy    = signals_df['Buy_Signal'].fill_null(False).to_numpy()
        sell   = signals_df['Sell_Signal'].fill_null(False).to_numpy()

        n = len(close)
        cost = commission + slippage

        cash = starting = 100_000.0
        pos = 0.0            # signed share count
        pos_type = 0         # 0 flat, 1 long, -1 short
        entry = 0.0
        trade_cash = 0.0     # cash committed to the open trade
        borrow = 0.0         # margin borrowed for the open trade
        high_water = low_water = 0.0
        sl = sp = None
        peak = cash
        max_dd = 0.0
        trades = 0
        wins = 0

        def equity(price: float) -> float:
            if pos_type == 0:
                return cash
            if pos_type == 1:
                return cash + pos * price - borrow
            # Short: margin held aside plus mark-to-market gain on the borrowed shares.
            return cash + trade_cash + abs(pos) * (entry - price)

        # Stop one bar early: bar i's signal needs bar i+1 to fill against.
        for i in range(n - 1):
            p    = close[i]
            fill = open_[i + 1]
            if not (np.isfinite(p) and np.isfinite(fill)) or fill <= 0:
                continue

            if pos_type == 0:
                if buy[i] or sell[i]:
                    direction = 1 if buy[i] else -1
                    # Buying pays above the quote, selling receives below it.
                    px = fill * (1 + cost) if direction == 1 else fill * (1 - cost)
                    trade_cash = cash * position_fraction
                    notional   = leverage * trade_cash
                    if trade_cash <= 0 or px <= 0:
                        continue
                    borrow   = notional - trade_cash if direction == 1 else 0.0
                    pos      = direction * notional / px
                    entry    = px
                    cash    -= trade_cash
                    pos_type = direction
                    high_water = low_water = px
                    if direction == 1:
                        sl = px * (1 - stop_loss_percent   / 100) if stop_loss_percent   else None
                        sp = px * (1 + stop_profit_percent / 100) if stop_profit_percent else None
                    else:
                        sl = px * (1 + stop_loss_percent   / 100) if stop_loss_percent   else None
                        sp = px * (1 - stop_profit_percent / 100) if stop_profit_percent else None
                    trades += 1

            elif pos_type == 1:
                if trailing_stop and p > high_water:
                    high_water = p
                    if stop_loss_percent:
                        sl = high_water * (1 - stop_loss_percent / 100)
                if (sl and p <= sl) or (sp and p >= sp) or sell[i]:
                    px = fill * (1 - cost)
                    cash += pos * px - borrow
                    if px > entry:
                        wins += 1
                    pos = 0.0; pos_type = 0; borrow = 0.0; trade_cash = 0.0
                    sl = sp = None

            else:  # short
                if trailing_stop and p < low_water:
                    low_water = p
                    if stop_loss_percent:
                        sl = low_water * (1 + stop_loss_percent / 100)
                if (sl and p >= sl) or (sp and p <= sp) or buy[i]:
                    px = fill * (1 + cost)
                    cash += trade_cash + abs(pos) * (entry - px)
                    if px < entry:
                        wins += 1
                    pos = 0.0; pos_type = 0; borrow = 0.0; trade_cash = 0.0
                    sl = sp = None

            eq = equity(p)
            if eq > peak:
                peak = eq
            if peak > 0:
                dd = (peak - eq) / peak
                if dd > max_dd:
                    max_dd = dd

        # Force-close anything still open at the last available price, and count
        # it — otherwise win_rate is measured against a shifting denominator.
        if pos_type != 0:
            fp = close[-1]
            if np.isfinite(fp) and fp > 0:
                if pos_type == 1:
                    px = fp * (1 - cost)
                    cash += pos * px - borrow
                    if px > entry:
                        wins += 1
                else:
                    px = fp * (1 + cost)
                    cash += trade_cash + abs(pos) * (entry - px)
                    if px < entry:
                        wins += 1
            pos = 0.0; pos_type = 0; borrow = 0.0; trade_cash = 0.0

        win_rate = wins / trades * 100 if trades else 0.0
        roi      = (cash - starting) / starting * 100

        # Buy-and-hold over the identical window, net of one round trip.
        finite = close[np.isfinite(close)]
        if len(finite) >= 2 and finite[0] > 0:
            bh_roi = ((finite[-1] * (1 - cost)) / (finite[0] * (1 + cost)) - 1) * 100
        else:
            bh_roi = 0.0

        try:
            days = int((dates[-1] - dates[0]) / np.timedelta64(1, 'D'))
        except Exception:
            days = n

        return cash - starting, {
            'win_rate':        round(win_rate, 2),
            'trades':          trades,
            'time_frame_days': days,
            'roi':             round(roi, 2),
            'max_drawdown':    round(max_dd * 100, 2),
            'benchmark_roi':   round(bh_roi, 2),
            'excess_roi':      round(roi - bh_roi, 2),
        }

    # -----------------------------------------------------------------------
    # Signal strategies
    # -----------------------------------------------------------------------

    def macd(self, df: pl.DataFrame) -> pl.DataFrame | None:
        """MACD (Moving Average Convergence Divergence)

        A trend-following momentum indicator built from the relationship between
        a 12- and 26-period EMA.
          MACD line:   EMA12 - EMA26
          Signal line: 9-period EMA of the MACD line
        Buy when the MACD line crosses above the signal line, sell when it
        crosses below. Use case: momentum shifts and entry/exit timing.
        """
        required_columns = ['MACD', 'MACD_Signal', 'Datetime']
        if not all(col in df.columns for col in required_columns):
            print("Required columns for MACD strategy are missing.")
            return None

        buy_signals = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
        sell_signals = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))

        return generate_signal(
            sell_signals.fill_null(False).to_list(),
            buy_signals.fill_null(False).to_list(),
            df['Datetime'].to_list(),
        )

    def rsi(self, df: pl.DataFrame, Upper_Band: float = 70, Lower_Band: float = 30) -> pl.DataFrame | None:
        """
        Relative Strength Index — a 0-100 momentum oscillator.

        Buy as RSI crosses down into oversold territory (< Lower_Band), sell as it
        crosses up into overbought (> Upper_Band). Use case: timing entries and
        exits against stretched momentum.

        Args:
            df: Historical price data, must carry an 'RSI' column.
            Upper_Band: Overbought threshold for sell signals.
            Lower_Band: Oversold threshold for buy signals.
        """
        if 'RSI' not in df.columns:
            print("Required column 'RSI' is missing.")
            return None

        buy_signals = (df['RSI'] < Lower_Band) & (df['RSI'].shift(1) >= Lower_Band)
        sell_signals = (df['RSI'] > Upper_Band) & (df['RSI'].shift(1) <= Upper_Band)

        return generate_signal(
            sell_signals.fill_null(False).to_list(),
            buy_signals.fill_null(False).to_list(),
            df['Datetime'].to_list(),
        )

    def ma(self, df: pl.DataFrame) -> pl.DataFrame | None:
        """
        Moving-average crossover: fast SMA20 against slow SMA150.

        Price above the average is a bullish trend, below is bearish. Buy on the
        upward cross, sell on the downward one. Use case: trend direction.
        """
        if not all(c in df.columns for c in ['SMA20', 'SMA150']):
            print("Required columns for MA strategy are missing.")
            return None

        df = df.with_columns(pl.col('SMA150').cast(pl.Float64, strict=False).alias('SMA150'))

        buy_signals = (df['SMA20'] > df['SMA150']) & (df['SMA20'].shift(1) <= df['SMA150'].shift(1))
        sell_signals = (df['SMA20'] < df['SMA150']) & (df['SMA20'].shift(1) >= df['SMA150'].shift(1))

        return generate_signal(
            sell_signals.fill_null(False).to_list(),
            buy_signals.fill_null(False).to_list(),
            df['Datetime'].to_list(),
        )

    def bollinger_bands(self, df: pl.DataFrame, window: int = 20,
                        num_std_dev: float = 2.0) -> pl.DataFrame | None:
        """
        Bollinger Bands — an SMA with volatility bands at ±num_std_dev.

        Buy when price crosses below the lower band (stretched down), sell when it
        crosses above the upper band. Use case: volatility extremes and reversals.
        """
        if not all(c in df.columns for c in ['Close', 'SMA20']):
            print("Required columns for Bollinger Bands are missing.")
            return None

        df = df.with_columns(pl.col('Close').rolling_std(window_size=window).alias('STD20'))
        df = df.filter(pl.col('SMA20').is_not_null() & pl.col('STD20').is_not_null())
        if df.is_empty():
            return None

        df = df.with_columns([
            (pl.col('SMA20') + (pl.col('STD20') * num_std_dev)).alias('Upper_Band'),
            (pl.col('SMA20') - (pl.col('STD20') * num_std_dev)).alias('Lower_Band'),
        ])

        buy_signals = (df['Close'] < df['Lower_Band']) & (df['Close'].shift(1) >= df['Lower_Band'].shift(1))
        sell_signals = (df['Close'] > df['Upper_Band']) & (df['Close'].shift(1) <= df['Upper_Band'].shift(1))

        return generate_signal(
            sell_signals.fill_null(False).to_list(),
            buy_signals.fill_null(False).to_list(),
            df['Datetime'].to_list(),
        )

    def ema_crossover(self, df: pl.DataFrame) -> pl.DataFrame | None:
        """
        EMA crossover: fast EMA12 against slow EMA26.

        Buy on the golden cross (EMA12 crossing above EMA26), sell on the death
        cross. Use case: trend changes, faster to react than the SMA version.
        """
        if not all(c in df.columns for c in ['EMA12', 'EMA26']):
            print("Required columns for EMA crossover are missing.")
            return None

        df = df.filter(pl.col('EMA12').is_not_null() & pl.col('EMA26').is_not_null())
        if df.is_empty():
            return None

        buy_signals = (df['EMA12'] > df['EMA26']) & (df['EMA12'].shift(1) <= df['EMA26'].shift(1))
        sell_signals = (df['EMA12'] < df['EMA26']) & (df['EMA12'].shift(1) >= df['EMA26'].shift(1))

        return generate_signal(
            sell_signals.fill_null(False).to_list(),
            buy_signals.fill_null(False).to_list(),
            df['Datetime'].to_list(),
        )

    def stochastic_oscillator(self, df: pl.DataFrame, k_window: int = 12, d_window: int = 3,
                              overbought: float = 80, oversold: float = 30) -> pl.DataFrame | None:
        """
        Stochastic oscillator — where the close sits within its recent range.

        Buy as %K crosses down into the oversold zone, sell as it crosses up into
        overbought. Use case: stretched momentum and reversals.

        Args:
            df: Historical price data with High/Low/Close.
            k_window: Lookback for %K.
            d_window: Smoothing window for the %D signal line.
            overbought: Sell threshold.
            oversold: Buy threshold.
        """
        if not isinstance(k_window, int) or k_window <= 0:
            raise ValueError(f"k_window must be a positive integer. Received k_window={k_window}")
        if not isinstance(d_window, int) or d_window <= 0:
            raise ValueError(f"d_window must be a positive integer. Received d_window={d_window}")

        required_columns = ['High', 'Low', 'Close']
        if not all(col in df.columns for col in required_columns):
            print(f"DataFrame must contain {required_columns} columns.")
            return _empty_signals(df)

        if len(df) < max(k_window, d_window):
            print(f"Insufficient data for Stochastic Oscillator. "
                  f"Required: {max(k_window, d_window)}, Available: {len(df)}")
            return _empty_signals(df)

        # Polars is immutable — the previous `df.sort(...)` and `df.drop(...)`
        # calls discarded their results and did nothing at all.
        df = df.sort('Datetime')

        df = df.with_columns([
            pl.col('Low').rolling_min(window_size=k_window, min_periods=1).alias('Lowest_Low'),
            pl.col('High').rolling_max(window_size=k_window, min_periods=1).alias('Highest_High'),
        ])
        df = df.with_columns(
            (pl.col('Highest_High') - pl.col('Lowest_Low')).alias('Denominator')
        )
        # A flat range would divide by zero; treat it as mid-range.
        df = df.with_columns(
            pl.when(pl.col('Denominator') == 0)
              .then(None)
              .otherwise(pl.col('Denominator'))
              .alias('Denominator')
        )
        df = df.with_columns(
            (((pl.col('Close') - pl.col('Lowest_Low')) / pl.col('Denominator')) * 100)
            .fill_null(50.0).alias('%K')
        )
        df = df.with_columns(
            pl.col('%K').rolling_mean(window_size=d_window, min_periods=1).alias('%D')
        )

        buy_signals = (df['%K'] < oversold) & (df['%K'].shift(1) >= oversold)
        sell_signals = (df['%K'] > overbought) & (df['%K'].shift(1) <= overbought)

        return generate_signal(
            sell_signals.fill_null(False).to_list(),
            buy_signals.fill_null(False).to_list(),
            df['Datetime'].to_list(),
        )

    def parabolic_sar(self, df: pl.DataFrame, step: float = 0.02,
                      max_step: float = 0.2) -> pl.DataFrame | None:
        """
        Parabolic SAR — a trailing dot that flips sides with the trend.

        Buy when the dot flips below price, sell when it flips above. Use case:
        trend reversals and trailing stop placement.

        Args:
            df: Historical price data with High/Low/Close.
            step: Acceleration factor increment.
            max_step: Acceleration factor ceiling.
        """
        required_columns = ['High', 'Low', 'Close']
        if not all(col in df.columns for col in required_columns):
            print(f"DataFrame must contain {required_columns} columns for Parabolic SAR.")
            return _empty_signals(df)

        # Pull to Python lists once. These must be complete: a single null High or
        # Low used to raise TypeError and kill this strategy for every symbol.
        highs = df['High'].to_list()
        lows  = df['Low'].to_list()
        n     = len(df)
        if n < 2 or highs[0] is None or lows[0] is None:
            return _empty_signals(df)

        sar   = [0.0] * n
        trend = [1] * n
        buy_signals  = [False] * n
        sell_signals = [False] * n

        ep = highs[0]
        af = step
        sar[0] = lows[0]

        for i in range(1, n):
            hi, lo = highs[i], lows[i]
            if hi is None or lo is None:
                # Carry the previous state through an incomplete bar.
                sar[i], trend[i] = sar[i - 1], trend[i - 1]
                continue

            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
            if trend[i - 1] == 1:
                if lo < sar[i]:
                    trend[i] = -1
                    sar[i] = ep
                    ep = lo
                    af = step
                    sell_signals[i] = True
                else:
                    trend[i] = 1
                    if hi > ep:
                        ep = hi
                        af = min(af + step, max_step)
            else:
                if hi > sar[i]:
                    trend[i] = 1
                    sar[i] = ep
                    ep = hi
                    af = step
                    buy_signals[i] = True
                else:
                    trend[i] = -1
                    if lo < ep:
                        ep = lo
                        af = min(af + step, max_step)

        return generate_signal(sell_signals, buy_signals, df['Datetime'].to_list())

    def atr_breakout(self, df: pl.DataFrame, multiplier: float = 1.2) -> pl.DataFrame | None:
        """
        ATR breakout — a volatility-scaled move away from the prior close.

        Buy when price closes more than multiplier × ATR above the previous
        close, sell on the equivalent move down. Use case: volatility-aware entries.
        """
        required_columns = ['Close', 'High', 'Low', 'ATR']
        if not all(col in df.columns for col in required_columns):
            print(f"DataFrame must contain {required_columns} columns for ATR Breakout.")
            return _empty_signals(df)

        df = df.with_columns([
            (pl.col('Close') + (pl.col('ATR') * multiplier)).alias('Upper_Breakout'),
            (pl.col('Close') - (pl.col('ATR') * multiplier)).alias('Lower_Breakout'),
        ])

        buy_signals = df['Close'] > df['Upper_Breakout'].shift(1)
        sell_signals = df['Close'] < df['Lower_Breakout'].shift(1)

        return generate_signal(
            sell_signals.fill_null(False).to_list(),
            buy_signals.fill_null(False).to_list(),
            df['Datetime'].to_list(),
        )

    def donchian_channel(self, df: pl.DataFrame, window: int = 20) -> pl.DataFrame | None:
        """
        Donchian channel — the highest high and lowest low over `window` bars.

        Buy on a close above the prior upper bound, sell below the lower bound.
        Use case: breakout and trend-following entries.
        """
        required_columns = ['High', 'Low', 'Close']
        if not all(col in df.columns for col in required_columns):
            print(f"DataFrame must contain {required_columns} columns.")
            return _empty_signals(df)

        df = df.with_columns([
            pl.col('High').rolling_max(window_size=window).alias('Donchian_High'),
            pl.col('Low').rolling_min(window_size=window).alias('Donchian_Low'),
        ])

        buy_signals = df['Close'] > df['Donchian_High'].shift(1)
        sell_signals = df['Close'] < df['Donchian_Low'].shift(1)

        return generate_signal(
            sell_signals.fill_null(False).to_list(),
            buy_signals.fill_null(False).to_list(),
            df['Datetime'].to_list(),
        )

    def ichimoku_cloud(self, df: pl.DataFrame, conversion_window: int = 9, base_window: int = 26,
                       leading_span_window: int = 52) -> pl.DataFrame | None:
        """
        Ichimoku cloud — trend, momentum and support/resistance in one overlay.

          Tenkan-sen (conversion): midpoint of the 9-bar range
          Kijun-sen  (base):       midpoint of the 26-bar range
          Senkou A/B (the cloud):  projected forward by base_window

        Buy when price crosses above the cloud, sell when it crosses below.

        Args:
            df: Historical price data with High/Low/Close.
            conversion_window: Lookback for the conversion line.
            base_window: Lookback for the base line, and the forward projection.
            leading_span_window: Lookback for leading span B.
        """
        required_columns = ['High', 'Low', 'Close']
        if not all(col in df.columns for col in required_columns):
            print(f"DataFrame must contain {required_columns} columns.")
            return _empty_signals(df)

        # Each line is the MIDPOINT of its range: (high + low) / 2. Without the
        # outer parentheses the division bound to the low alone, inflating the
        # conversion and base lines by roughly half the price.
        df = df.with_columns([
            ((pl.col('High').rolling_max(window_size=conversion_window)
              + pl.col('Low').rolling_min(window_size=conversion_window)) / 2).alias('Conversion_Line'),
            ((pl.col('High').rolling_max(window_size=base_window)
              + pl.col('Low').rolling_min(window_size=base_window)) / 2).alias('Base_Line'),
        ])
        df = df.with_columns([
            ((pl.col('Conversion_Line') + pl.col('Base_Line')) / 2)
            .shift(base_window).alias('Leading_Span_A'),
            ((pl.col('High').rolling_max(window_size=leading_span_window)
              + pl.col('Low').rolling_min(window_size=leading_span_window)) / 2)
            .shift(base_window).alias('Leading_Span_B'),
        ])
        df = df.with_columns([
            pl.max_horizontal(['Leading_Span_A', 'Leading_Span_B']).alias('Cloud_Top'),
            pl.min_horizontal(['Leading_Span_A', 'Leading_Span_B']).alias('Cloud_Bottom'),
        ])

        # Shift the price and the band together. Shifting the whole comparison
        # instead, as before, compared bar i-2 against band i-2 — off by one.
        buy_signals = (
            (pl.col('Close') > pl.col('Cloud_Top'))
            & (pl.col('Close').shift(1) <= pl.col('Cloud_Top').shift(1))
        )
        sell_signals = (
            (pl.col('Close') < pl.col('Cloud_Bottom'))
            & (pl.col('Close').shift(1) >= pl.col('Cloud_Bottom').shift(1))
        )
        df = df.with_columns([
            buy_signals.fill_null(False).alias('buy_signals'),
            sell_signals.fill_null(False).alias('sell_signals'),
        ])

        return generate_signal(
            df['sell_signals'].to_list(),
            df['buy_signals'].to_list(),
            df['Datetime'].to_list(),
        )

    def prev_high_low(self, df: pl.DataFrame, N: int = 20) -> pl.DataFrame | None:
        """
        Break of the previous N-bar high/low.

        At each bar, look back over the N bars ending one bar ago and take that
        range's extremes. Buy when the close breaks above the high, sell when it
        breaks below the low.

        N is in BARS, not minutes, so it must suit the timeframe being scanned —
        the old default of 390 ("one day of minute bars") swallowed 18 months of
        a daily series. It is clamped to a quarter of the available history so it
        can never consume the whole window.

        Args:
            df: Historical price data with Datetime/High/Low/Close.
            N: Lookback in bars.
        """
        required_columns = ['High', 'Low', 'Close']
        if not all(col in df.columns for col in required_columns):
            print(f"DataFrame must contain {required_columns} columns.")
            return _empty_signals(df)

        N = max(2, min(N, max(2, len(df) // 4)))

        df = df.with_columns([
            pl.col('Low').shift(1).rolling_min(window_size=N).alias('prev_n_min'),
            pl.col('High').shift(1).rolling_max(window_size=N).alias('prev_n_max'),
        ])

        buy_signals = df['Close'] >= df['prev_n_max']
        sell_signals = df['Close'] <= df['prev_n_min']

        return generate_signal(
            sell_signals.fill_null(False).to_list(),
            buy_signals.fill_null(False).to_list(),
            df['Datetime'].to_list(),
        )

    def vwap(self, df: pl.DataFrame, band: float = 0.02) -> pl.DataFrame | None:
        """
        Volume-weighted average price, traded as a mean-reversion band.

        Buy when price crosses above VWAP × (1 + band), sell when it crosses
        below VWAP × (1 - band).

        The band is symmetric. It previously used +2% for buys and -10% for
        sells, which made a sell arithmetically almost impossible — the strategy
        bought 19 times and sold 0 times over two years, so it scored like
        buy-and-hold and kept winning the "best strategy" slot on any rising stock.

        Uses the VWAP column from scraping.compute_indicators when present so the
        dashboard overlay and the signals agree on one definition.

        Args:
            df: Historical price data with Close and Volume.
            band: Half-width of the band, as a fraction of VWAP.
        """
        if 'Close' not in df.columns or 'Volume' not in df.columns:
            print("Required columns for VWAP are missing.")
            return _empty_signals(df)

        if 'VWAP' not in df.columns:
            if all(c in df.columns for c in ['High', 'Low']):
                typical = (pl.col('High') + pl.col('Low') + pl.col('Close')) / 3.0
            else:
                typical = pl.col('Close')
            df = df.with_columns(typical.alias('_tp'))
            df = df.with_columns([
                (pl.col('_tp') * pl.col('Volume')).cum_sum().alias('_ctpv'),
                pl.col('Volume').cum_sum().alias('_cvol'),
            ])
            df = df.with_columns(
                (pl.col('_ctpv') / (pl.col('_cvol') + 1e-10)).alias('VWAP')
            )

        upper = df['VWAP'] * (1 + band)
        lower = df['VWAP'] * (1 - band)

        buy_signals = (df['Close'] > upper) & (df['Close'].shift(1) <= upper.shift(1))
        sell_signals = (df['Close'] < lower) & (df['Close'].shift(1) >= lower.shift(1))

        return generate_signal(
            sell_signals.fill_null(False).to_list(),
            buy_signals.fill_null(False).to_list(),
            df['Datetime'].to_list(),
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def generate_signal(sell_signals: list, buy_signals: list, indexs: list) -> pl.DataFrame:
    """
    Build the standard signal frame.

    Args:
        sell_signals: Boolean list indicating sell signals.
        buy_signals: Boolean list indicating buy signals.
        indexs: The Datetime index for the signals.

    Returns:
        pl.DataFrame with Buy_Signal, Sell_Signal and Datetime columns.
    """
    if not (len(sell_signals) == len(buy_signals) == len(indexs)):
        raise ValueError("Length of sell_signals, buy_signals, and indexs must be equal.")

    return pl.DataFrame({
        'Buy_Signal':  pl.Series(buy_signals,  dtype=pl.Boolean),
        'Sell_Signal': pl.Series(sell_signals, dtype=pl.Boolean),
        'Datetime':    indexs,
    })


def _empty_signals(df: pl.DataFrame) -> pl.DataFrame:
    """
    An all-False signal frame aligned to df.

    The guard clauses used to return `pl.DataFrame(columns=..., index=...)`,
    which is not the Polars constructor signature and raised TypeError — so
    every one of those "safe" early exits actually crashed.
    """
    n = len(df)
    return pl.DataFrame({
        'Buy_Signal':  pl.Series([False] * n, dtype=pl.Boolean),
        'Sell_Signal': pl.Series([False] * n, dtype=pl.Boolean),
        'Datetime':    df['Datetime'] if 'Datetime' in df.columns else pl.Series([], dtype=pl.Datetime),
    })


def what_is_signal(best, backtest_res, n, min_margin: int = 2):
    """
    Reduce the strategy set to one verdict: True = buy, False = sell, None = none.

    Two paths, in order of evidence:

      1. The strategy selected on the training slice has fired in the last n
         bars — take its direction.
      2. Otherwise fall back to a vote of the remaining strategies, which must
         win by at least `min_margin` to count. Raising min_margin makes the
         scanner quieter and more selective; lowering it to 1 accepts a bare
         majority, which on the full index turned roughly half of all signals
         into 5-4 splits — noise dressed as consensus. The consensus threshold
         the original (never-called) `combine()` helper defined was 2.

    Three things this deliberately no longer does:

      * Gate on backtest ROI. That ROI is an in-sample, best-of-twelve selection
        artefact, and using it as an on/off switch suppressed every signal for
        every symbol — most mean-reversion strategies score negative in a rising
        market, so the gate was almost always shut.
      * Ignore `best`. The parameter was accepted and never read, so the chosen
        strategy had no influence on the verdict at all.
      * Resolve a tie to SELL, which it did by falling through to `return False`.

    Args:
        best: Name of the selected strategy, from evaluate_strategies.
        backtest_res: The results list from evaluate_strategies.
        n: How many recent bars count as "live".
        min_margin: Vote margin the fallback consensus must clear.

    Returns:
        True (buy), False (sell) or None (no actionable signal).
    """
    if not backtest_res or n < 1:
        return None

    votes: dict = {}
    for res in backtest_res:
        sig = res.get('signals')
        if sig is None or 'Buy_Signal' not in sig.columns or 'Sell_Signal' not in sig.columns:
            continue
        name = res.get('strategy_func')
        fired_buy  = bool(sig['Buy_Signal'][-n:].any())
        fired_sell = bool(sig['Sell_Signal'][-n:].any())
        # Silent, or contradicting itself — no clean vote either way.
        votes[name] = None if fired_buy == fired_sell else fired_buy

    if not votes:
        return None

    # The selected strategy speaks for itself when it has something to say.
    if best in votes and votes[best] is not None:
        return votes[best]

    clean = [v for v in votes.values() if v is not None]
    if not clean:
        return None

    buys  = sum(clean)
    sells = len(clean) - buys
    # A tie is no signal, not a sell. A one-vote win is not consensus either.
    if abs(buys - sells) < max(1, min_margin):
        return None
    return buys > sells
