# SmarTraid

A technical-analysis scanner for the S&P 500. It downloads price history for the
index, runs twelve independent signal strategies over each symbol, backtests them,
and surfaces the ones that currently have a live signal — in a terminal feed or a
Streamlit dashboard.

> **This is a research and learning tool, not trading advice.** Read
> [Reading the numbers](#reading-the-numbers) before you act on anything it prints.
> The scanner does not place orders and has no broker integration.

## Install

```bash
python -m venv env
env/Scripts/activate        # Windows;  source env/bin/activate elsewhere
pip install -r requirements.txt
```

## Run

```bash
# Terminal scanner
python server/main.py                                  # full index, daily bars
python server/main.py --symbols AAPL,MSFT --once       # one scan, two symbols
python server/main.py --timeframe 1h                   # intraday, gated on market hours

# Dashboard
python server/run.py            # or: streamlit run server/ui.py
```

`main.py --help` lists every option (timeframe, period, lookback, worker count).

## How it works

```
Wikipedia S&P 500 table  ──►  yfinance batch download  ──►  Polars indicators
                                                                    │
                                        ┌───────────────────────────┘
                                        ▼
                    12 signal strategies  ──►  backtest each on a training slice
                                        │
                                        ▼
                    best strategy selected on train, scored on held-out test
                                        │
                                        ▼
                          what_is_signal  ──►  BUY / SELL / nothing
```

### Indicators and strategies

| Indicator | Purpose | Signal |
|---|---|---|
| MACD | Momentum & trend | MACD line crosses the signal line |
| RSI | Momentum extremes | Crossing into >70 (sell) / <30 (buy) |
| MA | Trend | SMA20 crossing SMA150 |
| Bollinger Bands | Volatility & reversals | Close crossing a ±2σ band |
| VWAP | Price vs volume-weighted average | Close crossing a symmetric ±2% band |
| Ichimoku Cloud | Trend & momentum | Close crossing the cloud |
| Donchian Channel | Breakouts | Close beyond the prior 20-bar range |
| ATR Breakout | Volatility-scaled entries | Move beyond ±1.2 × ATR from the prior close |
| Parabolic SAR | Trend reversals | SAR dot flipping sides |
| Stochastic Oscillator | Momentum extremes | %K crossing into >80 / <30 |
| EMA Crossover | Trend changes | EMA12 crossing EMA26 |
| Previous High/Low | Breakouts | Close breaking the prior N-bar range |

No single indicator is reliable alone; the point of running twelve is to see
where they agree — and, just as usefully, where the backtest says they don't work.

## Reading the numbers

The dashboard and CLI report several columns, and the relationship between them
matters more than any one of them:

- **ROI** is measured **out-of-sample**. Strategy selection happens on the first
  70% of the history; the ROI you see comes from the remaining 30%, which played
  no part in choosing the strategy.
- **Train ROI** (dashboard only) is the in-sample figure that *did* the selecting.
  A large gap between Train ROI and ROI means the strategy is curve-fitted.
- **Buy & Hold** is holding the stock over the identical window, net of one round
  trip in costs. **Excess** is ROI minus that.
- **A strategy with negative Excess lost to doing nothing.** Most of them do, most
  of the time. That is the honest result, not a bug.

The backtest charges commission and slippage on both sides of every trade (5 bps
each by default) and fills a signal at the **next bar's open**, not the close it
was computed from. Stops are checked against the close and filled at the next
open, so intrabar gaps through a stop are not modelled — real stop fills can be
worse than reported.

## Layout

| Path | Role |
|---|---|
| [server/scraping.py](server/scraping.py) | yfinance downloads, Polars indicators, market calendar |
| [server/strategy.py](server/strategy.py) | The 12 strategies, backtest, selection, final verdict |
| [server/scanner.py](server/scanner.py) | Scan pipeline shared by the CLI and the dashboard |
| [server/main.py](server/main.py) | Terminal scanner CLI |
| [server/ui.py](server/ui.py) | Streamlit dashboard (live feed + per-stock analysis) |
| [server/plots.py](server/plots.py) | Plotly candlestick and signal markers |
| [server/run.py](server/run.py) | Dashboard launcher |
| [server/tests/](server/tests/) | pytest suite |

### Optional: regime and price models

[server/training.py](server/training.py) and [server/prediction.py](server/prediction.py)
fit a Gaussian HMM (market regime) and a RandomForest (next close). They need
MongoDB and are **not wired into the scanner** — the scanner's signals do not use
them. Enable them only if you want to experiment:

```bash
cp .env.example .env      # then fill in DB_USER, DB_PASSWORD, MODEL_STORE_KEY
docker compose up -d
```

Two things to know before trusting the price model: its features are raw price
levels, so it largely learns "tomorrow ≈ today" — compare `train_model()`'s
`naive_rmse` against its `rmse`. And stored models are signed with
`MODEL_STORE_KEY`, because deserialising a model is arbitrary code execution;
without a key set, the integrity check is a plain checksum and warns as much.

## Tests

```bash
python -m pytest server/tests -v
```

## Not implemented

Earlier versions of this README advertised features that do not exist. For the
record: there is **no** genetic-algorithm optimiser, **no** sentiment analysis, and
**no** live order routing. Position sizing exists only as a `position_fraction`
knob on the backtest, defaulting to all-in — do not read that default as a
recommendation.

## License

Not available for commercial use. Personal and research use only — see
[License.txt](License.txt).
