# SmarTraid

A technical-analysis scanner for the US stock market, with a browser-based
monitor station and a reinforcement-learning agent blended into every scan. It
downloads price history for every common stock above a market cap threshold
you choose, runs twelve independent signal strategies over each one, backtests
them, and surfaces the ones that currently have a live signal — live, in a
dashboard your browser connects to.

> **This is a research and learning tool, not trading advice.** Read
> [Reading the numbers](#reading-the-numbers) before you act on anything it
> shows. The scanner does not place orders and has no broker integration.

## Install

```bash
python -m venv env
env/Scripts/activate        # Windows;  source env/bin/activate elsewhere
pip install -r requirements.txt
```

Torch (needed only for the RL agent, `server/rl/`) is pinned in
`requirements.txt`; see the comment there for a smaller CPU-only install.

## Run

```bash
# Monitor station — open http://127.0.0.1:8000
python server/run.py
python server/run.py --port 8080 --reload

# Terminal scanner (no browser, one process)
python server/main.py                                  # all US stocks >= $2B market cap
python server/main.py --min-market-cap 10e9            # >= $10B instead
python server/main.py --symbols AAPL,MSFT --once       # one scan, two symbols
python server/main.py --timeframe 1h                   # intraday, gated on market hours
```

`main.py --help` lists every CLI option (timeframe, period, lookback, worker
count, consensus margin, market cap floor).

Both the CLI and the monitor station blend the RL policy into every scan
automatically whenever `server/rl/checkpoints/best.pt` exists — see
[The RL agent](#the-rl-agent-serverrl).

## The monitor station

`server/run.py` starts a FastAPI app (`server/web/app.py`) that serves a single
page and pushes live signals to it over Server-Sent Events — no polling, no
full-page reloads. The scanner runs as a background task inside that same
process, started and stopped from the page itself:

- **Today's Signals by Sector** — a row of cards at the top of the page,
  today's signals grouped by GICS sector (count, buy/sell split, % that beat
  buy-and-hold, average excess ROI). Sector comes from
  `scraping.get_sector_map` — the screener's quotes don't actually carry a
  `sector` field, so this queries `get_us_equities` once per GICS sector and
  tags every symbol with the sector that was queried for it, the same
  batched-not-per-symbol principle as the rest of universe fetching. Cached
  6h, same as the universe itself.
- **Scanner Controls** — Universe (all US exchanges / NASDAQ / NYSE / NYSE
  American), Sector (any GICS sector or all of them), and a market cap floor
  with quick-select tiers (Nano/Micro/Small/Mid/Large/Mega) alongside the
  slider, which now spans $0–3000B instead of $0.1–50B. There is no "custom
  symbol list" mode anymore — every scan goes through the same
  exchange/sector/market-cap universe fetch. Start/Stop is a single toggle
  switch rather than two buttons.
- **Live Signal Feed** — updates the moment a scan finds something. Click any
  row (here or in Signal History below) to load that symbol.
- **Chart & Backtest** — a candlestick chart (TradingView's Lightweight
  Charts, vendored locally, no CDN) with SMA overlays and buy/sell markers,
  plus the full 12-strategy backtest table for whatever symbol you pick.
- **Past Signals** — appears once you've loaded a symbol: every signal the
  scanner has ever recorded for it, plotted on the chart as hollow circles
  (distinct from the current backtest's arrows — these are what actually
  fired historically), with a table and a bar chart of each call's *real*
  return since it fired, signed so positive always means the call was right.
  That's the one number here that isn't a backtest artifact.
- **Company & News** — sector, industry, market cap, trailing/forward P/E,
  analyst recommendation, dividend, and business summary
  (`scraping.get_stock_data`'s existing `INFO`/`SUMMARY` fetch — this was
  already being pulled from yfinance and simply wasn't rendered anywhere until
  now), plus the latest headlines for the symbol (`scraping.get_stock_news`,
  `yf.Ticker(...).news`). Both are one-off lookups fired when you load a
  symbol, same rule as `current_stock_price` — never called inside a scan
  loop.
- **Suggested entry price** — once a symbol has a BUY/SELL verdict, a second
  banner suggests a limit price to get in at, using the last ~10 days of
  *hourly* bars (`GET /api/symbol/{symbol}/entry-price`,
  `strategy.suggest_entry_price`) rather than the daily/weekly signal price.
  The daily bar that fires a signal is too coarse to time an entry with —
  price may already be extended well past the level that made the setup
  attractive. The suggestion pulls the entry toward the hourly 20-period SMA
  when price is stretched away from it (a pullback), floored/ceilinged by the
  recent hourly swing low/high so it never suggests a price that hasn't
  actually traded recently. If price is already through its own SMA, the
  suggestion is just the current price — there's no better pullback level in
  the window.
- **Signal Distribution** — buy/sell split and an excess-ROI histogram,
  drawn on canvas.
- **Signal History** — filterable table, CSV export.

The app has **no authentication by default** — `--host` defaults to
`127.0.0.1` on purpose. Don't bind `0.0.0.0` outside a network you trust
unless you've set `SMARTRAID_USER`/`SMARTRAID_PASSWORD` (HTTP Basic Auth,
`server/web/auth.py`) to gate it first. See [DEPLOY.md](DEPLOY.md) for
putting this somewhere other people can reach it.

## How the rule-based scanner works

```
Yahoo screener: US-listed common stocks   ──►  yfinance batch download  ──►  Polars indicators
above a market cap floor (server/scraping.py)                                     │
                                        ┌───────────────────────────────────────────┘
                                        ▼
                    12 signal strategies  ──►  backtest each on a training slice
                                        │
                                        ▼
                    best strategy selected on train, scored on held-out test
                                        │
                                        ▼
                          what_is_signal  ──►  BUY / SELL / nothing
```

### Universe

The default universe is every NASDAQ/NYSE/NYSE-American common stock above a
market cap floor (`$2B` by default — set it with `--min-market-cap` on the CLI
or the slider in the monitor station), fetched via Yahoo's screener
(`yfinance`'s `yf.screen`/`EquityQuery`, `server/scraping.get_us_equities`).
That's a handful of paginated requests (250 results each — ~9 requests for the
$2B default's ~2,000 symbols), not one HTTP call per candidate symbol — the
same batching principle the rest of this pipeline already follows. Preferred
shares, warrants, units and rights are filtered out by ticker suffix (a
heuristic — Yahoo has no "is common stock" flag to check directly — documented
in `scraping._is_common_stock`). Results are cached in-process for 6 hours per
(market cap, exchange, sector) combination.

The dashboard narrows this same fetch two more ways, both filtered
server-side by the screener query rather than fetched-then-filtered:
**Universe** picks the exchange(s) (`scraping.EXCHANGE_GROUPS`), and
**Sector** restricts to one of `scraping.GICS_SECTORS`. `--symbols` still
exists on the CLI for an explicit ticker list; the dashboard has no
equivalent — every web scan goes through the universe fetch.

Lower market-cap thresholds mean more symbols and proportionally longer scans
— there's no hard ceiling, only what's practical to wait for.

### Rate limiting

Every yfinance call — batch OHLCV downloads, the screener, single-symbol
history/info/news — goes through `scraping.with_retries`: exponential backoff
with jitter, up to 3 retries, triggered on `YFRateLimitError` and network
timeouts/connection errors specifically (a bad symbol or a real bug fails the
same way on retry too, so those raise immediately instead of wasting time).
This matters most in `batch_download`, where a chunk covers up to 200 symbols
at once — before this existed, one transient rate-limit hit silently dropped
the whole chunk rather than retrying it.

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

## The RL agent (`server/rl/`)

An optional PPO-trained agent that trades the *same* symbols the scanner
already watches, using the *same* indicators, plus the 12 rule-based
strategies' own buy/sell flags as part of its input — it reads the scanner's
own output, rather than reinventing technical analysis from raw prices.

```
per-symbol OHLCV + indicators
        │
        ▼
scale-free features (returns, distance-from-MA, RSI/MACD/ATR/VWAP ratios)
   + the 12 rule-based strategies' Buy/Sell flags        (server/rl/features.py)
        │
        ▼
trailing window of N bars  ──►  Transformer encoder  ──►  actor-critic heads
  (time-based positional                                  (server/rl/model.py)
   encoding, last-bar token)
        │
        ▼
TradingEnv: next-bar-open fills, both-side commission+slippage —
the same execution model strategy.py's backtest uses            (server/rl/env.py)
        │
        ▼
PPO: rollout → GAE → clipped update, evaluated on a held-out
validation slice every few updates, best-on-val checkpointed     (server/rl/ppo.py, train.py)
```

### Train

```bash
python server/rl/train.py                                  # default 15-symbol universe
python server/rl/train.py --symbols AAPL,MSFT,NVDA --updates 60
```

Every symbol is split chronologically **train / val / test** (70/15/15 by
default). Training and PPO updates only ever see train. Checkpoint selection
watches val. `server/rl/checkpoints/best.pt` is whichever version had the best
mean out-of-sample excess return over buy-and-hold on val — that's the file
the monitor station and `backtest.py` load. Nothing touches test until you run:

```bash
python server/rl/backtest.py                                # test split, training symbols
python server/rl/backtest.py --symbols TSLA,DIS,KO          # symbols never trained on
```

Passing symbols outside the training set is a legitimate check, not a misuse —
it tells you whether the policy learned something that transfers, or just
memorised its 10 training tickers. The report prints the RL agent's result
next to strategy.py's own best rule-based strategy on the same window, so you
can see whether either approach is worth anything before trusting it.

**Read the backtest output skeptically.** A handful of PPO updates on a
handful of symbols is not enough data for genuine edge to emerge, and getting
this to actually beat buy-and-hold needs materially more updates, more
symbols, and probably more history than 2 years — this pipeline is built and
verified to run all of that correctly; it hasn't been trained at that scale.

That said, one specific failure mode *has* been diagnosed and fixed:
under-training used to collapse the policy onto "always sit flat" rather than
merely under-performing. A flat position costs nothing and returns exactly
zero every step, so to a not-yet-informative policy it looks like the
safest action available — measured on a real run, entropy fell from ~0.98 to
~0.29 within 25 updates and validation excess-ROI got monotonically *worse*
across evals as more symbols settled into zero trades. Three schedules in
`PPOConfig`, all driven by training progress in `train.py`'s loop, target
that directly:

- **Transaction-cost warm-up** (`cost_warmup_frac`, default 0.3) — training
  starts with `TradingEnv.cost_multiplier` at 0 (commission/slippage-free) so
  the policy can learn genuine directional signal before cost makes "do
  nothing" the locally rational move, then ramps linearly to the real cost
  rate by 30% of the way through training and stays there. This is the
  bigger lever of the three.
- **Entropy annealing** (`entropy_coef` → `entropy_coef_final`) — starts
  higher (0.02) and decays to 0.003 rather than sitting at one fixed value,
  keeping exploration alive longer instead of letting it collapse early.
- **Learning-rate annealing** (`lr` → `lr * lr_final_frac`) — standard linear
  decay to 10% of the initial rate, for late-training stability.

Re-run on the same 10 symbols and update count that originally exposed the
collapse, the policy kept trading (non-zero, varying ROI) across the entire
60-update run instead of decaying to exactly 0% by the end — the collapse is
gone. Validation excess-ROI still isn't consistently positive at this budget;
that remaining gap is the "needs more updates/symbols/history" problem above,
not the flat-collapse problem this fixes.

### Use it live

There is no toggle for this — the CLI and the monitor station both blend the
RL policy into every scan automatically whenever
`server/rl/checkpoints/best.pt` exists. Each rule-based signal gets `rl_action`
(BUY/SELL/FLAT), `rl_confidence`, and `rl_agrees` fields, computed from the
*same* batch-downloaded frame the rule-based scan already fetched — no extra
network calls, which is exactly the anti-pattern the rest of this codebase was
rewritten to remove (`Strategy.__init__` used to fetch its own quote per
symbol and defeat the whole point of batching). `server/rl/live.py` fails
soft: no checkpoint, a schema mismatch, or an inference error all just mean
the rule-based signal passes through unchanged — the only way to *not* see RL
output is to not have trained a checkpoint yet. The monitor station's "RL"
status pill and `/api/status`'s `rl_available` field report whether one
exists; neither one controls it.

### Evals

`server/rl/evaluate.py` is what both training's periodic validation check and
`backtest.py` call. Per symbol it reports: ROI, buy-and-hold ROI, excess,
Sharpe, Sortino, max drawdown, trade count, and a bar-level win rate — labelled
as such because it is **not the same statistic** as strategy.py's trade-level
win rate; comparing the two numbers as if they measured the same thing would
be a mistake the naming is trying to prevent.

## Layout

| Path | Role |
|---|---|
| [server/scraping.py](server/scraping.py) | yfinance downloads, Polars indicators, market calendar |
| [server/strategy.py](server/strategy.py) | The 12 strategies, backtest, selection, final verdict |
| [server/scanner.py](server/scanner.py) | Scan pipeline shared by the CLI and the monitor station |
| [server/signal_log.py](server/signal_log.py) | Persisted signal history (atomic writes, dedupe) |
| [server/main.py](server/main.py) | Terminal scanner CLI |
| [server/run.py](server/run.py) | Monitor station launcher |
| [server/web/app.py](server/web/app.py) | FastAPI app: REST + SSE, background scanner task |
| [server/web/static/](server/web/static/) | The dashboard itself (HTML/CSS/JS, vendored chart lib) |
| [server/rl/](server/rl/) | The RL agent — see above |
| [server/tests/](server/tests/) | pytest suite |

### Optional: legacy regime and price models

[server/training.py](server/training.py) and [server/prediction.py](server/prediction.py)
fit a Gaussian HMM (market regime) and a RandomForest (next close). They need
MongoDB and are **not wired into the scanner** — the scanner's signals do not use
them, and neither does the RL agent. Enable them only if you want to experiment:

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

`server/tests/conftest.py` imports torch before anything imports pandas — on at
least one Windows setup, the reverse order access-violates loading torch's
`c10.dll`. If you hit that error running an RL script directly, import `torch`
before `scraping`/`strategy` in your own entry point too.

## Not implemented

Earlier versions of this README advertised features that do not exist. For the
record: there is **no** genetic-algorithm optimiser and **no** live order
routing. Position sizing exists only as a `position_fraction` knob on the
rule-based backtest, defaulting to all-in — do not read that default as a
recommendation. The RL agent is real and runs end-to-end, but has only been
trained at smoke-test scale; see [The RL agent](#the-rl-agent-serverrl) above
before reading anything into its numbers.

## License

Not available for commercial use. Personal and research use only — see
[License.txt](License.txt).
