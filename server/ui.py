"""
SmarTraid — Real-Time Signal Dashboard
Run with: streamlit run server/ui.py
"""
import html
import json
import os
import re
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import polars as pl
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
import plots
import scanner
import scraping
import strategy

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SIGNALS_FILE    = Path(__file__).parent / "signals_log.json"
MAX_HISTORY     = 500
SIGNAL_LOOKBACK = scanner.DEFAULT_LOOKBACK

# Yahoo tickers: letters, digits, dots and hyphens only. Anything else is either
# a typo or an injection attempt — this input is rendered as raw HTML below.
_SYMBOL_RE = re.compile(r'^[A-Za-z0-9.\-]{1,10}$')

# The scanner thread and Streamlit's rerun both touch the signal file.
_log_lock = threading.Lock()

st.set_page_config(
    page_title="SmarTraid Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def esc(value) -> str:
    """
    Escape a value for interpolation into an unsafe_allow_html block.

    Symbols reach the feed from a free-text sidebar box and are persisted to
    signals_log.json, so an unescaped `<img onerror=...>` would be stored and
    re-rendered on every page load.
    """
    return html.escape(str(value), quote=True)


def valid_symbol(symbol: str) -> bool:
    return bool(_SYMBOL_RE.match(symbol or ''))


# ---------------------------------------------------------------------------
# Signal persistence (JSON file shared with the scanner thread)
# ---------------------------------------------------------------------------

def load_signals() -> list[dict]:
    if not SIGNALS_FILE.exists():
        return []
    try:
        data = json.loads(SIGNALS_FILE.read_text(encoding='utf-8'))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _write_atomic(signals: list[dict]):
    """
    Replace the log in one atomic step.

    `write_text` truncates before it writes, so an interrupted save — or a
    concurrent read — used to observe a half-written, invalid JSON file.
    """
    payload = json.dumps(signals, default=str, indent=2)
    fd, tmp = tempfile.mkstemp(dir=str(SIGNALS_FILE.parent), suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            fh.write(payload)
        os.replace(tmp, SIGNALS_FILE)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def save_signals(signals: list[dict]):
    with _log_lock:
        _write_atomic(signals)


def _dedupe_key(entry: dict) -> tuple:
    """One signal per symbol, direction and bar — not one per scan."""
    return (entry.get('symbol'), entry.get('direction'), str(entry.get('time'))[:10])


def add_signals(entries: list[dict]):
    """
    Append new signals, skipping duplicates.

    Read-modify-write under a lock: the previous version was called from ten
    concurrent workers with no synchronisation, so writes were silently lost.
    Repeated scans of the same daily bar also re-logged identical signals until
    the history cap flushed everything else out.
    """
    if not entries:
        return
    with _log_lock:
        current = load_signals()
        seen = {_dedupe_key(e) for e in current}
        fresh = [e for e in entries if _dedupe_key(e) not in seen]
        if not fresh:
            return
        merged = (fresh + current)[:MAX_HISTORY]
        _write_atomic(merged)


# ---------------------------------------------------------------------------
# Scanner (runs in a background thread; the heavy work is in scanner.scan)
# ---------------------------------------------------------------------------

def _scanner_thread(symbols: list[str], timeframe: str, lookback: int,
                    min_margin: int, stop_event: threading.Event):
    interval = scanner.scan_interval_seconds(timeframe)
    while not stop_event.is_set():
        try:
            if scanner.should_scan_now(timeframe):
                found = scanner.scan(symbols, timeframe=timeframe,
                                     lookback=lookback, quiet=True,
                                     min_margin=min_margin,
                                     stop_event=stop_event)
                add_signals(found)
                wait = interval
            else:
                wait = scanner.CLOSED_MARKET_SLEEP
        except Exception as e:
            print(f'[scanner thread] {e}')
            wait = interval

        # Wake promptly on stop instead of sleeping through it.
        if stop_event.wait(timeout=wait):
            return


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        'scanner_running': False,
        'stop_event':      None,
        'thread':          None,
        'scan_symbols':    [],
        'scan_timeframe':  scanner.DEFAULT_TIMEFRAME,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _start_scanner(symbols, timeframe, lookback, min_margin):
    stop = threading.Event()
    t = threading.Thread(target=_scanner_thread,
                         args=(symbols, timeframe, lookback, min_margin, stop),
                         daemon=True)
    t.start()
    st.session_state.update({
        'scanner_running': True,
        'stop_event':      stop,
        'thread':          t,
        'scan_symbols':    symbols,
        'scan_timeframe':  timeframe,
    })


def _stop_scanner():
    if st.session_state.get('stop_event'):
        st.session_state['stop_event'].set()
    st.session_state['scanner_running'] = False


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _signal_color(direction: str) -> str:
    return '#00c853' if direction == 'BUY' else '#ff1744'


def _badge(direction: str) -> str:
    icon = '▲' if direction == 'BUY' else '▼'
    return (f'<span style="color:{_signal_color(direction)};font-weight:bold;'
            f'font-size:1.1em">{icon} {esc(direction)}</span>')


def _metric_card(label, value, delta=None, suffix=''):
    delta_html = ''
    if delta is not None:
        color = '#00c853' if delta >= 0 else '#ff1744'
        sign  = '+' if delta >= 0 else ''
        delta_html = f'<div style="color:{color};font-size:0.85em">{sign}{delta:.1f}%</div>'
    st.markdown(f"""
    <div style="background:#1e1e2e;padding:16px;border-radius:10px;text-align:center">
        <div style="color:#888;font-size:0.8em;text-transform:uppercase">{esc(label)}</div>
        <div style="color:#fff;font-size:1.6em;font-weight:bold">{esc(value)}{esc(suffix)}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_scanner():
    st.markdown("## 📡 Live Signal Scanner")
    st.caption("ROI figures are measured **out-of-sample** — on the slice held back "
               "from strategy selection. **Excess** is ROI minus buy-and-hold over "
               "the same window; a strategy that cannot beat it is not worth trading.")

    # ── Sidebar controls ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Scanner Controls")

        scope = st.radio("Symbol scope", ["Full S&P 500", "Custom list"], index=0)
        symbols = None
        if scope == "Custom list":
            custom = st.text_area("Symbols (comma-separated)", value="AAPL,MSFT,NVDA,TSLA,AMZN")
            raw = [s.strip() for s in custom.split(',') if s.strip()]
            symbols = [scraping.yahoo_symbol(s) for s in raw if valid_symbol(s)]
            rejected = [s for s in raw if not valid_symbol(s)]
            if rejected:
                st.warning("Ignored invalid symbol(s): " + ", ".join(esc(r) for r in rejected[:5]))

        timeframe_ui = st.selectbox("Timeframe", ['1d', '1h', '5d', '1wk'], index=0)
        lookback_ui  = st.slider("Signal lookback (candles)", 1, 10, SIGNAL_LOOKBACK)
        margin_ui    = st.slider("Consensus margin", 1, 6, scanner.DEFAULT_MIN_MARGIN,
                                 help="Vote margin required when the selected strategy "
                                      "is silent. At 1, about half the index signals on "
                                      "any given day, mostly on 5-4 splits.")
        st.caption(f"Re-scans every {scanner.scan_interval_seconds(timeframe_ui) // 60} min "
                   f"on this timeframe.")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            start_btn = st.button("▶ Start", use_container_width=True,
                                  disabled=st.session_state.get('scanner_running', False))
        with col2:
            stop_btn = st.button("⏹ Stop", use_container_width=True,
                                 disabled=not st.session_state.get('scanner_running', False))

        if start_btn:
            if symbols is None:
                with st.spinner("Loading S&P 500 symbols..."):
                    try:
                        _, symbols = scraping.get_stocks()
                    except Exception as e:
                        st.error(f"Failed to load symbols: {e}")
                        symbols = []
            if symbols:
                _start_scanner(symbols, timeframe_ui, lookback_ui, margin_ui)
                st.rerun()
            else:
                st.error("No valid symbols to scan.")

        if stop_btn:
            _stop_scanner()
            st.rerun()

        st.markdown("---")
        if st.button("🗑 Clear history"):
            save_signals([])
            st.rerun()

        st.markdown("---")
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=True)

    # ── Status bar ────────────────────────────────────────────────────────
    running = st.session_state.get('scanner_running', False)
    if running:
        n_syms = len(st.session_state.get('scan_symbols', []))
        tf     = st.session_state.get('scan_timeframe', scanner.DEFAULT_TIMEFRAME)
        every  = scanner.scan_interval_seconds(tf) // 60
        st.success(f"🟢 Scanner running — {n_syms} symbols on {tf}, every {every} min")
        if scanner.is_intraday(tf) and not scraping.is_nyse_open():
            st.info("Exchange is closed — intraday scans are paused until it reopens.")
    else:
        st.info("⚪ Scanner stopped — press **▶ Start** to begin")

    # ── Metrics row ───────────────────────────────────────────────────────
    signals = load_signals()
    buys  = [s for s in signals if s.get('direction') == 'BUY']
    sells = [s for s in signals if s.get('direction') == 'SELL']
    beat  = [s for s in signals if s.get('excess_roi', 0) > 0]
    avg_excess = (sum(s.get('excess_roi', 0) for s in signals) / len(signals)) if signals else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1: _metric_card("Total Signals", len(signals))
    with c2: _metric_card("Buy / Sell", f"{len(buys)} / {len(sells)}")
    with c3: _metric_card("Beat Buy & Hold", f"{len(beat)}/{len(signals)}" if signals else "—")
    with c4: _metric_card("Avg Excess ROI", f"{avg_excess:+.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Live feed ─────────────────────────────────────────────────────────
    col_feed, col_chart = st.columns([1, 1])

    with col_feed:
        st.markdown("### 🔔 Signal Feed")

        fc1, fc2 = st.columns(2)
        with fc1:
            dir_filter = st.selectbox("Direction", ["All", "BUY", "SELL"])
        with fc2:
            sym_filter = st.text_input("Symbol filter", placeholder="e.g. AAPL")

        filtered = signals
        if dir_filter != "All":
            filtered = [s for s in filtered if s.get('direction') == dir_filter]
        if sym_filter:
            filtered = [s for s in filtered if sym_filter.upper() in s.get('symbol', '')]

        if not filtered:
            st.markdown(
                '<div style="text-align:center;padding:40px;color:#555">'
                'No signals yet — start the scanner to see live alerts</div>',
                unsafe_allow_html=True
            )
        else:
            for sig in filtered[:50]:
                color  = _signal_color(sig.get('direction', ''))
                excess = sig.get('excess_roi', 0)
                ex_col = '#00c853' if excess > 0 else '#ff1744'
                st.markdown(f"""
                <div style="border-left:4px solid {color};background:#1e1e2e;
                            padding:10px 14px;margin:6px 0;border-radius:6px;
                            display:flex;justify-content:space-between;align-items:center">
                    <div>
                        {_badge(sig.get('direction', ''))}
                        <span style="color:#fff;font-size:1.15em;margin-left:8px;font-weight:bold">{esc(sig.get('symbol', ''))}</span>
                        <span style="color:#aaa;font-size:0.85em;margin-left:8px">{esc(sig.get('time', ''))}</span>
                    </div>
                    <div style="text-align:right">
                        <span style="color:#fff;font-size:1.0em">${esc(sig.get('price', 0))}</span>
                        <span style="color:{ex_col};font-size:0.8em;margin-left:8px">excess {excess:+.1f}%</span>
                        <span style="color:#aaa;font-size:0.8em;margin-left:8px">{esc(sig.get('strategy', ''))}</span>
                    </div>
                </div>""", unsafe_allow_html=True)

    with col_chart:
        st.markdown("### 📊 Signal Distribution")
        if signals:
            fig_pie = go.Figure(go.Pie(
                labels=['BUY', 'SELL'], values=[len(buys), len(sells)],
                marker_colors=['#00c853', '#ff1744'], hole=0.4,
                textinfo='label+percent+value',
            ))
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                  font_color='#ccc', showlegend=False, height=250,
                                  margin=dict(t=10, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

            fig_hist = go.Figure(go.Histogram(
                x=[s.get('excess_roi', 0) for s in signals], nbinsx=20,
                marker_color='#5c6bc0', opacity=0.8,
            ))
            fig_hist.add_vline(x=0, line_dash='dash', line_color='#ff1744')
            fig_hist.update_layout(
                title="Excess ROI vs buy-and-hold (out-of-sample)",
                xaxis_title="Excess ROI (%)", yaxis_title="Count",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#ccc', height=250, margin=dict(t=40, b=10)
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.markdown('<div style="color:#555;text-align:center;padding:60px">No data yet</div>',
                        unsafe_allow_html=True)

    # ── Full table ────────────────────────────────────────────────────────
    if signals:
        st.markdown("### 📋 Signal History")
        table_df = pd.DataFrame(signals).rename(columns={
            'time': 'Bar', 'symbol': 'Symbol', 'direction': 'Direction',
            'price': 'Price ($)', 'roi': 'ROI (%)', 'benchmark_roi': 'Buy & Hold (%)',
            'excess_roi': 'Excess (%)', 'win_rate': 'Win Rate (%)',
            'trades': 'Trades', 'strategy': 'Best Strategy',
        })
        preferred = ['Bar', 'Symbol', 'Direction', 'Price ($)', 'ROI (%)',
                     'Buy & Hold (%)', 'Excess (%)', 'Win Rate (%)', 'Trades',
                     'Best Strategy']
        table_df = table_df[[c for c in preferred if c in table_df.columns]]

        st.dataframe(
            table_df.head(200), use_container_width=True, hide_index=True,
            column_config={
                'Price ($)':      st.column_config.NumberColumn(format="$%.2f"),
                'ROI (%)':        st.column_config.NumberColumn(format="%.2f%%"),
                'Buy & Hold (%)': st.column_config.NumberColumn(format="%.2f%%"),
                'Excess (%)':     st.column_config.NumberColumn(format="%.2f%%"),
                'Win Rate (%)':   st.column_config.NumberColumn(format="%.1f%%"),
            },
        )
        st.download_button("⬇ Export CSV", table_df.to_csv(index=False),
                           "signals.csv", "text/csv")

    # ── Auto-refresh ──────────────────────────────────────────────────────
    if auto_refresh and running:
        time.sleep(30)
        st.rerun()


def page_analysis():
    st.markdown("## 🔍 Stock Analysis")

    with st.sidebar:
        st.markdown("### Stock Picker")
        try:
            names, symbols_list = scraping.get_stocks()
        except Exception:
            names, symbols_list = ['SPDR S&P 500 ETF'], ['SPY']

        stock_name   = st.selectbox("Stock", names)
        stock_ticker = symbols_list[names.index(stock_name)]
        st.markdown(f"**Ticker:** `{stock_ticker}`")

        interval = st.selectbox("Interval", ['1d', '1h', '5d', '1wk', '1mo'], index=0)
        period   = st.selectbox("Period", ['2y', '1y', '6mo', '3mo', 'max'], index=0)

        indicators = st.multiselect(
            "Overlay indicators",
            ['SMA20', 'SMA50', 'SMA100', 'SMA150', 'SMA200', 'EMA12', 'EMA20', 'EMA26', 'VWAP'],
            default=['SMA20', 'SMA50'])

        run_btn = st.button("🔎 Analyse", use_container_width=True)

    if not run_btn:
        st.info("Select a stock in the sidebar and click **Analyse**.")
        return

    with st.spinner(f"Fetching {stock_ticker} data..."):
        try:
            data = scraping.get_stock_data(
                stock_ticker, interval=interval, period=period,
                return_flags={'DF': True, 'INDICATORS': True,
                              'MAX_KEY': True, 'SUMMARY': True, 'DIVD': True, 'INFO': True}
            )
        except Exception as e:
            st.error(f"Data fetch failed: {e}")
            return

    if not data or data.get('DF') is None or data['DF'].empty:
        st.error(f"No data returned for {stock_ticker}")
        return

    df_pl   = pl.from_pandas(data['DF'], include_index=True)
    info    = data.get('INFO', {})
    # These keys are 'SUMMARY' and 'DIVD' in scraping.get_stock_data; reading
    # 'SUMMERY'/'DIVID' as before always yielded an empty string.
    summary = data.get('SUMMARY', '')
    max_key = data.get('MAX_KEY', '')
    divid   = data.get('DIVD', '')

    # ── Key metrics ───────────────────────────────────────────────────────
    price = float(df_pl['Close'][-1])
    c1, c2, c3, c4 = st.columns(4)
    with c1: _metric_card("Last Close", f"${price:.2f}")
    with c2: _metric_card("Recommendation", max_key or "—")
    with c3: _metric_card("Dividend", divid or "—")
    with c4: _metric_card("Market Cap",
                          f"{info.get('marketCap', 0) / 1e9:.1f}B" if info.get('marketCap') else "—")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Strategy analysis ─────────────────────────────────────────────────
    with st.spinner("Running strategy analysis..."):
        try:
            stock_obj = strategy.Strategy(symbol=stock_ticker)
            best, backtest_res = stock_obj.evaluate_strategies(df_pl, timeframe=interval)
            sig = strategy.what_is_signal(best, backtest_res, SIGNAL_LOOKBACK,
                                          min_margin=scanner.DEFAULT_MIN_MARGIN)
        except Exception as e:
            st.warning(f"Strategy analysis failed: {e}")
            backtest_res, best, sig = [], None, None

    best_signals = None
    if backtest_res:
        best_signals = next((r['signals'] for r in backtest_res
                             if r['strategy_func'] == best), backtest_res[0]['signals'])

    # Pass the chosen overlays, not every column in the frame.
    fig = plots.plot_stock(df_pl, stock_ticker, indicators,
                           signals=best_signals, show='all', interval=interval)
    st.plotly_chart(fig, use_container_width=True)

    if sig is True:
        st.success(f"▲ **BUY signal** for {stock_ticker}  ·  Best strategy: `{best}`")
    elif sig is False:
        st.error(f"▼ **SELL signal** for {stock_ticker}  ·  Best strategy: `{best}`")
    else:
        st.info(f"No signal at this time for {stock_ticker}  ·  Best strategy: `{best}`")

    # ── Strategy results table ────────────────────────────────────────────
    if backtest_res:
        st.markdown("### Strategy Backtest Results")
        st.caption("**Train ROI** selected the strategy; **ROI** is the held-out "
                   "out-of-sample result. A large gap between them is curve-fitting.")
        rows = []
        for r in backtest_res:
            rm = r.get('risk_metrics', {})
            tm = r.get('train_metrics', {})
            last_buy  = r['signals']['Buy_Signal'][-SIGNAL_LOOKBACK:].to_list()
            last_sell = r['signals']['Sell_Signal'][-SIGNAL_LOOKBACK:].to_list()
            rows.append({
                'Strategy':       r['strategy_func'],
                'Train ROI (%)':  tm.get('roi', 0),
                'ROI (%)':        rm.get('roi', 0),
                'Buy & Hold (%)': rm.get('benchmark_roi', 0),
                'Excess (%)':     rm.get('excess_roi', 0),
                'Win Rate (%)':   rm.get('win_rate', 0),
                'Trades':         rm.get('trades', 0),
                'Max DD (%)':     rm.get('max_drawdown', 0),
                'Signal':         '▲ BUY' if any(last_buy) else ('▼ SELL' if any(last_sell) else '—'),
                'Selected':       '★' if r['strategy_func'] == best else '',
            })
        st.dataframe(
            pd.DataFrame(rows).sort_values('Excess (%)', ascending=False),
            use_container_width=True, hide_index=True,
            column_config={
                'Train ROI (%)':  st.column_config.NumberColumn(format="%.2f%%"),
                'ROI (%)':        st.column_config.NumberColumn(format="%.2f%%"),
                'Buy & Hold (%)': st.column_config.NumberColumn(format="%.2f%%"),
                'Excess (%)':     st.column_config.NumberColumn(format="%.2f%%"),
                'Win Rate (%)':   st.column_config.NumberColumn(format="%.1f%%"),
                'Max DD (%)':     st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

    if summary:
        with st.expander("Business Summary"):
            st.write(summary)


# ---------------------------------------------------------------------------
# App shell
# ---------------------------------------------------------------------------

_init_state()

tab1, tab2 = st.tabs(["📡 Live Scanner", "🔍 Stock Analysis"])
with tab1:
    page_scanner()
with tab2:
    page_analysis()

st.markdown("""<style>
    .stApp { background-color: #0e0e1a; }
    .block-container { padding-top: 1rem; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; padding: 8px 20px; }
</style>""", unsafe_allow_html=True)
