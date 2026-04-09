"""
SmarTraid — Real-Time Signal Dashboard
Run with: streamlit run server/ui.py
"""
import os, sys, json, threading, time
from datetime import datetime
from pathlib import Path

import streamlit as st
import polars as pl
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))
import scraping, strategy, plots

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SIGNALS_FILE   = Path(__file__).parent / "signals_log.json"
SCAN_INTERVAL  = 300   # seconds between full scans
MAX_WORKERS    = 10
SIGNAL_LOOKBACK = 4
TIMEFRAME      = '1d'

st.set_page_config(
    page_title="SmarTraid Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Signal persistence (JSON file shared with scanner thread)
# ---------------------------------------------------------------------------

def load_signals() -> list[dict]:
    if SIGNALS_FILE.exists():
        try:
            return json.loads(SIGNALS_FILE.read_text(encoding='utf-8'))
        except Exception:
            return []
    return []


def save_signals(signals: list[dict]):
    SIGNALS_FILE.write_text(json.dumps(signals, default=str, indent=2), encoding='utf-8')


def add_signal(entry: dict):
    signals = load_signals()
    signals.insert(0, entry)      # newest first
    signals = signals[:500]       # cap history
    save_signals(signals)


# ---------------------------------------------------------------------------
# Scanner (runs in a background thread)
# ---------------------------------------------------------------------------

def _analyse_symbol(symbol: str, df: pl.DataFrame) -> dict | None:
    """Run all strategies on one pre-fetched Polars DataFrame."""
    try:
        stock = strategy.Strategy(symbol=symbol)
        best, backtest_res = stock.get_strategy_func(df, timeframe=TIMEFRAME, plot=False)
        sig = strategy.what_is_signal(best, backtest_res, SIGNAL_LOOKBACK)
        if sig is None:
            return None

        rois  = [r['risk_metrics']['roi']      for r in backtest_res if r.get('risk_metrics')]
        wins  = [r['risk_metrics']['win_rate'] for r in backtest_res if r.get('risk_metrics')]
        price = float(df['Close'][-1])  # use last close from the batch — no extra HTTP call

        return {
            'time':      datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol':    symbol,
            'direction': 'BUY' if sig is True else 'SELL',
            'price':     round(price, 2),
            'avg_roi':   round(sum(rois) / len(rois), 2) if rois else 0,
            'win_rate':  round(sum(wins) / len(wins), 2) if wins else 0,
            'strategy':  best or 'combined',
        }
    except Exception:
        return None


def _scanner_thread(symbols: list[str], stop_event: threading.Event):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    while not stop_event.is_set():
        # ── Single batch download for all symbols ──────────────────────
        stock_data = scraping.batch_download(symbols, period='2y', interval=TIMEFRAME,
                                             chunk_size=200)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(_analyse_symbol, sym, df): sym
                       for sym, df in stock_data.items()}
            for future in as_completed(futures):
                if stop_event.is_set():
                    break
                result = future.result()
                if result:
                    add_signal(result)

        for _ in range(SCAN_INTERVAL):
            if stop_event.is_set():
                return
            time.sleep(1)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        'scanner_running': False,
        'stop_event':      None,
        'thread':          None,
        'scan_symbols':    [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _start_scanner(symbols):
    stop = threading.Event()
    t = threading.Thread(target=_scanner_thread, args=(symbols, stop), daemon=True)
    t.start()
    st.session_state['scanner_running'] = True
    st.session_state['stop_event']      = stop
    st.session_state['thread']          = t
    st.session_state['scan_symbols']    = symbols


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
    color = _signal_color(direction)
    return f'<span style="color:{color};font-weight:bold;font-size:1.1em">{icon} {direction}</span>'


def _metric_card(label, value, delta=None, suffix=''):
    delta_html = ''
    if delta is not None:
        color = '#00c853' if delta >= 0 else '#ff1744'
        sign  = '+' if delta >= 0 else ''
        delta_html = f'<div style="color:{color};font-size:0.85em">{sign}{delta:.1f}%</div>'
    st.markdown(f"""
    <div style="background:#1e1e2e;padding:16px;border-radius:10px;text-align:center">
        <div style="color:#888;font-size:0.8em;text-transform:uppercase">{label}</div>
        <div style="color:#fff;font-size:1.6em;font-weight:bold">{value}{suffix}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_scanner():
    st.markdown("## 📡 Live Signal Scanner")

    # ── Sidebar controls ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Scanner Controls")

        scope = st.radio("Symbol scope", ["Full S&P 500", "Custom list"], index=0)
        if scope == "Custom list":
            custom = st.text_area("Symbols (comma-separated)", value="AAPL,MSFT,NVDA,TSLA,AMZN")
            symbols = [s.strip().upper() for s in custom.split(',') if s.strip()]
        else:
            symbols = None   # resolved below on start

        timeframe_ui = st.selectbox("Timeframe", ['1d', '1h', '5d', '1wk'], index=0)
        lookback_ui  = st.slider("Signal lookback (candles)", 1, 10, 4)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            start_btn = st.button("▶ Start", use_container_width=True,
                                  disabled=st.session_state.get('scanner_running', False))
        with col2:
            stop_btn  = st.button("⏹ Stop",  use_container_width=True,
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
                _start_scanner(symbols)
                st.rerun()

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
        st.success(f"🟢 Scanner running — monitoring {n_syms} symbols  ·  refreshing every {SCAN_INTERVAL}s")
    else:
        st.info("⚪ Scanner stopped — press **▶ Start** to begin")

    # ── Metrics row ───────────────────────────────────────────────────────
    signals = load_signals()
    buys  = [s for s in signals if s['direction'] == 'BUY']
    sells = [s for s in signals if s['direction'] == 'SELL']
    avg_roi = (sum(s['avg_roi'] for s in signals) / len(signals)) if signals else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1: _metric_card("Total Signals",  len(signals))
    with c2: _metric_card("Buy Signals",    len(buys),  suffix='  ▲')
    with c3: _metric_card("Sell Signals",   len(sells), suffix='  ▼')
    with c4: _metric_card("Avg Strategy ROI", f"{avg_roi:+.1f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Live feed ─────────────────────────────────────────────────────────
    col_feed, col_chart = st.columns([1, 1])

    with col_feed:
        st.markdown("### 🔔 Signal Feed")

        # Filter controls
        fc1, fc2 = st.columns(2)
        with fc1:
            dir_filter = st.selectbox("Direction", ["All", "BUY", "SELL"])
        with fc2:
            sym_filter = st.text_input("Symbol filter", placeholder="e.g. AAPL")

        filtered = signals
        if dir_filter != "All":
            filtered = [s for s in filtered if s['direction'] == dir_filter]
        if sym_filter:
            filtered = [s for s in filtered if sym_filter.upper() in s['symbol']]

        if not filtered:
            st.markdown(
                '<div style="text-align:center;padding:40px;color:#555">'
                'No signals yet — start the scanner to see live alerts</div>',
                unsafe_allow_html=True
            )
        else:
            for sig in filtered[:50]:
                color   = _signal_color(sig['direction'])
                roi_str = f"{sig['avg_roi']:+.1f}%"
                wr_str  = f"{sig['win_rate']:.1f}%"
                st.markdown(f"""
                <div style="border-left:4px solid {color};background:#1e1e2e;
                            padding:10px 14px;margin:6px 0;border-radius:6px;
                            display:flex;justify-content:space-between;align-items:center">
                    <div>
                        {_badge(sig['direction'])}
                        <span style="color:#fff;font-size:1.15em;margin-left:8px;font-weight:bold">{sig['symbol']}</span>
                        <span style="color:#aaa;font-size:0.85em;margin-left:8px">{sig['time']}</span>
                    </div>
                    <div style="text-align:right">
                        <span style="color:#fff;font-size:1.0em">${sig['price']}</span>
                        <span style="color:#aaa;font-size:0.8em;margin-left:8px">ROI {roi_str} · WR {wr_str}</span>
                    </div>
                </div>""", unsafe_allow_html=True)

    with col_chart:
        st.markdown("### 📊 Signal Distribution")
        if signals:
            by_dir = {'BUY': len(buys), 'SELL': len(sells)}
            fig_pie = go.Figure(go.Pie(
                labels=list(by_dir.keys()),
                values=list(by_dir.values()),
                marker_colors=['#00c853', '#ff1744'],
                hole=0.4,
                textinfo='label+percent+value',
            ))
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                  font_color='#ccc', showlegend=False, height=250,
                                  margin=dict(t=10, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

            # ROI histogram
            rois = [s['avg_roi'] for s in signals]
            fig_hist = go.Figure(go.Histogram(
                x=rois, nbinsx=20,
                marker_color='#5c6bc0',
                opacity=0.8,
            ))
            fig_hist.update_layout(
                title="Strategy ROI distribution",
                xaxis_title="ROI (%)", yaxis_title="Count",
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
        table_df = pd.DataFrame(signals)
        table_df = table_df.rename(columns={
            'time': 'Time', 'symbol': 'Symbol', 'direction': 'Direction',
            'price': 'Price ($)', 'avg_roi': 'Avg ROI (%)', 'win_rate': 'Win Rate (%)',
            'strategy': 'Best Strategy',
        })
        st.dataframe(
            table_df.head(200),
            use_container_width=True,
            column_config={
                'Direction':  st.column_config.TextColumn(),
                'Price ($)':  st.column_config.NumberColumn(format="$%.2f"),
                'Avg ROI (%)': st.column_config.NumberColumn(format="%.2f%%"),
                'Win Rate (%)': st.column_config.NumberColumn(format="%.1f%%"),
            },
            hide_index=True,
        )

        csv = table_df.to_csv(index=False)
        st.download_button("⬇ Export CSV", csv, "signals.csv", "text/csv")

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
            names, symbols_list = ['SPY'], ['SPY']

        stock_name   = st.selectbox("Stock", names)
        sindex       = names.index(stock_name)
        stock_ticker = symbols_list[sindex]
        st.markdown(f"**Ticker:** `{stock_ticker}`")

        intervals = ['1d', '1h', '5d', '1wk', '1mo']
        interval  = st.selectbox("Interval", intervals, index=0)
        period    = st.selectbox("Period",   ['1y', '2y', '6mo', '3mo', 'max'], index=0)

        indicators = st.multiselect("Overlay indicators",
            ['SMA20', 'SMA50', 'SMA100', 'SMA150', 'SMA200', 'EMA20'],
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
                              'MAX_KEY': True, 'SUMMERY': True, 'DIVD': True, 'INFO': True}
            )
        except Exception as e:
            st.error(f"Data fetch failed: {e}")
            return

    if not data or 'DF' not in data or data['DF'] is None or data['DF'].empty:
        st.error(f"No data returned for {stock_ticker}")
        return

    df_pd  = data['DF']
    df_pl  = pl.from_pandas(df_pd, include_index=True)
    info   = data.get('INFO', {})
    summary = data.get('SUMMARY', '')
    max_key = data.get('MAX_KEY', '')
    divid   = data.get('DIVID', '')

    # ── Key metrics ────────────────────────────────────────────────────────
    price = scraping.current_stock_price(stock_ticker)
    c1, c2, c3, c4 = st.columns(4)
    with c1: _metric_card("Current Price", f"${price:.2f}")
    with c2: _metric_card("Recommendation", max_key or "—")
    with c3: _metric_card("Dividend", divid or "—")
    with c4: _metric_card("Market Cap", f"{info.get('marketCap', 0)/1e9:.1f}B" if info.get('marketCap') else "—")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Candlestick chart with signals ────────────────────────────────────
    with st.spinner("Running strategy analysis..."):
        try:
            stock_obj = strategy.Strategy(symbol=stock_ticker)
            best, backtest_res = stock_obj.get_strategy_func(df_pl, timeframe=interval, plot=False)
            sig = strategy.what_is_signal(best, backtest_res, SIGNAL_LOOKBACK)
        except Exception as e:
            st.warning(f"Strategy analysis failed: {e}")
            backtest_res, best, sig = [], None, None

    # Find best strategy signals for chart
    best_signals = None
    if backtest_res:
        for res in backtest_res:
            if res['strategy_func'] == best:
                best_signals = res['signals']
                break
        if best_signals is None:
            best_signals = backtest_res[0]['signals']

    fig = plots.plot_stock(df_pl, stock_ticker, df_pl.columns,
                           signals=best_signals, show='all', interval=interval)
    st.plotly_chart(fig, use_container_width=True)

    # ── Signal banner ─────────────────────────────────────────────────────
    if sig is True:
        st.success(f"▲ **BUY signal** detected for {stock_ticker}  ·  Best strategy: `{best}`")
    elif sig is False:
        st.error(f"▼ **SELL signal** detected for {stock_ticker}  ·  Best strategy: `{best}`")
    else:
        st.info(f"No signal at this time for {stock_ticker}  ·  Best strategy: `{best}`")

    # ── Strategy results table ────────────────────────────────────────────
    if backtest_res:
        st.markdown("### Strategy Backtest Results")
        rows = []
        for r in backtest_res:
            rm = r.get('risk_metrics', {})
            last_buy  = r['signals']['Buy_Signal'][-SIGNAL_LOOKBACK:].to_list() if r.get('signals') is not None else []
            last_sell = r['signals']['Sell_Signal'][-SIGNAL_LOOKBACK:].to_list() if r.get('signals') is not None else []
            rows.append({
                'Strategy':    r['strategy_func'],
                'ROI (%)':     rm.get('roi', 0),
                'Win Rate (%)': rm.get('win_rate', 0),
                'Max DD (%)':  rm.get('max_drawdown', 0),
                'Days':        rm.get('time_frame_days', 0),
                'Signal':      '▲ BUY' if any(last_buy) else ('▼ SELL' if any(last_sell) else '—'),
            })
        st.dataframe(
            pd.DataFrame(rows).sort_values('ROI (%)', ascending=False),
            use_container_width=True,
            column_config={
                'ROI (%)':     st.column_config.NumberColumn(format="%.2f%%"),
                'Win Rate (%)': st.column_config.NumberColumn(format="%.1f%%"),
                'Max DD (%)':  st.column_config.NumberColumn(format="%.2f%%"),
            },
            hide_index=True,
        )

    # ── Summary ───────────────────────────────────────────────────────────
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

# Dark theme tweak
st.markdown("""<style>
    .stApp { background-color: #0e0e1a; }
    .block-container { padding-top: 1rem; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; padding: 8px 20px; }
</style>""", unsafe_allow_html=True)
