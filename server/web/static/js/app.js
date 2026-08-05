/* App glue: SSE stream, controls, feed/table rendering, chart loading.
   DOM nodes for any server-supplied text are built with textContent, never
   innerHTML — the old Streamlit dashboard rendered symbols via
   unsafe_allow_html and had to escape by hand; here the injection class does
   not exist in the first place. */

const state = {
  signals: [],
  filterDirection: 'All',
  filterSymbol: '',
  sortBy: 'time',   // 'time' (newest first, server order) or 'confidence'
  status: null,
  nextScanAt: null,   // epoch ms — set by applyStatus(), ticked down by tickNextScan()
  scanBusy: false,    // true while a start/stop request is in flight — blocks double-clicks
};

const $ = (id) => document.getElementById(id);

function fmtPct(v) {
  if (v === null || v === undefined) return '—';
  const s = v >= 0 ? '+' : '';
  return `${s}${v.toFixed(1)}%`;
}
function fmtNum(v, digits = 2) {
  return v === null || v === undefined ? '—' : Number(v).toFixed(digits);
}
function signClass(v) {
  if (v === null || v === undefined) return '';
  return v > 0 ? 'num-pos' : v < 0 ? 'num-neg' : '';
}

/* ── Status / metrics ─────────────────────────────────────────────────── */

function fmtCountdown(ms) {
  const total = Math.max(0, Math.round(ms / 1000));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return m > 0 ? `${m}m ${String(s).padStart(2, '0')}s` : `${s}s`;
}

// Runs once a second (see init below) so the "Next scan" tile counts down
// live between status pushes, instead of the old static "~5m" estimate.
function tickNextScan() {
  const el = $('m-next');
  const status = state.status;
  if (!status || !status.running) { el.textContent = '—'; return; }
  if (!status.market_open) { el.textContent = 'mkt closed'; return; }
  if (state.nextScanAt == null) { el.textContent = 'starting…'; return; }
  const remaining = state.nextScanAt - Date.now();
  el.textContent = remaining <= 0 ? 'scanning…' : fmtCountdown(remaining);
}

function applyStatus(status) {
  state.status = status;

  const scanPill = $('scan-pill');
  scanPill.dataset.state = status.running ? 'running' : 'stopped';
  $('scan-label').textContent = status.running
    ? `scanning ${status.symbols_count} symbols (${status.timeframe})`
    : 'scanner stopped';

  const marketPill = $('market-pill');
  marketPill.dataset.state = status.market_open ? 'open' : 'closed';
  $('market-label').textContent = status.market_open ? 'market open' : 'market closed';

  const toggle = $('scan-toggle');
  toggle.setAttribute('aria-checked', String(status.running));
  toggle.disabled = state.scanBusy === true;
  $('scan-toggle-state').textContent = status.running ? 'Running' : 'Stopped';

  $('scan-error').hidden = !status.last_error;
  if (status.last_error) $('scan-error').textContent = `Error: ${status.last_error}`;

  if (status.running) {
    $('scan-hint').textContent = status.last_scan_at
      ? `Last scan: ${new Date(status.last_scan_at).toLocaleTimeString()} — ${status.last_scan_count} signal(s).`
      : 'First scan running…';
  } else {
    $('scan-hint').textContent = 'Not scanning.';
  }

  // Countdown target for the "Next scan" tile — recomputed on every status
  // push (scan start/finish, or the initial /api/status fetch) and ticked
  // down locally by tickNextScan() every second in between, so the UI
  // doesn't need the server to push once per second just for a clock.
  state.nextScanAt = (status.running && status.market_open)
    ? (status.last_scan_at ? Date.parse(status.last_scan_at) : Date.now()) + status.scan_interval_s * 1000
    : null;
  tickNextScan();

  // RL has no on/off control — this pill only ever reports what's true.
  const rlPill = $('rl-pill');
  rlPill.dataset.state = status.rl_available ? 'active' : 'unavailable';
  $('rl-label').textContent = status.rl_available
    ? 'RL active' : 'RL: no checkpoint';
}

function applySummary(summary) {
  $('m-total').textContent = summary.total;
  $('m-buysell').textContent = `${summary.buys} / ${summary.sells}`;
  $('m-beat').textContent = summary.total ? `${summary.beat_bench}/${summary.total}` : '—';

  const excessEl = $('m-excess');
  excessEl.textContent = fmtPct(summary.avg_excess);
  excessEl.className = 'tile-value ' + (summary.avg_excess > 0 ? 'pos' : summary.avg_excess < 0 ? 'neg' : '');

  Charts.renderDonut(summary.buys, summary.sells);
  Charts.renderHistogram(summary.excess_hist || []);
}

async function refreshSummary() {
  try { applySummary(await Api.signalsSummary()); } catch (_) {}
}

function renderSectorStats(data) {
  const grid = $('sector-stats-grid');
  $('sector-stats-date').textContent = data.date ? `as of ${data.date}` : '';
  const rows = data.sectors || [];
  grid.innerHTML = '';

  if (!rows.length) {
    const empty = document.createElement('div');
    empty.className = 'feed-empty';
    empty.textContent = 'No signals yet today.';
    grid.appendChild(empty);
    return;
  }

  for (const r of rows) {
    const card = document.createElement('div');
    card.className = 'sector-card';

    const name = document.createElement('div');
    name.className = 'sector-card-name';
    name.textContent = r.sector;
    name.title = r.sector;

    const countRow = document.createElement('div');
    countRow.className = 'sector-card-row';
    const totalSpan = document.createElement('span');
    totalSpan.textContent = `${r.total} signal${r.total === 1 ? '' : 's'}`;
    const beatSpan = document.createElement('span');
    beatSpan.textContent = `${r.beat_bench}/${r.total} beat`;
    countRow.append(totalSpan, beatSpan);

    const dirRow = document.createElement('div');
    dirRow.className = 'sector-card-row';
    const buySpan = document.createElement('span');
    buySpan.className = 'num-pos';
    buySpan.textContent = `${r.buys} BUY`;
    const sellSpan = document.createElement('span');
    sellSpan.className = 'num-neg';
    sellSpan.textContent = `${r.sells} SELL`;
    dirRow.append(buySpan, sellSpan);

    const excess = document.createElement('div');
    excess.className = 'sector-card-excess ' + signClass(r.avg_excess);
    excess.textContent = `${fmtPct(r.avg_excess)} avg excess`;

    card.append(name, countRow, dirRow, excess);
    grid.appendChild(card);
  }
}

async function refreshSectorStats() {
  try { renderSectorStats(await Api.sectorSummary()); } catch (_) {}
}

/* ── Feed ─────────────────────────────────────────────────────────────── */

function feedRowMatches(sig) {
  if (state.filterDirection !== 'All' && sig.direction !== state.filterDirection) return false;
  if (state.filterSymbol && !sig.symbol.toUpperCase().includes(state.filterSymbol)) return false;
  return true;
}

function sortSignals(signals) {
  if (state.sortBy !== 'confidence') return signals;
  // Highest RL confidence first; signals without one (no RL checkpoint yet,
  // or a symbol the live hook skipped) sink to the bottom rather than being
  // dropped, so switching sort mode never hides a signal.
  return [...signals].sort((a, b) => (b.rl_confidence ?? -1) - (a.rl_confidence ?? -1));
}

function buildFeedItem(sig) {
  const item = document.createElement('div');
  item.className = 'feed-item ' + (sig.direction === 'BUY' ? 'buy' : 'sell');
  item.addEventListener('click', () => loadSymbol(sig.symbol));

  const left = document.createElement('div');
  left.className = 'feed-left';
  const dir = document.createElement('span');
  dir.className = 'feed-dir ' + (sig.direction === 'BUY' ? 'buy' : 'sell');
  dir.textContent = sig.direction === 'BUY' ? '▲ BUY' : '▼ SELL';
  const sym = document.createElement('span');
  sym.className = 'feed-symbol';
  sym.textContent = sig.symbol;
  const time = document.createElement('span');
  time.className = 'feed-time';
  time.textContent = sig.time || '';
  left.append(dir, sym, time);

  const right = document.createElement('div');
  right.className = 'feed-right';
  const price = document.createElement('span');
  price.textContent = `$${fmtNum(sig.price)}`;
  const excess = document.createElement('span');
  excess.className = 'feed-excess ' + signClass(sig.excess_roi);
  excess.style.marginLeft = '8px';
  excess.textContent = `excess ${fmtPct(sig.excess_roi)}`;
  const strat = document.createElement('span');
  strat.className = 'feed-strategy';
  strat.textContent = sig.strategy || '';
  right.append(price, excess, strat);

  if (sig.rl_action) {
    const rl = document.createElement('span');
    rl.className = 'feed-strategy';
    const mark = sig.rl_agrees ? '✓' : '✗';
    rl.textContent = ` · RL ${sig.rl_action} ${mark} (${Math.round((sig.rl_confidence || 0) * 100)}%)`;
    rl.style.color = sig.rl_agrees ? 'var(--green)' : 'var(--amber)';
    right.append(rl);
  }

  item.append(left, right);
  return item;
}

function renderFeed() {
  const list = $('feed-list');
  list.innerHTML = '';
  const visible = sortSignals(state.signals.filter(feedRowMatches)).slice(0, 80);
  if (!visible.length) {
    const empty = document.createElement('div');
    empty.className = 'feed-empty';
    empty.textContent = 'No signals yet — start the scanner.';
    list.appendChild(empty);
    return;
  }
  for (const sig of visible) list.appendChild(buildFeedItem(sig));
}

function prependSignal(sig) {
  state.signals.unshift(sig);
  state.signals = state.signals.slice(0, 500);
  renderFeed();
  renderHistory();
  refreshSummary();
  refreshSectorStats();
}

async function loadSignalsFromServer() {
  try {
    state.signals = await Api.signals({ limit: 500 });
    renderFeed();
    renderHistory();
    refreshSummary();
    refreshSectorStats();
  } catch (e) { console.error(e); }
}

/* ── History table ────────────────────────────────────────────────────── */

function renderHistory() {
  const tbody = $('history-tbody');
  tbody.innerHTML = '';
  const rows = sortSignals(state.signals.filter(feedRowMatches)).slice(0, 200);

  for (const s of rows) {
    const tr = document.createElement('tr');
    tr.title = `Click to see ${s.symbol}'s past signals and chart`;
    tr.addEventListener('click', () => loadSymbol(s.symbol));
    const cells = [
      s.time || '', s.symbol, s.direction, `$${fmtNum(s.price)}`,
      fmtPct(s.roi), fmtPct(s.benchmark_roi), fmtPct(s.excess_roi),
      s.win_rate != null ? `${fmtNum(s.win_rate, 1)}%` : '—',
      s.trades ?? '—', s.strategy || '',
      s.rl_confidence != null ? `${Math.round(s.rl_confidence * 100)}%` : '—',
    ];
    cells.forEach((val, i) => {
      const td = document.createElement('td');
      td.textContent = val;
      if (i === 6) td.className = signClass(s.excess_roi);
      if (i === 2) td.className = s.direction === 'BUY' ? 'num-pos' : 'num-neg';
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  }
}

function exportCsv() {
  const rows = sortSignals(state.signals.filter(feedRowMatches));
  const header = ['time', 'symbol', 'direction', 'price', 'roi', 'benchmark_roi',
                  'excess_roi', 'win_rate', 'trades', 'strategy', 'rl_confidence'];
  const lines = [header.join(',')];
  for (const s of rows) {
    lines.push(header.map(k => JSON.stringify(s[k] ?? '')).join(','));
  }
  const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'signals.csv';
  a.click();
  URL.revokeObjectURL(a.href);
}

/* ── Chart / symbol analysis ─────────────────────────────────────────── */

async function loadSymbol(symbol) {
  $('symbol-input').value = symbol;
  const interval = $('chart-interval').value;
  const period = $('chart-period').value;
  let data;

  try {
    data = await Api.symbol(symbol, { interval, period, overlays: 'SMA20,SMA50,SMA150' });
    Charts.renderCandles(data);

    const banner = $('verdict-banner');
    banner.hidden = false;
    if (data.verdict === 'BUY') {
      banner.className = 'verdict-banner buy';
      banner.textContent = `▲ BUY signal for ${data.symbol} · best strategy: ${data.best_strategy}`;
    } else if (data.verdict === 'SELL') {
      banner.className = 'verdict-banner sell';
      banner.textContent = `▼ SELL signal for ${data.symbol} · best strategy: ${data.best_strategy}`;
    } else {
      banner.className = 'verdict-banner none';
      banner.textContent = `No signal at this time for ${data.symbol} · best strategy: ${data.best_strategy || '—'}`;
    }

    renderStrategyTable(data.strategies, data.best_strategy);
    renderCompanyInfo(data.meta);
    if (data.verdict !== 'BUY' && data.verdict !== 'SELL') renderEntryPrice(null);
  } catch (e) {
    alert(`Failed to load ${symbol}: ${e.message}`);
    return;
  }

  // Independent, slower lookups — each in its own try/catch and fired
  // together so one hiccuping (or being slow) doesn't hold up the others, and
  // none of them can take down the chart that already loaded fine.
  await Promise.allSettled([
    (async () => {
      try {
        const history = await Api.symbolSignalHistory(symbol);
        Charts.setHistoricalMarkers(history.signals);
        renderPastSignals(history);
      } catch (e) {
        console.error('past-signals load failed', e);
      }
    })(),
    (async () => {
      try {
        renderNews(await Api.symbolNews(symbol));
      } catch (e) {
        console.error('news load failed', e);
      }
    })(),
    (async () => {
      if (data.verdict !== 'BUY' && data.verdict !== 'SELL') return;
      try {
        renderEntryPrice(await Api.symbolEntryPrice(symbol, data.verdict));
      } catch (e) {
        console.error('entry-price load failed', e);
        renderEntryPrice(null);
      }
    })(),
  ]);
}

function renderEntryPrice(suggestion) {
  const el = $('entry-price-banner');
  if (!suggestion) { el.hidden = true; return; }
  el.hidden = false;

  const verb = suggestion.direction === 'BUY' ? 'Buy' : 'Sell';
  const away = suggestion.distance_pct === 0
    ? 'at the current price'
    : `${Math.abs(suggestion.distance_pct)}% ${suggestion.distance_pct < 0 ? 'below' : 'above'} the current price`;

  el.innerHTML = `Suggested entry (hourly): ${verb} near <strong>$${suggestion.entry_price.toFixed(2)}</strong> ` +
    `— ${away} · ${suggestion.basis} (last ${suggestion.lookback_bars} hourly bars)`;
}

function renderCompanyInfo(meta) {
  const panel = $('company-info');
  if (!meta) { panel.hidden = true; return; }
  panel.hidden = false;

  const badges = $('company-badges');
  badges.innerHTML = '';
  const addBadge = (text, cls) => {
    if (!text) return;
    const b = document.createElement('span');
    b.className = 'badge' + (cls ? ` ${cls}` : '');
    b.textContent = text;
    badges.appendChild(b);
  };
  addBadge(meta.sector);
  addBadge(meta.industry);
  if (meta.recommendation) {
    const rec = meta.recommendation.toLowerCase();
    const cls = rec.includes('buy') ? 'rec-buy' : rec.includes('sell') ? 'rec-sell' : 'rec-hold';
    addBadge(meta.recommendation.replace(/_/g, ' '), cls);
  }

  const facts = $('company-facts');
  facts.innerHTML = '';
  const addFact = (label, value, href) => {
    if (!value) return;
    const el = href ? document.createElement('a') : document.createElement('span');
    el.textContent = `${label}: ${value}`;
    if (href) { el.href = href; el.target = '_blank'; el.rel = 'noopener noreferrer'; el.textContent = label; }
    facts.appendChild(el);
  };
  addFact('Market cap', fmtMarketCap(meta.market_cap));
  addFact('P/E (TTM)', meta.pe_ratio ? meta.pe_ratio.toFixed(2) : null);
  addFact('Fwd P/E', meta.forward_pe_ratio ? meta.forward_pe_ratio.toFixed(2) : null);
  addFact('Employees', meta.employees ? meta.employees.toLocaleString() : null);
  addFact('Dividend', meta.dividend && meta.dividend !== 'No Dividend' ? meta.dividend : null);
  if (meta.website) addFact('Website', null, meta.website);

  const summaryEl = $('company-summary');
  const toggleEl = $('company-summary-toggle');
  summaryEl.textContent = meta.business_summary || '';
  summaryEl.classList.remove('expanded');
  toggleEl.hidden = !meta.business_summary;
  toggleEl.textContent = 'Show more';
  toggleEl.onclick = () => {
    const expanded = summaryEl.classList.toggle('expanded');
    toggleEl.textContent = expanded ? 'Show less' : 'Show more';
  };
}

function fmtMarketCap(v) {
  if (!v) return null;
  if (v >= 1e12) return `$${(v / 1e12).toFixed(2)}T`;
  if (v >= 1e9)  return `$${(v / 1e9).toFixed(2)}B`;
  if (v >= 1e6)  return `$${(v / 1e6).toFixed(1)}M`;
  return `$${v.toLocaleString()}`;
}

function fmtRelativeTime(iso) {
  if (!iso) return '';
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return '';
  const mins = Math.max(0, Math.round((Date.now() - then) / 60000));
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

function renderNews(data) {
  const section = $('news-section');
  const list = $('news-list');
  const news = (data && data.news) || [];

  section.hidden = false;
  list.innerHTML = '';
  $('news-empty').hidden = news.length > 0;

  for (const n of news) {
    if (!n.title) continue;
    // Anchor when there's somewhere to send the click; plain div otherwise —
    // no javascript: pseudo-href workaround for the no-link case.
    const a = document.createElement(n.link ? 'a' : 'div');
    a.className = 'news-item';
    if (n.link) { a.href = n.link; a.target = '_blank'; a.rel = 'noopener noreferrer'; }

    const title = document.createElement('span');
    title.className = 'news-title';
    title.textContent = n.title;

    const meta = document.createElement('span');
    meta.className = 'news-meta';
    const bits = [n.publisher, fmtRelativeTime(n.published_at)].filter(Boolean);
    meta.textContent = bits.join(' · ');

    a.append(title, meta);
    list.appendChild(a);
  }
}

function renderStrategyTable(rows, best) {
  const table = $('strategy-table');
  const tbody = $('strategy-tbody');
  tbody.innerHTML = '';
  table.hidden = !rows || !rows.length;

  for (const r of rows || []) {
    const tr = document.createElement('tr');
    if (r.strategy === best) tr.className = 'selected-row';

    const star = document.createElement('td');
    star.className = 'star';
    star.textContent = r.strategy === best ? '★' : '';

    const name = document.createElement('td');
    name.textContent = r.strategy;

    const vals = [
      fmtPct(r.train_roi), fmtPct(r.roi), fmtPct(r.benchmark_roi), fmtPct(r.excess_roi),
      r.win_rate != null ? `${fmtNum(r.win_rate, 1)}%` : '—',
      r.trades ?? '—', fmtPct(r.max_drawdown),
    ];

    tr.appendChild(star);
    tr.appendChild(name);
    vals.forEach((v, i) => {
      const td = document.createElement('td');
      td.textContent = v;
      if (i === 3) td.className = signClass(r.excess_roi);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  }
}

function renderPastSignals(history) {
  const { symbol, signals } = history;

  $('past-signals').hidden = false;
  $('past-signals-symbol').textContent = `— ${symbol}`;

  const tbody = $('past-signals-tbody');
  tbody.innerHTML = '';
  $('past-signals-table').hidden = !signals.length;
  $('past-signals-empty').hidden = signals.length > 0;

  // Newest first for the table (matches the main history table's convention);
  // the chart below reads oldest-to-newest, left to right.
  for (const s of [...signals].reverse()) {
    const tr = document.createElement('tr');

    const time = document.createElement('td');
    time.textContent = s.time || '';

    const dir = document.createElement('td');
    dir.textContent = s.direction;
    dir.className = s.direction === 'BUY' ? 'num-pos' : 'num-neg';

    const price = document.createElement('td');
    price.textContent = `$${fmtNum(s.price)}`;

    const callReturn = document.createElement('td');
    callReturn.textContent = fmtPct(s.call_return_pct);
    callReturn.className = signClass(s.call_return_pct);

    const roi = document.createElement('td');
    roi.textContent = fmtPct(s.roi);

    const excess = document.createElement('td');
    excess.textContent = fmtPct(s.excess_roi);
    excess.className = signClass(s.excess_roi);

    const strat = document.createElement('td');
    strat.textContent = s.strategy || '';

    const rl = document.createElement('td');
    if (s.rl_action) {
      rl.textContent = `${s.rl_action} (${Math.round((s.rl_confidence || 0) * 100)}%)`;
      rl.style.color = s.rl_agrees ? 'var(--green)' : 'var(--text-dim)';
    } else {
      rl.textContent = '—';
    }

    tr.append(time, dir, price, callReturn, roi, excess, strat, rl);
    tbody.appendChild(tr);
  }

  Charts.renderSignalPerformance(signals);
}

/* ── SSE ──────────────────────────────────────────────────────────────── */

function connectStream() {
  const pill = $('conn-pill');
  pill.dataset.state = 'connecting';
  $('conn-label').textContent = 'connecting…';

  const es = new EventSource('/api/stream');

  es.onopen = () => {
    pill.dataset.state = 'connected';
    $('conn-label').textContent = 'live';
  };
  es.onerror = () => {
    pill.dataset.state = 'disconnected';
    $('conn-label').textContent = 'reconnecting…';
  };
  es.addEventListener('status', (ev) => applyStatus(JSON.parse(ev.data)));
  es.addEventListener('signal', (ev) => prependSignal(JSON.parse(ev.data)));
  es.addEventListener('ping', () => {});
}

/* ── Controls ─────────────────────────────────────────────────────────── */

// Debounce so dragging the slider doesn't fire a request per pixel.
function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

const refreshUniverseCount = debounce(async () => {
  const note = $('market-cap-note');
  note.textContent = 'checking universe size…';
  try {
    const { count } = await Api.universeCount(currentUniverseParams());
    note.textContent = `~${count.toLocaleString()} symbols. Lower = more symbols, longer scans.`;
  } catch (e) {
    note.textContent = 'Could not check universe size — it will still resolve on Start.';
  }
}, 400);

function marketCapDollars() {
  return Number($('market-cap-range').value) * 1e9;
}

// {exchange, sector, min_market_cap} — the three filters the scanner, the
// universe-size check, and the symbol picker all resolve the same way.
function currentUniverseParams() {
  return {
    exchange:       $('exchange-select').value,
    sector:         $('sector-select').value,
    min_market_cap: marketCapDollars(),
  };
}

function fmtCapLabel(dollars) {
  if (dollars <= 0) return 'No minimum';
  if (dollars >= 1e12) return `$${(dollars / 1e12).toFixed(2)}T`;
  if (dollars >= 1e9)  return `$${(dollars / 1e9).toFixed(1)}B`;
  return `$${(dollars / 1e6).toFixed(0)}M`;
}

function applyMarketCap(capB) {
  $('market-cap-range').value = capB;
  $('market-cap-val').textContent = fmtCapLabel(capB * 1e9);
  for (const chip of document.querySelectorAll('.cap-chip')) {
    chip.classList.toggle('active', Number(chip.dataset.cap) === capB);
  }
  refreshUniverseCount();
}

function wireControls() {
  $('lookback-range').addEventListener('input', (e) => $('lookback-val').textContent = e.target.value);
  $('margin-range').addEventListener('input', (e) => $('margin-val').textContent = e.target.value);

  $('market-cap-range').addEventListener('input', (e) => applyMarketCap(Number(e.target.value)));
  for (const chip of document.querySelectorAll('.cap-chip')) {
    chip.addEventListener('click', () => applyMarketCap(Number(chip.dataset.cap)));
  }

  $('exchange-select').addEventListener('change', () => refreshUniverseCount());
  $('sector-select').addEventListener('change', () => refreshUniverseCount());

  $('scan-toggle').addEventListener('click', async () => {
    if (state.scanBusy) return;
    state.scanBusy = true;
    $('scan-toggle').disabled = true;
    try {
      if (state.status && state.status.running) {
        applyStatus(await Api.stopScanner());
      } else {
        const body = {
          ...currentUniverseParams(),
          timeframe:  $('timeframe-select').value,
          lookback:   Number($('lookback-range').value),
          min_margin: Number($('margin-range').value),
        };
        $('scan-error').hidden = true;
        applyStatus(await Api.startScanner(body));
      }
    } catch (e) {
      $('scan-error').hidden = false;
      $('scan-error').textContent = `Error: ${e.message}`;
    } finally {
      state.scanBusy = false;
      $('scan-toggle').disabled = false;
    }
  });

  $('clear-btn').addEventListener('click', async () => {
    if (!confirm('Clear all signal history?')) return;
    await Api.clearSignals();
    state.signals = [];
    renderFeed();
    renderHistory();
    refreshSummary();
  });

  $('feed-direction').addEventListener('change', (e) => {
    state.filterDirection = e.target.value;
    renderFeed();
    renderHistory();
  });
  $('feed-symbol').addEventListener('input', (e) => {
    state.filterSymbol = e.target.value.toUpperCase();
    renderFeed();
    renderHistory();
  });
  $('feed-sort').addEventListener('change', (e) => {
    state.sortBy = e.target.value;
    renderFeed();
    renderHistory();
  });

  $('analyse-btn').addEventListener('click', () => {
    const sym = $('symbol-input').value.trim();
    if (sym) loadSymbol(sym);
  });
  $('symbol-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') $('analyse-btn').click();
  });

  $('export-btn').addEventListener('click', exportCsv);
}

async function loadUniverse() {
  try {
    const list = await Api.universe();
    const datalist = $('universe-list');
    const frag = document.createDocumentFragment();
    for (const { name, symbol } of list) {
      const opt = document.createElement('option');
      opt.value = symbol;
      opt.label = name;
      frag.appendChild(opt);
    }
    datalist.appendChild(frag);
  } catch (e) { console.error('universe load failed', e); }
}

/* ── Init ─────────────────────────────────────────────────────────────── */

(async function init() {
  wireControls();
  connectStream();
  await Promise.all([loadSignalsFromServer(), loadUniverse()]);
  try { applyStatus(await Api.status()); } catch (_) {}
  applyMarketCap(Number($('market-cap-range').value));
  Charts.renderDonut(0, 0);
  Charts.renderHistogram([]);
  setInterval(tickNextScan, 1000);
})();
