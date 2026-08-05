/* Candlestick chart (Lightweight Charts) plus two hand-rolled canvas charts
   for the signal distribution, so the whole app depends on one small vendored
   library instead of a general-purpose charting stack. */
const Charts = (() => {
  let chart = null;
  let candleSeries = null;
  let volumeSeries = null;
  const overlaySeries = {};
  const overlayColors = ['#4d9dff', '#ffb454', '#c792ea', '#7ee8fa', '#ff8ba7'];

  // The backtest-derived markers (renderCandles) and the real historical
  // signal markers (setHistoricalMarkers) are set by two separate async calls
  // in app.js's loadSymbol(); both need to land on the chart together without
  // one clobbering the other, so each is tracked and merged before applying.
  let backtestMarkers = [];
  let historicalMarkers = [];

  function applyMarkers() {
    if (!candleSeries) return;
    candleSeries.setMarkers(
      [...backtestMarkers, ...historicalMarkers].sort((a, b) => a.time - b.time)
    );
  }

  function ensureChart() {
    if (chart) return;
    const el = document.getElementById('chart-container');
    document.getElementById('chart-placeholder').style.display = 'none';

    chart = LightweightCharts.createChart(el, {
      layout: { background: { color: 'transparent' }, textColor: '#8a90a6' },
      grid: {
        vertLines: { color: '#181c29' },
        horzLines: { color: '#181c29' },
      },
      rightPriceScale: { borderColor: '#232838' },
      timeScale: { borderColor: '#232838', timeVisible: false },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
      autoSize: true,
    });

    candleSeries = chart.addCandlestickSeries({
      upColor: '#00d97e', downColor: '#ff4d6a',
      borderUpColor: '#00d97e', borderDownColor: '#ff4d6a',
      wickUpColor: '#00d97e', wickDownColor: '#ff4d6a',
    });

    volumeSeries = chart.addHistogramSeries({
      color: '#2a3050', priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
    });
    chart.priceScale('volume').applyOptions({ scaleMargins: { top: 0.85, bottom: 0 } });
  }

  function renderCandles(payload) {
    ensureChart();

    candleSeries.setData(payload.candles.map(c => ({
      time: c.time, open: c.open, high: c.high, low: c.low, close: c.close,
    })));

    volumeSeries.setData(payload.candles.map(c => ({
      time: c.time, value: c.volume || 0,
      color: c.close >= c.open ? 'rgba(0,217,126,0.35)' : 'rgba(255,77,106,0.35)',
    })));

    Object.values(overlaySeries).forEach(s => chart.removeSeries(s));
    for (const key in overlaySeries) delete overlaySeries[key];

    let i = 0;
    for (const [name, points] of Object.entries(payload.overlays || {})) {
      if (!points.length) continue;
      const s = chart.addLineSeries({
        color: overlayColors[i++ % overlayColors.length],
        lineWidth: 1, priceLineVisible: false, lastValueVisible: false,
      });
      s.setData(points);
      overlaySeries[name] = s;
    }

    backtestMarkers = payload.markers || [];
    historicalMarkers = [];   // reset until the caller loads them for this symbol
    applyMarkers();
    chart.timeScale().fitContent();
  }

  /* Real past signals (server/signal_log), distinct from the backtest-derived
     markers above — small hollow circles so they read as "this actually
     happened" rather than "the current best strategy would have fired here". */
  function setHistoricalMarkers(signals) {
    historicalMarkers = (signals || [])
      .map(s => {
        const t = Date.parse(s.time.replace(' ', 'T') + 'Z');
        if (Number.isNaN(t)) return null;
        return {
          time: Math.floor(t / 1000),
          position: s.direction === 'BUY' ? 'belowBar' : 'aboveBar',
          color: s.direction === 'BUY' ? '#4d9dff' : '#ffb454',
          shape: 'circle',
          text: '',
        };
      })
      .filter(Boolean);
    applyMarkers();
  }

  /* ── Donut: buy vs sell ─────────────────────────────────────────────── */
  function renderDonut(buys, sells) {
    const cv = document.getElementById('donut-canvas');
    const ctx = cv.getContext('2d');
    const w = cv.width, h = cv.height, cx = w / 2, cy = h / 2, r = Math.min(w, h) / 2 - 10;
    ctx.clearRect(0, 0, w, h);

    const total = buys + sells;
    if (total === 0) {
      ctx.fillStyle = '#565c72';
      ctx.font = '13px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No data yet', cx, cy);
      return;
    }

    const segments = [
      { value: buys, color: '#00d97e', label: 'BUY' },
      { value: sells, color: '#ff4d6a', label: 'SELL' },
    ];
    let start = -Math.PI / 2;
    for (const seg of segments) {
      if (!seg.value) continue;
      const angle = (seg.value / total) * Math.PI * 2;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, r, start, start + angle);
      ctx.closePath();
      ctx.fillStyle = seg.color;
      ctx.fill();
      start += angle;
    }
    ctx.globalCompositeOperation = 'destination-out';
    ctx.beginPath();
    ctx.arc(cx, cy, r * 0.6, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalCompositeOperation = 'source-over';

    ctx.fillStyle = '#e8eaf0';
    ctx.font = 'bold 20px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(total, cx, cy + 2);
    ctx.font = '10px sans-serif';
    ctx.fillStyle = '#8a90a6';
    ctx.fillText('signals', cx, cy + 18);

    ctx.font = '11px sans-serif';
    ctx.fillStyle = '#00d97e';
    ctx.textAlign = 'left';
    ctx.fillText(`● BUY ${buys}`, 4, h - 6);
    ctx.fillStyle = '#ff4d6a';
    ctx.textAlign = 'right';
    ctx.fillText(`SELL ${sells} ●`, w - 4, h - 6);
  }

  /* ── Histogram: excess ROI distribution ───────────────────────────── */
  function renderHistogram(values) {
    const cv = document.getElementById('hist-canvas');
    const ctx = cv.getContext('2d');
    const w = cv.width, h = cv.height;
    const padL = 30, padB = 18, padT = 10, padR = 6;
    ctx.clearRect(0, 0, w, h);

    ctx.fillStyle = '#8a90a6';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Excess ROI distribution (out-of-sample, vs buy & hold)', w / 2, 12);

    if (!values.length) {
      ctx.fillStyle = '#565c72';
      ctx.fillText('No data yet', w / 2, h / 2);
      return;
    }

    const nBins = 16;
    const lo = Math.min(0, ...values), hi = Math.max(0, ...values);
    const span = (hi - lo) || 1;
    const bins = new Array(nBins).fill(0);
    for (const v of values) {
      let idx = Math.floor(((v - lo) / span) * nBins);
      idx = Math.max(0, Math.min(nBins - 1, idx));
      bins[idx]++;
    }
    const maxCount = Math.max(...bins, 1);

    const plotW = w - padL - padR, plotH = h - padT - padB - 12;
    const barW = plotW / nBins;

    const zeroFrac = (0 - lo) / span;
    const zeroX = padL + zeroFrac * plotW;
    if (lo < 0 && hi > 0) {
      ctx.strokeStyle = '#ff4d6a';
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(zeroX, padT + 12);
      ctx.lineTo(zeroX, padT + 12 + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    bins.forEach((count, i) => {
      const x = padL + i * barW;
      const barH = (count / maxCount) * plotH;
      const binCenter = lo + (i + 0.5) * (span / nBins);
      ctx.fillStyle = binCenter >= 0 ? 'rgba(0,217,126,0.7)' : 'rgba(255,77,106,0.7)';
      ctx.fillRect(x + 1, padT + 12 + plotH - barH, Math.max(barW - 2, 1), barH);
    });

    ctx.strokeStyle = '#232838';
    ctx.beginPath();
    ctx.moveTo(padL, padT + 12 + plotH);
    ctx.lineTo(w - padR, padT + 12 + plotH);
    ctx.stroke();

    ctx.fillStyle = '#565c72';
    ctx.font = '10px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(lo.toFixed(0) + '%', padL, h - 4);
    ctx.textAlign = 'right';
    ctx.fillText(hi.toFixed(0) + '%', w - padR, h - 4);
  }

  /* ── Per-symbol signal performance: call_return_pct, chronological ───
     One bar per past signal, oldest to newest. Green/above zero means the
     call was right (price moved the direction the signal predicted); red/
     below means it wasn't. This is a real price outcome, not a backtest
     number — see the note above the table in index.html. */
  function renderSignalPerformance(signals) {
    const cv = document.getElementById('perf-canvas');
    const ctx = cv.getContext('2d');
    const w = cv.width = cv.clientWidth || 640;
    const h = cv.height = 160;
    const padL = 36, padR = 8, padT = 10, padB = 20;
    ctx.clearRect(0, 0, w, h);

    const values = signals.map(s => s.call_return_pct).filter(v => v !== null && v !== undefined);

    if (!values.length) {
      ctx.fillStyle = '#565c72';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No past signals yet for this symbol', w / 2, h / 2);
      return;
    }

    const plotW = w - padL - padR, plotH = h - padT - padB;
    const maxAbs = Math.max(...values.map(Math.abs), 1);
    const zeroY = padT + plotH / 2;
    const barW = plotW / values.length;

    ctx.strokeStyle = '#232838';
    ctx.beginPath();
    ctx.moveTo(padL, zeroY);
    ctx.lineTo(w - padR, zeroY);
    ctx.stroke();

    values.forEach((v, i) => {
      const x = padL + i * barW;
      const barH = (Math.abs(v) / maxAbs) * (plotH / 2);
      ctx.fillStyle = v >= 0 ? 'rgba(0,217,126,0.75)' : 'rgba(255,77,106,0.75)';
      ctx.fillRect(x + 1, v >= 0 ? zeroY - barH : zeroY, Math.max(barW - 2, 1), barH);
    });

    ctx.fillStyle = '#565c72';
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(`+${maxAbs.toFixed(0)}%`, padL - 4, padT + 8);
    ctx.fillText(`-${maxAbs.toFixed(0)}%`, padL - 4, h - padB + 4);
    ctx.textAlign = 'left';
    ctx.fillText('oldest', padL, h - 4);
    ctx.textAlign = 'right';
    ctx.fillText('newest', w - padR, h - 4);
  }

  return { renderCandles, setHistoricalMarkers, renderDonut, renderHistogram, renderSignalPerformance };
})();
