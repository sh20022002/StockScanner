/* Candlestick chart (Lightweight Charts) plus two hand-rolled canvas charts
   for the signal distribution, so the whole app depends on one small vendored
   library instead of a general-purpose charting stack. */
const Charts = (() => {
  let chart = null;
  let candleSeries = null;
  let volumeSeries = null;
  const overlaySeries = {};
  const overlayColors = ['#4d9dff', '#ffb454', '#c792ea', '#7ee8fa', '#ff8ba7'];
  let hmmSeries = { mean: null, upper: null, lower: null };

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
    clearHmmProjection();   // last symbol's projection must not bleed onto new candles
    clearBounds();          // ditto for the peaks/troughs trendlines

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

  /* ── HMM regime projection (optional overlay) ─────────────────────────
     A fresh-fit-per-request, forward-looking line + uncertainty band —
     distinct from the historical overlays above (SMA), so it's tracked and
     cleared separately rather than going through overlaySeries. Bright,
     thick and solid on purpose: this used to blend into the SMA overlay
     color rotation, which defeats the point of a forecast being visually
     unmissable against actual price action. */
  function clearHmmProjection() {
    if (chart) Object.values(hmmSeries).forEach(s => s && chart.removeSeries(s));
    hmmSeries = { mean: null, upper: null, lower: null };
  }

  function renderHmmProjection(payload) {
    if (!chart) return;
    clearHmmProjection();
    if (!payload || !payload.available || !(payload.projection || []).length) return;

    const points = payload.projection;
    hmmSeries.mean = chart.addLineSeries({
      color: '#ff2ec4', lineWidth: 3, lineStyle: LightweightCharts.LineStyle.Solid,
      priceLineVisible: false, lastValueVisible: false, title: 'HMM projection',
    });
    hmmSeries.mean.setData(points.map(p => ({ time: p.time, value: p.price })));

    const bandOpts = {
      color: 'rgba(255,46,196,0.55)', lineWidth: 2,
      lineStyle: LightweightCharts.LineStyle.Dashed,
      priceLineVisible: false, lastValueVisible: false,
    };
    hmmSeries.upper = chart.addLineSeries(bandOpts);
    hmmSeries.upper.setData(points.map(p => ({ time: p.time, value: p.upper })));
    hmmSeries.lower = chart.addLineSeries(bandOpts);
    hmmSeries.lower.setData(points.map(p => ({ time: p.time, value: p.lower })));
  }

  /* ── Peaks & Troughs (optional overlay) ────────────────────────────────
     Two straight trendlines computed client-side from the loaded candles —
     not a rolling envelope. Both anchor to "the bottom candle" (the lowest
     Open in the loaded window): the Troughs line runs from its Open through
     the last confirming swing-low Open after it; the Peaks line from its
     High through the last confirming swing-high High after it. A swing
     point is the standard 3-candle fractal — more extreme than the candle
     immediately before and after it. Either line needs the bottom candle
     plus at least 2 confirming swing points after it (3 candles' worth of
     evidence) or it isn't drawn at all — two arbitrary points aren't a
     trend, and a misleading line is worse than no line. */
  let boundsSeries = { peaks: null, troughs: null };

  function clearBounds() {
    if (chart) Object.values(boundsSeries).forEach(s => s && chart.removeSeries(s));
    boundsSeries = { peaks: null, troughs: null };
  }

  function findSwingIndices(candles, valueOf, isMoreExtreme) {
    const swings = [];
    for (let i = 1; i < candles.length - 1; i++) {
      const v = valueOf(candles[i]);
      if (isMoreExtreme(v, valueOf(candles[i - 1])) && isMoreExtreme(v, valueOf(candles[i + 1]))) {
        swings.push(i);
      }
    }
    return swings;
  }

  function renderBounds(candles) {
    clearBounds();
    if (!chart || !candles || candles.length < 3) return;

    let bottomIdx = 0;
    for (let i = 1; i < candles.length; i++) {
      if (candles[i].open < candles[bottomIdx].open) bottomIdx = i;
    }

    const troughSwings = findSwingIndices(candles, c => c.open, (a, b) => a < b)
      .filter(i => i > bottomIdx);
    const peakSwings = findSwingIndices(candles, c => c.high, (a, b) => a > b)
      .filter(i => i > bottomIdx);

    if (troughSwings.length >= 2) {
      const lastIdx = troughSwings[troughSwings.length - 1];
      boundsSeries.troughs = chart.addLineSeries({
        color: '#3ddc84', lineWidth: 2, lineStyle: LightweightCharts.LineStyle.Solid,
        priceLineVisible: false, lastValueVisible: false, title: 'Troughs',
      });
      boundsSeries.troughs.setData([
        { time: candles[bottomIdx].time, value: candles[bottomIdx].open },
        { time: candles[lastIdx].time,   value: candles[lastIdx].open },
      ]);
    }

    if (peakSwings.length >= 2) {
      const lastIdx = peakSwings[peakSwings.length - 1];
      boundsSeries.peaks = chart.addLineSeries({
        color: '#ff6b81', lineWidth: 2, lineStyle: LightweightCharts.LineStyle.Solid,
        priceLineVisible: false, lastValueVisible: false, title: 'Peaks',
      });
      boundsSeries.peaks.setData([
        { time: candles[bottomIdx].time, value: candles[bottomIdx].high },
        { time: candles[lastIdx].time,   value: candles[lastIdx].high },
      ]);
    }
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

  return {
    renderCandles, setHistoricalMarkers, renderDonut, renderHistogram, renderSignalPerformance,
    renderHmmProjection, clearHmmProjection, renderBounds, clearBounds,
  };
})();
