/* Thin fetch wrappers over the FastAPI backend. */
const Api = (() => {
  async function req(path, options = {}) {
    const res = await fetch(path, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (_) {}
      throw new Error(detail);
    }
    if (res.status === 204) return null;
    return res.json();
  }

  return {
    status:        () => req('/api/status'),
    startScanner:  (body) => req('/api/scanner/start', { method: 'POST', body: JSON.stringify(body) }),
    stopScanner:   () => req('/api/scanner/stop', { method: 'POST' }),
    signals:       (params = {}) => req('/api/signals?' + new URLSearchParams(params)),
    signalsSummary:() => req('/api/signals/summary'),
    sectorSummary: () => req('/api/signals/sector-summary'),
    clearSignals:  () => req('/api/signals', { method: 'DELETE' }),
    universe:      (params = {}) => req('/api/universe?' + new URLSearchParams(params)),
    universeCount: (params = {}) => req('/api/universe/count?' + new URLSearchParams(params)),
    symbol:        (sym, params = {}) => req(`/api/symbol/${encodeURIComponent(sym)}?` + new URLSearchParams(params)),
    symbolSignalHistory: (sym) => req(`/api/symbol/${encodeURIComponent(sym)}/signal-history`),
    symbolNews:    (sym, count = 10) => req(`/api/symbol/${encodeURIComponent(sym)}/news?` + new URLSearchParams({ count })),
    symbolEntryPrice: (sym, direction) => req(`/api/symbol/${encodeURIComponent(sym)}/entry-price?` + new URLSearchParams({ direction })),
    symbolHmmProjection: (sym, params = {}) => req(`/api/symbol/${encodeURIComponent(sym)}/hmm-projection?` + new URLSearchParams(params)),
    longTermScreen: (params = {}) => req('/api/long-term-screen?' + new URLSearchParams(params)),
  };
})();
