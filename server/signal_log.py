"""
Shared persistence for scanner signals.

Extracted out of the old Streamlit dashboard so the web app's background
scanner task and its HTTP handlers can both use it. Signal semantics
(dedupe key, atomic write, history cap) are unchanged from before.
"""
import html
import json
import os
import re
import tempfile
import threading
from pathlib import Path

SIGNALS_FILE = Path(__file__).parent / "signals_log.json"
MAX_HISTORY  = 500

# Yahoo tickers: letters, digits, dots and hyphens only. Anything else is either
# a typo or an injection attempt — user-entered symbols are persisted and later
# rendered, so this is a security boundary, not just validation.
_SYMBOL_RE = re.compile(r'^[A-Za-z0-9.\-]{1,10}$')

# One process, potentially several concurrent requests/threads touching the file.
_log_lock = threading.Lock()


def esc(value) -> str:
    """Escape a value before it is interpolated into HTML."""
    return html.escape(str(value), quote=True)


def valid_symbol(symbol: str) -> bool:
    return bool(_SYMBOL_RE.match(symbol or ''))


def load_signals() -> list[dict]:
    if not SIGNALS_FILE.exists():
        return []
    try:
        data = json.loads(SIGNALS_FILE.read_text(encoding='utf-8'))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _write_atomic(signals: list[dict]):
    """Replace the log in one atomic step so a reader never sees a partial file."""
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


def add_signals(entries: list[dict]) -> list[dict]:
    """
    Append new signals, skipping duplicates.

    Returns the subset that was actually new — the caller (the web app) uses
    this to know what to push over SSE, rather than re-broadcasting the whole
    history on every scan tick.
    """
    if not entries:
        return []
    with _log_lock:
        current = load_signals()
        seen = {_dedupe_key(e) for e in current}
        fresh = [e for e in entries if _dedupe_key(e) not in seen]
        if not fresh:
            return []
        merged = (fresh + current)[:MAX_HISTORY]
        _write_atomic(merged)
        return fresh
