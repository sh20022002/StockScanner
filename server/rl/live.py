"""
Live inference hook: layer the trained RL policy on top of the rule-based
scanner's own output.

This does not replace strategy.py's scanner or run its own separate download —
it reuses the same batch-fetched frames the rule-based scan already pulled for
the symbols that fired a signal. Re-fetching per symbol here would reintroduce
exactly the anti-pattern the rest of this pipeline was rewritten to remove
(Strategy.__init__ used to fetch its own quote per symbol and defeat the whole
point of batching).

Opt-in and fail-soft by design: if there is no checkpoint, or a symbol's
feature schema doesn't match what the checkpoint was trained on, or inference
raises, the rule-based signal passes through unchanged. The RL layer can only
add information (rl_action / rl_confidence / rl_agrees fields), never remove
a signal the rule-based scanner already found.
"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # server/rl/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # server/

import polars as pl
import torch

from features import build_raw_features
from model import load_checkpoint

CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'checkpoints', 'best.pt')

_ACTIONS = {0: 'SELL', 1: 'FLAT', 2: 'BUY'}

_lock = threading.Lock()
_cache: dict = {}


def model_available() -> bool:
    return os.path.exists(CHECKPOINT_PATH)


def _load() -> dict:
    with _lock:
        if 'model' not in _cache:
            model, normalizer, feature_names, record = load_checkpoint(CHECKPOINT_PATH)
            _cache.update(
                model=model, normalizer=normalizer, feature_names=feature_names,
                window=record['feature_cfg']['window'],
                include_rule_signals=record['feature_cfg']['include_rule_signals'],
                extra=record.get('extra', {}),
            )
        return _cache


def infer_symbol(symbol: str, df: pl.DataFrame) -> dict | None:
    """
    Run the policy, deterministically, on a symbol's already-fetched,
    indicator-enriched frame. Returns None rather than guessing if the frame
    doesn't have enough history or doesn't match the checkpoint's feature
    schema (e.g. it was trained with include_rule_signals off).
    """
    c = _load()
    raw = build_raw_features(df, include_rule_signals=c['include_rule_signals'])
    if raw['feature_names'] != c['feature_names']:
        return None

    feats = raw['features'][raw['valid_from']:]
    window = c['window']
    if len(feats) < window:
        return None

    tail = c['normalizer'].transform(feats[-window:])
    obs = torch.from_numpy(tail).float().unsqueeze(0)
    action, _logp, _value, probs = c['model'].act(obs, deterministic=True)
    a = int(action.item())

    return {
        'rl_action':     _ACTIONS[a],
        'rl_confidence': round(float(probs[0, a].item()), 3),
        'rl_probs':      {_ACTIONS[i]: round(float(probs[0, i].item()), 3)
                          for i in range(probs.shape[-1])},
    }


def annotate(found: list[dict], stock_data: dict) -> list[dict]:
    """
    Enrich each rule-based signal in `found` with the RL policy's own read on
    that symbol, computed from the frame already in `stock_data[symbol]` — no
    extra network calls.
    """
    if not found or not model_available():
        return found

    try:
        _load()
    except Exception as e:
        print(f'[rl.live] failed to load checkpoint: {e}')
        return found

    for entry in found:
        df = stock_data.get(entry.get('symbol'))
        if df is None:
            continue
        try:
            rl = infer_symbol(entry['symbol'], df)
        except Exception as e:
            print(f"[rl.live] inference failed for {entry.get('symbol')}: {e}")
            rl = None
        if rl:
            entry.update(rl)
            entry['rl_agrees'] = (
                (entry.get('direction') == 'BUY' and rl['rl_action'] == 'BUY') or
                (entry.get('direction') == 'SELL' and rl['rl_action'] == 'SELL')
            )

    return found
