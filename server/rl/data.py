"""
Fetch, feature-engineer and chronologically split data for the RL pipeline.

One design choice worth being explicit about: the feature normalizer is fit
ONCE on the pooled training rows of every training symbol combined, not one
normalizer per symbol. A single agent is meant to trade across the scanned
universe, including symbols it never trained on — that only works if every
symbol's raw features are mapped into the same normalized space the same way.
Fitting per symbol would make the model's inputs mean something different
for AAPL than for a $4 penny stock, and the policy would have no way to
transfer.
"""
import os
import sys

# Import order matters on Windows — see the comment in features.py. This module
# imports scraping (pandas), so torch must load first if anything downstream in
# the same process will need it.
try:
    import torch  # noqa: F401
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # server/rl/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # server/

import numpy as np

import scraping
from config import FeatureConfig
from features import Normalizer, build_raw_features

# Smallest val/test slice worth reporting a metric on — below this a Sharpe or
# excess-ROI number is mostly noise, not a judgement of the policy.
MIN_EVAL_BARS = 30


def load_symbol_frames(symbols: list[str], timeframe: str, period: str) -> dict:
    """{symbol: pl.DataFrame} via the same batch download the scanner uses."""
    return scraping.batch_download(symbols, period=period, interval=timeframe, quiet=True)


def _split_bounds(n: int, train_fraction: float, val_fraction: float,
                  window: int, episode_length: int):
    """
    Chronological train/val/test cut points, or None if any split is too thin
    to be usable.

    Train needs room for at least one full episode (window + episode_length).
    Val/test only ever run a single deterministic full-series pass — they need
    window + 2 bars to construct a TradingEnv at all (see env.py), plus enough
    beyond that for the resulting metric to mean something (MIN_EVAL_BARS).
    Using the training minimum for val/test too — as an earlier version of this
    function did — rejected perfectly usable 2-year/daily/70-15-15 splits for
    no reason: a ~75-bar validation slice is plenty to evaluate on, and the
    only bar it needs to clear is "construct successfully".
    """
    train_end = int(n * train_fraction)
    val_end = train_end + int(n * val_fraction)
    min_train = max(window + episode_length, window + 2)
    min_eval = max(window + 2, MIN_EVAL_BARS)
    if (train_end < min_train
            or (val_end - train_end) < min_eval
            or (n - val_end) < min_eval):
        return None
    return train_end, val_end


def prepare_dataset(symbols: list[str], timeframe: str, period: str,
                    feature_cfg: FeatureConfig, train_fraction: float,
                    val_fraction: float, normalizer: Normalizer | None = None,
                    episode_length: int = 0) -> dict | None:
    """
    Returns:
        {
            'normalizer': Normalizer,        # fit on pooled training rows, unless one was passed in
            'feature_names': [...],
            'per_symbol': {
                symbol: {
                    'splits':    {'train': (feats, close, open), 'val': (...), 'test': (...)},
                    'df_splits': {'train': pl.DataFrame, 'val': ..., 'test': ...},
                }
            },
            'skipped': [symbols with too little history to split],
        }
        or None if no symbol had enough history at all.

    Pass `normalizer` to reuse a checkpoint's training-time normalizer instead
    of fitting a new one — required for backtesting or live inference, where
    fitting fresh stats on whatever symbols happen to be evaluated would score
    the model against a distribution it was never trained to see.
    """
    frames = load_symbol_frames(symbols, timeframe, period)

    prepared = {}
    skipped = []
    train_pool = []
    feature_names = None

    for symbol, df in frames.items():
        raw = build_raw_features(df, include_rule_signals=feature_cfg.include_rule_signals)
        feature_names = feature_names or raw['feature_names']
        vf = raw['valid_from']
        n_valid = len(raw['close']) - vf
        bounds = _split_bounds(n_valid, train_fraction, val_fraction,
                               feature_cfg.window, episode_length)
        if bounds is None:
            skipped.append(symbol)
            continue
        train_end, val_end = bounds

        feats = raw['features'][vf:]
        close = raw['close'][vf:]
        open_ = raw['open'][vf:]
        df_valid = df[vf:]

        prepared[symbol] = {
            'feats': feats, 'close': close, 'open': open_, 'df': df_valid,
            'bounds': (train_end, val_end),
        }
        train_pool.append(feats[:train_end])

    if not prepared:
        return None

    if normalizer is None:
        normalizer = Normalizer().fit(np.concatenate(train_pool, axis=0))

    per_symbol = {}
    for symbol, p in prepared.items():
        train_end, val_end = p['bounds']
        feats_n = normalizer.transform(p['feats'])
        per_symbol[symbol] = {
            'splits': {
                'train': (feats_n[:train_end], p['close'][:train_end], p['open'][:train_end]),
                'val':   (feats_n[train_end:val_end], p['close'][train_end:val_end], p['open'][train_end:val_end]),
                'test':  (feats_n[val_end:], p['close'][val_end:], p['open'][val_end:]),
            },
            'df_splits': {
                'train': p['df'][:train_end],
                'val':   p['df'][train_end:val_end],
                'test':  p['df'][val_end:],
            },
        }

    return {
        'normalizer': normalizer,
        'feature_names': feature_names,
        'per_symbol': per_symbol,
        'skipped': skipped,
    }
