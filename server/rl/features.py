"""
Turn an indicator-enriched OHLCV frame into the RL agent's per-bar feature vector.

Two feature groups:

  1. Scale-free technical features derived from scraping.compute_indicators —
     price expressed as ratios/returns rather than levels, so the same policy
     works across a $20 stock and a $2,000 one.
  2. Optionally, the 12 rule-based strategies' own Buy_Signal/Sell_Signal flags
     from strategy.Strategy.detect_signals(). This is what makes the agent
     "trade based on the result of this code" rather than reinventing technical
     analysis from raw prices: the transformer sees, at every bar, whether MACD,
     RSI, the Ichimoku cloud, and the rest already think something is happening.

All of this is computed causally — every column at row t is a function of data
up to and including t, never later. That is what makes windowing it into
non-overlapping episodes safe.
"""
import os
import sys

# Import torch before anything that pulls in pandas (strategy -> scraping does).
# On Windows, this process has been observed to access-violate loading torch's
# c10.dll if pandas' compiled extensions initialise first — some vcruntime DLL
# the two disagree on the version of. Reproduced outside pytest too: `import
# pandas; import torch` crashes, `import torch; import pandas` does not. Import
# order is the whole fix; it costs nothing once torch is actually installed.
try:
    import torch  # noqa: F401
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # server/ — scraping, strategy

import numpy as np
import polars as pl

import strategy as strategy_module

BASE_FEATURE_COLUMNS = [
    'ret_1', 'dist_sma20', 'dist_sma50', 'dist_sma150',
    'rsi_n', 'macd_hist_n', 'atr_n', 'dist_vwap', 'stoch_k_n', 'vol_z',
]

RULE_SIGNAL_COLUMNS = [f'{name}_{side}'
                       for name in strategy_module.STRATEGY_NAMES
                       for side in ('buy', 'sell')]


def derive_base_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add the scale-free engineered columns used as model input."""
    df = df.with_columns([
        (pl.col('Close') / pl.col('Close').shift(1)).log().alias('ret_1'),
        (pl.col('Close') / pl.col('SMA20') - 1).alias('dist_sma20'),
        (pl.col('Close') / pl.col('SMA50') - 1).alias('dist_sma50'),
        (pl.col('Close') / pl.col('SMA150') - 1).alias('dist_sma150'),
        (pl.col('RSI') / 100.0).alias('rsi_n'),
        (pl.col('MACD_Hist') / pl.col('Close')).alias('macd_hist_n'),
        (pl.col('ATR') / pl.col('Close')).alias('atr_n'),
        (pl.col('Close') / pl.col('VWAP') - 1).alias('dist_vwap'),
        (pl.col('STOCH_%K') / 100.0).alias('stoch_k_n'),
    ])
    vol_mean = pl.col('Volume').rolling_mean(window_size=20, min_periods=5)
    vol_std  = pl.col('Volume').rolling_std(window_size=20, min_periods=5)
    df = df.with_columns(
        ((pl.col('Volume') - vol_mean) / (vol_std + 1e-6)).alias('vol_z')
    )
    return df


def rule_signal_frame(df: pl.DataFrame) -> pl.DataFrame:
    """
    Run the 12 rule-based strategies once and return their flags as float
    columns, aligned 1:1 with df's rows.
    """
    strat = strategy_module.Strategy(symbol='__rl_features__')
    signals = strat.detect_signals(df)
    out = {}
    for name in strategy_module.STRATEGY_NAMES:
        sig = signals.get(name) if signals else None
        if sig is None:
            out[f'{name}_buy']  = np.zeros(len(df), dtype=np.float32)
            out[f'{name}_sell'] = np.zeros(len(df), dtype=np.float32)
        else:
            out[f'{name}_buy']  = sig['Buy_Signal'].fill_null(False).cast(pl.Float32).to_numpy()
            out[f'{name}_sell'] = sig['Sell_Signal'].fill_null(False).cast(pl.Float32).to_numpy()
    return pl.DataFrame(out)


class Normalizer:
    """
    Per-column z-score, fit once and frozen.

    Fit on the training split only and reused verbatim on val/test and at live
    inference — fitting on the full series would leak future distributional
    information (mean/variance of returns the model hasn't "seen" yet) into
    every training example, the same class of leak the old train_test_split
    shuffle produced in server/training.py.
    """
    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.mean = mean
        self.std = std

    def fit(self, x: np.ndarray) -> 'Normalizer':
        self.mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        self.std = np.where(std < 1e-6, 1.0, std)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / self.std
        return np.clip(np.nan_to_num(z, nan=0.0, posinf=8.0, neginf=-8.0), -8.0, 8.0)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def to_dict(self) -> dict:
        return {'mean': self.mean.tolist(), 'std': self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> 'Normalizer':
        return cls(mean=np.array(d['mean'], dtype=np.float32),
                   std=np.array(d['std'], dtype=np.float32))


def build_raw_features(df: pl.DataFrame, include_rule_signals: bool = True) -> dict:
    """
    Compute the full, un-normalised feature matrix for one symbol's frame.

    Returns:
        {
            'feature_names': [...],
            'features': np.ndarray (T, F)  — raw, may contain the warm-up NaNs
                                              indicator windows produce,
            'close': np.ndarray (T,),
            'open':  np.ndarray (T,),
            'dates': np.ndarray (T,) of datetime64,
            'valid_from': int  — first row with a complete feature vector,
        }
    """
    df = derive_base_features(df)
    names = list(BASE_FEATURE_COLUMNS)
    cols = [df[c].to_numpy().astype(np.float32) for c in BASE_FEATURE_COLUMNS]

    if include_rule_signals:
        rules = rule_signal_frame(df)
        names += RULE_SIGNAL_COLUMNS
        cols += [rules[c].to_numpy().astype(np.float32) for c in RULE_SIGNAL_COLUMNS]

    features = np.stack(cols, axis=1)
    valid_mask = ~np.isnan(features[:, :len(BASE_FEATURE_COLUMNS)]).any(axis=1)
    valid_from = int(np.argmax(valid_mask)) if valid_mask.any() else len(df)

    return {
        'feature_names': names,
        'features': features,
        'close': df['Close'].to_numpy().astype(np.float64),
        'open':  df['Open'].to_numpy().astype(np.float64),
        'dates': df['Datetime'].to_numpy(),
        'valid_from': valid_from,
    }
