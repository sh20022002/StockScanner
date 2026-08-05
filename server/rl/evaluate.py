"""
Deterministic evaluation of a trained policy, plus a rule-based comparison
baseline pulled straight from strategy.py so the two are read side by side.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # server/rl/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # server/

import math

import numpy as np
import torch

import strategy as strategy_module
from config import EnvConfig
from env import TradingEnv
from model import ActorCritic


def rollout_metrics(model: ActorCritic, features: np.ndarray, close: np.ndarray,
                    open_: np.ndarray, window: int, env_cfg: EnvConfig,
                    device: str = 'cpu') -> dict:
    """
    Run the policy deterministically (argmax) over one full series and report
    the same family of metrics strategy.backtest_strategy reports, so the two
    read side by side: roi, benchmark_roi, excess_roi, max_drawdown, trades.

    Adds Sharpe/Sortino, which strategy.py does not compute, and a bar-level
    win_rate — explicitly NOT the same statistic as strategy.py's trade-level
    win_rate (fraction of closed trades that were profitable). Comparing the
    two numbers directly would be comparing different things that happen to
    share a name.
    """
    env = TradingEnv(features, close, open_, window=window,
                     commission=env_cfg.commission, slippage=env_cfg.slippage,
                     reward_scale=env_cfg.reward_scale, episode_length=0)
    obs = env.reset(start=env._min_start)
    done = False
    while not done:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        action, _logp, _val, _probs = model.act(obs_t, deterministic=True)
        obs, _reward, done, _info = env.step(int(action.item()))

    step_returns = np.array(env.step_returns)
    equity_curve = np.array(env.equity_curve)
    positions = np.array(env.positions)

    roi = (equity_curve[-1] - 1) * 100
    bench_log_ret = math.log(open_[env._episode_end + 1] / open_[env._min_start + 1])
    benchmark_roi = (math.exp(bench_log_ret) - 1) * 100
    excess_roi = roi - benchmark_roi

    exposed = positions != 0
    bar_win_rate = float((step_returns[exposed] > 0).mean() * 100) if exposed.any() else 0.0

    ann = math.sqrt(252)
    mean_r, std_r = step_returns.mean(), step_returns.std()
    sharpe = float(mean_r / std_r * ann) if std_r > 1e-12 else 0.0
    downside = step_returns[step_returns < 0]
    down_std = downside.std() if len(downside) > 1 else 0.0
    sortino = float(mean_r / down_std * ann) if down_std > 1e-12 else 0.0

    return {
        'roi':            round(float(roi), 2),
        'benchmark_roi':  round(float(benchmark_roi), 2),
        'excess_roi':     round(float(excess_roi), 2),
        'sharpe':         round(sharpe, 3),
        'sortino':        round(sortino, 3),
        'max_drawdown':   round(float(env.max_drawdown) * 100, 2),
        'trades':         env.trades,
        'bar_win_rate':   round(bar_win_rate, 2),
        'n_steps':        len(step_returns),
        'equity_curve':   equity_curve.tolist(),
    }


def evaluate_dataset(model: ActorCritic, per_symbol: dict, split: str, window: int,
                     env_cfg: EnvConfig, device: str = 'cpu',
                     include_rule_baseline: bool = False,
                     train_fraction: float = 0.7, val_fraction: float = 0.15) -> dict:
    """
    Run rollout_metrics for every symbol's given split ('train'/'val'/'test')
    and average the results. Symbols too short for that split (env raises on
    construction) are skipped and reported, never silently dropped from the
    denominator without a trace.

    include_rule_baseline is only meaningful for split == 'test': it
    reconstructs the full (train+val+test) series and asks strategy.py to
    split it at (train_fraction + val_fraction), so its own out-of-sample
    tail lines up with the RL test split on materially the same window.
    """
    per_symbol_results = {}
    skipped = []
    want_baseline = include_rule_baseline and split == 'test'

    for symbol, data in per_symbol.items():
        feats, close, open_ = data['splits'][split]
        try:
            m = rollout_metrics(model, feats, close, open_, window, env_cfg, device)
        except ValueError:
            skipped.append(symbol)
            continue

        if want_baseline:
            full_df = (data['df_splits']['train']
                      .vstack(data['df_splits']['val'])
                      .vstack(data['df_splits']['test']))
            m['rule_baseline'] = rule_based_baseline(symbol, full_df,
                                                      train_fraction + val_fraction)

        per_symbol_results[symbol] = m

    if not per_symbol_results:
        return {'per_symbol': {}, 'skipped': skipped, 'avg_roi': 0.0,
               'avg_excess_roi': 0.0, 'avg_sharpe': 0.0, 'n_symbols': 0}

    excess = [m['excess_roi'] for m in per_symbol_results.values()]
    roi    = [m['roi'] for m in per_symbol_results.values()]
    sharpe = [m['sharpe'] for m in per_symbol_results.values()]

    return {
        'per_symbol':     per_symbol_results,
        'skipped':        skipped,
        'n_symbols':      len(per_symbol_results),
        'avg_roi':        round(float(np.mean(roi)), 2),
        'avg_excess_roi': round(float(np.mean(excess)), 2),
        'avg_sharpe':     round(float(np.mean(sharpe)), 3),
        'beat_benchmark': sum(1 for e in excess if e > 0),
    }


def rule_based_baseline(symbol: str, df, train_fraction: float) -> dict | None:
    """
    Run strategy.py's own strategy set on the same series, split at the same
    train_fraction boundary used for the RL split.

    This is an approximation, not a pixel-identical slice match: strategy.py
    computes its own train/test boundary internally from the frame it is
    given, so the two holdout windows line up on the same fraction of the same
    history without being defined by identical index arithmetic. Close enough
    to compare magnitudes; not close enough to subtract the two ROI numbers
    and call the difference precise.
    """
    strat = strategy_module.Strategy(symbol=symbol)
    best, results = strat.evaluate_strategies(df, train_fraction=train_fraction)
    if not results:
        return None
    chosen = next((r for r in results if r['strategy_func'] == best), results[0])
    rm = chosen['risk_metrics']
    return {
        'strategy':      chosen['strategy_func'],
        'roi':           rm.get('roi'),
        'benchmark_roi': rm.get('benchmark_roi'),
        'excess_roi':    rm.get('excess_roi'),
        'max_drawdown':  rm.get('max_drawdown'),
        'win_rate':      rm.get('win_rate'),
        'trades':        rm.get('trades'),
    }
