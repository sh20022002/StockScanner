"""
Train the RL trading agent with PPO.

    python server/rl/train.py
    python server/rl/train.py --symbols AAPL,MSFT,NVDA --updates 40
    python server/rl/train.py --run-name my-experiment --device cpu

Downloads data once, splits each symbol chronologically into train/val/test,
trains on the train split across all symbols, periodically evaluates the
(deterministic) policy on the val split, and checkpoints whichever version had
the best mean out-of-sample excess return over buy-and-hold on val.

That checkpoint — server/rl/checkpoints/best.pt — is what evaluate.py,
backtest.py and live.py load. Nothing here ever looks at the test split; that
stays untouched until backtest.py runs, which is the whole point of a holdout.
"""
import argparse
import json
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8')   # default Windows console codepage can't
except Exception:                              # encode some of the log line markers
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # server/rl/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # server/

import numpy as np
import torch

import data as data_module
import evaluate as evaluate_module
from config import TrainConfig
from env import TradingEnv
from model import ActorCritic, save_checkpoint
from ppo import collect_rollout, ppo_update

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
BEST_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, 'best.pt')


def build_train_envs(dataset: dict, cfg: TrainConfig, seed: int) -> list[TradingEnv]:
    envs = []
    for i, (symbol, d) in enumerate(sorted(dataset['per_symbol'].items())):
        feats, close, open_ = d['splits']['train']
        envs.append(TradingEnv(
            feats, close, open_, window=cfg.features.window,
            commission=cfg.env.commission, slippage=cfg.env.slippage,
            reward_scale=cfg.env.reward_scale, episode_length=cfg.env.episode_length,
            rng=np.random.default_rng(seed + i),
        ))
    return envs


def train(cfg: TrainConfig, run_name: str, device: str, log=print) -> str:
    torch.manual_seed(cfg.ppo.seed)
    rng = np.random.default_rng(cfg.ppo.seed)

    log(f'Downloading + preparing {len(cfg.symbols)} symbols ({cfg.timeframe}, {cfg.period})...')
    dataset = data_module.prepare_dataset(
        cfg.symbols, cfg.timeframe, cfg.period, cfg.features,
        cfg.train_fraction, cfg.val_fraction, episode_length=cfg.env.episode_length)
    if dataset is None:
        raise RuntimeError('No symbol had enough history to build train/val/test splits.')
    if dataset['skipped']:
        log(f"Skipped (too little history): {', '.join(dataset['skipped'])}")
    symbols_used = sorted(dataset['per_symbol'])
    log(f'Training on {len(symbols_used)} symbols: {", ".join(symbols_used)}')

    n_features = len(dataset['feature_names'])
    model = ActorCritic(n_features, cfg.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.ppo.lr)

    train_envs = build_train_envs(dataset, cfg, cfg.ppo.seed)

    run_dir = os.path.join(RUNS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    with open(os.path.join(run_dir, 'manifest.json'), 'w') as f:
        json.dump({
            'symbols_used': symbols_used, 'symbols_skipped': dataset['skipped'],
            'feature_names': dataset['feature_names'], 'n_features': n_features,
            'config': {
                'timeframe': cfg.timeframe, 'period': cfg.period,
                'train_fraction': cfg.train_fraction, 'val_fraction': cfg.val_fraction,
                'features': vars(cfg.features), 'env': vars(cfg.env),
                'model': vars(cfg.model), 'ppo': vars(cfg.ppo),
            },
        }, f, indent=2)

    history = []
    best_val_excess = float('-inf')
    t0 = time.time()

    for update in range(1, cfg.ppo.total_updates + 1):
        batch = collect_rollout(model, train_envs, cfg.ppo.rollout_steps, device, rng)
        stats = ppo_update(model, optimizer, batch, cfg.ppo, device, rng)
        stats['update'] = update
        stats['elapsed_s'] = round(time.time() - t0, 1)

        line = (f"[{update:>3}/{cfg.ppo.total_updates}] "
                f"reward={stats['mean_reward']:+.4f} "
                f"policy_loss={stats['policy_loss']:+.4f} "
                f"value_loss={stats['value_loss']:.4f} "
                f"entropy={stats['entropy']:.3f} "
                f"clip%={stats['clip_frac']*100:.0f}")

        if update % cfg.ppo.eval_every == 0 or update == cfg.ppo.total_updates:
            val = evaluate_module.evaluate_dataset(
                model, dataset['per_symbol'], 'val', cfg.features.window,
                cfg.env, device)
            stats['val'] = val
            line += (f" | val: roi={val['avg_roi']:+.1f}% excess={val['avg_excess_roi']:+.1f}% "
                     f"sharpe={val['avg_sharpe']:.2f} beat={val['beat_benchmark']}/{val['n_symbols']}")

            if val['avg_excess_roi'] > best_val_excess:
                best_val_excess = val['avg_excess_roi']
                extra = {
                    'update': update, 'val_avg_excess_roi': val['avg_excess_roi'],
                    'run_name': run_name, 'symbols_used': symbols_used,
                    'timeframe': cfg.timeframe, 'period': cfg.period,
                    'train_fraction': cfg.train_fraction, 'val_fraction': cfg.val_fraction,
                }
                save_checkpoint(os.path.join(run_dir, 'best.pt'), model,
                               dataset['normalizer'], dataset['feature_names'],
                               cfg.model, cfg.features, cfg.env, extra=extra)
                save_checkpoint(BEST_CHECKPOINT, model, dataset['normalizer'],
                               dataset['feature_names'], cfg.model, cfg.features, cfg.env,
                               extra=extra)
                line += '  <- new best, checkpointed'

        log(line)
        history.append(stats)

    save_checkpoint(os.path.join(run_dir, 'last.pt'), model, dataset['normalizer'],
                    dataset['feature_names'], cfg.model, cfg.features, cfg.env,
                    extra={'update': cfg.ppo.total_updates})
    with open(os.path.join(run_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2, default=float)

    log(f'\nDone in {time.time() - t0:.1f}s. Best val avg excess ROI: {best_val_excess:+.2f}%')
    log(f'Checkpoint: {BEST_CHECKPOINT}')
    return run_dir


def main():
    parser = argparse.ArgumentParser(description='Train the RL trading agent.')
    parser.add_argument('--symbols', default=None, help='Comma-separated symbols.')
    parser.add_argument('--timeframe', default=None)
    parser.add_argument('--period', default=None)
    parser.add_argument('--updates', type=int, default=None)
    parser.add_argument('--rollout-steps', type=int, default=None)
    parser.add_argument('--window', type=int, default=None)
    parser.add_argument('--no-rule-signals', action='store_true',
                        help='Drop the 12 rule-based strategy flags from the observation.')
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.symbols:
        cfg.symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    if args.timeframe:
        cfg.timeframe = args.timeframe
    if args.period:
        cfg.period = args.period
    if args.updates:
        cfg.ppo.total_updates = args.updates
    if args.rollout_steps:
        cfg.ppo.rollout_steps = args.rollout_steps
    if args.window:
        cfg.features.window = args.window
    if args.no_rule_signals:
        cfg.features.include_rule_signals = False
    if args.seed is not None:
        cfg.ppo.seed = args.seed

    run_name = args.run_name or time.strftime('run-%Y%m%d-%H%M%S')
    train(cfg, run_name, args.device)


if __name__ == '__main__':
    main()
