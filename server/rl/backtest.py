"""
Out-of-sample backtest for a trained RL checkpoint, reported side by side with
strategy.py's rule-based scanner on the same symbols.

    python server/rl/backtest.py
    python server/rl/backtest.py --symbols AAPL,MSFT,TSLA,DIS
    python server/rl/backtest.py --checkpoint server/rl/runs/run-.../best.pt

By default this evaluates on the TEST split only — the slice neither training
nor checkpoint selection ever looked at. Passing --symbols with tickers absent
from training is a legitimate and arguably more honest thing to do: it checks
whether the policy learned something that transfers, rather than something
specific to the training universe.
"""
import argparse
import json
import os
import sys

try:                                    # the default Windows console codepage
    sys.stdout.reconfigure(encoding='utf-8')   # can't encode the report's em-dashes
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # server/rl/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # server/

# evaluate (which imports torch) must load before data (which imports pandas via
# scraping) — see the import-order comment in features.py.
import evaluate as evaluate_module
import data as data_module
from config import FeatureConfig, TrainConfig
from model import load_checkpoint
from train import BEST_CHECKPOINT


def run_backtest(checkpoint_path: str, symbols: list[str] | None, device: str = 'cpu') -> dict:
    model, normalizer, feature_names, record = load_checkpoint(checkpoint_path, map_location=device)
    extra = record.get('extra', {})

    cfg = TrainConfig()
    cfg.features = FeatureConfig(**record['feature_cfg'])
    cfg.env.commission = record['env_cfg']['commission']
    cfg.env.slippage = record['env_cfg']['slippage']
    cfg.env.reward_scale = record['env_cfg']['reward_scale']
    cfg.timeframe = extra.get('timeframe', cfg.timeframe)
    cfg.period = extra.get('period', cfg.period)
    cfg.train_fraction = extra.get('train_fraction', cfg.train_fraction)
    cfg.val_fraction = extra.get('val_fraction', cfg.val_fraction)

    eval_symbols = symbols or extra.get('symbols_used') or cfg.symbols
    unseen = [s for s in eval_symbols if s not in (extra.get('symbols_used') or [])]

    print(f'Checkpoint: {checkpoint_path}')
    print(f"Trained on {len(extra.get('symbols_used', []))} symbols at update "
          f"{extra.get('update', '?')} (val excess ROI {extra.get('val_avg_excess_roi', '?')}%)")
    print(f'Backtesting {len(eval_symbols)} symbols ({cfg.timeframe}, {cfg.period}), '
          f'test split only (untouched by training or checkpoint selection).')
    if unseen:
        print(f"  {len(unseen)} of these were NOT in the training set: {', '.join(unseen)}")

    # Reuse the checkpoint's own training-time normalizer rather than fitting a
    # fresh one on this backtest's symbol set — those are two different
    # distributions, and only the checkpoint's is what the model was trained on.
    dataset = data_module.prepare_dataset(
        eval_symbols, cfg.timeframe, cfg.period, cfg.features,
        cfg.train_fraction, cfg.val_fraction, normalizer=normalizer)
    if dataset is None:
        raise RuntimeError('No symbol had enough history to backtest.')
    if dataset['skipped']:
        print(f"  Skipped (too little history): {', '.join(dataset['skipped'])}")

    result = evaluate_module.evaluate_dataset(
        model, dataset['per_symbol'], 'test', cfg.features.window,
        cfg.env, device, include_rule_baseline=True,
        train_fraction=cfg.train_fraction, val_fraction=cfg.val_fraction)

    _print_report(result, unseen)
    return result


def _print_report(result: dict, unseen: list[str]):
    rows = sorted(result['per_symbol'].items(), key=lambda kv: -kv[1]['excess_roi'])

    print(f"\n{'symbol':<8} {'roi':>8} {'bench':>8} {'excess':>8} {'sharpe':>7} "
          f"{'trades':>7} | {'rule best':<20} {'rule excess':>12}")
    print('-' * 92)
    for symbol, m in rows:
        rb = m.get('rule_baseline') or {}
        flag = ' *' if symbol in unseen else ''
        rb_excess = rb.get('excess_roi')
        rb_excess_str = f'{rb_excess:+.1f}%' if rb_excess is not None else '—'
        print(f"{symbol + flag:<8} {m['roi']:>+7.1f}% {m['benchmark_roi']:>+7.1f}% "
              f"{m['excess_roi']:>+7.1f}% {m['sharpe']:>7.2f} {m['trades']:>7} | "
              f"{rb.get('strategy', '—'):<20} {rb_excess_str:>12}")

    n = result['n_symbols']
    print('-' * 92)
    print(f"RL agent   — avg ROI {result['avg_roi']:+.1f}%  avg excess {result['avg_excess_roi']:+.1f}%  "
          f"avg Sharpe {result['avg_sharpe']:.2f}  beat buy&hold {result['beat_benchmark']}/{n}")

    rule_excess = [m['rule_baseline']['excess_roi'] for m in result['per_symbol'].values()
                  if m.get('rule_baseline') and m['rule_baseline'].get('excess_roi') is not None]
    if rule_excess:
        rule_beat = sum(1 for e in rule_excess if e > 0)
        print(f"Rule-based — avg excess {sum(rule_excess)/len(rule_excess):+.1f}%  "
              f"beat buy&hold {rule_beat}/{len(rule_excess)}")
    if unseen:
        print('(*  not in the training set — a generalization check, not curve-fit)')


def main():
    parser = argparse.ArgumentParser(description='Backtest a trained RL checkpoint.')
    parser.add_argument('--checkpoint', default=BEST_CHECKPOINT)
    parser.add_argument('--symbols', default=None, help='Comma-separated. Default: the training set.')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--json-out', default=None, help='Optional path to dump the full report as JSON.')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f'No checkpoint at {args.checkpoint}. Run server/rl/train.py first.')
        sys.exit(1)

    symbols = [s.strip().upper() for s in args.symbols.split(',')] if args.symbols else None
    result = run_backtest(args.checkpoint, symbols, args.device)

    if args.json_out:
        serializable = {k: v for k, v in result.items() if k != 'per_symbol'}
        serializable['per_symbol'] = {
            s: {k: v for k, v in m.items() if k != 'equity_curve'}
            for s, m in result['per_symbol'].items()
        }
        with open(args.json_out, 'w') as f:
            json.dump(serializable, f, indent=2, default=float)
        print(f'\nWrote {args.json_out}')


if __name__ == '__main__':
    main()
