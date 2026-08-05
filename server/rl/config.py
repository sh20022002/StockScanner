"""Configuration for the RL trading agent."""
from dataclasses import dataclass, field


@dataclass
class FeatureConfig:
    """What goes into the policy's observation at each bar."""
    window: int = 32          # trailing bars per observation — the transformer's context length
    include_rule_signals: bool = True   # append strategy.py's 12 buy/sell flags as extra channels


@dataclass
class EnvConfig:
    """
    Trading environment mechanics.

    Deliberately mirrors strategy.backtest_strategy: next-bar-open fills, both-side
    commission and slippage, no leverage. Sharing the cost model is what makes the
    RL backtest and the rule-based backtest comparable rather than two different
    fantasies of how a fill works.
    """
    commission: float = 0.0005          # matches strategy.DEFAULT_COMMISSION
    slippage: float = 0.0005            # matches strategy.DEFAULT_SLIPPAGE
    reward_scale: float = 100.0         # log-return reward, scaled for gradient magnitude
    episode_length: int = 128           # bars per training episode (0 = whole split)


@dataclass
class ModelConfig:
    """Time-based Transformer encoder + actor-critic heads."""
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    n_actions: int = 3          # 0 short, 1 flat, 2 long


@dataclass
class PPOConfig:
    """
    PPO hyperparameters.

    entropy_coef / entropy_coef_final, lr / lr_final_frac, and
    cost_warmup_frac all define linear schedules over total_updates rather
    than fixed values — see train.py. Fixed values let this collapse onto a
    degenerate "always flat" policy: a flat position costs nothing and
    returns exactly zero every step, so under a not-yet-informative policy
    it looks like the safest action available, and once entropy has decayed
    the policy stops exploring its way out. Observed empirically here —
    entropy fell from ~0.98 to ~0.29 within 25 updates and validation
    excess-ROI got monotonically worse across evals as more symbols settled
    into zero trades. The schedules below target that failure mode directly:
    entropy starts higher and decays slower, and — the bigger lever —
    training starts commission/slippage-free so the policy can learn
    genuine directional edge before cost makes "do nothing" the locally
    rational move, then ramps cost up to the real rate it will face at
    inference.
    """
    total_updates: int = 60
    rollout_steps: int = 512          # env steps collected per update, across symbols
    epochs_per_update: int = 4
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.02         # starting entropy bonus
    entropy_coef_final: float = 0.003  # annealed linearly to this by the last update
    lr: float = 3e-4
    lr_final_frac: float = 0.1         # lr anneals linearly to lr * this by the last update
    cost_warmup_frac: float = 0.3      # fraction of total_updates ramping cost 0 -> full
    max_grad_norm: float = 0.5
    eval_every: int = 10               # updates between validation evals
    seed: int = 42


@dataclass
class TrainConfig:
    symbols: list = field(default_factory=lambda: [
        'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'JPM', 'XOM',
        'JNJ', 'WMT', 'PG', 'HD', 'V', 'MA', 'KO',
    ])
    timeframe: str = '1d'
    period: str = '2y'
    train_fraction: float = 0.7
    val_fraction: float = 0.15         # remainder is test/holdout
    features: FeatureConfig = field(default_factory=FeatureConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
