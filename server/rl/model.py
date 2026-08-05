"""
Time-based Transformer encoder + actor-critic heads.

"Time-based" here means the standard sinusoidal positional encoding from
Vaswani et al. — each position in the window gets a fixed sin/cos fingerprint
so attention can tell bar t-31 from bar t-1 (self-attention alone is
permutation-invariant and would otherwise see a bag of bars, not a sequence).
The encoder attends over a trailing window of bars and the representation of
the most recent bar (last token, after it has attended to the whole window) is
what feeds the policy and value heads — that token has context on the entire
window but is anchored at "now", which is what a live trading decision needs.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))       # server/rl/ — sibling modules

import torch
import torch.nn as nn
from torch.distributions import Categorical

from config import ModelConfig


class TimeEncoding(nn.Module):
    """Fixed sinusoidal positional encoding — no learned parameters."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TimeSeriesTransformer(nn.Module):
    """Encodes a (batch, window, n_features) window into a (batch, d_model) embedding."""

    def __init__(self, n_features: int, cfg: ModelConfig):
        super().__init__()
        self.input_proj = nn.Linear(n_features, cfg.d_model)
        self.time_encoding = TimeEncoding(cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout, batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, window, n_features) -> (batch, d_model), the last bar's embedding."""
        h = self.input_proj(x)
        h = self.time_encoding(h)
        h = self.encoder(h)
        return self.norm(h[:, -1, :])       # the most-recent-bar token


class ActorCritic(nn.Module):
    def __init__(self, n_features: int, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_features = n_features
        self.backbone = TimeSeriesTransformer(n_features, cfg)
        self.policy_head = nn.Linear(cfg.d_model, cfg.n_actions)
        self.value_head = nn.Linear(cfg.d_model, 1)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """
        obs: (batch, window, n_features).
        Returns (action, log_prob, value, probs) as tensors of shape (batch,)
        (probs is (batch, n_actions), kept for confidence reporting at inference).
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value, dist.probs

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """For PPO updates: log_prob/entropy/value of given actions under the current policy."""
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


def build_model(n_features: int, cfg: ModelConfig) -> ActorCritic:
    return ActorCritic(n_features, cfg)


def save_checkpoint(path, model: ActorCritic, normalizer, feature_names: list[str],
                    model_cfg: ModelConfig, feature_cfg, env_cfg, extra: dict | None = None):
    """
    Bundle everything needed to reproduce inference: weights, config, and the
    frozen feature normalizer. Without the normalizer a checkpoint is useless —
    it was fit on a specific training distribution and inference must match it.
    """
    torch.save({
        'state_dict':     model.state_dict(),
        'n_features':     model.n_features,
        'model_cfg':      vars(model_cfg),
        'feature_cfg':    vars(feature_cfg),
        'env_cfg':        vars(env_cfg),
        'feature_names':  feature_names,
        'normalizer':     normalizer.to_dict(),
        'extra':          extra or {},
    }, path)


def load_checkpoint(path, map_location='cpu'):
    """Returns (model, normalizer, feature_names, record) — record is the raw dict for metadata."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from features import Normalizer
    record = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ModelConfig(**record['model_cfg'])
    model = build_model(record['n_features'], cfg)
    model.load_state_dict(record['state_dict'])
    model.eval()
    normalizer = Normalizer.from_dict(record['normalizer'])
    return model, normalizer, record['feature_names'], record
