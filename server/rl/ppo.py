"""
Proximal Policy Optimization — rollout collection, GAE, and the clipped update.

Training runs on CPU across a handful of TradingEnv instances (one per training
symbol), round-robin, rather than a vectorized batch of environments. That
keeps the implementation simple; it costs wall-clock time, not correctness —
see train.py for the actual budget this repo runs by default.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))    # server/rl/ — sibling modules

import numpy as np
import torch
import torch.nn.functional as F

from config import PPOConfig
from env import TradingEnv
from model import ActorCritic


class RolloutBatch:
    __slots__ = ('obs', 'actions', 'logprobs', 'values', 'rewards', 'dones')

    def __init__(self, obs, actions, logprobs, values, rewards, dones):
        self.obs = obs
        self.actions = actions
        self.logprobs = logprobs
        self.values = values
        self.rewards = rewards
        self.dones = dones

    def __len__(self):
        return len(self.rewards)


def collect_rollout(model: ActorCritic, envs: list[TradingEnv], min_steps: int,
                    device: str, rng: np.random.Generator) -> RolloutBatch:
    """
    Collect at least `min_steps` transitions, always finishing whichever
    episode is in progress.

    Every episode is treated as terminal for GAE purposes (each TradingEnv
    episode has a fixed length), which is the standard fixed-horizon
    simplification — it avoids having to bootstrap a value estimate mid-series,
    at the cost of a small amount of variance at episode boundaries.
    """
    obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], []
    collected = 0

    model.eval()
    while collected < min_steps:
        env = envs[int(rng.integers(len(envs)))]
        obs = env.reset()
        done = False
        while not done:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            action, logprob, value, _ = model.act(obs_t, deterministic=False)
            a = int(action.item())

            next_obs, reward, done, _info = env.step(a)

            obs_buf.append(obs)
            act_buf.append(a)
            logp_buf.append(float(logprob.item()))
            val_buf.append(float(value.item()))
            rew_buf.append(reward)
            done_buf.append(done)

            obs = next_obs
            collected += 1

    return RolloutBatch(
        obs=np.stack(obs_buf).astype(np.float32),
        actions=np.array(act_buf, dtype=np.int64),
        logprobs=np.array(logp_buf, dtype=np.float32),
        values=np.array(val_buf, dtype=np.float32),
        rewards=np.array(rew_buf, dtype=np.float32),
        dones=np.array(done_buf, dtype=bool),
    )


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
                gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation over a flat buffer of concatenated episodes."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = 0.0 if dones[t] else values[t + 1]
        next_nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def ppo_update(model: ActorCritic, optimizer: torch.optim.Optimizer,
               batch: RolloutBatch, cfg: PPOConfig, device: str,
               rng: np.random.Generator) -> dict:
    """One PPO update: several epochs of clipped-surrogate minibatch SGD over the batch."""
    advantages, returns = compute_gae(batch.rewards, batch.values, batch.dones,
                                      cfg.gamma, cfg.gae_lambda)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    obs_t     = torch.from_numpy(batch.obs).float().to(device)
    actions_t = torch.from_numpy(batch.actions).long().to(device)
    old_logp_t = torch.from_numpy(batch.logprobs).float().to(device)
    adv_t     = torch.from_numpy(advantages).float().to(device)
    ret_t     = torch.from_numpy(returns).float().to(device)

    n = len(batch)
    stats = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'clip_frac': 0.0, 'n_updates': 0}

    model.train()
    for _epoch in range(cfg.epochs_per_update):
        idx = rng.permutation(n)
        for start in range(0, n, cfg.minibatch_size):
            mb = idx[start:start + cfg.minibatch_size]
            if len(mb) < 2:
                continue
            mb_t = torch.from_numpy(mb).long().to(device)

            logp, entropy, value = model.evaluate_actions(obs_t[mb_t], actions_t[mb_t])
            ratio = torch.exp(logp - old_logp_t[mb_t])

            surr1 = ratio * adv_t[mb_t]
            surr2 = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * adv_t[mb_t]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value, ret_t[mb_t])
            entropy_loss = -entropy.mean()

            loss = policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                clip_frac = ((ratio - 1.0).abs() > cfg.clip_ratio).float().mean().item()

            stats['policy_loss']  += policy_loss.item()
            stats['value_loss']   += value_loss.item()
            stats['entropy']      += entropy.mean().item()
            stats['clip_frac']    += clip_frac
            stats['n_updates']    += 1

    n_upd = max(stats['n_updates'], 1)
    return {
        'policy_loss': stats['policy_loss'] / n_upd,
        'value_loss':  stats['value_loss'] / n_upd,
        'entropy':     stats['entropy'] / n_upd,
        'clip_frac':   stats['clip_frac'] / n_upd,
        'mean_reward': float(batch.rewards.mean()),
    }
