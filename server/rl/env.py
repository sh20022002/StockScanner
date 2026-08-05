"""
Single-asset trading environment for the RL agent.

Deliberately mirrors strategy.backtest_strategy's execution model so the RL
backtest and the rule-based backtest are measuring the same thing:

  * A decision made from bar t's observation is filled at bar t+1's OPEN, never
    at a price the agent could not yet have known.
  * Commission and slippage are charged, combined, on the notional traded —
    the same `commission + slippage` fraction strategy.py uses, applied per
    unit of position changed (flat->long trades 1 unit, long->short trades 2).
  * No lookahead: the observation at step t is a window of features computed
    from bars up to and including t.

Actions are discrete: 0 = short, 1 = flat, 2 = long. No leverage, no partial
sizing — matching strategy.py's binary long/short/flat stance keeps the RL and
rule-based results comparable rather than two different fantasies of exposure.
"""
import math

import numpy as np


class TradingEnv:
    def __init__(self, features: np.ndarray, close: np.ndarray, open_: np.ndarray,
                window: int, commission: float, slippage: float,
                reward_scale: float = 100.0, episode_length: int = 0,
                rng: np.random.Generator | None = None):
        if len(features) != len(close) or len(close) != len(open_):
            raise ValueError('features/close/open must be the same length')

        self.features = features
        self.close = close
        self.open = open_
        self.window = window
        self.cost_rate = commission + slippage
        self.reward_scale = reward_scale
        self.episode_length = episode_length
        self.rng = rng or np.random.default_rng()

        n = len(close)
        self._min_start = window - 1
        self._max_start = n - 3           # step() reads open[t+1] and open[t+2]
        if self._max_start < self._min_start:
            raise ValueError(f'series too short for window={window}: only {n} bars')

        self.t = self._min_start
        self.pos = 0
        self.equity = 1.0
        self._episode_end = self._max_start

    @property
    def n_features(self) -> int:
        return self.features.shape[1]

    def reset(self, start: int | None = None) -> np.ndarray:
        if start is None:
            if self.episode_length and self.episode_length > 0:
                latest = self._max_start - self.episode_length
                start = (self._min_start if latest <= self._min_start
                         else int(self.rng.integers(self._min_start, latest + 1)))
            else:
                start = self._min_start
        self.t = start
        self.pos = 0
        self.equity = 1.0
        self.peak_equity = 1.0
        self.max_drawdown = 0.0
        self.trades = 0
        self.equity_curve = [1.0]
        self.positions = []
        self.step_returns = []
        self._episode_end = (min(self._max_start, start + self.episode_length)
                             if self.episode_length else self._max_start)
        return self._obs()

    def _obs(self) -> np.ndarray:
        start = self.t - self.window + 1
        return self.features[start:self.t + 1]

    def step(self, action: int):
        if action not in (0, 1, 2):
            raise ValueError(f'action must be 0/1/2, got {action}')
        target_pos = action - 1
        delta = abs(target_pos - self.pos)
        cost = self.cost_rate * delta
        if delta:
            self.trades += 1

        o1, o2 = self.open[self.t + 1], self.open[self.t + 2]
        price_log_ret = math.log(o2 / o1) if o1 > 0 and o2 > 0 and np.isfinite(o1) and np.isfinite(o2) else 0.0
        net_log_ret = target_pos * price_log_ret - cost

        self.equity *= math.exp(net_log_ret)
        self.peak_equity = max(self.peak_equity, self.equity)
        if self.peak_equity > 0:
            dd = (self.peak_equity - self.equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, dd)

        self.equity_curve.append(self.equity)
        self.positions.append(target_pos)
        self.step_returns.append(net_log_ret)

        self.pos = target_pos
        self.t += 1
        done = self.t >= self._episode_end

        info = {'equity': self.equity, 'log_ret': net_log_ret, 'cost': cost,
                'max_drawdown': self.max_drawdown, 'trades': self.trades}
        obs = None if done else self._obs()
        return obs, net_log_ret * self.reward_scale, done, info
