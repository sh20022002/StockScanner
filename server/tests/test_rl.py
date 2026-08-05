"""
Tests for the RL trading agent (server/rl/).

Run with: pytest server/tests -v
"""
import math
import os
import sys

import pytest

torch = pytest.importorskip('torch')   # must import before scraping (pandas) — see below

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rl'))

# scraping pulls in pandas; on Windows this process has been observed to
# access-violate loading torch's c10.dll if pandas' compiled extensions
# initialise first. `torch = pytest.importorskip('torch')` above must run
# before this import, not after — see the same note in server/rl/features.py.
import scraping
from config import EnvConfig, FeatureConfig, ModelConfig, PPOConfig
from env import TradingEnv
from features import Normalizer, build_raw_features
from model import ActorCritic, load_checkpoint, save_checkpoint
from ppo import collect_rollout, compute_gae, ppo_update


# ---------------------------------------------------------------------------
# Synthetic data — real compute_indicators output, not a hand-rolled stand-in,
# so these tests exercise the exact columns features.py reads in production.
# ---------------------------------------------------------------------------

def make_indicator_df(n=200, trend='up', seed=0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    from datetime import datetime, timedelta
    dates = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(n)]

    if trend == 'up':
        drift = np.linspace(0, 20, n)
    elif trend == 'down':
        drift = np.linspace(0, -20, n)
    else:
        drift = np.zeros(n)
    close = 100.0 + drift + rng.standard_normal(n).cumsum() * 0.3
    close = np.maximum(close, 1.0)
    high = close * 1.01
    low = close * 0.99
    open_ = close * (1 + rng.standard_normal(n) * 0.001)
    volume = rng.integers(500_000, 1_500_000, n).astype(np.float64)

    df = pl.DataFrame({
        'Datetime': dates, 'Open': open_, 'High': high, 'Low': low,
        'Close': close, 'Volume': volume,
    })
    return scraping.compute_indicators(df)


def make_flat_df(n=60) -> pl.DataFrame:
    """Zero-movement series — useful for exact cost-accounting assertions."""
    from datetime import datetime, timedelta
    dates = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(n)]
    price = [100.0] * n
    df = pl.DataFrame({
        'Datetime': dates, 'Open': price, 'High': price, 'Low': price,
        'Close': price, 'Volume': [1_000_000.0] * n,
    })
    return scraping.compute_indicators(df)


TINY_MODEL_CFG = ModelConfig(d_model=8, n_heads=2, n_layers=1, d_ff=16, dropout=0.0)


# ---------------------------------------------------------------------------
# features.py
# ---------------------------------------------------------------------------

class TestFeatures:
    def test_shapes_and_names(self):
        df = make_indicator_df()
        raw = build_raw_features(df, include_rule_signals=False)
        assert raw['features'].shape == (len(df), 10)
        assert len(raw['feature_names']) == 10

    def test_rule_signal_columns_included(self):
        df = make_indicator_df()
        raw = build_raw_features(df, include_rule_signals=True)
        assert raw['features'].shape[1] == 10 + 24     # 12 strategies x buy/sell
        assert raw['features'].shape[1] == len(raw['feature_names'])

    def test_no_nan_after_valid_from(self):
        df = make_indicator_df()
        raw = build_raw_features(df, include_rule_signals=True)
        tail = raw['features'][raw['valid_from']:]
        assert not np.isnan(tail).any()

    def test_valid_from_skips_only_warmup(self):
        df = make_indicator_df()
        raw = build_raw_features(df)
        # ret_1's first-row NaN plus vol_z's rolling_std(min_periods=5) are the
        # only sources of warm-up NaN — every SMA/EMA/RSI/etc. column from
        # compute_indicators uses min_periods=1, so none of those force a skip.
        assert raw['valid_from'] <= 5


class TestNormalizer:
    def test_fit_transform_is_standardised(self):
        rng = np.random.default_rng(0)
        x = rng.normal(loc=5.0, scale=2.0, size=(500, 4)).astype(np.float32)
        norm = Normalizer().fit(x)
        z = norm.transform(x)
        assert np.abs(z.mean(axis=0)).max() < 0.05
        assert np.abs(z.std(axis=0) - 1.0).max() < 0.05

    def test_clips_extreme_values(self):
        x = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
        norm = Normalizer().fit(x)
        huge = np.array([[1e9]], dtype=np.float32)
        z = norm.transform(huge)
        assert np.all(np.abs(z) <= 8.0)

    def test_handles_zero_variance_column(self):
        x = np.ones((10, 2), dtype=np.float32)
        norm = Normalizer().fit(x)
        z = norm.transform(x)
        assert np.isfinite(z).all()

    def test_round_trip_dict(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(50, 3)).astype(np.float32)
        norm = Normalizer().fit(x)
        restored = Normalizer.from_dict(norm.to_dict())
        np.testing.assert_allclose(norm.transform(x), restored.transform(x))


# ---------------------------------------------------------------------------
# env.py
# ---------------------------------------------------------------------------

class TestTradingEnv:
    def _tiny_env(self, n=40, window=8, **kwargs):
        feats = np.random.default_rng(0).normal(size=(n, 3)).astype(np.float32)
        close = np.full(n, 100.0)
        open_ = np.full(n, 100.0)
        return TradingEnv(feats, close, open_, window=window,
                          commission=0.0005, slippage=0.0005, **kwargs)

    def test_raises_when_series_too_short(self):
        feats = np.zeros((5, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            TradingEnv(feats, np.ones(5), np.ones(5), window=8,
                      commission=0.0, slippage=0.0)

    def test_reset_obs_shape(self):
        env = self._tiny_env(window=8)
        obs = env.reset(start=env._min_start)
        assert obs.shape == (8, 3)

    def test_flat_price_flat_action_costs_nothing(self):
        env = self._tiny_env()
        env.reset(start=env._min_start)
        _obs, reward, _done, info = env.step(1)     # 1 == flat
        assert reward == pytest.approx(0.0, abs=1e-9)
        assert info['cost'] == 0.0
        assert env.equity == pytest.approx(1.0)

    def test_going_long_on_flat_price_costs_exactly_the_spread(self):
        """
        Flat->long is a delta of 1 unit; cost must be exactly commission+slippage,
        and since price never moves, reward is entirely the (negative) cost.
        """
        env = self._tiny_env()
        env.reset(start=env._min_start)
        _obs, reward, _done, info = env.step(2)      # 2 == long
        assert info['cost'] == pytest.approx(0.0005 + 0.0005)
        assert reward == pytest.approx(-info['cost'] * env.reward_scale, rel=1e-6)

    def test_short_to_long_costs_double(self):
        """long<->short is a delta of 2 units of notional traded."""
        env = self._tiny_env()
        env.reset(start=env._min_start)
        env.step(0)                                   # go short (delta 1 from flat)
        _obs, _reward, _done, info = env.step(2)       # short -> long (delta 2)
        assert info['cost'] == pytest.approx((0.0005 + 0.0005) * 2)

    def test_fill_uses_next_bar_open_not_current_close(self):
        """
        The position decided from the observation at t must be priced off
        open[t+1]->open[t+2], never off close[t] — that would be trading on a
        price the agent could not yet know, the same bug fixed in
        strategy.backtest_strategy's fill timing.
        """
        n = 10
        feats = np.zeros((n, 2), dtype=np.float32)
        close = np.full(n, 999.0)              # deliberately wrong if ever read
        open_ = np.full(n, 100.0)
        open_[3] = 50.0                        # the fill price for a decision made at t=1
        env = TradingEnv(feats, close, open_, window=2, commission=0.0, slippage=0.0)
        env.reset(start=1)
        _obs, _reward, _done, info = env.step(2)   # long, filled at open[2]->open[3]
        expected = math.log(open_[3] / open_[2])
        assert info['log_ret'] == pytest.approx(expected)

    def test_episode_length_bounds_the_rollout(self):
        env = self._tiny_env(n=100, window=8, episode_length=5)
        env.reset(start=env._min_start)
        steps = 0
        done = False
        while not done:
            _obs, _r, done, _i = env.step(1)
            steps += 1
        assert steps == 5

    def test_random_start_stays_in_bounds(self):
        env = self._tiny_env(n=100, window=8, episode_length=5,
                             rng=np.random.default_rng(3))
        for _ in range(20):
            env.reset()
            assert env._min_start <= env.t <= env._max_start

    def test_max_drawdown_tracks_losses(self):
        n = 20
        feats = np.zeros((n, 2), dtype=np.float32)
        open_ = np.array([100.0 * (0.9 ** i) for i in range(n)])   # steadily falling
        close = open_.copy()
        env = TradingEnv(feats, close, open_, window=2, commission=0.0, slippage=0.0)
        env.reset(start=env._min_start)
        done = False
        while not done:
            _obs, _r, done, info = env.step(2)     # long into a falling market
        assert info['max_drawdown'] > 0.05


# ---------------------------------------------------------------------------
# model.py
# ---------------------------------------------------------------------------

class TestModel:
    def test_forward_shapes(self):
        model = ActorCritic(n_features=5, cfg=TINY_MODEL_CFG)
        x = torch.randn(4, 8, 5)
        logits, value = model(x)
        assert logits.shape == (4, 3)
        assert value.shape == (4,)

    def test_act_deterministic_is_reproducible(self):
        model = ActorCritic(n_features=5, cfg=TINY_MODEL_CFG)
        model.eval()
        x = torch.randn(1, 8, 5)
        a1, _, _, _ = model.act(x, deterministic=True)
        a2, _, _, _ = model.act(x, deterministic=True)
        assert a1.item() == a2.item()

    def test_probs_sum_to_one(self):
        model = ActorCritic(n_features=5, cfg=TINY_MODEL_CFG)
        _a, _lp, _v, probs = model.act(torch.randn(3, 8, 5))
        assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-5)

    def test_evaluate_actions_matches_forward(self):
        model = ActorCritic(n_features=5, cfg=TINY_MODEL_CFG)
        x = torch.randn(6, 8, 5)
        actions = torch.randint(0, 3, (6,))
        logp, entropy, value = model.evaluate_actions(x, actions)
        assert logp.shape == (6,)
        assert entropy.shape == (6,)
        assert value.shape == (6,)
        assert (entropy >= 0).all()

    def test_checkpoint_round_trip(self, tmp_path):
        model = ActorCritic(n_features=4, cfg=TINY_MODEL_CFG)
        normalizer = Normalizer().fit(np.random.default_rng(0).normal(size=(20, 4)).astype(np.float32))
        path = tmp_path / 'ckpt.pt'

        save_checkpoint(str(path), model, normalizer, ['a', 'b', 'c', 'd'],
                        TINY_MODEL_CFG, FeatureConfig(window=8), EnvConfig(),
                        extra={'note': 'test'})
        loaded_model, loaded_norm, names, record = load_checkpoint(str(path))

        x = torch.randn(2, 8, 4)
        with torch.no_grad():
            logits_a, _ = model(x)
            logits_b, _ = loaded_model(x)
        assert torch.allclose(logits_a, logits_b, atol=1e-6)
        assert names == ['a', 'b', 'c', 'd']
        assert record['extra']['note'] == 'test'
        np.testing.assert_allclose(normalizer.mean, loaded_norm.mean)


# ---------------------------------------------------------------------------
# ppo.py
# ---------------------------------------------------------------------------

class TestGAE:
    def test_single_step_episodes_have_no_bootstrap(self):
        # Every step is terminal -> advantage collapses to reward - value.
        rewards = np.array([1.0, 2.0, -1.0], dtype=np.float32)
        values = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        dones = np.array([True, True, True])
        adv, ret = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
        np.testing.assert_allclose(adv, rewards - values, atol=1e-6)
        np.testing.assert_allclose(ret, rewards, atol=1e-6)

    def test_bootstraps_within_an_episode(self):
        rewards = np.array([1.0, 1.0], dtype=np.float32)
        values = np.array([0.0, 0.0], dtype=np.float32)
        dones = np.array([False, True])
        adv, _ret = compute_gae(rewards, values, dones, gamma=1.0, lam=1.0)
        # adv[1] = r1 - v1 = 1.0 ; adv[0] = r0 + v1 - v0 + lam*adv[1] = 1 + 1 = 2
        assert adv[1] == pytest.approx(1.0)
        assert adv[0] == pytest.approx(2.0)


class TestPPOTraining:
    def _envs(self, n_envs=2, n=60, window=6):
        rng = np.random.default_rng(7)
        envs = []
        for _ in range(n_envs):
            feats = rng.normal(size=(n, 4)).astype(np.float32)
            close = 100 + rng.normal(size=n).cumsum().astype(np.float64)
            open_ = close.copy()
            envs.append(TradingEnv(feats, close, open_, window=window,
                                   commission=0.0005, slippage=0.0005,
                                   reward_scale=100.0, episode_length=15, rng=rng))
        return envs

    def test_collect_rollout_meets_min_steps_and_respects_episodes(self):
        model = ActorCritic(n_features=4, cfg=TINY_MODEL_CFG)
        envs = self._envs()
        batch = collect_rollout(model, envs, min_steps=20, device='cpu',
                                rng=np.random.default_rng(0))
        assert len(batch) >= 20
        # Episodes are fixed-length (15) and always run to completion, so a done
        # flag must appear at least once every 15 steps.
        assert batch.dones.any()

    def test_update_runs_and_changes_weights(self):
        model = ActorCritic(n_features=4, cfg=TINY_MODEL_CFG)
        envs = self._envs()
        rng = np.random.default_rng(1)
        batch = collect_rollout(model, envs, min_steps=30, device='cpu', rng=rng)

        before = [p.clone() for p in model.parameters()]
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        stats = ppo_update(model, optimizer, batch, PPOConfig(epochs_per_update=2,
                          minibatch_size=8), 'cpu', rng)

        assert math.isfinite(stats['policy_loss'])
        assert math.isfinite(stats['value_loss'])
        changed = any(not torch.allclose(b, a) for b, a in zip(before, model.parameters()))
        assert changed


# ---------------------------------------------------------------------------
# End-to-end smoke: a couple of real PPO updates on real indicator data
# ---------------------------------------------------------------------------

class TestSmokeTrainAndEvaluate:
    def test_tiny_training_loop_then_eval_produces_finite_metrics(self):
        import evaluate as evaluate_module

        df = make_indicator_df(n=220, trend='up')
        feature_cfg = FeatureConfig(window=8, include_rule_signals=True)
        raw = build_raw_features(df, include_rule_signals=True)
        vf = raw['valid_from']
        feats, close, open_ = raw['features'][vf:], raw['close'][vf:], raw['open'][vf:]

        split = int(len(close) * 0.7)
        normalizer = Normalizer().fit(feats[:split])
        feats_n = normalizer.transform(feats)

        env_cfg = EnvConfig(commission=0.0005, slippage=0.0005, episode_length=20)
        model = ActorCritic(n_features=feats.shape[1], cfg=TINY_MODEL_CFG)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        rng = np.random.default_rng(0)

        train_env = TradingEnv(feats_n[:split], close[:split], open_[:split],
                               window=feature_cfg.window, commission=env_cfg.commission,
                               slippage=env_cfg.slippage, reward_scale=env_cfg.reward_scale,
                               episode_length=env_cfg.episode_length, rng=rng)

        for _ in range(2):
            batch = collect_rollout(model, [train_env], min_steps=40, device='cpu', rng=rng)
            stats = ppo_update(model, optimizer, batch, PPOConfig(epochs_per_update=1,
                              minibatch_size=8), 'cpu', rng)
            assert math.isfinite(stats['policy_loss'])

        metrics = evaluate_module.rollout_metrics(
            model, feats_n[split:], close[split:], open_[split:],
            feature_cfg.window, env_cfg)

        for key in ('roi', 'benchmark_roi', 'excess_roi', 'sharpe', 'sortino', 'max_drawdown'):
            assert math.isfinite(metrics[key]), f'{key} is not finite: {metrics[key]}'


# ---------------------------------------------------------------------------
# live.py
# ---------------------------------------------------------------------------

class TestLiveHook:
    def test_unavailable_without_checkpoint(self, monkeypatch):
        import live as live_module
        monkeypatch.setattr(live_module, 'CHECKPOINT_PATH', '/nonexistent/path.pt')
        assert live_module.model_available() is False

    def test_annotate_passes_through_when_unavailable(self, monkeypatch):
        import live as live_module
        monkeypatch.setattr(live_module, 'CHECKPOINT_PATH', '/nonexistent/path.pt')
        found = [{'symbol': 'AAPL', 'direction': 'BUY'}]
        out = live_module.annotate(found, stock_data={})
        assert out == found
        assert 'rl_action' not in out[0]

    def test_annotate_enriches_with_a_real_checkpoint(self, tmp_path, monkeypatch):
        import live as live_module

        df = make_indicator_df(n=100)
        feature_cfg = FeatureConfig(window=6, include_rule_signals=False)
        raw = build_raw_features(df, include_rule_signals=False)
        normalizer = Normalizer().fit(raw['features'][raw['valid_from']:])
        model = ActorCritic(n_features=raw['features'].shape[1], cfg=TINY_MODEL_CFG)

        ckpt = tmp_path / 'best.pt'
        save_checkpoint(str(ckpt), model, normalizer, raw['feature_names'],
                        TINY_MODEL_CFG, feature_cfg, EnvConfig())

        monkeypatch.setattr(live_module, 'CHECKPOINT_PATH', str(ckpt))
        live_module._cache.clear()

        found = [{'symbol': 'TEST', 'direction': 'BUY'}]
        out = live_module.annotate(found, stock_data={'TEST': df})

        assert out[0]['rl_action'] in ('BUY', 'SELL', 'FLAT')
        assert 0.0 <= out[0]['rl_confidence'] <= 1.0
        assert 'rl_agrees' in out[0]

    def test_annotate_skips_symbol_missing_from_stock_data(self, tmp_path, monkeypatch):
        import live as live_module

        df = make_indicator_df(n=100)
        feature_cfg = FeatureConfig(window=6, include_rule_signals=False)
        raw = build_raw_features(df, include_rule_signals=False)
        normalizer = Normalizer().fit(raw['features'][raw['valid_from']:])
        model = ActorCritic(n_features=raw['features'].shape[1], cfg=TINY_MODEL_CFG)

        ckpt = tmp_path / 'best.pt'
        save_checkpoint(str(ckpt), model, normalizer, raw['feature_names'],
                        TINY_MODEL_CFG, feature_cfg, EnvConfig())
        monkeypatch.setattr(live_module, 'CHECKPOINT_PATH', str(ckpt))
        live_module._cache.clear()

        found = [{'symbol': 'MISSING', 'direction': 'SELL'}]
        out = live_module.annotate(found, stock_data={})
        assert 'rl_action' not in out[0]
