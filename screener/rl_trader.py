"""
Layer 4: RL Portfolio Trader (DQN wrapper)

Thin wrapper around Stable Baselines 3 DQN for training, saving, and
loading portfolio agents.  Depends on ``stable-baselines3`` and ``gymnasium``.
"""

from __future__ import annotations

import time

from screener.config import ScreenerConfig


class RLTrader:
    """Train and run a DQN agent for portfolio management."""

    def __init__(self, cfg: ScreenerConfig):
        self.cfg = cfg

    def train(self, env):
        """Train a DQN model on the given PortfolioEnv.

        Returns the trained SB3 DQN model.
        """
        from stable_baselines3 import DQN
        from stable_baselines3.common.callbacks import BaseCallback

        total_steps = self.cfg.rl_total_timesteps

        class _ProgressCallback(BaseCallback):
            """Print a concise one-liner every 10k steps."""

            def __init__(self):
                super().__init__()
                self._t0 = time.time()
                self._next_log = 10_000

            def _on_step(self) -> bool:
                if self.num_timesteps >= self._next_log:
                    elapsed = time.time() - self._t0
                    fps = self.num_timesteps / max(elapsed, 1)
                    ep_info = self.locals.get("infos", [{}])
                    loss = self.model.logger.name_to_value.get(
                        "train/loss", float("nan")
                    )
                    print(
                        f"    DQN {self.num_timesteps:>6}/{total_steps} "
                        f"({elapsed:.0f}s, {fps:.0f}fps) "
                        f"loss={loss:.2f}"
                    )
                    self._next_log += 10_000
                return True

        t0 = time.time()
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=self.cfg.rl_learning_rate,
            buffer_size=self.cfg.rl_buffer_size,
            batch_size=self.cfg.rl_batch_size,
            gamma=self.cfg.rl_gamma,
            exploration_fraction=self.cfg.rl_exploration_fraction,
            exploration_final_eps=self.cfg.rl_exploration_final_eps,
            target_update_interval=self.cfg.rl_target_update_interval,
            policy_kwargs=dict(net_arch=self.cfg.rl_net_arch),
            verbose=0,
        )
        model.learn(
            total_timesteps=total_steps,
            callback=_ProgressCallback(),
        )
        elapsed = time.time() - t0
        print(f"    DQN training complete: {total_steps} steps in {elapsed:.0f}s")
        return model

    @staticmethod
    def save(model, path: str):
        """Save a trained DQN model to disk."""
        model.save(path)

    @staticmethod
    def load(path: str):
        """Load a DQN model from disk."""
        from stable_baselines3 import DQN

        return DQN.load(path)
