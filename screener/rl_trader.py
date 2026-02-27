"""
Layer 4: RL Portfolio Trader (DQN wrapper)

Thin wrapper around Stable Baselines 3 DQN for training, saving, and
loading portfolio agents.  Depends on ``stable-baselines3`` and ``gymnasium``.
"""

from __future__ import annotations

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
            verbose=1,
        )
        model.learn(total_timesteps=self.cfg.rl_total_timesteps)
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
