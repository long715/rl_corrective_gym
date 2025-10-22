"""
Responsible for initialising the environment for the task.
- "domain" [single, multi]: single sample refers to resetting at the same state and multi-sample resets at different statees
"""

import gymnasium as gym


from rl_corrective_gym.environments.single_node_correction import (
    SingleCorrectiveTransferEnvironment,
)
from rl_corrective_gym.environments.corrective_transfer_env import (
    CorrectiveTransferEnvironment,
)
from rl_corrective_gym.space_env_config import SpaceEnvironmentConfig


class SpaceEnvironmentFactory:
    def __init__(self):
        pass

    def create_environment(self, env_config: SpaceEnvironmentConfig) -> gym.Env:
        domain: str = env_config.domain

        if domain == "single":
            return SingleCorrectiveTransferEnvironment(env_config)
        elif domain == "multi":
            return CorrectiveTransferEnvironment(env_config)
        else:
            raise ValueError(f"Invalid domain: {domain}")
