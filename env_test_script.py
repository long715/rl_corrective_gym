"""
Author: Lee Violet Ong
Date: 15/07/25

This file defines the test suite for the corrective transfer gym, build to validate the logical implementation of the gym.
This also contains test training scripts for the gym.
"""

import gymnasium as gym

from corrective_transfer_env import CorrectiveTransferEnvironment


def test_prop():
    # Tests if the propagator works without perturbation and control
    env: gym.Env = CorrectiveTransferEnvironment("SCP_impulsive_traj.csv", "SCP_dV.csv")
    final_state, reward, _, _, _ = env.step([-1, 1, 1, 1])
    print(env.nominal_traj[-1, :] - final_state, reward)

    # error ~1e-7 which could be attributed to rounding within propagator
    assert abs(reward) < 1e-6, "Propagator error exceeds acceptable threshold"


if __name__ == "__main__":
    test_prop()
