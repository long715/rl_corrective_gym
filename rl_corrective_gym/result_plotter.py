"""
Author: Lee Violet Ong
Date: 07/08/25

p lots we want (for both train and eval):
- plot of each of the rewards
- plot fo the terminal state
- trajectory plots (need to propagate)
"""

import argparse

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from rl_corrective_gym.env_test_script import test_init
from rl_corrective_gym.gym_env_setup.corrective_transfer_env import (
    CorrectiveTransferEnvironment,
)

df = pd.read_csv(
    "../../SAC-mars-25_09_30_14-05-02/10/data/eval.csv",
    on_bad_lines="skip",
    engine="python",
)


def plot_rewards():
    plt.figure(1)

    plt.subplot(2, 2, 1)
    plt.title("Reward Effort")
    plt.ylabel("Reward")
    df_rew_effort: pd.Series = df["reward_effort"]
    plt.plot(range(len(df_rew_effort)), df_rew_effort, ".")

    # TODO: check on unconstrained controls
    # unsure where the source of error is, seems to be giving a slightly diff mass
    plt.subplot(2, 2, 2)
    plt.title("Reward Control Penalty")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    df_rew_control_penalty: pd.Series = df["reward_control_penalty"][0:1000]
    plt.plot(range(len(df_rew_control_penalty)), df_rew_control_penalty, ".")

    plt.subplot(2, 2, 3)
    plt.title("Reward Dynamics")
    plt.xlabel("Episode")
    # plt.ylabel("Reward")
    df_rew_dyn: pd.Series = df["reward_dyn"][0:1000]
    plt.plot(range(len(df_rew_dyn)), df_rew_dyn, ".")

    plt.show()


def plot_terminal():
    """
    Plots the terminal pos magnitude in the x axis and terminal
    vel magnitude in the y axis. Used to compare the deviation of
    different control schemes: optimal, guided, unguided
    """
    # desired terminal state
    nominal_terminal_state: np.ndarray = pd.read_csv(
        "nominal_trajectory/SCP_impulsive_traj.csv"
    ).to_numpy()[-1, :]

    # eval terminal state
    state_pos: np.ndarray = np.array([])
    state_vel: np.ndarray = np.array([])

    for state in df["gui_terminal_state"][-1000:].to_numpy():
        # ignore the mass for now
        state_numpy: np.ndarray = (
            np.fromstring(state.strip("[]"), sep=" ") - nominal_terminal_state
        )

        state_pos = np.append(state_pos, np.linalg.norm(state_numpy[0:3]))
        state_vel = np.append(state_vel, np.linalg.norm(state_numpy[3:6]))

    plt.figure()
    plt.title("Terminal State Deviation")
    plt.xlabel("Position magnitude")
    plt.ylabel("Velocity magnitude")

    plt.plot(state_pos, state_vel, "bx", label="guid")

    # reset for no_guid plot
    state_pos: np.ndarray = np.array([])
    state_vel: np.ndarray = np.array([])

    for state in df["no_gui_terminal_state"][-1000:].to_numpy():
        # ignore the mass for now
        state_numpy: np.ndarray = (
            np.fromstring(state.strip("[]"), sep=" ") - nominal_terminal_state
        )

        state_pos = np.append(state_pos, np.linalg.norm(state_numpy[0:3]))
        state_vel = np.append(state_vel, np.linalg.norm(state_numpy[3:6]))

    plt.plot(state_pos, state_vel, "rx", label="no_guid")

    # OPTIMAL CONTROL - assume single sample and take final eval scenario
    env: CorrectiveTransferEnvironment = test_init()
    env.chosen_timestamp = df["timestep"].iloc[-1]
    env.noise = np.fromstring(df["noise"].iloc[-1].strip("[]"), sep=" ")
    env.state = env.nominal_traj[env.chosen_timestamp] + env.noise
    env._init_logs()
    opt_control: np.ndarray = env._optimal_control()

    nom_imp: np.ndarray = env.nominal_imp[env.chosen_timestamp]
    corr_imp: np.ndarray = np.fromstring(
        df["corrective_impulse"].iloc[-1].strip("[]"), sep=" "
    )
    total: np.ndarray = nom_imp + corr_imp

    # check for the optimal control
    if np.linalg.norm(opt_control + nom_imp) > df["vmax"].iloc[-1]:
        print("FAIL: Optimal Control Not Feasible")
    else:
        print("PASS: Optimal Control Feasible")

    # check for effort reduction
    if np.linalg.norm(total) >= np.linalg.norm(nom_imp):
        print("FAIL: No improvement in effort ")
    else:
        print("PASS: Effort reduced")

    optimal_dev: np.ndarray = env._propagate(True, opt_control) - nominal_terminal_state
    opt_pos: float = np.linalg.norm(optimal_dev[0:3])
    opt_vel: float = np.linalg.norm(optimal_dev[3:6])
    plt.plot(opt_pos, opt_vel, "kx", label="optimal")

    plt.legend()
    plt.show()


def plot_control():
    env: CorrectiveTransferEnvironment = test_init()
    print(env.max_thrust, env.timestep)

    pass


def plot_trajectory():
    """
    Use env propagator to obtain whole trajectory states, plot the position
    to show the deviation of both guided and unguided.
    """
    env: CorrectiveTransferEnvironment = test_init()

    plt.figure()
    ax = plt.axes(projection="3d")

    # plot the desired trajectory

    ax.plot(env.nominal_traj[:, 0], env.nominal_traj[:, 1], env.nominal_traj[:, 2])

    # need to manually do reset setup ie. state, chosen_timestep
    sum = 0
    for i in range(1, 1001):
        env.chosen_timestamp = int(df["timestep"].iloc[-i])
        env.noise = np.fromstring(df["noise"].iloc[-i].strip("[]"), sep=" ")
        env.state = env.nominal_traj[env.chosen_timestamp] + env.noise

        corrective_impulse: np.ndarray = np.fromstring(
            df["corrective_impulse"].iloc[-i].strip("[]"), sep=" "
        )
        sum += np.linalg.norm(corrective_impulse)

        env._init_logs()
        env._propagate(True, corrective_impulse)
        env._propagate(False)
        ax.plot(
            env.gui_log_pos[:, 0], env.gui_log_pos[:, 1], env.gui_log_pos[:, 2], "r"
        )
        ax.plot(
            env.nogui_log_pos[:, 0],
            env.nogui_log_pos[:, 1],
            env.nogui_log_pos[:, 2],
            "g",
        )

    plt.show()


def plot_loss():
    plt.figure()

    critic_loss_one = df["critic_loss_total"][1000:]

    plt.plot(range(len(critic_loss_one)), critic_loss_one)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Critic Loss")

    plt.show()


def bar_control():
    """
    Aims to show the proportion of nominal to corrective in vmax, this is for
    the initial analysis for feasibility.
    """
    nominal_imp: np.ndarray = pd.read_csv("nominal_trajectory/SCP_dV.csv").to_numpy()
    nom_prop: np.ndarray = np.array([])
    corr_prop: np.ndarray = np.array([])

    for i in range(1000):
        chosen_timestamp: int = df["timestep"][i]
        vmax: int = df["vmax"][i]

        corrective_imp: np.ndarray = np.fromstring(
            df["corrective_impulse"][i].strip("[]"), sep=" "
        )
        n_imp: np.ndarray = nominal_imp[chosen_timestamp, :]

        total_imp: np.ndarray = corrective_imp + n_imp
        total_unit: np.ndarray = total_imp / np.linalg.norm(total_imp)

        n_imp_mag = np.dot(n_imp, total_unit)
        corrective_impulse_mag = np.dot(corrective_imp, total_unit)

        if n_imp_mag < 0:
            corrective_impulse_mag += n_imp_mag
            n_imp_mag = 0
        elif corrective_impulse_mag < 0:
            n_imp_mag += corrective_impulse_mag
            corrective_impulse_mag = 0

        nom_prop = np.append(nom_prop, n_imp_mag / vmax)
        corr_prop = np.append(corr_prop, corrective_impulse_mag / vmax)

    plt.figure()
    x = range(100)
    vmax_prop = np.array([1] * 100)

    plt.bar(x, corr_prop[900:], alpha=0.8, color="r")
    plt.bar(x, nom_prop[900:], bottom=corr_prop[900:], alpha=0.6, color="b")
    plt.bar(x, vmax_prop, alpha=0.3, color="g")

    plt.xlabel("Episodes")
    plt.ylabel("Proportion")
    plt.legend(["Corrective", "Nominal"])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot",
        required=False,
        type=str,
        default="terminal",
        choices=["reward", "terminal", "control", "trajectory", "loss", "bar_prop"],
    )

    args = parser.parse_args()
    if args.plot == "reward":
        plot_rewards()
    elif args.plot == "terminal":
        plot_terminal()
    elif args.plot == "control":
        plot_control()
    elif args.plot == "trajectory":
        plot_trajectory()
    elif args.plot == "loss":
        plot_loss()
    elif args.plot == "bar_prop":
        bar_control()
