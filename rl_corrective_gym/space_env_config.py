from cares_reinforcement_learning.util.configurations import SubscriptableClass


class SpaceEnvironmentConfig(SubscriptableClass):

    traj_filename: str
    impulse_filename: str

    # defines if we want to reset to the same timestep
    single_run: bool

    # defines the action space to use
    action_config: int

    # defines the reward function to use
    dyn_rew: int
    effort_rew: int

    tof: float  # days
    max_thrust: float  # kg*km/s^2
    exhaust_vel: float  # km/s

    dyn_pos_sd: float = 1.0  # km
    dyn_vel_sd: float = 0.05  # km/s

    max_corr: float = 0.1  # km/s
