class SpaceEnvironmentConfig:
    traj_filename: str
    impulse_filename: str

    # defines if we want to reset to the same timestep
    single_run: bool

    tof: float  # days
    max_thrust: float  # kg*km/s^2
    exhaust_vel: float  # km/s

    dyn_pos_sd: float = 1.0  # km
    dyn_vel_sd: float = 0.05  # km/s

    max_corr: float = 0.01  # km/s
