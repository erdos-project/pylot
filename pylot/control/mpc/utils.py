import numpy as np

global_config = {
    'vehicle': { # configured for lincoln mkz
        'length': 4.93,
        'width': 1.86,
        'offset': 1.0,
        'wheel_length': 0.3,
        'wheel_width': 0.2,
        'track': 0.7,
        'wheelbase': 2.85,
        'max_steer': np.deg2rad(37.5),
        'min_steer': np.deg2rad(-37.5),
        'max_steer_speed': np.deg2rad(22.5),
        'min_steer_speed': np.deg2rad(-22.5),
        'max_vel': 20,
        'min_vel': 0,
        'max_accel': 3.0,
        'min_accel': -5.0,
    },
    'controller': {
        'R': np.diag([0.01, 0.10]),  # Input cost
        'Rd': np.diag([0.01, 1.0]),  # Input difference cost
        'Q': np.diag([1.0, 1.0, 0.01, 0.01]),  # State cost
        'Qf': np.diag([1.0, 1.0, 0.01, 0.01]),  # Terminal state cost
        'goal_threshold': 1.0,  # Threshold for goal test [m]
        'expiration_time': 100.0,  # Expiration time [s]
        'max_iteration': 5,  # Max step iterations
        'convergence_threshold': 0.1,  # Threshold for convergence test
        'horizon': 5,  # Horizon
        'index_horizon': 5,  # Index horizon
    },
}


def compute_curvature(vel, accel, yaw):
    dx = vel * np.tan(yaw)
    ddx = accel * np.tan(yaw)
    dy = vel * np.tan(yaw)
    ddy = accel * np.tan(yaw)
    return (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2) ** (3 / 2))


def zero_to_2_pi(angle):
    return (angle + 360) % 360