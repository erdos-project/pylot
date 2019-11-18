import numpy as np


class Vehicle:
    def __init__(self, config):
        self.config = config
        self.t = None  # Time [s]
        self.s = None  # Arc distance [m]
        self.x = None  # X coordinate [m]
        self.y = None  # Y coordinate [m]
        self.k = None  # Curvature [1/m]
        self.vel = None  # Tangential velocity [m/s]
        self.yaw = None  # Yaw [rad]
        self.accel = None  # Acceleration [m/s2]
        self.steer = None  # Steering [rad]

    def update(self, t, s, x, y, k, vel, yaw, accel, steer):
        self.t = t
        self.s = s
        self.x = x
        self.y = y
        self.k = k
        self.vel = vel
        self.yaw = yaw
        self.accel = accel
        self.steer = steer

    def get_position(self):
        return np.asarray([self.x, self.y])

    def get_state(self):
        return np.asarray([self.x, self.y, self.vel, self.yaw])