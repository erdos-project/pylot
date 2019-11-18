import numpy as np


class Trajectory:
    def __init__(self, t_list, s_list, x_list, y_list, k_list, vel_list,
                 yaw_list, accel_list=None, steer_list=None):
        self.t_list = list(t_list)  # Time [s]
        self.s_list = list(s_list)  # Arc distance list [m]
        self.x_list = list(x_list)  # X coordinate list [m]
        self.y_list = list(y_list)  # Y coordinate list [m]
        self.k_list = list(k_list)  # Curvature list [1/m]
        self.vel_list = list(vel_list)  # Tangential velocity list [m/s]
        self.yaw_list = list(self._normalize_yaw(yaw_list))  # Yaw list [rad]
        if accel_list is not None:
            self.accel_list = list(accel_list)  # Acceleration list [m/s2]
        else:
            self.accel_list = accel_list
        if steer_list is not None:
            self.steer_list = list(steer_list)  # Steering list [rad]
        else:
            self.steer_list = steer_list

    def append_vel(self, vel):
        self.vel_list.append(vel)

    def append(self, t, s, x, y, k, vel, yaw, accel=None, steer=None):
        self.t_list.append(t)
        self.s_list.append(s)
        self.x_list.append(x)
        self.y_list.append(y)
        self.k_list.append(k)
        self.vel_list.append(vel)
        self.yaw_list.append(yaw)
        if accel is not None:
            self.accel_list.append(accel)
        if steer is not None:
            self.steer_list.append(steer)

    @staticmethod
    def _normalize_yaw(yaw_list):
        if len(yaw_list) > 1:
            for i in range(len(yaw_list) - 1):
                delta_yaw = yaw_list[i + 1] - yaw_list[i]
                while delta_yaw >= np.pi / 2.0:
                    yaw_list[i + 1] -= np.pi * 2.0
                    delta_yaw = yaw_list[i + 1] - yaw_list[i]
                while delta_yaw <= -np.pi / 2.0:
                    yaw_list[i + 1] += np.pi * 2.0
                    delta_yaw = yaw_list[i + 1] - yaw_list[i]
        return yaw_list