"""This module implements a longitudinal and lateral controller."""

import math
import time
from collections import deque

import numpy as np

import pylot.utils


class PIDLongitudinalController(object):
    """Implements longitudinal control using a PID.

    Args:
       K_P (:obj:`float`): Proportional term.
       K_D (:obj:`float`): Differential term.
       K_I (:obj:`float`): Integral term.
       dt (:obj:`float`): time differential in seconds.
    """
    def __init__(self,
                 K_P: float = 1.0,
                 K_D: float = 0.0,
                 K_I: float = 0.0,
                 dt: float = 0.03,
                 use_real_time: bool = False):
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._use_real_time = use_real_time
        self._last_time = time.time()
        self._error_buffer = deque(maxlen=10)

    def run_step(self, target_speed: float, current_speed: float):
        """Computes the throttle/brake based on the PID equations.

        Args:
            target_speed (:obj:`float`): Target speed in m/s.
            current_speed (:obj:`float`): Current speed in m/s.

        Returns:
            Throttle and brake values.
        """
        # Transform to km/h
        error = (target_speed - current_speed) * 3.6
        self._error_buffer.append(error)

        if self._use_real_time:
            time_now = time.time()
            dt = time_now - self._last_time
            self._last_time = time_now
        else:
            dt = self._dt
        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / dt
            _ie = sum(self._error_buffer) * dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip(
            (self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0,
            1.0)


class PIDLateralController(object):
    """Implements lateral control using a PID.

    Args:
       K_P (:obj:`float`): Proportional term.
       K_D (:obj:`float`): Differential term.
       K_I (:obj:`float`): Integral term.
       dt (:obj:`float`): time differential in seconds.
    """
    def __init__(self,
                 K_P: float = 1.0,
                 K_D: float = 0.0,
                 K_I: float = 0.0,
                 dt: float = 0.03,
                 use_real_time: bool = False):
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._use_real_time = use_real_time
        self._last_time = time.time()
        self._e_buffer = deque(maxlen=10)

    def run_step(self, waypoint, vehicle_transform):
        v_begin = vehicle_transform.location
        v_end = v_begin + pylot.utils.Location(
            x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
            y=math.sin(math.radians(vehicle_transform.rotation.yaw)))

        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        w_vec = np.array([
            waypoint.location.x - v_begin.x, waypoint.location.y - v_begin.y,
            0.0
        ])
        _dot = math.acos(
            np.clip(
                np.dot(w_vec, v_vec) /
                (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

        _cross = np.cross(v_vec, w_vec)

        if _cross[2] < 0:
            _dot *= -1.0

        if self._use_real_time:
            time_now = time.time()
            dt = time_now - self._last_time
            self._last_time = time_now
        else:
            dt = self._dt

        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / dt
            _ie = sum(self._e_buffer) * dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip(
            (self._k_p * _dot) + (self._k_d * _de) + (self._k_i * _ie), -1.0,
            1.0)
