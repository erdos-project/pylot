import numpy as np


def radians_to_steer(rad, steer_gain):
    """Converts radians to steer input.

    Returns:
        :obj:`float`: Between [-1.0, 1.0].
    """
    steer = steer_gain * rad
    if steer > 0:
        steer = min(steer, 1)
    else:
        steer = max(steer, -1)
    return steer


def steer_to_radians(steer, steer_gain):
    """Converts radians to steer input.

    Assumes max steering angle is -45, 45 degrees.

    Returns:
        :obj:`float`: Steering in radians.
    """
    rad = steer / steer_gain
    if rad > 0:
        rad = min(rad, np.pi / 2)
    else:
        rad = max(rad, -np.pi / 2)
    return rad


def compute_throttle_and_brake(pid, current_speed, target_speed, flags):
    """Computes the throttle/brake required to reach the target speed.

    It uses the longitudinal controller to derive the required information.

    Args:
        pid: The pid controller.
        current_speed (:obj:`float`): The current speed of the ego vehicle
            (in m/s).
        target_speed (:obj:`float`): The target speed to reach (in m/s).
        flags (absl.flags): The flags object.

    Returns:
        Throttle and brake values.
    """
    pid.target = target_speed
    pid_gain = pid(feedback=current_speed)
    throttle = min(max(flags.default_throttle - 1.3 * pid_gain, 0),
                   flags.throttle_max)
    if pid_gain > 0.5:
        brake = min(0.35 * pid_gain * flags.brake_strength, 1)
    else:
        brake = 0
    return throttle, brake
