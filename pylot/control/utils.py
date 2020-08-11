import numpy as np


def radians_to_steer(rad: float, steer_gain: float):
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


def steer_to_radians(steer: float, steer_gain: float):
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


def compute_throttle_and_brake(pid, current_speed: float, target_speed: float,
                               flags, logger):
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
    if current_speed < 0:
        logger.warning('Current speed is negative: {}'.format(current_speed))
        non_negative_speed = 0
    else:
        non_negative_speed = current_speed
    acceleration = pid.run_step(target_speed, non_negative_speed)
    if acceleration >= 0.0:
        throttle = min(acceleration, flags.throttle_max)
        brake = 0
    else:
        throttle = 0.0
        brake = min(abs(acceleration), flags.brake_max)
    # Keep the brake pressed when stopped or when sliding back on a hill.
    if (current_speed < 1 and target_speed == 0) or current_speed < -0.3:
        brake = 1.0
    return throttle, brake
