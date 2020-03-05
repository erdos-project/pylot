import ctypes
import numpy as np
from ctypes import c_double, c_int

cdll = ctypes.CDLL("pylot/planning/frenet_optimal_trajectory/FrenetOptimalTrajectory/build/libFrenetOptimalTrajectory.so")
_get_fot_frenet_space = cdll.get_fot_frenet_space
_get_fot_frenet_space.argtypes = (
    c_double, c_double, c_double, c_double, c_double,  # s0, c_speed, c_d, c_d_d, c_d_dd
    ctypes.POINTER(c_double), ctypes.POINTER(c_double),  # wx, wy
    ctypes.POINTER(c_double), ctypes.POINTER(c_double),  # ox, oy
    c_int, c_int, c_double,  # path length, num obstacles, target speed
    ctypes.POINTER(c_double), ctypes.POINTER(c_double),  # result x, result y
    ctypes.POINTER(c_double), ctypes.POINTER(c_double)  # speeds, misc
)
_get_fot_frenet_space.restype = ctypes.c_int
_compute_initial_conditions = cdll.compute_initial_conditions
_compute_initial_conditions.restype = None
_compute_initial_conditions.argtypes = (
    c_double, c_double, c_double, c_double, c_double, c_double,
    ctypes.POINTER(c_double), ctypes.POINTER(c_double),
    c_int, ctypes.POINTER(c_double)
)


def get_fot_frenet_space(s0, c_speed, c_d, c_d_d, c_d_dd, wx, wy, obs,
                         target_speed):
    """ Return the frenet optimal trajectory given initial conditions in
    cartesian space.

    Args:
        s0 (float): initial longitudinal position
        c_speed (float): initial forward speed
        c_d (float): initial lateral position
        c_d_d (float): initial lateral velocity
        c_d_dd (float): initial lateral acceleration
        wx (list(float)): list of global x waypoints
        wy (list(float)): list of global y waypoints
        obs (list((float, float))): list of obstacle locations
        target_speed (float): target speed

    Returns:
        result_x (list(float)): x positions of fot, if it exists
        result_y (list(float)): y positions of fot, if it exists
        speeds (list(float)): velocities of fot, if it exists
        params (dict): next frenet coordinates, if they exist
        success (bool): whether a fot was found or not
    """
    result_x = np.zeros(100)
    result_y = np.zeros(100)
    speeds = np.zeros(100)
    misc = np.zeros(5)
    if obs.shape[0] == 0:
        obs = np.empty((0, 2))
    success = _get_fot_frenet_space(
        c_double(s0), c_double(c_speed), c_double(c_d),
        c_double(c_d_d), c_double(c_d_dd),
        wx.ctypes.data_as(ctypes.POINTER(c_double)),
        wy.ctypes.data_as(ctypes.POINTER(c_double)),
        obs[:, 0].ctypes.data_as(ctypes.POINTER(c_double)),
        obs[:, 1].ctypes.data_as(ctypes.POINTER(c_double)),
        c_int(len(wx)), c_int(len(obs)), c_double(target_speed),
        result_x.ctypes.data_as(ctypes.POINTER(c_double)),
        result_y.ctypes.data_as(ctypes.POINTER(c_double)),
        speeds.ctypes.data_as(ctypes.POINTER(c_double)),
        misc.ctypes.data_as(ctypes.POINTER(c_double))
    )
    params = {
        "s0": misc[0],
        "c_speed": misc[1],
        "c_d": misc[2],
        "c_d_d": misc[3],
        "c_d_dd": misc[4]
    }
    ind = -1
    if success:
        ind = np.where(np.isnan(result_x))[0][0]

    return result_x[:ind], result_y[:ind], speeds[:ind], params, success


def compute_initial_conditions(s0, x, y, vx, vy, forward_speed, wx, wy):
    """ Return the frenet optimal trajectory given initial conditions in
    cartesian space.

    Args:
        s0 (float): previous longitudinal position
        x (float): initial x position
        y (float): initial y position
        vx (float): initial x velocity
        vy (float): initial y velocity
        forward_speed (float): initial speed
        wx (list(float)): list of global x waypoints
        wy (list(float)): list of global y waypoints

    Returns:
        s_c (float): current longitudinal position
        c_speed (float): current speed
        c_d (float): current lateral offset
        c_d_d (float): current lateral velocity
        c_d_dd (float): current lateral acceleration
    """
    misc = np.zeros(5)
    _compute_initial_conditions(
        c_double(s0), c_double(x), c_double(y), c_double(vx), c_double(vy),
        c_double(forward_speed), wx.ctypes.data_as(ctypes.POINTER(c_double)),
        wy.ctypes.data_as(ctypes.POINTER(c_double)), c_int(len(wx)),
        misc.ctypes.data_as(ctypes.POINTER(c_double))
    )

    return misc[0], misc[1], misc[2], misc[3], misc[4]
