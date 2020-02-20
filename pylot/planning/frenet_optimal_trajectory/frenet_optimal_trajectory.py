"""
Author: Edward Fang
Email: edward.fang@berkeley.edu

This code is adapted from https://github.com/AtsushiSakai/PythonRobotics/tree/
master/PathPlanning/FrenetOptimalTrajectory.
Its author is Atsushi Sakai.

Reference Papers:
- [Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame]
(https://www.researchgate.net/profile/Moritz_Werling/publication/224156269_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame/links/54f749df0cf210398e9277af.pdf)
- [Optimal trajectory generation for dynamic street scenarios in a Frenet Frame]
(https://www.youtube.com/watch?v=Cj6tAQe7UCY)

"""

import numpy as np
import copy
import math
from shapely.geometry import box, Point

from pylot.control.mpc.utils import CubicSpline2D
from pylot.planning.frenet_optimal_trajectory.constants import *
from pylot.planning.frenet_optimal_trajectory.quartic_polynomials \
    import QuarticPolynomial
from pylot.planning.frenet_optimal_trajectory.quintic_polynomials \
    import QuinticPolynomial


class FrenetPath:
    def __init__(self):
        self.t = []         # time
        self.d = []         # lateral offset
        self.d_d = []       # lateral speed
        self.d_dd = []      # lateral acceleration
        self.d_ddd = []     # lateral jerk
        self.s = []         # s position along spline
        self.s_d = []       # s speed
        self.s_dd = []      # s acceleration
        self.s_ddd = []     # s jerk
        self.cd = 0.0       # lateral cost
        self.cv = 0.0       # longitudinal cost
        self.cf = 0.0       # final cost

        self.x = []         # x position
        self.y = []         # y position
        self.yaw = []       # yaw in rads
        self.ds = []        # speed
        self.c = []         # curvature


def _calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0, target_speed):
    """
    Calculate frenet paths as described in the references.
    """
    frenet_paths = []

    # generate path to each offset goal
    for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W):

        # Lateral motion planning
        for Ti in np.arange(MINT, MAXT, DT):
            fp = FrenetPath()

            # lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
            lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

            # Longitudinal motion planning (Velocity keeping)
            for tv in np.arange(target_speed - D_T_S * N_S_SAMPLE,
                                target_speed + D_T_S * N_S_SAMPLE, D_T_S):
                tfp = copy.deepcopy(fp)
                lon_qp = QuarticPolynomial(s0, c_speed, 0.0, tv, 0.0, Ti)

                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                # square of diff from target speed
                ds = (target_speed - tfp.s_d[-1])**2

                tfp.cd = KJ * Jp + KT * Ti + KD * tfp.d[-1]**2
                tfp.cv = KJ * Js + KT * Ti + KD * ds
                tfp.cf = KLAT * tfp.cd + KLON * tfp.cv

                frenet_paths.append(tfp)

    return frenet_paths


def _calc_global_paths(fplist, csp):
    """
    Convert the frenet path to a global path in terms of x, y, yaw, velocity.
    """
    for fp in fplist:

        # calc global positions
        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            iyaw = csp.calc_yaw(fp.s[i])
            di = fp.d[i]
            fx = ix + di * math.cos(iyaw + math.pi / 2.0)
            fy = iy + di * math.sin(iyaw + math.pi / 2.0)
            fp.x.append(fx)
            fp.y.append(fy)

        # calc yaw and ds
        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(math.atan2(dy, dx))
            fp.ds.append(math.hypot(dx, dy))

        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])

        # calc curvature
        for i in range(len(fp.yaw) - 1):
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

    return fplist


def _check_collision(fp, ob):
    """
    Check the frenet path for collision with an obstacle list.
    """
    if len(ob) == 0:
        return True

    for obstacle in ob:
        b = box(obstacle[0][0], obstacle[0][1], obstacle[1][0], obstacle[1][1])
        for (ix, iy) in zip(fp.x, fp.y):
            if np.isnan(ix) or np.isnan(iy):
                continue
            c = Point(ix, iy).buffer(OBSTACLE_RADIUS)
            if b.intersects(c):
                return False

    return True


def _check_paths(fplist, ob):
    okind = []
    for i, _ in enumerate(fplist):
        # Max speed check
        if np.any(np.abs(fplist[i].s_d) > MAX_SPEED):
            continue
        # Max accel check
        elif np.any(np.abs(fplist[i].s_dd) > MAX_ACCEL):
            continue
        # Max curvature check
        elif np.any(np.abs(fplist[i].c) > MAX_CURVATURE):
            continue
        # Collision check
        elif not _check_collision(fplist[i], ob):
            continue

        okind.append(i)

    return [fplist[i] for i in okind]


def frenet_optimal_planning(csp,
                            s0,
                            c_speed,
                            c_d,
                            c_d_d,
                            c_d_dd,
                            ob,
                            target_speed):
    """Find the frenet optimal trajectory.

    Args:
        csp (:py:class:`planning.control.mpc.utils.CubicSpline2D`): Cubic
            spline defining the intended trajectory to follow
        s0 (:float:): s-position along the spline
        c_speed (:float:): the speed at which the vehicle is moving
        c_d (:float:): the lateral offset d from the frenet frame
        c_d_d (:float:): the lateral speed of the vehicle
        c_d_dd (:float:): the lateral acceleration of the vehicle
        ob (:list([x, y]:): list of obstacle origins
        target_speed (:float:): target speed to reach

    Returns:
        bestpath
            (:py:class:`~pylot.planning.frenet_optimal_trajectory.FrenetPath`):
            frenet optimal trajectory
    """
    fplist = _calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0, target_speed)
    fplist = _calc_global_paths(fplist, csp)
    fplist = _check_paths(fplist, ob)

    # find minimum cost path
    mincost = float("inf")
    bestpath = None
    for fp in fplist:
        if mincost >= fp.cf:
            mincost = fp.cf
            bestpath = fp

    return bestpath


def generate_target_course(x, y):
    """
    Return the global course and frenet path defined by a list of waypoints.

    Args:
        x (:list(float):): list of x positions [m]
        y (:list(float):): list of y positions [m]

    Returns:
        rx (:list(float):): list of x positions [m]
        ry (:list(float):): list of y positions [m]
        ryaw (:list(float):): list of yaws [rads]
        rk (:list(float):): list of curvatures [1/m]
        csp (:py:class:`~pylot.control.mpc.utils.CubicSpline2D`): Cubic spline
            defined by input x, y
    """
    csp = CubicSpline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, csp
