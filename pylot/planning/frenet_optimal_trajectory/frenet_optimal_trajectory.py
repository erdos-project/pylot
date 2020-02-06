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
import matplotlib.pyplot as plt
import copy
import math

from pylot.control.mpc.utils import CubicSpline2D
from pylot.planning.frenet_optimal_trajectory.constants import *
from pylot.planning.frenet_optimal_trajectory.quartic_polynomials \
    import QuarticPolynomial
from pylot.planning.frenet_optimal_trajectory.quintic_polynomials \
    import QuinticPolynomial


class Frenet_path:
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
            fp = Frenet_path()

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

    for i in range(len(ob[:, 0])):
        d = [((ix - ob[i, 0])**2 + (iy - ob[i, 1])**2)
             for (ix, iy) in zip(fp.x, fp.y)]

        collision = any([di <= OBSTACLE_RADIUS**2 for di in d])

        if collision:
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
        csp (:py:class:`planning.control.mpc.utils.CubicSpline2D`): Cubic spline
            defining the intended trajectory to follow
        s0 (:float:): s-position along the spline
        c_speed (:float:): the speed at which the vehicle is moving
        c_d (:float:): the lateral offset d from the frenet frame
        c_d_d (:float:): the lateral speed of the vehicle
        c_d_dd (:float:): the lateral acceleration of the vehicle
        ob (:list([x, y]): list of obstacle origins
        target_speed (:float:): target speed to reach
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
        x (:list(float):): list of x positions
        y (:list(float):): list of y positions
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


def main():
    print(__file__ + " start!!")
    sim_loop = 500
    show_animation = True

    conds = {'s0': 4.200000000000001, 'c_speed': 1.763241783911561, 'c_d': -0.7015454162686688, 'c_d_d': -0.01485228361137665, 'c_d_dd': 0.0, 'wx': [-1.9890024662017822, -1.9891330003738403, -1.989263653755188, -1.989394187927246, -1.9895247220993042, -1.9896552562713623, -1.98978590965271, -1.9900000095367432, -1.9900000095367432, -1.9724081754684448, -1.8622934818267822, -1.65205979347229, -1.343270182609558, -0.9382200241088867, -0.43992316722869873, 0.1479179859161377, 0.8209325075149536, 1.6044834852218628, 2.500894069671631, 3.4878153800964355, 4.552231311798096, 5.680100917816162, 6.856550216674805, 8.066060066223145, 10.008929252624512, 10.008929252624512, 11.008929252624512, 12.008929252624512, 13.008929252624512, 14.008929252624512, 15.008928298950195, 16.008926391601562, 17.008926391601562, 18.008926391601562, 19.008926391601562, 20.008926391601562, 21.008926391601562, 22.008926391601562, 23.008926391601562, 24.008926391601562, 25.008926391601562, 26.008926391601562, 27.008926391601562, 28.008926391601562, 29.008926391601562, 30.00892448425293, 31.00892448425293, 32.0089225769043, 33.0089225769043, 34.0089225769043, 35.0089225769043, 36.0089225769043, 37.0089225769043, 38.0089225769043, 39.0089225769043, 40.0089225769043, 41.0089225769043, 42.0089225769043, 43.0089225769043, 44.0089225769043, 45.0089225769043, 46.0089225769043, 47.0089225769043, 48.0089225769043, 49.0089225769043, 50.0089225769043, 51.0089225769043, 52.0089225769043, 53.0089225769043, 54.0089225769043, 55.0089225769043, 56.0089225769043, 57.0089225769043, 58.0089225769043, 59.0089225769043, 60.0089225769043, 61.0089225769043, 62.0089225769043, 63.0089225769043, 64.00892639160156, 65.00892639160156, 66.00892639160156, 67.00892639160156, 68.00892639160156, 69.00892639160156, 70.00892639160156, 71.00892639160156, 72.00892639160156, 73.00892639160156, 74.00892639160156, 75.00892639160156, 76.00892639160156, 77.00892639160156, 78.00892639160156, 79.41213989257812, 79.41213989257812, 80.41213989257812, 81.41213989257812, 82.53720092773438, 83.7699203491211], 'wy': [309.9596862792969, 310.9596862792969, 311.9596862792969, 312.9596862792969, 313.9596862792969, 314.9596862792969, 315.9596862792969, 317.6001281738281, 317.6001281738281, 318.7019958496094, 319.86895751953125, 321.0220947265625, 322.15283203125, 323.25274658203125, 324.313720703125, 325.3277893066406, 326.2874450683594, 327.2090759277344, 328.050048828125, 328.78271484375, 329.3973388671875, 329.88592529296875, 330.241943359375, 330.46075439453125, 330.53997802734375, 330.53997802734375, 330.54052734375, 330.5410461425781, 330.5415954589844, 330.5421142578125, 330.54266357421875, 330.5431823730469, 330.5437316894531, 330.54425048828125, 330.5447998046875, 330.54534912109375, 330.5458679199219, 330.5464172363281, 330.54693603515625, 330.5474853515625, 330.5480041503906, 330.5485534667969, 330.549072265625, 330.54962158203125, 330.5501708984375, 330.5506896972656, 330.5512390136719, 330.5517578125, 330.55230712890625, 330.5528259277344, 330.5533752441406, 330.55389404296875, 330.554443359375, 330.55499267578125, 330.5555114746094, 330.5560607910156, 330.55657958984375, 330.55712890625, 330.5576477050781, 330.5581970214844, 330.5587158203125, 330.55926513671875, 330.5597839355469, 330.5603332519531, 330.5608825683594, 330.5614013671875, 330.56195068359375, 330.5624694824219, 330.5630187988281, 330.56353759765625, 330.5640869140625, 330.5646057128906, 330.5651550292969, 330.5657043457031, 330.56622314453125, 330.5667724609375, 330.5672912597656, 330.5678405761719, 330.568359375, 330.56890869140625, 330.5694274902344, 330.5699768066406, 330.5705261230469, 330.571044921875, 330.57159423828125, 330.5721130371094, 330.5726623535156, 330.57318115234375, 330.57373046875, 330.5742492675781, 330.5747985839844, 330.5753479003906, 330.57586669921875, 330.576416015625, 330.5771484375, 330.5771484375, 330.57769775390625, 330.5782165527344, 330.5586853027344, 330.4064025878906], 'obstacle_list': [[11.432791981999445, 300.0154646474434], [11.432045645406461, 300.68699929076814], [11.431299308813477, 301.35853393409286], [11.430552972220493, 302.0300685774176], [11.429806635627509, 302.7016032207423], [11.429060299034525, 303.3731378640671], [11.428313962441539, 304.0446725073918], [11.427567625848555, 304.7162071507165], [11.426821289255571, 305.38774179404123], [11.426074952662587, 306.05927643736595]]}

    # way points
    wx = conds['wx']
    wy = conds['wy']

    # obstacle lists
    ob = np.array(conds['obstacle_list'])

    tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)

    # initial state
    c_speed = conds['c_speed']  # current speed [m/s]
    c_d = conds['c_d']  # current lateral position [m]
    c_d_d = conds['c_d_d']  # current lateral speed [m/s]
    c_d_dd = conds['c_d_dd']  # current latral acceleration [m/s]
    s0 = conds['s0']  # current course position

    area = 20.0  # animation area length [m]

    for i in range(sim_loop):
        path = frenet_optimal_planning(
            csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob, 10)

        s0 = path.s[1]
        c_d = path.d[1]
        c_d_d = path.d_d[1]
        c_d_dd = path.d_dd[1]
        c_speed = path.s_d[1]

        if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 1.0:
            print("Goal")
            break

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(tx, ty)
            plt.plot(ob[:, 0], ob[:, 1], "xk")
            plt.plot(path.x[1:], path.y[1:], "-or")
            plt.plot(path.x[1], path.y[1], "vc")
            plt.xlim(path.x[1] - area, path.x[1] + area)
            plt.ylim(path.y[1] - area, path.y[1] + area)
            plt.title("v[m/s]:" + str(c_speed)[0:4])
            plt.grid(True)
            plt.pause(0.001)

    print("Finish")
    if show_animation:  # pragma: no cover
        plt.grid(True)
        plt.pause(0.001)
        plt.show()


if __name__ == '__main__':
    main()
