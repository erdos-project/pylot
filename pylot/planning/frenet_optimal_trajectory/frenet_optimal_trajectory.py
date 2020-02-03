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
from pylot.planning.frenet_optimal_trajectory.quartic_polynomials \
    import QuarticPolynomial
from pylot.planning.frenet_optimal_trajectory.quintic_polynomials \
    import QuinticPolynomial


SIM_LOOP = 500

# Parameter
MAX_SPEED = 25.0  # maximum speed [m/s]
MAX_ACCEL = 6.0  # maximum acceleration [m/ss]
MAX_CURVATURE = 4.0  # maximum curvature [1/m]
MAX_ROAD_WIDTH = 12.0  # maximum road width [m]
D_ROAD_W = 1.0  # road width sampling length [m]
DT = 0.25  # time tick [s]
MAXT = 6  # max prediction time [m]
MINT = 5  # min prediction time [m]
D_T_S = 1.0  # target speed sampling length [m/s]
N_S_SAMPLE = 1  # sampling number of target speed
OBSTACLE_RADIUS = 6  # obstacle radius [m]

# cost weights
KJ = 0.1
KT = 0.1
KD = 1.0
KLAT = 1.0
KLON = 1.0

show_animation = True


class Frenet_path:
    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []


def calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0, target_speed):
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
            for tv in np.arange(target_speed - D_T_S * N_S_SAMPLE, target_speed + D_T_S * N_S_SAMPLE, D_T_S):
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


def calc_global_paths(fplist, csp):
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


def check_collision(fp, ob):
    if len(ob) == 0:
        return True

    for i in range(len(ob[:, 0])):
        d = [((ix - ob[i, 0])**2 + (iy - ob[i, 1])**2)
             for (ix, iy) in zip(fp.x, fp.y)]

        collision = any([di <= OBSTACLE_RADIUS**2 for di in d])

        if collision:
            return False

    return True


def check_paths(fplist, ob):
    okind = []
    for i, _ in enumerate(fplist):
        if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
            # print("Exceeded max speed")
            continue
        elif any([abs(a) > MAX_ACCEL for a in fplist[i].s_dd]):  # Max accel check
            # print("Exceeded max accel")
            # print(fplist[i].s_dd)
            continue
        elif any([abs(c) > MAX_CURVATURE for c in fplist[i].c]):  # Max curvature check
            # print("Exceeded max curvature")
            continue
        elif not check_collision(fplist[i], ob):
            # print("Collision")
            continue

        okind.append(i)

    return [fplist[i] for i in okind]


def frenet_optimal_planning(csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob, target_speed):
    fplist = calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0, target_speed)
    fplist = calc_global_paths(fplist, csp)
    fplist = check_paths(fplist, ob)

    # find minimum cost path
    mincost = float("inf")
    bestpath = None
    for fp in fplist:
        if mincost >= fp.cf:
            mincost = fp.cf
            bestpath = fp

    return bestpath


def generate_target_course(x, y):
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

    # way points
    wx = [158.6698760986328, 157.6698760986328, 156.6698760986328, 155.6698760986328, 154.6698760986328, 153.6698760986328, 152.6698760986328, 151.6698760986328, 150.6698760986328, 149.6698760986328, 148.6698760986328, 147.6698760986328, 146.6698760986328, 145.6698760986328, 144.6698760986328, 143.6698760986328, 142.6698760986328, 141.6698760986328, 140.6698760986328, 139.6698760986328, 138.6698760986328, 137.6698760986328, 136.6698760986328, 135.6698760986328, 134.6698760986328, 133.6698760986328, 132.6698760986328, 131.6698760986328, 130.6698760986328, 129.6698760986328, 128.6698760986328, 127.66987609863281, 126.66987609863281, 125.66987609863281, 124.66987609863281, 123.66987609863281, 122.66987609863281, 121.66987609863281, 120.66987609863281, 119.66987609863281, 118.66987609863281, 117.66987609863281, 116.66987609863281, 115.66987609863281, 114.66987609863281, 113.66987609863281, 112.66987609863281, 111.66987609863281, 110.66987609863281, 109.66987609863281]
    wy = [195.14553833007812, 195.1454620361328, 195.14537048339844, 195.14529418945312, 195.1452178955078, 195.14512634277344, 195.14505004882812, 195.1449737548828, 195.14488220214844, 195.14480590820312, 195.1447296142578, 195.14463806152344, 195.14456176757812, 195.1444854736328, 195.14439392089844, 195.14431762695312, 195.1442413330078, 195.14414978027344, 195.14407348632812, 195.1439971923828, 195.14390563964844, 195.14382934570312, 195.1437530517578, 195.14366149902344, 195.14358520507812, 195.1435089111328, 195.14341735839844, 195.14334106445312, 195.1432647705078, 195.14317321777344, 195.14309692382812, 195.1430206298828, 195.14292907714844, 195.14285278320312, 195.1427764892578, 195.14268493652344, 195.14260864257812, 195.1425323486328, 195.14244079589844, 195.14236450195312, 195.1422882080078, 195.1422119140625, 195.14212036132812, 195.1420440673828, 195.1419677734375, 195.14187622070312, 195.1417999267578, 195.1417236328125, 195.14163208007812, 195.1415557861328]
    ob = np.array([[153.31408925442975, 187.5718973329686]])

    tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)
    # initial state
    c_speed = 6.738670652134567  # current speed [m/s]
    c_d = 1.388534968537939  # current lateral position [m]
    c_d_d = 0.014405145689946532  # current lateral speed [m/s]
    c_d_dd = 0.0  # current latral acceleration [m/s]
    s0 = 0.0  # current course position

    area = 40.0  # animation area length [m]

    for i in range(SIM_LOOP):
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
            plt.pause(0.0001)

    print("Finish")
    if show_animation:  # pragma: no cover
        plt.grid(True)
        plt.pause(0.0001)
        plt.show()


if __name__ == '__main__':
    main()
