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
OBSTACLE_RADIUS = 4  # obstacle radius [m]

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
            # print(ob[i])
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

    wx = [-1.9890024662017822, -1.9891330003738403, -1.989263653755188,
          -1.989394187927246, -1.9895247220993042, -1.9896552562713623,
          -1.98978590965271, -1.9900000095367432, -1.9900000095367432,
          -1.9724081754684448, -1.8622934818267822, -1.65205979347229,
          -1.343270182609558, -0.9382200241088867, -0.43992316722869873,
          0.1479179859161377, 0.8209325075149536, 1.6044834852218628,
          2.500894069671631, 3.4878153800964355, 4.552231311798096,
          5.680100917816162, 6.856550216674805, 8.066060066223145,
          10.008929252624512, 10.008929252624512, 11.008929252624512,
          12.008929252624512, 13.008929252624512, 14.008929252624512,
          15.008928298950195, 16.008926391601562, 17.008926391601562,
          18.008926391601562, 19.008926391601562, 20.008926391601562,
          21.008926391601562, 22.008926391601562, 23.008926391601562,
          24.008926391601562, 25.008926391601562, 26.008926391601562,
          27.008926391601562, 28.008926391601562, 29.008926391601562,
          30.00892448425293, 31.00892448425293, 32.0089225769043,
          33.0089225769043, 34.0089225769043, 35.0089225769043,
          36.0089225769043, 37.0089225769043, 38.0089225769043,
          39.0089225769043, 40.0089225769043, 41.0089225769043,
          42.0089225769043, 43.0089225769043, 44.0089225769043,
          45.0089225769043, 46.0089225769043, 47.0089225769043,
          48.0089225769043, 49.0089225769043, 50.0089225769043,
          51.0089225769043, 52.0089225769043, 53.0089225769043,
          54.0089225769043, 55.0089225769043, 56.0089225769043,
          57.0089225769043, 58.0089225769043, 59.0089225769043,
          60.0089225769043, 61.0089225769043, 62.0089225769043,
          63.0089225769043, 64.00892639160156, 65.00892639160156,
          66.00892639160156, 67.00892639160156, 68.00892639160156,
          69.00892639160156, 70.00892639160156, 71.00892639160156,
          72.00892639160156, 73.00892639160156, 74.00892639160156,
          75.00892639160156, 76.00892639160156, 77.00892639160156,
          78.00892639160156, 79.41213989257812, 79.41213989257812,
          80.41213989257812, 81.41213989257812, 82.53720092773438,
          83.7699203491211]
    wy = [309.9596862792969, 310.9596862792969, 311.9596862792969,
          312.9596862792969, 313.9596862792969, 314.9596862792969,
          315.9596862792969, 317.6001281738281, 317.6001281738281,
          318.7019958496094, 319.86895751953125, 321.0220947265625,
          322.15283203125, 323.25274658203125, 324.313720703125,
          325.3277893066406, 326.2874450683594, 327.2090759277344,
          328.050048828125, 328.78271484375, 329.3973388671875,
          329.88592529296875, 330.241943359375, 330.46075439453125,
          330.53997802734375, 330.53997802734375, 330.54052734375,
          330.5410461425781, 330.5415954589844, 330.5421142578125,
          330.54266357421875, 330.5431823730469, 330.5437316894531,
          330.54425048828125, 330.5447998046875, 330.54534912109375,
          330.5458679199219, 330.5464172363281, 330.54693603515625,
          330.5474853515625, 330.5480041503906, 330.5485534667969,
          330.549072265625, 330.54962158203125, 330.5501708984375,
          330.5506896972656, 330.5512390136719, 330.5517578125,
          330.55230712890625, 330.5528259277344, 330.5533752441406,
          330.55389404296875, 330.554443359375, 330.55499267578125,
          330.5555114746094, 330.5560607910156, 330.55657958984375,
          330.55712890625, 330.5576477050781, 330.5581970214844,
          330.5587158203125, 330.55926513671875, 330.5597839355469,
          330.5603332519531, 330.5608825683594, 330.5614013671875,
          330.56195068359375, 330.5624694824219, 330.5630187988281,
          330.56353759765625, 330.5640869140625, 330.5646057128906,
          330.5651550292969, 330.5657043457031, 330.56622314453125,
          330.5667724609375, 330.5672912597656, 330.5678405761719,
          330.568359375, 330.56890869140625, 330.5694274902344,
          330.5699768066406, 330.5705261230469, 330.571044921875,
          330.57159423828125, 330.5721130371094, 330.5726623535156,
          330.57318115234375, 330.57373046875, 330.5742492675781,
          330.5747985839844, 330.5753479003906, 330.57586669921875,
          330.576416015625, 330.5771484375, 330.5771484375, 330.57769775390625,
          330.5782165527344, 330.5586853027344, 330.4064025878906]

    ob = np.array([[139.0164317535845, 163.86427598223275], [139.0087429032337, 164.23922913387364], [139.00105405288286, 164.61418228551452], [138.99336520253206, 164.98913543715537], [138.98567635218126, 165.36408858879625], [138.97798750183046, 165.73904174043713], [138.97029865147965, 166.113994892078], [138.96260980112882, 166.4889480437189], [138.95492095077802, 166.86390119535977], [138.94723210042721, 167.23885434700065], [138.0551741610554, 181.14294132004517], [138.07841767604464, 181.07500995504824], [138.10166119103388, 181.0070785900513], [138.12490470602313, 180.93914722505437], [138.1481482210124, 180.87121586005742], [138.17139173600165, 180.8032844950605], [138.1946352509909, 180.73535313006354], [138.21787876598015, 180.6674217650666], [138.2411222809694, 180.59949040006967], [138.26436579595864, 180.53155903507272], [136.1957625179805, 185.05278938275384], [136.17782069208474, 184.95244214275644], [136.159878866189, 184.852094902759], [136.14193704029321, 184.75174766276157], [136.12399521439747, 184.65140042276417], [136.1060533885017, 184.55105318276674], [136.08811156260595, 184.45070594276933], [136.07016973671017, 184.3503587027719], [136.05222791081442, 184.25001146277447], [136.03428608491865, 184.14966422277706]])

    tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)
    # initial state
    c_speed =  0 # current speed [m/s]
    c_d = 0  # current lateral position [m]
    c_d_d = 0  # current lateral speed [m/s]
    c_d_dd = 0.0  # current latral acceleration [m/s]
    s0 = 0  # current course position

    area = 100.0  # animation area length [m]

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
