import numpy as np

import bisect

global_config = {
    'vehicle': { # configured for lincoln mkz
        'length': 4.93,
        'width': 1.86,
        'offset': 1.0,
        'wheel_length': 0.3,
        'wheel_width': 0.2,
        'track': 0.7,
        'wheelbase': 2.85,
        'max_steer': np.deg2rad(37.5),
        'min_steer': np.deg2rad(-37.5),
        'max_steer_speed': np.deg2rad(22.5),
        'min_steer_speed': np.deg2rad(-22.5),
        'max_vel': 20,
        'min_vel': 0,
        'max_accel': 3.0,
        'min_accel': -5.0,
    },
    'controller': {
        'R': np.diag([0.01, 0.10]),  # Input cost
        'Rd': np.diag([0.01, 1.0]),  # Input difference cost
        'Q': np.diag([1.0, 1.0, 0.01, 0.01]),  # State cost
        'Qf': np.diag([1.0, 1.0, 0.01, 0.01]),  # Terminal state cost
        'goal_threshold': 1.0,  # Threshold for goal test [m]
        'expiration_time': 100.0,  # Expiration time [s]
        'max_iteration': 5,  # Max step iterations
        'convergence_threshold': 0.1,  # Threshold for convergence test
        'horizon': 5,  # Horizon
        'index_horizon': 5,  # Index horizon
    },
}


def compute_curvature(vel, accel, yaw):
    dx = vel * np.tan(yaw)
    ddx = accel * np.tan(yaw)
    dy = vel * np.tan(yaw)
    ddy = accel * np.tan(yaw)
    return (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2) ** (3 / 2))


def normalize_yaw(yaw_list):
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


def zero_to_2_pi(angle):
    return (angle + 360) % 360


class CubicSpline1D:
    """
    1-dimensional cubic spline class. For technical details see: http://mathworld.wolfram.com/CubicSpline.html
    """

    def __init__(self, x, y):
        """
        Construct the 1-dimensional cubic spline.
        :param x: list
            List of x values.
        :param y: list
            List of y values.
        """
        self.a = [item for item in y]
        self.b, self.c, self.d, self.w = [], [], [], []
        self.x = x
        self.y = y
        self.nx = len(x)
        h = np.diff(x)
        matrix_a = self._matrix_a(h)
        matrix_b = self._matrix_b(h)
        self.c = np.linalg.solve(matrix_a, matrix_b)
        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
            tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * \
                 (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)

    def calc_der0(self, t):
        """
        Calculate the 1st derivative evaluated at t.

        :param t: float
            Position along the 1-dimensional spline.
        :return: float
            1st derivative evaluated at t.
        """
        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None
        i = self._search_index(t)
        dx = t - self.x[i]
        result = \
            self.a[i] + self.b[i] * dx + \
            self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0
        return result

    def calc_der1(self, t):
        """
        Calculate the 2nd derivative evaluated at t.

        :param t: float
            Position along the 1-dimensional spline.
        :return: float
            2nd derivative evaluated at t.
        """
        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None
        i = self._search_index(t)
        dx = t - self.x[i]
        result = \
            self.b[i] + 2.0 * self.c[i] * dx + \
            3.0 * self.d[i] * dx ** 2.0
        return result

    def calc_der2(self, t):
        """
        Calculate the 3rd derivative evaluated at t.

        :param t: float
            Position along the 1-dimensional spline.
        :return: float
            3rd derivative evaluated at t.
        """
        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None
        i = self._search_index(t)
        dx = t - self.x[i]
        result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return result

    def _search_index(self, x):
        """
        Search the spline for index closest to x.

        :param x: float
            Position along the 1-dimensional spline.
        :return: int
            Index closest to x.
        """
        return bisect.bisect(self.x, x) - 1

    def _matrix_a(self, h):
        """
        Create the constants matrix a used in spline construction.

        :param h: np.ndarray
            List of deltas between values.
        :return: np.ndarray
            Constants matrix.
        """
        matrix_a = np.zeros((self.nx, self.nx))
        matrix_a[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                matrix_a[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            matrix_a[i + 1, i] = h[i]
            matrix_a[i, i + 1] = h[i]

        matrix_a[0, 1] = 0.0
        matrix_a[self.nx - 1, self.nx - 2] = 0.0
        matrix_a[self.nx - 1, self.nx - 1] = 1.0
        return matrix_a

    def _matrix_b(self, h):
        """
        Create the 1st derivative matrix b used in spline construction.

        :param h: np.ndarray
            List of deltas between values
        :return: np.ndarray
            1st derivative matrix
        """
        matrix_b = np.zeros(self.nx)
        for i in range(self.nx - 2):
            matrix_b[i + 1] = \
                3.0 * (self.a[i + 2] - self.a[i + 1]) / \
                h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
        return matrix_b


class CubicSpline2D:
    """
    2-dimensional cubic spline class. For technical details see: http://mathworld.wolfram.com/CubicSpline.html
    """

    def __init__(self, x, y, delta_s=1):
        """
        Construct the 2-dimensional cubic spline.

        :param x: list
            List of x values.
        :param y: list
            List of y values
        :param delta_s: float.
            Distance between interpolated points.
        """
        self.delta_s = delta_s  # [m]
        self.s = self._calc_s(x, y)
        self.sx = CubicSpline1D(self.s, x)
        self.sy = CubicSpline1D(self.s, y)

    def calc_x(self, s):
        """
        Calculate the x position along the spline at given s.

        :param s: float
            s position along the 2-dimensional spline.
        :return: float
            x position along the 2-dimensional spline.
        """
        return self.sx.calc_der0(s)

    def calc_y(self, s):
        """
        Calculate the y position along the spline at given s.

        :param s: float
            s position along the 2-dimensional spline.
        :return: float
            y position along the 2-dimensional spline.
        """
        return self.sy.calc_der0(s)

    def calc_position(self, s):
        """
        Calculate the x, y position along the spline at given s.

        :param s: float
            s position along the 2-dimensional spline.
        :return: (float, float)
            x, y position along the 2-dimensional spline.
        """
        x = self.sx.calc_der0(s)
        y = self.sy.calc_der0(s)
        return x, y

    def calc_curvature(self, s):
        """
        Calculate the curvature along the spline at given s.

        :param s: float
            s position along the 2-dimensional spline.
        :return: float
            Curvature along the 2-dimensional spline.
        """
        dx = self.sx.calc_der1(s)
        ddx = self.sx.calc_der2(s)
        dy = self.sy.calc_der1(s)
        ddy = self.sy.calc_der2(s)
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2) ** (3 / 2))
        return k

    def calc_yaw(self, s):
        """
        Calculate the yaw in radians along the spline at given s.

        :param s: float
            s position along the 2-dimensional spline.
        :return: float
            Yaw along the 2-dimensional spline.
        """
        dx = self.sx.calc_der1(s)
        dy = self.sy.calc_der1(s)
        yaw = np.arctan2(dy, dx)
        return yaw

    def _calc_s(self, x, y):
        """
        Calculate the s values for interpolation given x, y.

        :param x: list
            List of x values.
        :param y: list
            List of y values.
        :return: np.ndarray
            List of s values for interpolation.
        """
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = [np.sqrt(idx ** 2 + idy ** 2)
                   for (idx, idy) in zip(dx, dy)]
        s = [0]
        s.extend(np.cumsum(self.ds))
        s = np.unique(s)
        return s


class Vehicle:
    def __init__(self, config):
        self.config = config
        self.time = None  # Time [s]
        self.distance = None  # Arc distance [m]
        self.x = None  # X coordinate [m]
        self.y = None  # Y coordinate [m]
        self.curvature = None  # Curvature [1/m]
        self.vel = None  # Tangential velocity [m/s]
        self.yaw = None  # Yaw [rad]
        self.accel = None  # Acceleration [m/s2]
        self.steer = None  # Steering [rad]

    def update(self, time, distance, x, y, curvature, vel, yaw, accel, steer):
        self.time = time
        self.distance = distance
        self.x = x
        self.y = y
        self.curvature = curvature
        self.vel = vel
        self.yaw = yaw
        self.accel = accel
        self.steer = steer

    def get_position(self):
        return np.asarray([self.x, self.y])

    def get_state(self):
        return np.asarray([self.x, self.y, self.vel, self.yaw])


class Trajectory:
    def __init__(self, t_list, s_list, x_list, y_list, k_list, vel_list,
                 yaw_list, accel_list=None, steer_list=None):
        self.t_list = list(t_list)  # Time [s]
        self.s_list = list(s_list)  # Arc distance list [m]
        self.x_list = list(x_list)  # X coordinate list [m]
        self.y_list = list(y_list)  # Y coordinate list [m]
        self.k_list = list(k_list)  # Curvature list [1/m]
        self.vel_list = list(vel_list)  # Tangential velocity list [m/s]
        self.yaw_list = list(normalize_yaw(yaw_list))  # Yaw list [rad]
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