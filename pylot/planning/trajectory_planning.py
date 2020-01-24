import math
import numpy as np

EXPECTED_JERK_IN_ONE_SEC = 2  # m/s/s
EXPECTED_ACC_IN_ONE_SEC = 1  # m/s
MAX_ACCELERATION_THRESHOLD = 10  # m/s/s
MAX_JERK_THRESHOLD = 10  # m/s/s/s


class PlanningObstaclePrediction(object):
    """Non-ego obstacles move with constant acceleration.
    """
    def __init__(self, s, v_s, acc_s, d, v_d, acc_d):
        self.s = s
        self.v_s = v_s
        self.acc_s = acc_s
        self.d = d
        self.v_d = v_d
        self.acc_d = acc_d

    def state_at(self, t):
        s_t = self.s + (self.v_s * t) + self.acc_s * t**2 / 2.0
        v_s_t = self.v_s + self.acc_s * t
        d_t = self.d + (self.v_d * t) + self.acc_d * t**2 / 2.0
        v_d_t = self.v_d + self.acc_d * t
        return s_t, v_s_t, d_t, v_d_t


def logistic(x):
    """
    Returns a value between 0 and 1 for x in the range [0, inf] and -1 to 1
    for x in the range [-inf, inf].
    """
    return 2.0 / (1 + math.exp(-x)) - 1.0


def get_polynomial_func(coefficients):
    """ Returns a function that computes the polynomial. """
    def f(x):
        total = 0.0
        for i, coef in enumerate(coefficients):
            total += coef * x**i
        return total

    return f


def differentiate_polynomial(coefficients):
    """Calculates the derivative of a polynomial and returns
    the corresponding coefficients.
    """
    new_coeffs = []
    for deg, prev_coef in enumerate(coefficients[1:]):
        new_coeffs.append((deg + 1) * prev_coef)
    return new_coeffs


def get_nth_derivative_for_polynomial(coeffs, n):
    for i in range(n):
        coeffs = differentiate_polynomial(coeffs)
    return get_polynomial_func(coeffs)


def get_min_distance_to_any_obstacle(s_coeffs, d_coeffs, duration, obstacles):
    """ Returns the minimum future distance the ego-vehicle will be
    from any obstacle if the ego follows the given trajectory.

    Args:
        s_coeffs: Coefficients of ego's trajectory s polynomial.
        d_coeffs: Coefficients of ego's trajectory d polynomial.
        duration: Duration of the movement.
        obstacles: A list of objects describing obstacles state and future
            positions.
    """
    min_dist = 100000000
    for obstacle in obstacles:
        dist = get_min_future_obstacle_distance(s_coeffs, d_coeffs, duration,
                                                obstacle)
        min_dist = min(min_dist, dist)
    return min_dist


def get_min_future_obstacle_distance(s_coeffs, d_coeffs, duration, obstacle):
    """ Returns the minimum future distance the ego-vehicle will be
    from the obstacle if the ego follows the given trajectory.

    Args:
        s_coeffs: Coefficients of ego's trajectory s polynomial.
        d_coeffs: Coefficients of ego's trajectory d polynomial.
        duration: Duration of the movement.
        obstacle: An object describing the obstacle state and future
            positions.
    """
    min_dist = 100000000
    s_func = get_polynomial_func(s_coeffs)
    d_func = get_polynomial_func(d_coeffs)
    # Discretize time.
    for percentage in range(100):
        t = float(percentage) / 100 * duration
        s_t = s_func(t)
        d_t = d_func(t)
        obs_s_t, _, obs_d_t, _ = obstacle.state_at(t)
        dist = math.sqrt((s_t - obs_s_t)**2 + (d_t - obs_d_t)**2)
        min_dist = min(min_dist, dist)
    return min_dist


def jerk_minimizing_trajectory(s_initial, v_initial, acc_initial, s_final,
                               v_final, acc_final, duration):
    """ Computes the jerk minimization trajectory.

    Args:
        s_initial: Initial vehicle position.
        v_initial: Initial vehicle velocity.
        acc_initial: Initial vehicle acceleration.
        s_final: Final vehicle position.
        v_final: Final vehicle velocity.
        acc_final: Final vehicle accleration.
        duration: The duration over which the maneuver should occur.
    Returns:
        A list of 6 elements, each corresponding to a coefficient in the
        polynomial:
        s(t) = a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5
    """
    matrix_a = np.array([[duration**3, duration**4, duration**5],
                         [3 * duration**2, 4 * duration**3, 5 * duration**4],
                         [6 * duration, 12 * duration**2, 20 * duration**3]])

    matrix_b = np.array([[
        s_final -
        (s_initial + v_initial * duration + 0.5 * acc_initial * duration**2)
    ], [v_final - (v_initial + acc_initial * duration)],
                         [acc_final - acc_initial]])

    matrix_c = math.inv(matrix_a) * matrix_b

    a0 = s_initial
    a1 = v_initial
    a2 = 0.5 * acc_initial
    a3 = matrix_c[0]
    a4 = matrix_c[1]
    a5 = matrix_c[2]
    return [a0, a1, a2, a3, a4, a5]


def max_jerk_cost(s_coeffs, d_coeffs, duration):
    jerk_func = get_nth_derivative_for_polynomial(d_coeffs, 3)
    max_jerk = max([
        jerk_func(duration * float(percentage) / 100.0)
        for percentage in range(100)
    ],
                   key=abs)
    # Step function.
    if max_jerk > MAX_JERK_THRESHOLD:
        return 1
    else:
        return 0


def total_jerk_cost(s_coeffs, d_coeffs, duration):
    jerk_func = get_nth_derivative_for_polynomial(d_coeffs, 3)
    total_jerk = 0.0
    dt = float(duration) / 100.0
    for percentage in range(100):
        future_time = dt * float(percentage)
        total_jerk += abs(jerk_func(future_time) * dt)
    jerk_per_second = total_jerk / duration
    return logistic(jerk_per_second / EXPECTED_JERK_IN_ONE_SEC)


def max_acceleration_cost(s_coeffs, d_coeffs, duration):
    acc_func = get_nth_derivative_for_polynomial(s_coeffs, 3)
    max_acc = max([
        acc_func(duration * float(percentage) / 100.0)
        for percentage in range(100)
    ],
                  key=abs)
    # Step function.
    if max_acc > MAX_ACCELERATION_THRESHOLD:
        return 1
    else:
        return 0


def total_acceleration_cost(s_coeffs, d_coeffs, duration):
    acc_func = get_nth_derivative_for_polynomial(s_coeffs, 3)
    total_acc = 0.0
    dt = float(duration) / 100.0
    for percentage in range(100):
        future_time = dt * float(percentage)
        total_acc += abs(acc_func(future_time) * dt)
    acc_per_second = total_acc / float(duration)
    return logistic(acc_per_second / EXPECTED_ACC_IN_ONE_SEC)


def efficiency_cost(s_coeffs, d_coeffs, duration):
    """ Cost of slow average speed trajectories."""
    return 0


def exceeds_speed_limit_cost(s_coeffs, d_coeffs, duration):
    return 0


def stays_on_road_cost(s_coeffs, d_coeffs, duration):
    return 0


def obstacle_collision_cost(s_coeffs, d_coeffs, duration):
    """ Binary cost function which penalizes collisions."""
    return 0


def too_close_to_obstacle_cost(s_coeffs, d_coeffs, duration):
    """ Cost of getting too close to other obstacles."""
    return 0


def time_diff_cost(s_coeffs, d_coeffs, duration):
    """ Cost of trajectories that span a duration which is longer or
    shorter than the duration requested."""
    return 0


def s_diff_cost(s_coeffs, d_coeffs, duration):
    """ Cost of trajectories whose s coordinate differ from the goal s."""
    return 0


def d_diff_cost(s_coeffs, d_coeffs, duration):
    """ Cost of trajectories whose d coordinate differ from the goal d."""
    return 0


WEIGHTED_COST_FUNCTIONS = [
    (time_diff_cost, 2 * 10),
    (s_diff_cost, 5 * 10),
    (d_diff_cost, 5 * 10),
    (efficiency_cost, 2 * 10),
    (max_jerk_cost, 4 * 10),
    (total_jerk_cost, 3 * 10),
    (obstacle_collision_cost, 10**3),
    (too_close_to_obstacle_cost, 10**2),
    (max_acceleration_cost, 3 * 10),
    (total_acceleration_cost, 2 * 10),
]


def find_best_trajectory(s_initial, v_s_initial, acc_s_initial, d_initial,
                         v_d_initial, acc_d_initial, duration, goals):
    min_traj_cost = 10**9
    best_trajectory = None
    # 1) Generate trajectories.
    trajectories = []
    for (s_final, v_s_final, acc_s_final, d_final, v_d_final,
         acc_d_final) in goals:
        s_coeffs = jerk_minimizing_trajectory(s_initial, v_s_initial,
                                              acc_s_initial, s_final,
                                              v_s_final, acc_s_final, duration)
        d_coeffs = jerk_minimizing_trajectory(d_initial, v_d_initial,
                                              acc_d_initial, d_final,
                                              v_d_final, acc_d_final, duration)
        trajectories.append((s_coeffs, d_coeffs, duration))
    # 2) Find the best trajectory.
    for (s_coeffs, d_coeffs, duration) in trajectories:
        traj_cost = calculate_trajectory_cost(s_coeffs, d_coeffs, duration)
        if traj_cost < min_traj_cost:
            min_traj_cost = traj_cost
            best_trajectory = (s_coeffs, d_coeffs, duration)
    return best_trajectory


def calculate_trajectory_cost(s_coeffs, d_coeffs, duration):
    total_cost = 0.0
    for (cost_func, weight) in WEIGHTED_COST_FUNCTIONS:
        total_cost += cost_func(s_coeffs, d_coeffs, duration) * weight
    return total_cost
