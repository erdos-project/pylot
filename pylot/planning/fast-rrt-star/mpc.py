"""
Author: Fangyu Wu
Email: fangyuwu@berkeley.edu
The code is adapted from
https://github.com/AtsushiSakai/PythonRobotics/tree/master/
PathTracking/model_predictive_speed_and_steer_control
"""

import numpy as np
import cvxpy
from cvxpy.expressions import constants
from fast_rrt_star import *
import networkx as nx
import matplotlib.pyplot as plt
import bisect
import time
import warnings
import sys

plt.rcParams['font.family'] = 'FreeSans'
plt.style.use('dark_background')
MPH2MPS = 0.44704


class CubicSpline1D:
    def __init__(self, x, y):
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
        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None
        i = self._search_index(t)
        dx = t - self.x[i]
        result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return result

    def _search_index(self, x):
        return bisect.bisect(self.x, x) - 1

    def _matrix_a(self, h):
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
        matrix_b = np.zeros(self.nx)
        for i in range(self.nx - 2):
            matrix_b[i + 1] = \
                3.0 * (self.a[i + 2] - self.a[i + 1]) / \
                h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
        return matrix_b


class CubicSpline2D:
    def __init__(self, x, y, delta_s=1):
        self.delta_s = delta_s  # [m]
        self.s = self._calc_s(x, y)
        self.sx = CubicSpline1D(self.s, x)
        self.sy = CubicSpline1D(self.s, y)

    def calc_x(self, s):
        return self.sx.calc_der0(s)

    def calc_y(self, s):
        return self.sy.calc_der0(s)

    def calc_position(self, s):
        x = self.sx.calc_der0(s)
        y = self.sy.calc_der0(s)
        return x, y

    def calc_curvature(self, s):
        dx = self.sx.calc_der1(s)
        ddx = self.sx.calc_der2(s)
        dy = self.sy.calc_der1(s)
        ddy = self.sy.calc_der2(s)
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))
        return k

    def calc_yaw(self, s):
        dx = self.sx.calc_der1(s)
        dy = self.sy.calc_der1(s)
        yaw = np.arctan2(dy, dx)
        return yaw

    def _calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = [np.sqrt(idx ** 2 + idy ** 2)
                   for (idx, idy) in zip(dx, dy)]
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s


class Env:
    def __init__(self, config):
        self.config = config
        self.boundaries = config['boundaries']
        self.static_obstacles = list(config['static_obstacles'])
        self.dynamic_obstacles = list(config['dynamic_obstacles'])

    def update(self, dynamic_obstacles):
        self.dynamic_obstacles = dynamic_obstacles


class Planner:
    def __init__(self, config, env):
        self.config = config
        self.env = env

    def recalc_vel(self, reference, vehicle, path_index):
        s_list = reference.s_list
        vel_list = [vehicle.config['max_vel']]
        s = vehicle.s
        t = vehicle.t
        for obstacle in self.env.dynamic_obstacles:
            s_next = obstacle[2] - \
                vehicle.config['length'] + \
                vehicle.config['offset']  # Approximately true when k is large
            t_next = obstacle[1] + 3  # Apply a virtual time buffer
            vel_next = (s_next - s) / (t_next - t)
            while s < s_next:
                path_index = path_index + 1
                s = s_list[path_index]
                vel_list.append(vel_next)
            t = t_next
        while s != s_list[-1]:
            path_index = path_index + 1
            s = s_list[path_index]
            vel_list.append(vehicle.config['max_vel'])
        return vel_list

    def recalc_reference(self, reference, vehicle):
        waypoints_x_list = []
        waypoints_y_list = []
        current_s = vehicle.s
        current_x = vehicle.x
        current_y = vehicle.y
        waypoints_x_list.append(current_x)
        waypoints_y_list.append(current_y)
        target_x = reference.x_list[-1]
        target_y = reference.y_list[-1]
        waypoints_x_list.append(target_x)
        waypoints_y_list.append(target_y)
        recalc_waypoints = np.asarray([waypoints_x_list, waypoints_y_list])
        recalc_spline = CubicSpline2D(
            recalc_waypoints[0, :], recalc_waypoints[1, :])
        recalc_s_list = np.arange(
            0, recalc_spline.s[-1], recalc_spline.delta_s)
        recalc_x_list = [recalc_spline.calc_x(s) for s in recalc_s_list]
        recalc_y_list = [recalc_spline.calc_y(s) for s in recalc_s_list]
        recalc_k_list = [
            recalc_spline.calc_curvature(s) for s in recalc_s_list]
        recalc_vel_list = [25 * MPH2MPS for _ in recalc_s_list]
        recalc_yaw_list = [recalc_spline.calc_yaw(s) for s in recalc_s_list]
        return {
            's_list': recalc_s_list + current_s,  # Arc distance [m]
            'x_list': recalc_x_list,  # Desired X coordinates [m]
            'y_list': recalc_y_list,  # Desired Y coordinates [m]
            'k_list': recalc_k_list,  # Curvatures [1/m]
            'vel_list': recalc_vel_list,  # Desired velocities [m/s]
            'yaw_list': recalc_yaw_list,  # Yaws [rad]
        }

    def rrt_star_path_plan(self, state_space, starting_state, target_space, obstacle_map, n_samples=200,
                           granularity=0.2,
                           d_threshold=1):
        rrt_star, rrt_star_final_state, rrt_star_cost = apply_rrt_star(state_space=state_space,
                                                                       starting_state=starting_state,
                                                                       target_space=target_space,
                                                                       obstacle_map=obstacle_map,
                                                                       d_threshold=d_threshold,
                                                                       n_samples=n_samples,
                                                                       granularity=granularity)



        path = nx.shortest_path(rrt_star, starting_state, rrt_star_final_state)
        path = np.array([[p[0], p[1]] for p in path])
        if rrt_star_cost is None:
            print("Failed to find plan.")
        else:
            print("Found")
        spline = CubicSpline2D(path[:, 0], path[:, 1])
        recalc_s_list = np.arange(
            0, spline.s[-1], spline.delta_s)
        recalc_x_list = [spline.calc_x(s) for s in recalc_s_list]
        recalc_y_list = [spline.calc_y(s) for s in recalc_s_list]
        recalc_k_list = [
            spline.calc_curvature(s) for s in recalc_s_list]
        recalc_vel_list = [25 * MPH2MPS for _ in recalc_s_list]
        recalc_yaw_list = [spline.calc_yaw(s) for s in recalc_s_list]
        return {
            's_list': recalc_s_list,  # Arc distance [m]
            'x_list': recalc_x_list,  # Desired X coordinates [m]
            'y_list': recalc_y_list,  # Desired Y coordinates [m]
            'k_list': recalc_k_list,  # Curvatures [1/m]
            'vel_list': recalc_vel_list,  # Desired velocities [m/s]
            'yaw_list': recalc_yaw_list,  # Yaws [rad]
        }

    def render(self, t, color='w'):
        for obstacle in self.env.static_obstacles:
            plt.plot(obstacle[:, 0].flatten(),
                     obstacle[:, 1].flatten(), color)
        for obstacle in self.env.dynamic_obstacles:
            if obstacle[0] <= t <= obstacle[1]:
                plt.plot(obstacle[3:4], obstacle[4:5], 'or')


class Trajectory:
    def __init__(self, t_list, s_list, x_list, y_list, k_list, vel_list,
                 yaw_list, accel_list=None, steer_list=None):
        self.t_list = list(t_list)  # Time [s]
        self.s_list = list(s_list)  # Arc distance list [m]
        self.x_list = list(x_list)  # X coordinate list [m]
        self.y_list = list(y_list)  # Y coordinate list [m]
        self.k_list = list(k_list)  # Curvature list [1/m]
        self.vel_list = list(vel_list)  # Tangential velocity list [m/s]
        self.yaw_list = list(self._normalize_yaw(yaw_list))  # Yaw list [rad]
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

    def render(self, color='w'):
        plt.plot(self.x_list, self.y_list, '-{}'.format(color))

    @staticmethod
    def _normalize_yaw(yaw_list):
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


class Vehicle:
    def __init__(self, config):
        self.config = config
        self.t = None  # Time [s]
        self.s = None  # Arc distance [m]
        self.x = None  # X coordinate [m]
        self.y = None  # Y coordinate [m]
        self.k = None  # Curvature [1/m]
        self.vel = None  # Tangential velocity [m/s]
        self.yaw = None  # Yaw [rad]
        self.accel = None  # Acceleration [m/s2]
        self.steer = None  # Steering [rad]

    def update(self, t, s, x, y, k, vel, yaw, accel, steer):
        self.t = t
        self.s = s
        self.x = x
        self.y = y
        self.k = k
        self.vel = vel
        self.yaw = yaw
        self.accel = accel
        self.steer = steer

    def get_position(self):
        return np.asarray([self.x, self.y])

    def get_state(self):
        return np.asarray([self.x, self.y, self.vel, self.yaw])

    def render(self, color='w'):
        bounding_box = np.asarray([
            [-self.config['offset'],
             (self.config['length'] - self.config['offset']),
             (self.config['length'] - self.config['offset']),
             -self.config['offset'],
             -self.config['offset']],
            [self.config['width'] / 2,
             self.config['width'] / 2,
             -self.config['width'] / 2,
             -self.config['width'] / 2,
             self.config['width'] / 2]])
        front_right_wheel = np.asarray([
            [self.config['wheel_length'],
             -self.config['wheel_length'],
             -self.config['wheel_length'],
             self.config['wheel_length'],
             self.config['wheel_length']],
            [-self.config['wheel_width'] - self.config['track'],
             -self.config['wheel_width'] - self.config['track'],
             self.config['wheel_width'] - self.config['track'],
             self.config['wheel_width'] - self.config['track'],
             -self.config['wheel_width'] - self.config['track']]])
        rear_right_wheel = np.copy(front_right_wheel)
        front_left_wheel = np.copy(front_right_wheel)
        front_left_wheel[1, :] *= -1
        rear_left_wheel = np.copy(rear_right_wheel)
        rear_left_wheel[1, :] *= -1

        rotation_yaw = np.array([[np.cos(self.yaw), np.sin(self.yaw)],
                                 [-np.sin(self.yaw), np.cos(self.yaw)]])
        rotation_steer = np.array([[np.cos(self.steer), np.sin(self.steer)],
                                   [-np.sin(self.steer), np.cos(self.steer)]])
        front_right_wheel = (front_right_wheel.T.dot(rotation_steer)).T
        front_left_wheel = (front_left_wheel.T.dot(rotation_steer)).T
        front_right_wheel[0, :] += self.config['wheelbase']
        front_left_wheel[0, :] += self.config['wheelbase']
        front_right_wheel = (front_right_wheel.T.dot(rotation_yaw)).T
        front_left_wheel = (front_left_wheel.T.dot(rotation_yaw)).T
        bounding_box = (bounding_box.T.dot(rotation_yaw)).T
        rear_right_wheel = (rear_right_wheel.T.dot(rotation_yaw)).T
        rear_left_wheel = (rear_left_wheel.T.dot(rotation_yaw)).T

        plt.plot(
            bounding_box[0, :].flatten() + self.x,
            bounding_box[1, :].flatten() + self.y,
            color
        )
        for wheel in [front_right_wheel, front_left_wheel,
                      rear_right_wheel, rear_left_wheel]:
            plt.plot(
                wheel[0, :].flatten() + self.x,
                wheel[1, :].flatten() + self.y,
                color
            )
        plt.plot(self.x, self.y, '+{}'.format(color))


class ModelPredictiveController:
    def __init__(self, config):
        self.env = Env(config['env'])
        self.planner = Planner(config['planner'], self.env)
        self.reference = Trajectory(**config['reference'])
        self.vehicle = Vehicle(config['vehicle'])
        self.path_length = len(self.reference.s_list)
        self.path_index = 0
        self.t = 0.0  # [s]
        self.step_count = 0
        initial_condition = {
            't_list': [self.t],  # Initial time [s]
            's_list': self.reference.s_list[0:1],  # Initial arc distance [m]
            'x_list': self.reference.x_list[0:1],  # Initial X coordinate [m]
            'y_list': self.reference.y_list[0:1],  # Initial Y coordinate [m]
            'k_list': self.reference.k_list[0:1],  # Initial curvature [1/m]
            'vel_list': self.reference.vel_list[0:1],  # Initial velocity [m/s]
            'yaw_list': self.reference.yaw_list[0:1],  # Initial yaw [rad]
            'accel_list': np.asarray([]),  # Initial acceleration [m/s2]
            'steer_list': np.asarray([]),  # Initial steering [rad]
        }
        self.solution = Trajectory(**initial_condition)
        self.vehicle.update(
            self.t,
            self.solution.s_list[-1],
            self.solution.x_list[-1],
            self.solution.y_list[-1],
            self.solution.k_list[-1],
            self.solution.vel_list[-1],
            self.solution.yaw_list[-1],
            None,
            None,
        )
        self.config = config['controller']
        self.num_state = 4  # Number of states: X = [x, y, vel, yaw]
        self.num_input = 2  # Number of inputs: U = [accel, steer]
        self.position = self.vehicle.get_position()
        self.goal = np.asarray(
            [self.reference.x_list[-1], self.reference.y_list[-1]])
        self.delta_t = 0.1  # [s]
        self.delta_s = self.reference.s_list[1] - self.reference.s_list[0]
        self.horizon_accel = np.zeros((self.config['horizon'], 1))  # [m/s2]
        self.horizon_steer = np.zeros((self.config['horizon'], 1))  # [rad]

    def step(self):
        self.t = self.t + self.delta_t
        self.step_count = self.step_count + 1
        # Solve control for the next step
        if not self.is_completed():
            self._update_path_index()
            reference_state, reference_steer = \
                self._retrieve_imminent_reference()
            try:
                for _ in range(self.config['max_iteration']):
                    is_converged = self._iterative_control(
                        reference_state, reference_steer)
                    if is_converged:
                        break
            except KeyboardInterrupt:  # TODO: Replace with Erdos interrupt
                warnings.warn('Control is interrupted.')
        # Update vehicle
        t = self.t
        s = self.path_index
        accel = self.horizon_accel[0]
        steer = self.horizon_steer[0]
        state = self._step_state(self.vehicle.get_state(), accel, steer)
        x = state[0]
        y = state[1]
        vel = state[2]
        yaw = state[3]
        k = self._compute_curvature(vel, accel, yaw)
        self.vehicle.update(t, s, x, y, k, vel, yaw, accel, steer)
        # Update solution
        self.solution.append(t, s, x, y, k, vel, yaw, accel, steer)

    def render(self, show_grid=False):
        plt.cla()
        self.planner.render(self.t, color='w')
        self.reference.render(color='c')
        self.solution.render(color='m')
        self.vehicle.render(color='y')
        ax = plt.gca()
        ax.set_xlim([-20, 20])
        ax.set_ylim([0, 100])
        ax.set_aspect(1)
        plt.grid(show_grid)
        plt.title('Time: {:.1f} s Speed: {:.1f} m/s'.
                  format(self.vehicle.t, self.vehicle.vel))
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.pause(0.0001)
        plt.savefig('demo_steeraround/{:05d}.png'.format(self.step_count))

    def is_completed(self):
        is_expired = self.config['expiration_time'] < self.t
        is_arrived = \
            self.config['goal_threshold'] > \
            np.linalg.norm(self.goal - self.vehicle.get_position())
        return is_expired or is_arrived

    def _update_path_index(self):
        dx = [self.vehicle.x - x for x in self.reference.x_list[
              self.path_index:self.path_index+self.config['index_horizon']]]
        dy = [self.vehicle.y - y for y in self.reference.y_list[
              self.path_index:self.path_index+self.config['index_horizon']]]
        dxy = [np.sqrt(x**2 + y**2) for x, y in zip(dx, dy)]
        self.path_index = np.argmin(dxy) + self.path_index

    def _retrieve_imminent_reference(self):
        reference_state = np.zeros((self.num_state, self.config['horizon']+1))
        reference_steer = np.zeros((1, self.config['horizon']+1))

        arc_displacement = 0.0
        for t in range(self.config['horizon'] + 1):
            offset = int(round(arc_displacement / self.delta_s))
            if (self.path_index + offset) < self.path_length:
                reference_state[0, t] = \
                    self.reference.x_list[self.path_index + offset]
                reference_state[1, t] = \
                    self.reference.y_list[self.path_index + offset]
                reference_state[2, t] = \
                    self.reference.vel_list[self.path_index + offset]
                reference_state[3, t] = \
                    self.reference.yaw_list[self.path_index + offset]
            else:
                reference_state[0, t] = \
                    self.reference.x_list[self.path_length - 1]
                reference_state[1, t] = \
                    self.reference.y_list[self.path_length - 1]
                reference_state[2, t] = \
                    self.reference.vel_list[self.path_length - 1]
                reference_state[3, t] = \
                    self.reference.yaw_list[self.path_length - 1]
            arc_displacement = \
                arc_displacement + abs(self.vehicle.vel) * self.delta_t
        return reference_state, reference_steer

    def _iterative_control(self, reference_state, reference_steer):
        is_converged = False
        predicted_state = self._predict_state()
        horizon_x, horizon_y, horizon_vel, \
            horizon_yaw, horizon_accel, horizon_steer = \
            self._control(reference_state, predicted_state, reference_steer)
        iteration_difference = \
            np.linalg.norm(self.horizon_accel - horizon_accel, ord=1) + \
            np.linalg.norm(self.horizon_steer - horizon_steer, ord=1)
        self.horizon_accel = horizon_accel
        self.horizon_steer = horizon_steer
        if iteration_difference < self.config['convergence_threshold']:
            is_converged = True
        return is_converged

    def _predict_state(self):
        predicted_state = np.zeros((self.num_state, self.config['horizon']+1))
        predicted_state[:, 0] = self.vehicle.get_state()
        state = predicted_state[:, 0]
        for accel, steer, t in zip(self.horizon_accel,
                                   self.horizon_steer,
                                   range(1, self.config['horizon']+1)):
            state = self._step_state(state, accel, steer)
            predicted_state[:, t] = state
        return predicted_state

    def _step_state(self, state, accel, steer):
        steer = np.clip(
            steer,
            self.vehicle.config['min_steer'],
            self.vehicle.config['max_steer']
        )
        state[0] = state[0] + state[2] * np.cos(state[3]) * self.delta_t
        state[1] = state[1] + state[2] * np.sin(state[3]) * self.delta_t
        state[2] = state[2] + accel * self.delta_t
        state[3] = state[3] + state[2] / self.vehicle.config['wheelbase'] * \
            np.tan(steer) * self.delta_t
        state[2] = np.clip(
            state[2],
            self.vehicle.config['min_vel'],
            self.vehicle.config['max_vel']
        )
        return state

    def _control(self, reference_state, predicted_state, reference_steer):
        x = cvxpy.Variable((self.num_state, self.config['horizon']+1))
        u = cvxpy.Variable((self.num_input, self.config['horizon']))

        cost = constants.Constant(0.0)
        constraints = []

        for t in range(self.config['horizon']):
            cost += cvxpy.quad_form(u[:, t], self.config['R'])

            if t != 0:
                cost += cvxpy.quad_form(
                    reference_state[:, t] - x[:, t], self.config['Q'])

            matrix_a, matrix_b, matrix_c = self._linearized_model_matrix(
                predicted_state[2, t],
                predicted_state[3, t],
                reference_steer[0, t]
            )
            constraints += [x[:, t + 1] ==
                            matrix_a * x[:, t] + matrix_b * u[:, t] + matrix_c]

            if t < (self.config['horizon'] - 1):
                cost += cvxpy.quad_form(
                    u[:, t + 1] - u[:, t], self.config['Rd'])
                constraints += [
                    cvxpy.abs(u[1, t + 1] - u[1, t]) <=
                    self.vehicle.config['max_steer_speed'] * self.delta_t]

        cost += cvxpy.quad_form(
            reference_state[:, self.config['horizon']] -
            x[:, self.config['horizon']], self.config['Qf'])

        constraints += [x[:, 0] == self.vehicle.get_state()]
        constraints += [x[2, :] <= self.vehicle.config['max_vel']]
        constraints += [x[2, :] >= self.vehicle.config['min_vel']]
        constraints += [u[0, :] <= self.vehicle.config['max_accel']]
        constraints += [u[0, :] >= self.vehicle.config['min_accel']]
        constraints += [u[1, :] <= self.vehicle.config['max_steer']]
        constraints += [u[1, :] >= self.vehicle.config['min_steer']]

        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)

        if prob.status == cvxpy.OPTIMAL or \
                prob.status == cvxpy.OPTIMAL_INACCURATE:
            horizon_x = np.array(x.value[0, :]).flatten()
            horizon_y = np.array(x.value[1, :]).flatten()
            horizon_vel = np.array(x.value[2, :]).flatten()
            horizon_yaw = np.array(x.value[3, :]).flatten()
            horizon_accel = np.array(u.value[0, :]).flatten()
            horizon_steer = np.array(u.value[1, :]).flatten()
        else:
            raise RuntimeError('Planning problem is infeasible.')

        return horizon_x, horizon_y, horizon_vel, \
            horizon_yaw, horizon_accel, horizon_steer

    def _linearized_model_matrix(self, vel, yaw, steer):
        matrix_a = np.zeros((self.num_state, self.num_state))
        matrix_a[0, 0] = 1.0
        matrix_a[1, 1] = 1.0
        matrix_a[2, 2] = 1.0
        matrix_a[3, 3] = 1.0
        matrix_a[0, 2] = self.delta_t * np.cos(yaw)
        matrix_a[0, 3] = -self.delta_t * vel * np.sin(yaw)
        matrix_a[1, 2] = self.delta_t * np.sin(yaw)
        matrix_a[1, 3] = self.delta_t * vel * np.cos(yaw)
        matrix_a[3, 2] = \
            self.delta_t * np.tan(steer) / self.vehicle.config['wheelbase']

        matrix_b = np.zeros((self.num_state, self.num_input))
        matrix_b[2, 0] = self.delta_t
        matrix_b[3, 1] = self.delta_t * vel / \
            (self.vehicle.config['wheelbase'] * np.cos(steer)**2)

        matrix_c = np.zeros(self.num_state)
        matrix_c[0] = self.delta_t * vel * np.sin(yaw) * yaw
        matrix_c[1] = - self.delta_t * vel * np.cos(yaw) * yaw
        matrix_c[3] = - self.delta_t * vel * steer / \
            (self.vehicle.config['wheelbase'] * np.cos(steer)**2)
        return matrix_a, matrix_b, matrix_c

    @staticmethod
    def _compute_curvature(vel, accel, yaw):
        dx = vel * np.tan(yaw)
        ddx = accel * np.tan(yaw)
        dy = vel * np.tan(yaw)
        ddy = accel * np.tan(yaw)
        return (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))


if __name__ == '__main__':
    waypoints = np.asarray([
        [0, 0, 0, 0],
        [0, 60, 70, 100]
    ])
    spline = CubicSpline2D(waypoints[0, :], waypoints[1, :])
    reference_s_list = np.arange(0, spline.s[-1], spline.delta_s)
    reference_x_list = [spline.calc_x(s) for s in reference_s_list]
    reference_y_list = [spline.calc_y(s) for s in reference_s_list]
    reference_k_list = [spline.calc_curvature(s) for s in reference_s_list]
    reference_vel_list = [25 * MPH2MPS for s in reference_s_list]
    reference_yaw_list = [spline.calc_yaw(s) for s in reference_s_list]

    global_config = {
        'env': {
            'boundaries': None,
            'static_obstacles': [np.asarray([
                [-8, 50],
                [-8, 60],
                [-3, 60],
                [-3, 50],
                [-8, 50]
            ])],
            # Each dynamic obstacle is formatted as [s, x, y, t_enter, t_exit]
            'dynamic_obstacles': [],
        },
        'planner': {},
        'reference': {
            't_list': [],  # Time [s]
            's_list': reference_s_list,  # Arc distance [m]
            'x_list': reference_x_list,  # Desired X coordinates [m]
            'y_list': reference_y_list,  # Desired Y coordinates [m]
            'k_list': reference_k_list,  # Curvatures [1/m]
            'vel_list': reference_vel_list,  # Desired velocities [m/s]
            'yaw_list': reference_yaw_list,  # Yaws [rad]
        },
        'vehicle': {
            'length': 4.5,
            'width': 2.0,
            'offset': 1.0,
            'wheel_length': 0.3,
            'wheel_width': 0.2,
            'track': 0.7,
            'wheelbase': 2.5,
            'max_steer': np.deg2rad(45.0),
            'min_steer': np.deg2rad(-45.0),
            'max_steer_speed': np.deg2rad(30.0),
            'min_steer_speed': np.deg2rad(0.0),
            'max_vel': 25 * MPH2MPS,
            'min_vel': -5 * MPH2MPS,
            'max_accel': 2.0,
            'min_accel': -8.0,
        },
        'controller': {
            'R': np.diag([0.01, 0.01]),  # Input cost
            'Rd': np.diag([0.01, 1.0]),  # Input difference cost
            'Q': np.diag([1.0, 1.0, 0.5, 0.5]),  # State cost
            'Qf': np.diag([1.0, 1.0, 0.5, 0.5]),  # Terminal state cost
            'goal_threshold': 1.0,  # Threshold for goal test [m]
            'expiration_time': 100.0,  # Expiration time [s]
            'max_iteration': 5,  # Max step iterations
            'convergence_threshold': 0.1,  # Threshold for convergence test
            'horizon': 5,  # Horizon
            'index_horizon': 5,  # Index horizon
        },
    }
    mpc = ModelPredictiveController(global_config)
    reference = ModelPredictiveController(global_config).reference
    obstacle_map = {1: ((-8, 50), (7, 12))}

    dangerous_kid = [4.5, 8.5, 65.0, 0.0, 65.0]  # [t_enter, t_exit, s, x, y]
    while not mpc.is_completed():
        start = time.time()
        mpc.step()
        end = time.time()

        # detect pedestrian
        if mpc.vehicle.t - dangerous_kid[0] >= 0 and np.abs(mpc.vehicle.t - dangerous_kid[0]) < 2.5:
            mpc.env.update([dangerous_kid])
            obstacle_map = {0: ((-1.5, 64.5), (3, 3)), 1: ((-8, 50), (7, 12))}

            starting_state = mpc.vehicle.get_position()
            starting_state = (starting_state[0], starting_state[1])
            target_state = (mpc.reference.x_list[-1], mpc.reference.y_list[-1])
            target_space = ((-0.5, starting_state[1] + 10), (1.0, 1.0))
            state_space = ((-5, starting_state[1]-1), (5, starting_state[1] + 11))
            recalc_config = mpc.planner.rrt_star_path_plan(state_space, starting_state, target_space, obstacle_map)

            mpc.reference.s_list[mpc.path_index:] = recalc_config['s_list']
            mpc.reference.s_list[mpc.path_index:] = recalc_config['s_list']
            mpc.reference.x_list[mpc.path_index:] = recalc_config['x_list']
            mpc.reference.y_list[mpc.path_index:] = recalc_config['y_list']
            mpc.reference.k_list[mpc.path_index:] = recalc_config['k_list']
            mpc.reference.vel_list[mpc.path_index:] = recalc_config['vel_list']
            mpc.reference.yaw_list[mpc.path_index:] = recalc_config['yaw_list']
        else:
            recalc_config = mpc.planner.recalc_reference(reference, mpc.vehicle)

            mpc.reference.s_list[mpc.path_index:] = recalc_config['s_list']
            mpc.reference.s_list[mpc.path_index:] = recalc_config['s_list']
            mpc.reference.x_list[mpc.path_index:] = recalc_config['x_list']
            mpc.reference.y_list[mpc.path_index:] = recalc_config['y_list']
            mpc.reference.k_list[mpc.path_index:] = recalc_config['k_list']
            mpc.reference.vel_list[mpc.path_index:] = recalc_config['vel_list']
            mpc.reference.yaw_list[mpc.path_index:] = recalc_config['yaw_list']




        mpc.render()
    sys.stdout.write('\nDone.\n')
