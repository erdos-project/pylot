"""
Author: Fangyu Wu, Edward Fang
Email: fangyuwu@berkeley.edu, edward.fang@berkeley.edu

The code is adapted from https://github.com/AtsushiSakai/PythonRobotics/tree/
master/PathTracking/model_predictive_speed_and_steer_control.
Its author is Atsushi Sakai.

Reference Papers:
- [Kinematic and Dynamic Vehicle Models for Autonomous Driving Control Design]
(https://borrelli.me.berkeley.edu/pdfpub/IV_KinematicMPC_jason.pdf)
"""
import cvxpy
from cvxpy.expressions import constants

import numpy as np

from pylot.control.mpc.utils import Trajectory, Vehicle, compute_curvature


class ModelPredictiveController:
    def __init__(self, config):
        self.reference = Trajectory(**config['reference'])
        self.vehicle = Vehicle(config['vehicle'])
        self.path_length = len(self.reference.s_list)
        self.path_index = 0
        self.t = 0.0  # [s]
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
        """
        Solve the mpc problem and step the vehicle forward in time with solved
        controls.

        :return: None
        """
        self.t = self.t + self.delta_t
        # Solve control for the next step
        self._update_path_index()
        reference_state, reference_steer = \
            self._retrieve_imminent_reference()

        for _ in range(self.config['max_iteration']):
            is_converged = self._iterative_control(reference_state,
                                                   reference_steer)
            if is_converged:
                break

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
        k = compute_curvature(vel, accel, yaw)
        self.vehicle.update(t, s, x, y, k, vel, yaw, accel, steer)

        # Update solution
        self.solution.append(t, s, x, y, k, vel, yaw, accel, steer)

    def _update_path_index(self):
        """
        Update the index along the path where the vehicle lies.

        :return: None
        """
        dx = [
            self.vehicle.x - x
            for x in self.reference.x_list[self.path_index:self.path_index +
                                           self.config['index_horizon']]
        ]
        dy = [
            self.vehicle.y - y
            for y in self.reference.y_list[self.path_index:self.path_index +
                                           self.config['index_horizon']]
        ]
        dxy = [np.sqrt(x**2 + y**2) for x, y in zip(dx, dy)]
        self.path_index = np.argmin(dxy) + self.path_index

    def _retrieve_imminent_reference(self):
        """
        Retrieve the reference state and reference steer in the imminent
        horizon.

        :return: reference state and reference steer
        """
        reference_state = np.zeros(
            (self.num_state, self.config['horizon'] + 1))
        reference_steer = np.zeros((1, self.config['horizon'] + 1))

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
        """
        Helper function to iteratively optimize mpc output.

        :param reference_state: np.array of reference states
        :param reference_steer: np.array of reference steers
        :return: convergence and optimality as boolean
        """
        is_converged = False
        predicted_state = self._predict_state()
        horizon_x, horizon_y, horizon_vel, horizon_yaw, horizon_accel, \
            horizon_steer, solved = \
            self._control(reference_state, predicted_state,
                          reference_steer)
        iteration_difference = \
            np.linalg.norm(self.horizon_accel - horizon_accel, ord=1) + \
            np.linalg.norm(self.horizon_steer - horizon_steer, ord=1)
        self.horizon_accel = horizon_accel
        self.horizon_steer = horizon_steer
        if iteration_difference < self.config['convergence_threshold']:
            is_converged = True
        return is_converged and solved

    def _predict_state(self):
        """
        Propogate controls through the horizon to predict the next state.

        :return: predicted state at end of horizon
        """
        predicted_state = np.zeros(
            (self.num_state, self.config['horizon'] + 1))
        predicted_state[:, 0] = self.vehicle.get_state()
        state = predicted_state[:, 0]
        for accel, steer, t in zip(self.horizon_accel, self.horizon_steer,
                                   range(1, self.config['horizon'] + 1)):
            state = self._step_state(state, accel, steer)
            predicted_state[:, t] = state
        return predicted_state

    def _step_state(self, state, accel, steer):
        """
        Calculate the next state given current state, accel, and steer input.

        :param state: current state as np.array
        :param accel: accel in m/s^2
        :param steer: steer in radians
        :return: next state
        """
        steer = np.clip(steer, self.vehicle.config['min_steer'],
                        self.vehicle.config['max_steer'])
        state[0] = state[0] + state[2] * np.cos(state[3]) * self.delta_t
        state[1] = state[1] + state[2] * np.sin(state[3]) * self.delta_t
        state[2] = state[2] + accel * self.delta_t
        state[3] = state[3] + state[2] / self.vehicle.config['wheelbase'] * \
            np.tan(steer) * self.delta_t
        state[2] = np.clip(state[2], self.vehicle.config['min_vel'],
                           self.vehicle.config['max_vel'])
        return state

    def _control(self, reference_state, predicted_state, reference_steer):
        """
        Solve the MPC control problem.

        :param reference_state: np.array of reference states
        :param predicted_state: np.array of predicted states obtained using
            propogated controls
        :param reference_steer: np.array of reference steering
        :return:
        """
        # intialize problem
        x = cvxpy.Variable((self.num_state, self.config['horizon'] + 1))
        u = cvxpy.Variable((self.num_input, self.config['horizon']))
        cost = constants.Constant(0.0)
        constraints = []

        # iterate over the horizon
        for t in range(self.config['horizon']):
            cost += cvxpy.quad_form(u[:, t], self.config['R'])

            if t != 0:
                cost += cvxpy.quad_form(reference_state[:, t] - x[:, t],
                                        self.config['Q'])

            matrix_a, matrix_b, matrix_c = self._linearized_model_matrix(
                predicted_state[2, t], predicted_state[3, t],
                reference_steer[0, t])
            constraints += [
                x[:,
                  t + 1] == matrix_a @ x[:, t] + matrix_b @ u[:, t] + matrix_c
            ]

            if t < (self.config['horizon'] - 1):
                cost += cvxpy.quad_form(u[:, t + 1] - u[:, t],
                                        self.config['Rd'])
                constraints += [
                    cvxpy.abs(u[1, t + 1] - u[1, t]) <=
                    self.vehicle.config['max_steer_speed'] * self.delta_t
                ]

        # set the cost
        cost += cvxpy.quad_form(
            reference_state[:, self.config['horizon']] -
            x[:, self.config['horizon']], self.config['Qf'])

        # set the constraints
        constraints += [x[:, 0] == self.vehicle.get_state()]
        constraints += [x[2, :] <= self.vehicle.config['max_vel']]
        constraints += [x[2, :] >= self.vehicle.config['min_vel']]
        constraints += [u[0, :] <= self.vehicle.config['max_accel']]
        constraints += [u[0, :] >= self.vehicle.config['min_accel']]
        constraints += [u[1, :] <= self.vehicle.config['max_steer']]
        constraints += [u[1, :] >= self.vehicle.config['min_steer']]

        # solve the problem
        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)

        # keep track of optimality
        solved = False
        if prob.status == cvxpy.OPTIMAL or \
                prob.status == cvxpy.OPTIMAL_INACCURATE:
            solved = True

        # return solution
        horizon_x = np.array(x.value[0, :]).flatten()
        horizon_y = np.array(x.value[1, :]).flatten()
        horizon_vel = np.array(x.value[2, :]).flatten()
        horizon_yaw = np.array(x.value[3, :]).flatten()
        horizon_accel = np.array(u.value[0, :]).flatten()
        horizon_steer = np.array(u.value[1, :]).flatten()

        return horizon_x, horizon_y, horizon_vel, horizon_yaw, horizon_accel, \
            horizon_steer, solved

    def _linearized_model_matrix(self, vel, yaw, steer):
        """
        Return the transition matrices linearized around vel, yaw, steer.
        Transition matrices A, B, C are of the form:
            Ax_t + Bu_t + C = x_t+1

        :param vel: reference velocity in m/s
        :param yaw: reference yaw in radians
        :param steer: reference steer in radians
        :return: transition matrices
        """
        # state matrix
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

        # input matrix
        matrix_b = np.zeros((self.num_state, self.num_input))
        matrix_b[2, 0] = self.delta_t
        matrix_b[3, 1] = self.delta_t * vel / \
            (self.vehicle.config['wheelbase'] * np.cos(steer)**2)

        # constant matrix
        matrix_c = np.zeros(self.num_state)
        matrix_c[0] = self.delta_t * vel * np.sin(yaw) * yaw
        matrix_c[1] = -self.delta_t * vel * np.cos(yaw) * yaw
        matrix_c[3] = - self.delta_t * vel * steer / \
            (self.vehicle.config['wheelbase'] * np.cos(steer)**2)

        return matrix_a, matrix_b, matrix_c
