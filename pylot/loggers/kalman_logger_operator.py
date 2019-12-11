from collections import deque
from erdos.op import Op
from erdos.utils import setup_csv_logging, setup_logging, time_epoch_ms
from pylot.map.hd_map import HDMap
from pylot.simulation.utils import kalman_step, steer2rad, get_transition_matrix

import numpy as np

import carla
import collections
import itertools
import math
import pylot.control.utils
import pylot.utils
import threading


class KalmanLoggerOp(Op):
    """ Apply Kalman Filtering to Estimate Poisiton, Speed, and Orientation
    Simulate real-world sensing with Perfect Sensor + Gaussian Noise
    Operator logging ground truth, dead reckoning estimates, and Kalman estimates

    Attributes:
    init_X - initial/previous Kalman state estimate
    init_V - error covariance,
    Q - process noise,
    R - measurement noise,
    prev_speed - log previous speed used to approximate acceleration
    dt - time gap between frames used to approximate acceleration
    naive_pred - state stimates for naive dead reckoning approach
    """
    def __init__(self,
                 name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        super(KalmanLoggerOp, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._flags = flags
        self._can_bus_msgs = deque()
        self._imu_msgs = deque()
        self._control_msgs = deque()

        # Initialize Kalman Variables, state - [x,y,velocity,yaw]
        self._init_X = None
        self._init_V = np.eye(4)
        self.Q = np.eye(4) * .1
        self.R = np.eye(4)
        self.R[2] *= .1
        self.R[3] *= .1
        self.prev_speed = None
        self.dt = .1
        self.naive_pred = None
        self._lock = threading.Lock()

    @staticmethod
    def setup_streams(input_streams):
        input_streams.filter(pylot.utils.is_can_bus_stream).add_callback(
            KalmanLoggerOp.on_can_bus_update)
        input_streams.filter(pylot.utils.is_past_control_stream).add_callback(
            KalmanLoggerOp.on_control_update)

        return []

    def on_can_bus_update(self, msg):
        with self._lock:
            self._can_bus_msgs.append(msg)


    def on_control_update(self, msg):
        with self._lock:
            self._control_msgs.append(msg)

        #Hack for synchronizing control update with can bus without downstream operator
        can_bus_msg = self._can_bus_msgs.popleft()
        vehicle_transform = can_bus_msg.data.transform
        vehicle_speed = can_bus_msg.data.forward_speed

        control_msg = self._control_msgs.popleft()
        steer = control_msg.data.steer
        steer_rad = steer2rad(steer)

        #perfect sensor data
        vel_gt = vehicle_speed
        yaw_gt = np.radians(vehicle_transform.rotation.yaw)
        x_gt = vehicle_transform.location.x
        y_gt = vehicle_transform.location.y
        timestamp = can_bus_msg.timestamp

        #simulate noise
        vel = vel_gt + np.random.normal(0, .1)
        yaw = yaw_gt + np.random.normal(0, .1)
        x = x_gt + np.random.normal(0, 1)
        y = y_gt + np.random.normal(0, 1)

        #noisy kalman measurement
        X_meas = np.array([x, y, vel, yaw])

        #Initialize state
        if self._init_X is None:
            self._init_X = X_meas
            self.prev_speed = vel
            self.naive_pred = X_meas
        else:
            accel = (vel- self.prev_speed)/self.dt
            self.prev_speed = vel
            u = np.array([accel, steer_rad])

            #kalman filtering
            A, B, c = get_transition_matrix(vel, yaw, steer_rad)
            X_filt, V_filt = kalman_step(
                X_meas, A, B, c, np.eye(*np.shape(X_meas)), 0,
                u, self.Q, self.R, self._init_X, self._init_V)

            self._init_X = X_filt
            self._init_V = V_filt

            #dead reckoning
            A, B, c = get_transition_matrix(self.naive_pred[2], self.naive_pred[3], steer_rad)
            self.naive_pred = A.dot(self.naive_pred) + B.dot(u) + c

            #logging
            self._logger.info('{} GT Measure: x {}, y {}, vel {}, yaw {}'.format(
                timestamp, x_gt, y_gt, vel_gt, yaw_gt))
            self._logger.info('{} Dead Reckoning: x {}, y {}, vel {}, yaw {}'.format(
                timestamp, self.naive_pred[0], self.naive_pred[1],
                self.naive_pred[2], self.naive_pred[3]))
            self._logger.info('{} Filter: x {}, y {}, vel {}, yaw {}'.format(
                timestamp, self._init_X[0], self._init_X[1], self._init_X[2], self._init_X[3]))
            self._logger.info('{} Variance: ['.format(timestamp) \
                + '\n'.join([''.join(['{:4} '.format(item) for item in row]) \
                for row in V_filt]) + ']')
            self._logger.info('{} Control: Acceleration {}, Steer {}'.format(
                timestamp, u[0], u[1]))

            self._csv_logger.info('{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(
                time_epoch_ms(), x_gt, y_gt, vel_gt, yaw_gt,
                self.naive_pred[0], self.naive_pred[1], self.naive_pred[2], self.naive_pred[3],
                self._init_X[0], self._init_X[1], self._init_X[2], self._init_X[3]))
