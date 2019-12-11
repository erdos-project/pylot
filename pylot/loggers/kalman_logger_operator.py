from collections import deque
from erdos.op import Op
from erdos.utils import setup_csv_logging, setup_logging, time_epoch_ms
from pylot.map.hd_map import HDMap
from pylot.simulation.utils import kalman_step, steer2rad

import numpy as np

import carla
import collections
import itertools
import math
import pylot.control.utils
import pylot.utils
import threading


class KalmanLoggerOp(Op):
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
        self._init_X = None
        self._init_V = np.eye(4)
        self.prev_speed = None
        self.naive_pred = None
        self._lock = threading.Lock()

    @staticmethod
    def setup_streams(input_streams):
        print("SETTING UP STREAMS")
        input_streams.filter(pylot.utils.is_can_bus_stream).add_callback(
            KalmanLoggerOp.on_can_bus_update)
        # input_streams.filter(pylot.utils.is_imu_stream).add_callback(
        #     KalmanLoggerOp.on_imu_update)
        input_streams.filter(pylot.utils.is_past_control_stream).add_callback(
            KalmanLoggerOp.on_control_update)

        #input_streams.add_completion_callback(
        #   KalmanLoggerOp.on_notification)
        # Set no watermark on the output stream so that we do not
        # close the watermark loop with the carla operator.
        return []

    def on_can_bus_update(self, msg):
        self._logger.info("CAN BUS UPDATE")
        with self._lock:
            self._can_bus_msgs.append(msg)

    # def on_imu_update(self, msg):
    #     with self._lock:
    #         self._imu_msgs.append(msg)

    def on_control_update(self, msg):
        self._logger.info("CONTROL UPDATE")
        with self._lock:
            self._control_msgs.append(msg)

        can_bus_msg = self._can_bus_msgs.popleft()
        vehicle_transform = can_bus_msg.data.transform
        vehicle_speed = can_bus_msg.data.forward_speed
        # imu_msg = self._imu_msgs.popleft()
        # accel = imu_msg.accelerometer

        control_msg = self._control_msgs.popleft()
        steer = control_msg.steer
        steer_rad = steer2rad(steer)

        vel_gt = vehicle_speed
        yaw_gt = np.radian(vehicle_transform.rotation.yaw)
        x_gt = vehicle_transform.location.x
        y_gt = vehicle_transform.location.y
        timestamp = can_bus_msg.timestamp
       
        
        #simulate noise
        vel = vel_gt + np.random.normal(0,.1)
        yaw = yaw_gt + np.random.normal(0,.1)
        x = x_gt + np.random.normal(0,1)
        y = y_gt + np.random.normal(0,1)
    
        X_meas = np.array([x,y, vel,yaw])
        
        if self._init_X is None:
            self._init_X = X_meas
            self.prev_speed = vel
            self.naive_pred = X_meas 
        else:
            dt = .1
            accel = (vel- self.prev_speed)/dt
            self.prev_speed = vel

            u = np.array([accel,steer_rad])
            Q = np.eye(4) * .1
            R = np.eye(4)
            R[2] *= .1
            R[3] *=.1

            A,B,c = self.get_transition_matrix( vel, yaw, steer_rad)
            X_filt, V_filt = kalman_step(X_meas, A, B, c, np.eye(*np.shape(X_meas)), 0, u, Q,R,self._init_X,self._init_V)
            self._init_X = X_filt
            self._init_V = V_filt
            
            A,B,c = self.get_transition_matrix(self.naive_pred[2], self.naive_pred[3], steer_rad)
            self.naive_pred = A.dot(self.naive_pred) + B.dot(u) + c
            
        
            self._logger.info('{} GT Measure: x {}, y {}, vel {}, yaw {}'.format(
                timestamp,x_gt, y_gt,vel_gt, yaw_gt))
            self._logger.info('{} Dead Reckoning: x {}, y {}, vel {}, yaw {}'.format(
                timestamp, self.naive_pred[0], self.naive_pred[1], self.naive_pred[2], self.naive_pred[3]))
            self._logger.info('{} Filter: x {}, y {}, vel {}, yaw {}'.format(
                  timestamp, self._init_X[0], self._init_X[1], self._init_X[2], self._init_X[3]))
            self._logger.info('{} Variance: ['.format(timestamp) \
                + '\n'.join([''.join(['{:4} '.format(item) for item in row]) for row in V_filt]) + ']')
            self._logger.info('{} Control: Acceleration {}, Steer {}'.format(
                  timestamp, u[0],u[1]))
            
            self._csv_logger.info('{},{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(
                timestamp,time_epoch_ms(), x_gt,y_gt,vel_gt,yaw_gt, self.naive_pred[0], self.naive_pred[1],
                self.naive_pred[2], self.naive_pred[3], self._init_X[0], self._init_X[1], self._init_X[2], self._init_X[3]))



    # def on_notification(self, msg):
    #     # Get hero vehicle info.
    #     self._logger.info("NOTIFICATION UPDATE")
    #     can_bus_msg = self._can_bus_msgs.popleft()
    #     vehicle_transform = can_bus_msg.data.transform
    #     speed = can_bus_msg.data.forward_speed
    #     yaw = vehicle_transform.rotation.yaw
    #     x = vehicle_transform.location.x
    #     y = vehicle_transform.location.y
    #     timestamp = can_bus_msg.timestamp

    #     accel = (speed - self.speed)

    #     # imu_msg = self._imu_msgs.popleft()
    #     # accel = imu_msg.accelerometer

    #     control_msg = self._control_msgs.popleft()
    #     steer = control_msg.steer

    #     X_meas = [x,y, vel,yaw]
    #     u = [acccel,steer]
    #     Q = np.eye(4) * .1
    #     R = np.eye(4)
    #     R[2] *= .1
    #     R[3] *=.1

    #     # prev_vel = self._init_X[2]
    #     # prev_yaw = self._init_X[3]
    #     # prev_steer = self._init_V[1]
    #     A,B,c = self.get_transition_matrix(self, vel, yaw, steer)
    #     X_filt, V_filt = kalman_step(X_meas, A, B, c, np.eye(np.shape(X_meas)), 0, u, Q,R,self._init_X,self._init_V)
    #     self._init_X = X_filt
    #     self._init_V = V_filt
    #     self._logger.info('{} Measure: x {}, y {}, vel {}, yaw {}'.format(
    #         timestamp,X_meas[0],X_meas[1],X_meas[2],X_meas[3]))
    #     self._logger.info('{} Filter: x {}, y {}, vel {}, yaw {}'.format(
    #           timestamp, X_filt[0],X_filt[1],X_filt[2],X_filt[3]))
    #     self._logger.info('{} Variance: '.format(timestamp) \
    #         + '\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in Vfilt]))
    #     self._logger.info('{} Control: Acceleration {}, Steer {}'.format(
    #           timestamp, u[0],u[1]))


    def get_transition_matrix(self, vel, yaw, steer):
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
        delta_t = .1
        wheelbase = 2.85
        matrix_a = np.zeros((4,4))
        matrix_a[0, 0] = 1.0
        matrix_a[1, 1] = 1.0
        matrix_a[2, 2] = 1.0
        matrix_a[3, 3] = 1.0
        matrix_a[0, 2] = delta_t * np.cos(yaw)
        matrix_a[0, 3] = -delta_t * vel * np.sin(yaw)
        matrix_a[1, 2] = delta_t * np.sin(yaw)
        matrix_a[1, 3] = delta_t * vel * np.cos(yaw)
        matrix_a[3, 2] = \
            delta_t * np.tan(steer) / wheelbase

        # input matrix
        matrix_b = np.zeros((4,2))
        matrix_b[2, 0] = delta_t
        matrix_b[3, 1] = delta_t * vel / \
            (wheelbase * np.cos(steer)**2)

        # constant matrix
        matrix_c = np.zeros(4)
        matrix_c[0] = delta_t * vel * np.sin(yaw) * yaw
        matrix_c[1] = - delta_t * vel * np.cos(yaw) * yaw
        matrix_c[3] = - delta_t * vel * steer / \
            (wheelbase * np.cos(steer)**2)

        return matrix_a, matrix_b, matrix_c

