"""This module implements EKF localization using GNSS and IMU."""

from collections import deque
from typing import Union
from functools import partial

import erdos
from erdos.operator import OneInOneOut
from erdos.context import OneInOneOutContext
from erdos import Message, ReadStream, Timestamp, WriteStream

import numpy as np

import pylot.utils
from pylot.localization.messages import GNSSMessageTuple, IMUMessageTuple

from pylot.utils import Location, Pose, Quaternion, Rotation, Transform, \
        Vector3D


class LocalizationOperator(OneInOneOut):
    """Localizes vehicle using GPS and IMU.

    The operator implements localization using a Kalman filter.

    Args:
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, flags):
        # Initialize queues to store incoming data.
        self._imu_updates = deque()
        self._gnss_updates = deque()
        self._ground_pose_updates = deque()

        # Initialize a logger.
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)

        # Gravity vector.
        self._g = np.array([0, 0, -9.81])

        # Previous timestamp values.
        self._last_pose_estimate = None
        self._last_timestamp = None

        # NOTE: At the start of the simulation, the vehicle drops down from
        # the sky, during which the IMU values screw up the calculations.
        # This boolean flag takes care to start the prediction only when the
        # values have stabilized.
        self._is_started = False

        # Constants required for the Kalman filtering.
        var_imu_f, var_imu_w, var_gnss = 0.5, 0.5, 0.1
        self.__Q = np.identity(6)
        self.__Q[0:3, 0:3] = self.__Q[0:3, 0:3] * var_imu_f
        self.__Q[3:6, 3:6] = self.__Q[3:6, 3:6] * var_imu_w

        self.__F = np.identity(9)

        self.__L = np.zeros([9, 6])
        self.__L[3:9, :] = np.identity(6)

        self.__R_GNSS = np.identity(3) * var_gnss

        self._last_covariance = np.zeros((9, 9))

    def on_data(self, context: OneInOneOutContext,
                data: Union[IMUMessageTuple, GNSSMessageTuple,
                            pylot.utils.Pose]):
        if isinstance(data, IMUMessageTuple):
            self._logger.debug('@{}: IMU update'.format(context.timestamp))
            self._imu_updates.append((context.timestamp, data))
        elif isinstance(data, GNSSMessageTuple):
            self._logger.debug('@{}: GNSS update'.format(context.timestamp))
            self._gnss_updates.append((context.timestamp, data))
        elif isinstance(data, pylot.utils.Pose):
            self._logger.debug('@{}: Pose update'.format(context.timestamp))
            self._ground_pose_updates.append((context.timestamp, data))
        else:
            raise ValueError('Unexpected data type')

    def get_message(self, queue: deque, timestamp: Timestamp, name: str):
        msg = None
        if queue:
            while len(queue) > 0:
                retrieved_msg = queue.popleft()
                # Messages are stored as tuples: (timestamp, data)
                if retrieved_msg[0] == timestamp:
                    msg = retrieved_msg[1]
                    break
            if not msg:
                raise ValueError(
                    "The message for {} with timestamp {} was not found".
                    format(name, timestamp))
        return msg

    def __skew_symmetric(self, v):
        """Skew symmetric form of a 3x1 vector."""
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]],
                        dtype=np.float64)

    def __update_using_gnss(self, location_estimate, velocity_estimate,
                            rotation_estimate, gnss_reading, delta_t):
        # Construct H_k = [I, 0, 0] (shape=(3, 9))
        H_k = np.zeros((3, 9))
        H_k[:, :3] = np.identity(3)

        # Propogate uncertainty.
        Q = self.__Q * delta_t * delta_t
        self._last_covariance = (self.__F.dot(self._last_covariance).dot(
            self.__F.T)) + (self.__L.dot(Q).dot(self.__L.T))

        # Compute Kalman gain. (shape=(9, 3))
        K_k = self._last_covariance.dot(
            H_k.T.dot(
                np.linalg.inv(
                    H_k.dot(self._last_covariance.dot(H_k.T)) +
                    self.__R_GNSS)))

        # Compute error state. (9x3) x ((3x1) - (3x1)) = shape(9, 1)
        delta_x_k = K_k.dot(gnss_reading - location_estimate)

        # Correct predicted state.
        corrected_location_estimate = location_estimate + delta_x_k[0:3]
        corrected_velocity_estimate = velocity_estimate + delta_x_k[3:6]
        roll, pitch, yaw = delta_x_k[6:]
        corrected_rotation_estimate = Quaternion.from_rotation(
            Rotation(roll=roll, pitch=pitch, yaw=yaw)) * rotation_estimate

        # Fix the covariance.
        self._last_covariance = (np.identity(9) - K_k.dot(H_k)).dot(
            self._last_covariance)

        return (
            corrected_location_estimate,
            corrected_velocity_estimate,
            corrected_rotation_estimate,
        )

    @erdos.profile_method()
    def on_watermark(self, context: OneInOneOutContext):
        self._logger.debug("@{}: received watermark.".format(
            context.timestamp))
        if context.timestamp.is_top:
            context.write_stream.send(erdos.WatermarkMessage(
                context.timestamp))
            return

        # Retrieve the messages for this timestamp.
        pose_msg = self.get_message(self._ground_pose_updates,
                                    context.timestamp, "Pose")
        gnss_msg = self.get_message(self._gnss_updates, context.timestamp,
                                    "GNSS")
        imu_msg = self.get_message(self._imu_updates, context.timestamp, "IMU")

        if self._last_pose_estimate is None or \
           (abs(imu_msg.acceleration.y) > 100 and not self._is_started):
            self._logger.debug(
                "@{}: The initial pose estimate is not initialized.".format(
                    context.timestamp))
            # If this is the first update or values have not stabilized,
            # save the estimates.
            if pose_msg:
                self._last_pose_estimate = pose_msg
                self._last_timestamp = context.timestamp.coordinates[0]
                context.write_stream.send(
                    erdos.Message(context.timestamp, pose_msg))
                context.write_stream.send(
                    erdos.WatermarkMessage(context.timestamp))
            else:
                raise NotImplementedError(
                    "Need pose message to initialize the estimates.")
        else:
            self._is_started = True

            # Initialize the delta_t
            current_ts = context.timestamp.coordinates[0]
            delta_t = (current_ts - self._last_timestamp) / 1000.0

            # Estimate the rotation.
            last_rotation_estimate = Quaternion.from_rotation(
                self._last_pose_estimate.transform.rotation)
            rotation_estimate = (
                last_rotation_estimate *
                Quaternion.from_angular_velocity(imu_msg.gyro, delta_t))

            # Transform the IMU accelerometer data from the body frame to the
            # world frame, and retrieve location and velocity estimates.
            accelerometer_data = last_rotation_estimate.matrix.dot(
                imu_msg.acceleration.as_numpy_array()) + self._g
            last_location_estimate = \
                self._last_pose_estimate.transform.location.as_numpy_array()
            last_velocity_estimate = \
                self._last_pose_estimate.velocity_vector.as_numpy_array()

            # Estimate the location.
            location_estimate = last_location_estimate + (
                delta_t * last_velocity_estimate) + ((
                    (delta_t**2) / 2.0) * accelerometer_data)

            # Estimate the velocity.
            velocity_estimate = last_velocity_estimate + (delta_t *
                                                          accelerometer_data)

            # Fuse the GNSS values using an EKF to fix drifts and noise in
            # the estimates.

            # Linearize the motion model and compute Jacobians.
            self.__F[0:3, 3:6] = np.identity(3) * delta_t
            self.__F[3:6, 6:9] = last_rotation_estimate.matrix.dot(
                -self.__skew_symmetric(accelerometer_data.reshape(
                    (3, 1)))) * delta_t

            # Fix estimates using GNSS
            gnss_reading = Location.from_gps(
                gnss_msg.latitude, gnss_msg.longitude,
                gnss_msg.altitude).as_numpy_array()
            (
                location_estimate,
                velocity_estimate,
                rotation_estimate,
            ) = self.__update_using_gnss(location_estimate, velocity_estimate,
                                         rotation_estimate, gnss_reading,
                                         delta_t)

            # Create the PoseMessage and send it downstream.
            current_pose = Pose(
                transform=Transform(location=Location(*location_estimate),
                                    rotation=rotation_estimate.as_rotation()),
                forward_speed=Vector3D(*velocity_estimate).magnitude(),
                velocity_vector=Vector3D(*velocity_estimate),
                localization_time=current_ts,
            )
            self._logger.debug("@{}: Predicted pose: {}".format(
                context.timestamp, current_pose))
            context.write_stream.send(
                erdos.Message(context.timestamp, current_pose))
            context.write_stream.send(erdos.WatermarkMessage(
                context.timestamp))

            # Set the estimates for the next iteration.
            self._last_timestamp = current_ts
            self._last_pose_estimate = current_pose

    def destroy(self):
        self._logger.warn('destroying {}'.format(self.config.name))
