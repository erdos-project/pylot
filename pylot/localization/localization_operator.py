import erdos
from pylot.utils import Quaternion, Pose, Transform, Location, Vector3D

import numpy as np

from functools import partial
from collections import deque


class LocalizationOperator(erdos.Operator):
    def __init__(self, imu_stream, gnss_stream, ground_pose_stream,
                 pose_stream, flags):
        # Queue of saved messages.
        self._imu_updates = deque()
        self._gnss_updates = deque()
        self._ground_pose_updates = None

        # Streams to register a watermark on.
        watermark_streams = []

        # Register callbacks on both the IMU and GNSS streams.
        imu_stream.add_callback(
            partial(self.save, msg_type="IMU", queue=self._imu_updates))
        watermark_streams.append(imu_stream)

        gnss_stream.add_callback(
            partial(self.save, msg_type="GNSS", queue=self._gnss_updates))
        watermark_streams.append(gnss_stream)

        # Register the ground pose stream, if provided.
        if ground_pose_stream:
            self._ground_pose_updates = deque()
            ground_pose_stream.add_callback(
                partial(self.save,
                        msg_type="pose",
                        queue=self._ground_pose_updates))
            watermark_streams.append(ground_pose_stream)
        erdos.add_watermark_callback(watermark_streams, [], self.on_watermark)

        # Save the output stream.
        self._pose_stream = pose_stream

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

    @staticmethod
    def connect(imu_stream, gnss_stream, ground_pose_stream):
        pose_stream = erdos.WriteStream()
        return [pose_stream]

    def save(self, msg, msg_type, queue):
        self._logger.debug("@{}: received {} message.".format(
            msg.timestamp, msg_type))
        queue.append(msg)

    def get_message(self, queue, timestamp, name):
        msg = None
        if queue:
            while len(queue) > 0:
                retrieved_msg = queue.popleft()
                if retrieved_msg.timestamp == timestamp:
                    msg = retrieved_msg
                    break
            if not msg:
                raise ValueError(
                    "The message for {} with timestamp {} was not found".
                    format(name, timestamp))
        return msg

    @erdos.profile_method()
    def on_watermark(self, timestamp):
        self._logger.debug("@{}: received watermark.".format(timestamp))

        # Retrieve the messages for this timestamp.
        pose_msg = self.get_message(self._ground_pose_updates, timestamp,
                                    "pose")
        gnss_msg = self.get_message(self._gnss_updates, timestamp, "GNSS")
        imu_msg = self.get_message(self._imu_updates, timestamp, "IMU")

        if (self._last_pose_estimate is None or self._last_timestamp is None
            ) or (abs(imu_msg.acceleration.y) > 100 and not self._is_started):
            self._logger.debug(
                "@{}: The initial pose estimate is not initialized.".format(
                    timestamp))
            # If this is the first update or values have not stabilized,
            # save the estimates.
            if pose_msg:
                self._last_pose_estimate = pose_msg.data
                self._last_timestamp = timestamp.coordinates[0]
            else:
                raise NotImplementedError(
                    "Need pose message to initialize the estimates.")
        else:
            self._is_started = True

            # Initialize the delta_t
            current_ts = timestamp.coordinates[0]
            delta_t = (current_ts - self._last_timestamp) / 1000.0

            # Estimate the rotation.
            last_rotation_estimate = Quaternion.from_rotation(
                self._last_pose_estimate.transform.rotation)
            rotation_estimate = (last_rotation_estimate *
                                 Quaternion.from_angular_velocity(
                                     imu_msg.gyro, delta_t)).as_rotation()

            # Transform the IMU accelerometer data from the body frame to the
            # world frame, and retrieve location and velocity estimates.
            accelerometer_data = last_rotation_estimate.matrix.dot(
                imu_msg.acceleration.as_numpy_array()) + self._g
            last_location_estimate = self._last_pose_estimate.transform.\
                    location.as_numpy_array()
            last_velocity_estimate = self._last_pose_estimate.velocity_vector.\
                    as_numpy_array()

            # Estimate the location.
            location_estimate_np = last_location_estimate + (
                delta_t * last_velocity_estimate) + ((
                    (delta_t**2) / 2.0) * accelerometer_data)
            location_estimate = Location(*location_estimate_np)

            # Estimate the velocity.
            velocity_estimate_np = last_velocity_estimate + (
                delta_t * accelerometer_data)
            velocity_estimate = Vector3D(*velocity_estimate_np)

            # Create the PoseMessage and send it downstream.
            current_pose = Pose(
                transform=Transform(location=location_estimate,
                                    rotation=rotation_estimate),
                forward_speed=velocity_estimate.magnitude(),
                velocity_vector=velocity_estimate,
                localization_time=current_ts,
            )
            self._logger.debug("@{}: Predicted pose: {}".format(
                timestamp, current_pose))
            self._pose_stream.send(erdos.Message(timestamp, current_pose))

            # Set the estimates for the next iteration.
            self._last_timestamp = current_ts
            self._last_pose_estimate = current_pose
