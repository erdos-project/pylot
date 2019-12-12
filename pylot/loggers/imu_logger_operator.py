import json

from erdos.op import Op
from erdos.utils import setup_logging

import pylot.utils
import pylot.simulation.carla_utils


class IMULoggerOperator(Op):
    """ Subscribes to IMU streams and logs IMU measurements. This will log
    every frame to preserve linearization when approximating jerk in smoothness
    evaluation metrics."""

    def __init__(self, name, flags, log_file_name=None):
        super(IMULoggerOperator, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self._flags = flags

    @staticmethod
    def setup_streams(input_streams):
        input_streams.filter(pylot.utils.is_imu_stream).add_callback(
            IMULoggerOperator.on_imu_update)
        return []

    def on_imu_update(self, msg):
        """ The callback function that gets called upon receipt of the
        IMU message to be logged.

        Args:
            msg: A message of type `pylot.simulation.messages.IMUMessage` to
                be logged.
        """

        transform = msg.transform
        acceleration = msg.acceleration
        gyro = msg.gyro
        compass = msg.compass
        timestamp = msg.timestamp.coordinates[0]
        file_name = '{}imu-{}.json'.format(
            self._flags.data_path, timestamp)
        measurements = {
            "transform": str(transform),
            "acceleration": str(acceleration),
            "gyro": str(gyro),
            "compass": str(compass),
            "timestamp": str(timestamp)
        }
        with open(file_name, 'w') as outfile:
            json.dump(measurements, outfile)
