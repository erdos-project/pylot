"""This module implements an operator that logs IMU sensor readings."""

import json
import os

import erdos
from erdos.operator import OneInOneOut
from erdos.context import OneInOneOutContext
from pylot.localization.messages import IMUMessageTuple


class IMULoggerOperator(OneInOneOut):
    """Subscribes to IMU streams and logs IMU measurements.

    This will log every frame to preserve linearization when approximating jerk
    in smoothness evaluation metrics.

    Args:
        flags (absl.flags): Object to be used to access absl flags.

    Attributes:
        _logger (:obj:`logging.Logger`): Instance to be used to log messages.
        _flags (absl.flags): Object to be used to access absl flags.
        _msg_cnt (:obj:`int`): Number of messages received.
        _data_path (:obj:`str`): Directory to which to log files.
    """
    def __init__(self, flags):
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags
        self._msg_cnt = 0
        self._data_path = os.path.join(self._flags.data_path, 'imu')
        os.makedirs(self._data_path, exist_ok=True)

    def on_data(self, context: OneInOneOutContext, data: IMUMessageTuple):
        """Invoked upon receipt of an IMU message.

        Args:
            data (:py:class:`pylot.localization.messages.IMUMessageTuple`): Message
                to be logged.
        """
        self._logger.debug('@{}: {} received message'.format(
            context.timestamp, self.config.name))
        self._msg_cnt += 1
        if self._msg_cnt % self._flags.log_every_nth_message != 0:
            return
        assert len(context.timestamp.coordinates) == 1
        timestamp = context.timestamp.coordinates[0]
        file_name = os.path.join(self._data_path,
                                 'imu-{}.json'.format(timestamp))
        measurements = {
            "transform": str(data.transform),
            "acceleration": str(data.acceleration),
            "gyro": str(data.gyro),
            "compass": str(data.compass),
            "timestamp": str(timestamp)
        }
        with open(file_name, 'w') as outfile:
            json.dump(measurements, outfile)
