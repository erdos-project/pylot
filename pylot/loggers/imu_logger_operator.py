"""This module implements an operator that logs IMU sensor readings."""

import erdos
import json
import os


class IMULoggerOperator(erdos.Operator):
    """Subscribes to IMU streams and logs IMU measurements.

    This will log every frame to preserve linearization when approximating jerk
    in smoothness evaluation metrics.

    Args:
        imu_stream: (:py:class:`erdos.streams.ReadStream`): The stream on which
            :py:class:`~pylot.perception.messages.IMUMessage` are received.
        name (str): The name of the operator.
        flags (absl.flags): Object to be used to access absl flags.
        log_file_name (str, optional): Name of file where log messages are
            written to. If None, then messages are written to stdout.

    Attributes:
        _name (str): The name of the operator.
        _logger (:obj:`logging.Logger`): Instance to be used to log messages.
        _flags (absl.flags): Object to be used to access absl flags.
        _msg_cnt (:obj:`int`): Number of messages received.
    """
    def __init__(self, imu_stream, name, flags, log_file_name=None):
        imu_stream.add_callback(self.on_imu_update)
        self._name = name
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._flags = flags
        self._msg_cnt = 0

    @staticmethod
    def connect(imu_stream):
        return []

    def on_imu_update(self, msg):
        """Invoked upon receipt of an IMU message.

        Args:
            msg (:py:class:`pylot.perception.messages.IMUMessage`): Message to
                be logged.
        """
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self._name))
        self._msg_cnt += 1
        if self._msg_cnt % self._flags.log_every_nth_message != 0:
            return
        assert len(msg.timestamp.coordinates) == 1
        timestamp = msg.timestamp.coordinates[0]
        file_name = os.path.join(self._flags.data_path,
                                 'imu-{}.json'.format(timestamp))
        measurements = {
            "transform": str(msg.transform),
            "acceleration": str(msg.acceleration),
            "gyro": str(msg.gyro),
            "compass": str(msg.compass),
            "timestamp": str(timestamp)
        }
        with open(file_name, 'w') as outfile:
            json.dump(measurements, outfile)
