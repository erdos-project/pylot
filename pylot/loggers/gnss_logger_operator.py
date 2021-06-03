"""This module implements an operator that logs GNSS sensor readings."""

import json
import os

import erdos


class GNSSLoggerOperator(erdos.Operator):
    """Subscribes to GNSS streams and logs GNSS measurements.

    This will log every frame to preserve linearization when approximating jerk
    in smoothness evaluation metrics.

    Args:
        gnss_stream: (:py:class:`erdos.ReadStream`): The stream on which
            :py:class:`~pylot.localization.messages.GNSSMessage` are received.
        flags (absl.flags): Object to be used to access absl flags.

    Attributes:
        _logger (:obj:`logging.Logger`): Instance to be used to log messages.
        _flags (absl.flags): Object to be used to access absl flags.
        _msg_cnt (:obj:`int`): Number of messages received.
        _data_path (:obj:`str`): Directory to which to log files.
    """
    def __init__(self, gnss_stream: erdos.ReadStream,
                 finished_indicator_stream: erdos.WriteStream, flags):
        gnss_stream.add_callback(self.on_gnss_update)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags
        self._msg_cnt = 0
        self._data_path = os.path.join(self._flags.data_path, 'gnss')
        os.makedirs(self._data_path, exist_ok=True)

    @staticmethod
    def connect(gnss_stream: erdos.ReadStream):
        finished_indicator_stream = erdos.WriteStream()
        return [finished_indicator_stream]

    def on_gnss_update(self, msg):
        """Invoked upon receipt of an GNSS message.

        Args:
            msg (:py:class:`pylot.localization.messages.GNSSMessage`): Message
                to be logged.
        """
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self.config.name))
        self._msg_cnt += 1
        if self._msg_cnt % self._flags.log_every_nth_message != 0:
            return
        assert len(msg.timestamp.coordinates) == 1
        timestamp = msg.timestamp.coordinates[0]
        file_name = os.path.join(self._data_path,
                                 'gnss-{}.json'.format(timestamp))
        measurements = {
            "latitude": str(msg.latitude),
            "longitude": str(msg.longitude),
            "altitude": str(msg.altitude),
            "timestamp": str(timestamp)
        }
        with open(file_name, 'w') as outfile:
            json.dump(measurements, outfile)
