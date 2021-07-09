"""This module implements an operator that logs bounding boxes."""
import json
import os

import erdos


class BoundingBoxLoggerOperator(erdos.Operator):
    """Logs bounding boxes of obstacles to files.

    Args:
        obstacles_stream (:py:class:`erdos.ReadStream`): The stream on which
            :py:class:`~pylot.perception.messages.ObstaclesMessage` are
            received.
        flags (absl.flags): Object to be used to access absl flags.

    Attributes:
        _logger (:obj:`logging.Logger`): Instance to be used to log messages.
        _flags (absl.flags): Object to be used to access absl flags.
        _msg_cnt (:obj:`int`): Number of messages received.
        _data_path (:obj:`str`): Directory to which to log files.
    """
    def __init__(self, obstacles_stream: erdos.ReadStream,
                 finished_indicator_stream: erdos.WriteStream, flags,
                 file_base_name):
        obstacles_stream.add_callback(self.on_obstacles_msg)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags
        self._file_base_name = file_base_name
        self._msg_cnt = 0
        self._data_path = os.path.join(self._flags.data_path, file_base_name)
        os.makedirs(self._data_path, exist_ok=True)

    @staticmethod
    def connect(obstacles_stream: erdos.ReadStream):
        """Connects the operator to other streams.

        The operator receives an obstacles stream.
        """
        finished_indicator_stream = erdos.WriteStream()
        return [finished_indicator_stream]

    def on_obstacles_msg(self, msg: erdos.Message):
        """Logs bounding boxes to files.

        Invoked upon the receipt of a msg on the obstacles stream.

        Args:
            msg (:py:class:`~pylot.perception.messages.ObstaclesMessage`):
                Received message.
        """
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self.config.name))
        self._msg_cnt += 1
        if self._msg_cnt % self._flags.log_every_nth_message != 0:
            return
        bboxes = [obstacle.get_in_log_format() for obstacle in msg.obstacles]
        assert len(msg.timestamp.coordinates) == 1
        timestamp = msg.timestamp.coordinates[0]
        # Write the bounding boxes.
        file_name = os.path.join(
            self._data_path, '{}-{}.json'.format(self._file_base_name,
                                                 timestamp))
        with open(file_name, 'w') as outfile:
            json.dump(bboxes, outfile)
