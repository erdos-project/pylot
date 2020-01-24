"""This module implements an operator that logs tracked obstacles."""

import erdos
import os


class MultipleObjectTrackerLoggerOperator(erdos.Operator):
    """Operator that logs tracked obstacles in the MOT16 file format.

    Args:
        obstacles_stream (:py:class:`erdos.ReadStream`): The stream on which
            :py:class:`~pylot.perception.messages.ObstaclesMessage` are
            received.
        name (:obj:`str`): The name of the operator.
        flags (absl.flags): Object to be used to access absl flags.
        log_file_name (:obj:`str`, optional): Name of file where log messages
            are written to. If None, then messages are written to stdout.

    Attributes:
        _name (:obj:`str`): The name of the operator.
        _logger (:obj:`logging.Logger`): Instance to be used to log messages.
        _flags (absl.flags): Object to be used to access absl flags.
        _msg_cnt (:obj:`int`): Number of messages received.
    """
    def __init__(self, obstacles_stream, name, flags, log_file_name=None):
        # Register a callback on obstacles data stream.
        obstacles_stream.add_callback(self.on_obstacles_msg)
        self._name = name
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._flags = flags
        self._msg_cnt = 0

    @staticmethod
    def connect(obstacles_stream):
        return []

    def on_obstacles_msg(self, msg):
        """Logs obstacles to files.

        Invoked upon the receipt of a msg on the obstacles stream.

        Args:
            msg (:py:class:`~pylot.perception.messages.ObstaclesMessage`):
                Received message.
        """
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self._name))
        self._msg_cnt += 1
        if self._msg_cnt % self._flags.log_every_nth_message != 0:
            return
        assert len(msg.timestamp.coordinates) == 1
        timestamp = msg.timestamp.coordinates[0]
        lines = []
        for obstacle in msg.obstacles:
            if obstacle.label in {'person', 'vehicle'}:
                lines.append(obstacle.as_mot16_str(timestamp))

        # Write the data, MOT16 style: https://motchallenge.net/instructions/
        file_name = os.path.join(self._flags.data_path,
                                 'mot-{}.txt'.format(timestamp))
        with open(file_name, 'w') as outfile:
            outfile.writelines(lines)
