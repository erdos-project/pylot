"""This module implements an operator that logs tracked obstacles."""

import os

import erdos
from erdos.context import OneInOneOutContext
from erdos.operator import OneInOneOut

from pylot.perception.messages import ObstaclesMessageTuple


class MultipleObjectTrackerLoggerOperator(OneInOneOut):
    """Operator that logs tracked obstacles in the MOT16 file format.

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
    def __init__(self, flags):
        # Register a callback on obstacles data stream.
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags
        self._msg_cnt = 0
        self._data_path = os.path.join(self._flags.data_path,
                                       'multiple_object_tracker')
        os.makedirs(self._data_path, exist_ok=True)

    def on_data(self, context: OneInOneOutContext,
                data: ObstaclesMessageTuple):
        """Logs obstacles to files.

        Invoked upon the receipt of a msg on the obstacles stream.

        Args:
            data (:py:class:`~pylot.perception.messages.ObstaclesMessageTuple`):
                Received message.
        """
        self._logger.debug('@{}: {} received message'.format(
            context.timestamp, self.config.name))
        self._msg_cnt += 1
        if self._msg_cnt % self._flags.log_every_nth_message != 0:
            return
        assert len(context.timestamp.coordinates) == 1
        timestamp = context.timestamp.coordinates[0]
        lines = []
        for obstacle in data.obstacles:
            if obstacle.is_person() or obstacle.is_vehicle():
                lines.append(obstacle.as_mot16_str(timestamp))

        # Write the data, MOT16 style: https://motchallenge.net/instructions/
        file_name = os.path.join(self._data_path,
                                 'mot-{}.txt'.format(timestamp))
        with open(file_name, 'w') as outfile:
            outfile.writelines(lines)
