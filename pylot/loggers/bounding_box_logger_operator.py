"""This module implements an operator that logs bounding boxes."""
import json
import os

import erdos
from erdos.context import OneInOneOutContext
from erdos.operator import OneInOneOut

from pylot.perception.messages import ObstaclesMessageTuple


class BoundingBoxLoggerOperator(OneInOneOut):
    """Logs bounding boxes of obstacles to files.

    Args:
        flags (absl.flags): Object to be used to access absl flags.

    Attributes:
        _logger (:obj:`logging.Logger`): Instance to be used to log messages.
        _flags (absl.flags): Object to be used to access absl flags.
        _msg_cnt (:obj:`int`): Number of messages received.
        _data_path (:obj:`str`): Directory to which to log files.
    """
    def __init__(self, flags, file_base_name: str):
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags
        self._file_base_name = file_base_name
        self._msg_cnt = 0
        self._data_path = os.path.join(self._flags.data_path, file_base_name)
        os.makedirs(self._data_path, exist_ok=True)

    def on_data(self, context: OneInOneOutContext,
                data: ObstaclesMessageTuple):
        """Logs bounding boxes to files.

        Invoked upon the receipt of a msg on the obstacles stream.

        Args:
            data: The obstacle to log.
        """
        self._logger.debug('@{}: {} received message'.format(
            context.timestamp, self.config.name))
        self._msg_cnt += 1
        if self._msg_cnt % self._flags.log_every_nth_message != 0:
            return
        bboxes = [obstacle.get_in_log_format() for obstacle in data.obstacles]
        assert len(context.timestamp.coordinates) == 1
        timestamp = context.timestamp.coordinates[0]
        # Write the bounding boxes.
        file_name = os.path.join(
            self._data_path, '{}-{}.json'.format(self._file_base_name,
                                                 timestamp))
        with open(file_name, 'w') as outfile:
            json.dump(bboxes, outfile)
