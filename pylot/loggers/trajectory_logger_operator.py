"""This module implements an operator that logs obstacle trajectories."""

import json
import os

import erdos
from erdos.operator import OneInOneOut
from erdos.context import OneInOneOutContext

from pylot.perception.messages import ObstacleTrajectoriesMessageTuple


class TrajectoryLoggerOperator(OneInOneOut):
    """Logs tracked obstacles trajectories to files.

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
        self._data_path = os.path.join(self._flags.data_path, 'trajectories')
        os.makedirs(self._data_path, exist_ok=True)

    def on_data(self, context: OneInOneOutContext,
                data: ObstacleTrajectoriesMessageTuple):
        """Logs obstacle trajectories to files.

        Invoked upon the receipt of trajectories on the obstacles trajectories stream.

        Args:
            msg (:py:class:`~pylot.perception.messages.ObstacleTrajectoriesMessageTuple`):  # noqa: E501
                Received trajectories.
        """
        self._logger.debug('@{}: {} received message'.format(
            context.timestamp, self.config.name))
        self._msg_cnt += 1
        if self._msg_cnt % self._flags.log_every_nth_message != 0:
            return
        trajectories = [
            str(trajectory) for trajectory in data.obstacle_trajectories
        ]
        assert len(context.timestamp.coordinates) == 1
        timestamp = context.timestamp.coordinates[0]
        # Write the trajectories.
        file_name = os.path.join(self._data_path,
                                 'trajectories-{}.json'.format(timestamp))
        with open(file_name, 'w') as outfile:
            json.dump(trajectories, outfile)
