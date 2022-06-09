"""This module implements an operator that logs lidar point clouds."""
import os

import erdos
from erdos.operator import OneInOneOut
from erdos.context import OneInOneOutContext

from pylot.perception.point_cloud import PointCloud


class LidarLoggerOperator(OneInOneOut):
    """Operator that logs point cloud messages.

    Args:
        flags (absl.flags): Object to be used to access absl flags.
        filename_prefix (:obj:`str`): Used to build the names of the files it
            logs to.

    Attributes:
        _logger (:obj:`logging.Logger`): Instance to be used to log messages.
        _flags (absl.flags): Object to be used to access absl flags.
        _pc_msg_cnt (:obj:`int`): Number of messages received.
        _filename_prefix (:obj:`str`): Used to construct the names of the files
             it logs to.
        _data_path (:obj:`str`): Directory to which to log files.
    """
    def __init__(self, flags, filename_prefix: str):
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags
        self._pc_msg_cnt = 0
        self._filename_prefix = filename_prefix
        self._data_path = os.path.join(self._flags.data_path, filename_prefix)
        os.makedirs(self._data_path, exist_ok=True)

    def on_data(self, context: OneInOneOutContext, data: PointCloud):
        """Invoked upon the receipt of PointCloud object on input stream.

        Args:
            data (:py:class:`~pylot.perception.PointCloud`):
                Received PointCloud object.
        """
        self._logger.debug('@{}: {} received message'.format(
            context.timestamp, self.config.name))
        self._pc_msg_cnt += 1
        if self._pc_msg_cnt % self._flags.log_every_nth_message != 0:
            return
        assert len(context.timestamp.coordinates) == 1
        # Write the lidar information.
        data.save(context.timestamp.coordinates[0], self._data_path,
                  self._filename_prefix)
