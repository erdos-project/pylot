"""This module implements an operator that logs lidar point clouds."""
import erdos


class LidarLoggerOperator(erdos.Operator):
    """Operator that logs point cloud messages.

    Args:
        lidar_stream (:py:class:`erdos.ReadStream`): The stream on which
            :py:class:`~pylot.perception.messages.PointCloudMessage` are
            received.
        name (:obj:`str`): The name of the operator.
        flags (absl.flags): Object to be used to access absl flags.
        filename_prefix (:obj:`str`): Used to build the names of the files it
            logs to.
        log_file_name (:obj:`str`, optional): Name of file where log messages
            are written to. If None, then messages are written to stdout.

    Attributes:
        _name (:obj:`str`): The name of the operator.
        _logger (:obj:`logging.Logger`): Instance to be used to log messages.
        _flags (absl.flags): Object to be used to access absl flags.
        _pc_msg_cnt (:obj:`int`): Number of messages received.
        _filename_prefix (:obj:`str`): Used to construct the names of the files
             it logs to.
    """
    def __init__(self,
                 lidar_stream,
                 name,
                 flags,
                 filename_prefix,
                 log_file_name=None):
        lidar_stream.add_callback(self.on_lidar_frame)
        self._name = name
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._flags = flags
        self._pc_msg_cnt = 0
        self._filename_prefix = filename_prefix

    @staticmethod
    def connect(lidar_stream):
        return []

    def on_lidar_frame(self, msg):
        """Invoked upon the receipt of a msg on the point cloud stream.

        Args:
            msg (:py:class:`~pylot.perception.messages.PointCloudMessage`):
                Received message.
        """
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self._name))
        self._pc_msg_cnt += 1
        if self._pc_msg_cnt % self._flags.log_every_nth_message != 0:
            return
        assert len(msg.timestamp.coordinates) == 1
        # Write the lidar information.
        msg.point_cloud.save(msg.timestamp.coordinates[0],
                             self._flags.data_path, self._filename_prefix)
