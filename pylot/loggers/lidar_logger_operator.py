"""This module implements an operator that logs lidar point clouds."""
import erdos


class LidarLoggerOperator(erdos.Operator):
    """Operator that logs point cloud messages.

    Args:
        lidar_stream (:py:class:`erdos.ReadStream`): The stream on which
            :py:class:`~pylot.perception.messages.PointCloudMessage` are
            received.
        flags (absl.flags): Object to be used to access absl flags.
        filename_prefix (:obj:`str`): Used to build the names of the files it
            logs to.

    Attributes:
        _logger (:obj:`logging.Logger`): Instance to be used to log messages.
        _flags (absl.flags): Object to be used to access absl flags.
        _pc_msg_cnt (:obj:`int`): Number of messages received.
        _filename_prefix (:obj:`str`): Used to construct the names of the files
             it logs to.
    """
    def __init__(self, lidar_stream: erdos.ReadStream,
                 finished_indicator_stream: erdos.WriteStream, flags,
                 filename_prefix: str):
        lidar_stream.add_callback(self.on_lidar_frame)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags
        self._pc_msg_cnt = 0
        self._filename_prefix = filename_prefix

    @staticmethod
    def connect(lidar_stream: erdos.ReadStream):
        finished_indicator_stream = erdos.WriteStream()
        return [finished_indicator_stream]

    def on_lidar_frame(self, msg):
        """Invoked upon the receipt of a msg on the point cloud stream.

        Args:
            msg (:py:class:`~pylot.perception.messages.PointCloudMessage`):
                Received message.
        """
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self.config.name))
        self._pc_msg_cnt += 1
        if self._pc_msg_cnt % self._flags.log_every_nth_message != 0:
            return
        assert len(msg.timestamp.coordinates) == 1
        # Write the lidar information.
        msg.point_cloud.save(msg.timestamp.coordinates[0],
                             self._flags.data_path, self._filename_prefix)
