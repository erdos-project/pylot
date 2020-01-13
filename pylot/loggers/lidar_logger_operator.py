import erdos


class LidarLoggerOperator(erdos.Operator):
    """ Logs point cloud messages."""
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
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self._name))
        self._pc_msg_cnt += 1
        if self._pc_msg_cnt % self._flags.log_every_nth_message != 0:
            return
        assert len(msg.timestamp.coordinates) == 1
        # Write the lidar information.
        msg.point_cloud.save(msg.timestamp.coordinates[0],
                             self._flags.data_path, self._filename_prefix)
