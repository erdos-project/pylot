import erdos


class CameraLoggerOperator(erdos.Operator):
    """ Logs camera frames."""
    def __init__(self,
                 camera_stream,
                 name,
                 flags,
                 filename_prefix,
                 log_file_name=None):
        camera_stream.add_callback(self.on_frame)
        self._name = name
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._flags = flags
        self._frame_cnt = 0
        self._filename_prefix = filename_prefix

    @staticmethod
    def connect(camera_stream):
        return []

    def on_frame(self, msg):
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self._name))
        self._frame_cnt += 1
        if self._frame_cnt % self._flags.log_every_nth_message != 0:
            return
        msg.frame.save(msg.timestamp.coordinates[0], self._flags.data_path,
                       self._filename_prefix)
