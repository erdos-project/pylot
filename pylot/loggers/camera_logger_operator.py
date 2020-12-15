"""This module implements an operator that logs camera frames."""

import erdos


class CameraLoggerOperator(erdos.Operator):
    """Logs camera frames to files.

    Args:
        camera_stream (:py:class:`erdos.ReadStream`): The stream on which
            camera frames are received.
        flags (absl.flags): Object to be used to access absl flags.
        filename_prefix (:obj:`str`): Used to build the names of the files it
            logs to.

    Attributes:
        _logger (:obj:`logging.Logger`): Instance to be used to log messages.
        _flags (absl.flags): Object to be used to access absl flags.
        _frame_cnt (:obj:`int`): Number of messages received.
        _filename_prefix (:obj:`str`): Used to construct the names of the files
            it logs to.
    """
    def __init__(self, camera_stream: erdos.ReadStream,
                 finished_indicator_stream: erdos.WriteStream, flags,
                 filename_prefix: str):
        camera_stream.add_callback(self.on_frame)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags
        self._frame_cnt = 0
        self._filename_prefix = filename_prefix

    @staticmethod
    def connect(camera_stream: erdos.ReadStream):
        """Connects the operator to other streams.

        The operator receives a camera stream.
        """
        finished_indicator_stream = erdos.WriteStream()
        return [finished_indicator_stream]

    def on_frame(self, msg: erdos.Message):
        """Invoked whenever a frame message is received on the stream."""
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self.config.name))
        self._frame_cnt += 1
        if self._frame_cnt % self._flags.log_every_nth_message != 0:
            return
        msg.frame.save(msg.timestamp.coordinates[0], self._flags.data_path,
                       self._filename_prefix)
