"""This module implements an operator that logs camera frames."""

import erdos


class CameraLoggerOperator(erdos.Operator):
    """Logs camera frames to files.

    Args:
        camera_stream (:py:class:`erdos.streams.ReadStream`): The stream on
             which camera frames are received.
        name (str): The name of the operator.
        flags (absl.flags): Object to be used to access absl flags.
        filename_prefix (str): Used to build the names of the files it logs to.
        log_file_name (str, optional): Name of file where log messages are
            written to. If None, then messages are written to stdout.

    Attributes:
        _name (str): The string name of the operator.
        _logger (:obj:`logging.Logger`): Instance to be used to log messages.
        _flags (absl.flags): Object to be used to access absl flags.
        _frame_cnt (:obj:`int`): Number of messages received.
        _filename_prefix (str): Used to construct the names of the files it
             logs to.
    """
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
        """Connects the operator to other streams.

        The operator receives an obstacles stream and does not write to any
        output stream.
        """
        return []

    def on_frame(self, msg):
        """Invoked whenever a frame message is received on the stream."""
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self._name))
        self._frame_cnt += 1
        if self._frame_cnt % self._flags.log_every_nth_message != 0:
            return
        msg.frame.save(msg.timestamp.coordinates[0], self._flags.data_path,
                       self._filename_prefix)
