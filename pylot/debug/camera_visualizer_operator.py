"""This module implements an operator that visualizes camera frames."""

import erdos


class CameraVisualizerOperator(erdos.Operator):
    """Subscribes to a camera stream, and visualizes frames.

    Args:
        camera_stream (:py:class:`erdos.ReadStream`): The stream on which
            camera frames are received.
        name (:obj:`str`): The name of the operator.
        log_file_name (:obj:`str`, optional): Name of file where log messages
            are written to. If None, then messages are written to stdout.

    Attributes:
        _name (:obj:`str`): The string name of the operator.
        _logger (:obj:`logging.Logger`): Instance to be used to log messages.
    """
    def __init__(self, camera_stream, name, log_file_name=None):
        camera_stream.add_callback(self.display_frame)
        self._name = name
        self._logger = erdos.utils.setup_logging(name, log_file_name)

    @staticmethod
    def connect(camera_stream):
        return []

    def display_frame(self, msg):
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self._name))
        msg.frame.visualize(self._name, msg.timestamp)
