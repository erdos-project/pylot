"""This module implements an operator that visualizes camera frames."""

import erdos

import pylot.utils


class CameraVisualizerOperator(erdos.Operator):
    """Subscribes to a camera stream, and visualizes frames.

    Args:
        camera_stream (:py:class:`erdos.ReadStream`): The stream on which
            camera frames are received.

    Attributes:
        _logger (:obj:`logging.Logger`): Instance to be used to log messages.
    """
    def __init__(self, camera_stream):
        camera_stream.add_callback(self.display_frame)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)

    @staticmethod
    def connect(camera_stream):
        return []

    def display_frame(self, msg):
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self.config.name))
        msg.frame.visualize(pylot.utils.PYGAME_DISPLAY, msg.timestamp)
