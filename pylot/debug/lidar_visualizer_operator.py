"""This module implements an operator that visualizes point clouds."""

import erdos
import pptk


class LidarVisualizerOperator(erdos.Operator):
    """Subscribes to point cloud streams and visualizes point clouds.

    Args:
        point_cloud_stream (:py:class:`erdos.streams.ReadStream`): The stream
            on which :py:class:`~pylot.perception.messages.PointCloudMessage`
            are received.
        name (:obj:`str`): The name of the operator.
        log_file_name (:obj:`str`, optional): Name of file where log messages
            are written to. If None, then messages are written to stdout.

    Attributes:
        _name (:obj:`str`): The name of the operator.
        _logger (:obj:`logging.Logger`): Instance to be used to log messages.
    """
    def __init__(self, point_cloud_stream, name, log_file_name=None):
        point_cloud_stream.add_callback(self.display_point_cloud)
        self._name = name
        self._logger = erdos.utils.setup_logging(name, log_file_name)

    @staticmethod
    def connect(point_cloud_stream):
        return []

    def display_point_cloud(self, msg):
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self._name))
        pptk.viewer(msg.point_cloud.points)
