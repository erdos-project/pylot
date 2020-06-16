"""This module implements an operator that visualizes point clouds."""

import erdos

import pylot.utils


class LidarVisualizerOperator(erdos.Operator):
    """Subscribes to point cloud streams and visualizes point clouds.

    Args:
        point_cloud_stream (:py:class:`erdos.ReadStream`): The stream on which
            :py:class:`~pylot.perception.messages.PointCloudMessage`
            are received.

    Attributes:
        _logger (:obj:`logging.Logger`): Instance to be used to log messages.
    """
    def __init__(self, point_cloud_stream, flags):
        point_cloud_stream.add_callback(self.display_point_cloud)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags

    @staticmethod
    def connect(point_cloud_stream):
        return []

    def display_point_cloud(self, msg):
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self.config.name))
        msg.point_cloud.visualize(pylot.utils.PYGAME_DISPLAY,
                                  self._flags.carla_camera_image_width,
                                  self._flags.carla_camera_image_height)
