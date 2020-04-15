"""This module implements an operator that visualizes point clouds."""

import erdos

import numpy as np

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

        if self._flags.visualizer_backend == 'pygame':
            import pygame
            # Transform point cloud to top down view.
            lidar_data = np.array(msg.point_cloud.global_points[:, :2])
            lidar_data *= min(self._flags.carla_camera_image_width,
                              self._flags.carla_camera_image_height) / 100
            lidar_data += (0.5 * self._flags.carla_camera_image_width,
                           0.5 * self._flags.carla_camera_image_height)
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self._flags.carla_camera_image_width,
                              self._flags.carla_camera_image_height, 3)
            lidar_img = np.zeros((lidar_img_size), dtype=int)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            pygame.surfarray.blit_array(pylot.utils.PYGAME_DISPLAY, lidar_img)
            pygame.display.flip()
        else:
            import pptk
            pptk.viewer(msg.point_cloud.points)
