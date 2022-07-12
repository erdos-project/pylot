from __future__ import annotations  # Nicer syntax for Union types (PEP 604)

from collections import deque
from typing import List, Union

import erdos
from erdos.operator import OneInOneOut
from erdos.context import OneInOneOutContext
from pylot.perception.point_cloud import PointCloud

import pylot.utils
from pylot.perception.detection.utils import get_obstacle_locations
from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.depth_frame import DepthFrame
from pylot.perception.messages import ObstaclesMessageTuple
from pylot.drivers.sensor_setup import CameraSetup


class ObstacleLocationFinderOperator(OneInOneOut[Union[ObstaclesMessageTuple,
                                                       DepthFrame, PointCloud,
                                                       pylot.utils.Pose],
                                                 List[Obstacle]]):
    """Computes the world location of the obstacle.

    The operator uses a point cloud, which may come from a depth frame to
    compute the world location of an obstacle. It populates the location
    attribute in each obstacle object.

    Warning:
        An obstacle will be ignored if the operator cannot find its location.

    Args:
        flags (absl.flags): Object to be used to access absl flags.
        camera_setup: The setup of the center camera. This setup is used to
            calculate the real-world location of the camera, which in turn is
            used to convert detected obstacles from camera coordinates to
            real-world coordinates.
    """
    def __init__(self, flags, camera_setup: CameraSetup):
        self._flags = flags
        self._camera_setup = camera_setup
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        # Queues in which received messages are stored.
        self._obstacles_msgs = deque()
        self._depth_msgs = deque()
        self._pose_msgs = deque()

    def on_data(self, context: OneInOneOutContext, data: ObstaclesMessageTuple
                | DepthFrame | PointCloud | pylot.utils.Pose):
        if isinstance(data, ObstaclesMessageTuple):
            self.on_obstacles_update(context, data)
        elif isinstance(data, (DepthFrame, PointCloud)):
            self.on_depth_update(context, data)
        elif isinstance(data, pylot.utils.Pose):
            self.on_pose_update(context, data)
        else:
            raise ValueError('Unexpected data type')

    def on_obstacles_update(self, context: OneInOneOutContext[List[Obstacle]],
                            data: ObstaclesMessageTuple):
        self._logger.debug('@{}: obstacles update'.format(context.timestamp))
        self._obstacles_msgs.append(data)

    def on_depth_update(self, context: OneInOneOutContext[List[Obstacle]],
                        data: DepthFrame | PointCloud):
        self._logger.debug('@{}: depth update'.format(context.timestamp))
        self._depth_msgs.append(data)

    def on_pose_update(self, context: OneInOneOutContext[List[Obstacle]],
                       data: pylot.utils.Pose):
        self._logger.debug('@{}: pose update'.format(context.timestamp))
        self._pose_msgs.append(data)

    @erdos.profile_method()
    def on_watermark(self, context: OneInOneOutContext[List[Obstacle]]):
        """Invoked when all input streams have received a watermark.

        Args:
            timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of
                the watermark.
        """
        self._logger.debug('@{}: received watermark'.format(context.timestamp))
        if context.timestamp.is_top:
            return
        obstacles_msg = self._obstacles_msgs.popleft()
        depth_msg = self._depth_msgs.popleft()
        vehicle_transform = self._pose_msgs.popleft().transform
        obstacles_with_location = get_obstacle_locations(
            obstacles_msg.obstacles, depth_msg, vehicle_transform,
            self._camera_setup, self._logger)
        self._logger.debug('@{}: {}'.format(context.timestamp,
                                            obstacles_with_location))
        context.write_stream.send(
            erdos.Message(context.timestamp, obstacles_with_location))

    def destroy(self):
        self._logger.warn('destroying {}'.format(self.config.name))
