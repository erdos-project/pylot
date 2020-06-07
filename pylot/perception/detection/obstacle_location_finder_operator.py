from collections import deque

import erdos

from pylot.perception.detection.utils import get_obstacle_locations
from pylot.perception.messages import ObstaclesMessage


class ObstacleLocationFinderOperator(erdos.Operator):
    """Computes the world location of the obstacle.

    The operator uses a point cloud, which may come from a depth frame to
    compute the world location of an obstacle. It populates the location
    attribute in each obstacle object.

    Warning:
        An obstacle will be ignored if the operator cannot find its location.

    Args:
        obstacles_stream (:py:class:`erdos.ReadStream`): Stream on which
            detected obstacles are received.
        depth_stream (:py:class:`erdos.ReadStream`): Stream on which
            either point cloud messages or depth frames are received. The
            message type differs dependening on how data-flow operators are
            connected.
        pose_stream (:py:class:`erdos.ReadStream`): Stream on which pose
            info is received.
        obstacles_output_stream (:py:class:`erdos.WriteStream`): Stream on
            which the operator sends detected obstacles with their world
            location set.
        flags (absl.flags): Object to be used to access absl flags.
        camera_setup (:py:class:`~pylot.drivers.sensor_setup.CameraSetup`):
            The setup of the center camera. This setup is used to calculate the
            real-world location of the camera, which in turn is used to convert
            detected obstacles from camera coordinates to real-world
            coordinates.
    """
    def __init__(self, obstacles_stream, depth_stream, pose_stream,
                 obstacles_output_stream, flags, camera_setup):
        obstacles_stream.add_callback(self.on_obstacles_update)
        depth_stream.add_callback(self.on_depth_update)
        pose_stream.add_callback(self.on_pose_update)
        erdos.add_watermark_callback(
            [obstacles_stream, depth_stream, pose_stream],
            [obstacles_output_stream], self.on_watermark)
        self._flags = flags
        self._camera_setup = camera_setup
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        # Queues in which received messages are stored.
        self._obstacles_msgs = deque()
        self._depth_msgs = deque()
        self._pose_msgs = deque()

    @staticmethod
    def connect(obstacles_stream, depth_stream, pose_stream):
        obstacles_output_stream = erdos.WriteStream()
        return [obstacles_output_stream]

    @erdos.profile_method()
    def on_watermark(self, timestamp, obstacles_output_stream):
        """Invoked when all input streams have received a watermark.

        Args:
            timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of
                the watermark.
        """
        self._logger.debug('@{}: received watermark'.format(timestamp))
        obstacles_msg = self._obstacles_msgs.popleft()
        depth_msg = self._depth_msgs.popleft()
        vehicle_transform = self._pose_msgs.popleft().data.transform

        obstacles_with_location = get_obstacle_locations(
            obstacles_msg.obstacles, depth_msg, vehicle_transform,
            self._camera_setup, self._logger)

        self._logger.debug('@{}: {}'.format(timestamp,
                                            obstacles_with_location))

        obstacles_output_stream.send(
            ObstaclesMessage(timestamp, obstacles_with_location))

    def on_obstacles_update(self, msg):
        self._logger.debug('@{}: obstacles update'.format(msg.timestamp))
        self._obstacles_msgs.append(msg)

    def on_depth_update(self, msg):
        self._logger.debug('@{}: depth update'.format(msg.timestamp))
        self._depth_msgs.append(msg)

    def on_pose_update(self, msg):
        self._logger.debug('@{}: pose update'.format(msg.timestamp))
        self._pose_msgs.append(msg)
