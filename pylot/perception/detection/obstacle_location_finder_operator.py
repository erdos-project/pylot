from collections import deque
import copy
import erdos

from pylot.perception.messages import ObstaclesMessage
from pylot.utils import Rotation, Transform


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
        point_cloud_stream (:py:class:`erdos.ReadStream`): Stream on which
            point cloud messages are received.
        can_bus_stream (:py:class:`erdos.ReadStream`): Stream on which can
            bus info is received.
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
    def __init__(self, obstacles_stream, point_cloud_stream, can_bus_stream,
                 obstacles_output_stream, flags, camera_setup):
        obstacles_stream.add_callback(self.on_obstacles_update)
        point_cloud_stream.add_callback(self.on_point_cloud_update)
        can_bus_stream.add_callback(self.on_can_bus_update)
        erdos.add_watermark_callback(
            [obstacles_stream, point_cloud_stream, can_bus_stream],
            [obstacles_output_stream], self.on_watermark)
        self._flags = flags
        self._camera_setup = camera_setup
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        # Queues in which received messages are stored.
        self._obstacles_msgs = deque()
        self._point_cloud_msgs = deque()
        self._can_bus_msgs = deque()

    @staticmethod
    def connect(obstacles_stream, point_cloud_stream, can_bus_stream):
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
        point_cloud = self._point_cloud_msgs.popleft().point_cloud
        vehicle_transform = self._can_bus_msgs.popleft().data.transform

        transformed_camera_setup = copy.deepcopy(self._camera_setup)
        transformed_camera_setup.set_transform(
            vehicle_transform * transformed_camera_setup.transform)

        obstacles_with_location = []
        for obstacle in obstacles_msg.obstacles:
            location = point_cloud.get_pixel_location(
                obstacle.bounding_box.get_center_point(),
                transformed_camera_setup)
            if location is not None:
                obstacle.transform = Transform(location, Rotation())
                obstacles_with_location.append(obstacle)
            else:
                self._logger.error(
                    'Could not find world location for obstacle {}'.format(
                        obstacle))

        obstacles_output_stream.send(
            ObstaclesMessage(timestamp, obstacles_with_location))

    def on_obstacles_update(self, msg):
        self._logger.debug('@{}: obstacles update'.format(msg.timestamp))
        self._obstacles_msgs.append(msg)

    def on_point_cloud_update(self, msg):
        self._logger.debug('@{}: point cloud update'.format(msg.timestamp))
        self._point_cloud_msgs.append(msg)

    def on_can_bus_update(self, msg):
        self._logger.debug('@{}: can bus update'.format(msg.timestamp))
        self._can_bus_msgs.append(msg)
