"""This module implements an operator that visualizes planning waypoints."""
from collections import deque
import erdos

import pylot.utils

DEFAULT_VIS_TIME = 0.1


class WaypointVisualizerOperator(erdos.Operator):
    """Visualizes the waypoints released by a planner.

    This operator listens on the `wp_debug` feed and draws the waypoints on the
    world simulation screen.

    Args:
        waypoints_stream (:py:class:`erdos.ReadStream`): The stream on which
            :py:class:`~pylot.planning.messages.WaypointsMessage` are received.
        camera_stream (:py:class:`erdos.ReadStream`): The stream on which
            camera frames are received.
        flags (absl.flags): Object to be used to access absl flags.

    Attributes:
        _logger (:obj:`logging.Logger`): Instance to be used to log messages.
        _flags (absl.flags): Object to be used to access absl flags.
        _bgr_msgs (:obj:`collections.deque`): Buffer of received ground BGR
            image messages.
        _waypoints_msgs (:obj:`collections.deque`): Buffer of received
            waypoints messages.
        _pose_msgs (:obj:`collections.deque`): Buffer of received ground pose
            messages.
        _world (carla.World): A handle to the world to draw the waypoints on.
    """

    def __init__(self, waypoints_stream, camera_stream, pose_stream, flags):
        waypoints_stream.add_callback(self.on_wp_update)
        camera_stream.add_callback(self.on_bgr_frame)
        pose_stream.add_callback(self.on_pose_update)
        erdos.add_watermark_callback([camera_stream, waypoints_stream], [],
                                     self.on_watermark)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags
        self._bgr_msgs = deque()
        self._waypoints_msgs = deque()
        self._pose_msgs = deque()

    @staticmethod
    def connect(waypoints_stream, camera_stream, pose_stream):
        return []

    def run(self):
        # Run method is invoked after all operators finished initializing,
        # including the CARLA operator, which reloads the world. Thus, if
        # we get the world here we're sure it is up-to-date.
        if self._flags.draw_waypoints_on_world:
            from pylot.simulation.utils import get_world
            _, self._world = get_world(self._flags.carla_host,
                                       self._flags.carla_port,
                                       self._flags.carla_timeout)
            if self._world is None:
                raise ValueError("Error connecting to the simulator.")

    def on_watermark(self, timestamp):
        """Invoked when all input streams have received a watermark.

        Args:
            timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of
                the watermark.
        """
        self._logger.debug('@{}: received watermark'.format(timestamp))
        bgr_msg = self._bgr_msgs.popleft()
        waypoints_msg = self._waypoints_msgs.popleft()
        pose_msg = self._pose_msgs.popleft()
        bgr_frame = bgr_msg.frame
        vehicle_transform = pose_msg.data.transform
        if self._flags.draw_waypoints_on_camera_frames:
            bgr_frame.camera_setup.set_transform(
                vehicle_transform * bgr_frame.camera_setup.transform)
            extrinsic_matrix = bgr_frame.camera_setup.get_extrinsic_matrix()
            intrinsic_matrix = bgr_frame.camera_setup.get_intrinsic_matrix()
            for waypoint in waypoints_msg.waypoints:
                pixel_location = waypoint.location.to_camera_view(
                    extrinsic_matrix, intrinsic_matrix)
                bgr_frame.draw_point(pixel_location, [0, 0, 0])
            bgr_frame.visualize(self.config.name)

    def on_bgr_frame(self, msg):
        """Invoked when a msg on the camera stream is received.

        Args:
            msg (:py:class:`~pylot.prediction.messages.FrameMessage`):
                Received message.
        """
        self._logger.debug('@{}: {} received bgr message'.format(
            msg.timestamp, self.config.name))
        self._bgr_msgs.append(msg)

    def on_wp_update(self, msg):
        """Callback that draws waypoints on the screen.

        Invoked when a msg on the waypoints stream is received.

        Args:
            msg (:py:class:`~pylot.planning.messages.WaypointsMessage`): A
                message containing waypoints to be drawn on the screen.
        """
        self._logger.debug('@{}: {} received waypoints message'.format(
            msg.timestamp, self.config.name))
        self._waypoints_msgs.append(msg)
        if self._flags.draw_waypoints_on_world:
            for waypoint in msg.waypoints:
                # Adds 0.5 to z to ensure that the point is above the road
                # surface.
                loc = (waypoint.location +
                       pylot.utils.Location(0, 0, 0.5)).as_carla_location()
                self._world.debug.draw_point(loc,
                                             size=0.1,
                                             life_time=DEFAULT_VIS_TIME)

    def on_pose_update(self, msg):
        """Invoked when a msg on the pose stream is received.

        Args:
            msg (:py:class:`erdos.message.Message`): The data field of the
                message contains the :py:class:`~pylot.utils.Pose` object.
        """
        self._logger.debug('@{}: {} received pose message'.format(
            msg.timestamp, self.config.name))
        self._pose_msgs.append(msg)
