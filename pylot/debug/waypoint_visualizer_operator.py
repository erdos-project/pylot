from absl import flags
import carla
from collections import deque
import erdos

import pylot.simulation.utils
import pylot.utils

DEFAULT_VIS_TIME = 0.1

flags.DEFINE_bool('draw_waypoints_on_world', True,
                  'True to enable drawing on the Carla world')
flags.DEFINE_bool('draw_waypoints_on_camera_frames', False,
                  'True to enable drawing on camera frames')


class WaypointVisualizerOperator(erdos.Operator):
    """ WaypointVisualizerOperator visualizes the waypoints released by a
    global route planner.

    This operator listens on the `wp_debug` feed and draws the waypoints on the
    world simulation screen.

    Attributes:
        _world: A handle to the world to draw the waypoints on.
        _bgr_msgs: Buffer of received ground BGR image messages.
        _waypoints_msgs: Buffer of received waypoints messages.
        _can_bus_msgs: Buffer of received ground can bus messages.
    """
    def __init__(self,
                 waypoints_stream,
                 camera_stream,
                 can_bus_stream,
                 name,
                 flags,
                 log_file_name=None):
        """ Initializes the WaypointVisualizerOperator with the given
        parameters.

        Args:
            waypoints_stream: A stream on which we receive
               `planning.messages.WaypointsMessage` messages, which must be
                drawn on the screen.
            camera_stream: A stream on which we receive
                perception.camera_frame.CameraFrames.
            name: The name of the operator.
            flags: A handle to the global flags instance to retrieve the
                configuration.
        """
        waypoints_stream.add_callback(self.on_wp_update)
        camera_stream.add_callback(self.on_bgr_frame)
        can_bus_stream.add_callback(self.on_can_bus_update)
        erdos.add_watermark_callback([camera_stream, waypoints_stream], [],
                                     self.on_watermark)
        self._name = name
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._flags = flags
        self._bgr_msgs = deque()
        self._waypoints_msgs = deque()
        self._can_bus_msgs = deque()
        if self._flags.draw_waypoints_on_world:
            _, self._world = pylot.simulation.utils.get_world(
                self._flags.carla_host, self._flags.carla_port,
                self._flags.carla_timeout)
            if self._world is None:
                raise ValueError("Error connecting to the simulator.")

    @staticmethod
    def connect(waypoints_stream, camera_stream, can_bus_stream):
        return []

    def on_watermark(self, timestamp):
        self._logger.debug('@{}: received watermark'.format(timestamp))
        bgr_msg = self._bgr_msgs.popleft()
        waypoints_msg = self._waypoints_msgs.popleft()
        can_bus_msg = self._can_bus_msgs.popleft()
        bgr_frame = bgr_msg.frame
        vehicle_transform = can_bus_msg.data.transform
        if self._flags.draw_waypoints_on_camera_frames:
            bgr_frame.camera_setup.set_transform(
                vehicle_transform * bgr_frame.camera_setup.transform)
            extrinsic_matrix = bgr_frame.camera_setup.get_extrinsic_matrix()
            intrinsic_matrix = bgr_frame.camera_setup.get_intrinsic_matrix()
            for waypoint in waypoints_msg.waypoints:
                pixel_location = waypoint.location.to_camera_view(
                    extrinsic_matrix, intrinsic_matrix)
                bgr_frame.draw_point(pixel_location, [0, 0, 0])
            bgr_frame.visualize(self._name)

    def on_bgr_frame(self, msg):
        self._logger.debug('@{}: {} received bgr message'.format(
            msg.timestamp, self._name))
        self._bgr_msgs.append(msg)

    def on_wp_update(self, msg):
        """ The callback function that gets called upon receipt of the
        waypoint to be drawn on the screen.

        Args:
            msg: A message of type `planning.messages.WaypointsMessage` to
                be drawn on the screen.
        """
        self._logger.debug('@{}: {} received waypoints message'.format(
            msg.timestamp, self._name))
        self._waypoints_msgs.append(msg)
        if self._flags.draw_waypoints_on_world:
            for waypoint in msg.waypoints:
                # Adds 0.5 to z to ensure that the point is above the road
                # surface.
                loc = (waypoint.location +
                       pylot.utils.Location(0, 0, 0.5)).as_carla_location()
                self._world.debug.draw_point(loc,
                                             size=0.1,
                                             life_time=DEFAULT_VIS_TIME,
                                             color=carla.Color(0, 0, 255))

    def on_can_bus_update(self, msg):
        self._logger.debug('@{}: {} received can bus message'.format(
            msg.timestamp, self._name))
        self._can_bus_msgs.append(msg)
