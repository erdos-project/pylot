import carla
import erdos

# Pylot specific imports.
import pylot.simulation.carla_utils

DEFAULT_VIS_TIME = 0.1


class WaypointVisualizerOperator(erdos.Operator):
    """ WaypointVisualizerOperator visualizes the waypoints released by a
    global route planner.

    This operator listens on the `wp_debug` feed and draws the waypoints on the
    world simulation screen.

    Attributes:
        _world: A handle to the world to draw the waypoints on.
    """
    def __init__(self, waypoints_stream, name, flags, log_file_name=None):
        """ Initializes the WaypointVisualizerOperator with the given
        parameters.

        Args:
            waypoints_stream: A stream on which we receive
                              `planning.messages.WaypointsMessage` messages,
                              which must be drawn on the screen.
            name: The name of the operator.
            flags: A handle to the global flags instance to retrieve the
                configuration.
        """
        waypoints_stream.add_callback(self.on_wp_update)
        self._name = name
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._flags = flags
        _, self._world = pylot.simulation.carla_utils.get_world(
            self._flags.carla_host, self._flags.carla_port,
            self._flags.carla_timeout)
        if self._world is None:
            raise ValueError("Error connecting to the simulator.")

    @staticmethod
    def connect(waypoints_stream):
        return []

    def on_wp_update(self, msg):
        """ The callback function that gets called upon receipt of the
        waypoint to be drawn on the screen.

        Args:
            msg: A message of type `planning.messages.WaypointsMessage` to
                be drawn on the screen.
        """
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self._name))
        for waypoint in msg.waypoints:
            loc = carla.Location(waypoint.location.x, waypoint.location.y,
                                 waypoint.location.z + 0.5)
            self._world.debug.draw_point(loc,
                                         size=0.2,
                                         life_time=DEFAULT_VIS_TIME,
                                         color=carla.Color(0, 0, 255))
