import carla
import erdust

# Pylot specific imports.
import pylot.simulation.carla_utils


class WaypointVisualizerOperator(erdust.Operator):
    """ WaypointVisualizerOperator visualizes the waypoints released by a
    global route planner.

    This operator listens on the `wp_debug` feed and draws the waypoints on the
    world simulation screen.

    Attributes:
        _world: A handle to the world to draw the waypoints on.
    """

    def __init__(self, waypoints_stream, name, flags):
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
        self._flags = flags
        _, self._world = pylot.simulation.carla_utils.get_world(
            self._flags.carla_host,
            self._flags.carla_port,
            self._flags.carla_timeout)
        if self._world is None:
            raise ValueError("Error connecting to the simulator.")
        self._colors = [carla.Color(255, 0, 0),
                        carla.Color(0, 255, 0),
                        carla.Color(0, 0, 255),
                        carla.Color(128, 128, 0),
                        carla.Color(0, 128, 128),
                        carla.Color(128, 0, 128),
                        carla.Color(64, 64, 0),
                        carla.Color(64, 0, 64),
                        carla.Color(0, 64, 64)]

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
        assert len(msg.waypoints) <= len(self._colors)
        index = 0
        for waypoint in msg.waypoints:
            loc = carla.Location(waypoint.location.x,
                                 waypoint.location.y,
                                 waypoint.location.z)
            begin = loc + carla.Location(z=0.5)
            end = begin + carla.Location(waypoint.forward_vector.x,
                                         waypoint.forward_vector.y,
                                         waypoint.forward_vector.z)
            self._world.debug.draw_arrow(begin,
                                         end,
                                         arrow_size=0.3,
                                         life_time=30.0,
                                         color=self._colors[index])
            index += 1
