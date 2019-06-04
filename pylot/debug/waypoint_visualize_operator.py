import carla

# ERDOS specific imports.
from erdos.op import Op
from erdos.utils import setup_logging

# Pylot specific imports.
import pylot_utils
import simulation.carla_utils


class WaypointVisualizerOperator(Op):
    """ WaypointVisualizerOperator visualizes the waypoints released by a
    global route planner.

    This operator listens on the `wp_debug` feed and draws the waypoints on the
    world simulation screen.

    Attributes:
        _world: A handle to the world to draw the waypoints on.
    """

    def __init__(self, name, flags, log_file_name=None):
        """ Initializes the WaypointVisualizerOperator with the given
        parameters.

        Args:
            name: The name of the operator.
            flags: A handle to the global flags instance to retrieve the
                configuration.
            log_file_name: The file to log the required information to.
        """
        super(WaypointVisualizerOperator, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self._flags = flags
        _, self._world = simulation.carla_utils.get_world(
            self._flags.carla_host,
            self._flags.carla_port)
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
    def setup_streams(input_streams):
        """ This method takes in a single input stream called `wp_debug` which
        sends a `planning.messages.WaypointsMessage` to be drawn on
        the screen.

        Args:
            input_streams: A list of streams to take the input from (length=2)

        Returns:
            An empty list representing that this operator does not publish
            any output streams.
        """
        if len(input_streams) > 1:
            raise ValueError(
                "The WaypointVisualizerOperator should not receive more than"
                " two inputs. Please check the graph connections.")

        input_streams.filter(pylot_utils.is_waypoints_stream).add_callback(
            WaypointVisualizerOperator.on_wp_update)
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
            end = begin + carla.Location(waypoint.orientation.x,
                                         waypoint.orientation.y,
                                         waypoint.orientation.z)
            self._world.debug.draw_arrow(begin,
                                         end,
                                         arrow_size=0.3,
                                         life_time=30.0,
                                         color=self._colors[index])
            index += 1
