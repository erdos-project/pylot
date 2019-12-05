import carla

# ERDOS specific imports.
from erdos.op import Op
from erdos.utils import setup_logging

# Pylot specific imports.
import pylot.utils
import pylot.simulation.carla_utils


class PlanningVisualizeOperator(Op):
    """ PlanningVisualizeOperator visualizes the waypoints released by planner.

    This operator listens on the `waypoints` feed and draws the waypoints on the
    world simulation screen.

    Attributes:
        _world: A handle to the world to draw the waypoints on.
    """
    def __init__(self, name, flags, log_file_name=None):
        """ Initializes the PlanningVisualizeOperator with the given
        parameters.

        Args:
            name: The name of the operator.
            flags: A handle to the global flags instance to retrieve the
                configuration.
            log_file_name: The file to log the required information to.
        """
        super(PlanningVisualizeOperator, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self._flags = flags
        _, self._world = pylot.simulation.carla_utils.get_world(
            self._flags.carla_host,
            self._flags.carla_port,
            self._flags.carla_timeout)
        if self._world is None:
            raise ValueError("Error connecting to the simulator.")

    @staticmethod
    def setup_streams(input_streams):
        """ This method takes in a single input stream called `waypoints` which
        sends a `planning.messages.WaypointsMessage` to be drawn on
        the screen.

        Args:
            input_streams: A list of streams to take the input from

        Returns:
            An empty list representing that this operator does not publish
            any output streams.
        """
        input_streams.filter(pylot.utils.is_waypoints_stream).add_callback(
            PlanningVisualizeOperator.on_wp_update)
        return []

    def on_wp_update(self, msg):
        """ The callback function that gets called upon receipt of the
        waypoint to be drawn on the screen.

        Args:
            msg: A message of type `planning.messages.WaypointsMessage` to
                be drawn on the screen.
        """
        for waypoint in msg.waypoints:
            loc = carla.Location(waypoint.location.x,
                                 waypoint.location.y,
                                 0.5)

            self._world.debug.draw_point(
                loc,
                size=0.2,
                life_time=self._flags.planning_vis_life_time,
                color=carla.Color(0, 0, 255)
            )
