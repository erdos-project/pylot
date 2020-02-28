import erdos

# Pylot specific imports.
import pylot.utils

DEFAULT_VIS_TIME = 30000.0


class CanBusVisualizerOperator(erdos.Operator):
    """ CanBusVisualizerOperator visualizes the CanBus locations.

    This operator listens on the `can_bus` feed and draws the locations on the
    world simulation screen.

    Attributes:
        _world: A handle to the world to draw the locations on.
    """
    def __init__(self, can_bus_stream, flags, log_file_name=None):
        """ Initializes the CanBusVisualizerOperator with the given
        parameters.

        Args:
            flags: A handle to the global flags instance to retrieve the
                configuration.
            log_file_name: The file to log the required information to.
        """
        can_bus_stream.add_callback(self.on_can_bus_update)
        self._logger = erdos.utils.setup_logging(self.name, log_file_name)
        self._flags = flags

    @staticmethod
    def connect(can_bus_stream):
        return []

    def run(self):
        # Run method is invoked after all operators finished initializing,
        # including the CARLA operator, which reloads the world. Thus, if
        # we get the world here we're sure it is up-to-date.
        from pylot.simulation.utils import get_world
        _, self._world = get_world(self._flags.carla_host,
                                   self._flags.carla_port,
                                   self._flags.carla_timeout)
        if self._world is None:
            raise ValueError("Error connecting to the simulator.")

    def on_can_bus_update(self, msg):
        """ The callback function that gets called upon receipt of the
        CanBus location to be drawn on the screen.

        Args:
            msg: CanBus message
        """
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self.name))
        vehicle_transform = msg.data.transform
        # Draw position. We add 0.5 to z to ensure that the point is above the
        # road surface.
        loc = (vehicle_transform.location +
               pylot.utils.Location(0, 0, 0.5)).as_carla_location()
        self._world.debug.draw_point(loc, size=0.2, life_time=DEFAULT_VIS_TIME)
