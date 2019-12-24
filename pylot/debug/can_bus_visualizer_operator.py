import carla
import erdust

# Pylot specific imports.
import pylot.utils
import pylot.simulation.carla_utils

DEFAULT_VIS_TIME = 30000.0


class CanBusVisualizerOperator(erdust.Operator):
    """ CanBusVisualizerOperator visualizes the CanBus locations.

    This operator listens on the `can_bus` feed and draws the locations on the
    world simulation screen.

    Attributes:
        _world: A handle to the world to draw the locations on.
    """
    def __init__(self, can_bus_stream, name, flags, log_file_name=None):
        """ Initializes the CanBusVisualizerOperator with the given
        parameters.

        Args:
            name: The name of the operator.
            flags: A handle to the global flags instance to retrieve the
                configuration.
            log_file_name: The file to log the required information to.
        """
        can_bus_stream.add_callback(self.on_can_bus_update)
        self._name = name
        self._logger = erdust.utils.setup_logging(name, log_file_name)
        self._flags = flags
        _, self._world = pylot.simulation.carla_utils.get_world(
            self._flags.carla_host, self._flags.carla_port,
            self._flags.carla_timeout)
        if self._world is None:
            raise ValueError("Error connecting to the simulator.")

    @staticmethod
    def connect(can_bus_stream):
        return []

    def on_can_bus_update(self, msg):
        """ The callback function that gets called upon receipt of the
        CanBus location to be drawn on the screen.

        Args:
            msg: CanBus message
        """
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self._name))
        vehicle_transform = msg.data.transform
        # draw position
        self._world.debug.draw_point(carla.Location(
            x=vehicle_transform.location.x,
            y=vehicle_transform.location.y,
            z=vehicle_transform.location.z + 0.5),
                                     size=0.2,
                                     life_time=DEFAULT_VIS_TIME,
                                     color=carla.Color(255, 0, 0))
