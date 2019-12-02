import carla

# ERDOS specific imports.
from erdos.op import Op
from erdos.utils import setup_logging

# Pylot specific imports.
import pylot.utils
import pylot.simulation.carla_utils


class CanBusVisualizeOperator(Op):
    """ CanBusVisualizeOperator visualizes the CanBus locations.

    This operator listens on the `can_bus` feed and draws the locations on the
    world simulation screen.

    Attributes:
        _world: A handle to the world to draw the locations on.
    """

    def __init__(self, name, flags, log_file_name=None):
        """ Initializes the CanBusVisualizeOperator with the given
        parameters.

        Args:
            name: The name of the operator.
            flags: A handle to the global flags instance to retrieve the
                configuration.
            log_file_name: The file to log the required information to.
        """
        super(CanBusVisualizeOperator, self).__init__(name)
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
        """ This method takes in a single input stream called `can_bus` which
        sends a CanBus message to be drawn on the screen.

        Args:
            input_streams: A list of streams to take the input from

        Returns:
            An empty list representing that this operator does not publish
            any output streams.
        """

        input_streams.filter(pylot.utils.is_can_bus_stream).add_callback(
            CanBusVisualizeOperator.on_can_bus_update)
        return []

    def on_can_bus_update(self, msg):
        """ The callback function that gets called upon receipt of the
        CanBus location to be drawn on the screen.

        Args:
            msg: CanBus message
        """
        vehicle_transform = msg.data.transform

        # draw position
        self._world.debug.draw_point(
            carla.Location(x=vehicle_transform.location.x, y=vehicle_transform.location.y, z=0.5),
            size=0.2,
            life_time=30000.0, color=carla.Color(255, 0, 0))

