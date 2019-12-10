from collections import deque
import threading

import carla

# ERDOS specific imports.
from erdos.op import Op
from erdos.utils import setup_logging

# Pylot specific imports.
import pylot.utils
import pylot.simulation.carla_utils
import pylot.simulation.utils

DEFAULT_VIS_TIME = 0.1


class PredictionVisualizeOperator(Op):
    """ PredictionVisualizeOperator visualizes the predictions released by the
    prediction module.

    This operator listens on the `predictions` and 'can_bus' feed and draws the
    predicted waypoints in the world.

    Attributes:
        _world: A handle to the world to draw the waypoints on.
    """
    def __init__(self, name, flags, log_file_name=None):
        """ Initializes the PredictionVisualizeOperator with the given
        parameters.

        Args:
            name: The name of the operator.
            flags: A handle to the global flags instance to retrieve the
                configuration.
            log_file_name: The file to log the required information to.
        """
        super(PredictionVisualizeOperator, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self._flags = flags
        _, self._world = pylot.simulation.carla_utils.get_world(
            self._flags.carla_host,
            self._flags.carla_port,
            self._flags.carla_timeout)
        if self._world is None:
            raise ValueError("Error connecting to the simulator.")

        self._lock = threading.Lock()
        self._prediction_msgs = deque()
        self._can_bus_msgs = deque()

    @staticmethod
    def setup_streams(input_streams):
        """ This method takes in input streams called `prediction` which
        sends a `prediction.messages.PredictionMessage` to be drawn on
        the screen. `can_bus` stream is used because prediction messages are
        ego-centric.

        Args:
            input_streams: A list of streams to take the input from

        Returns:
            An empty list representing that this operator does not publish
            any output streams.
        """
        input_streams.filter(pylot.utils.is_prediction_stream).add_callback(
            PredictionVisualizeOperator.on_prediction_update)
        input_streams.filter(pylot.utils.is_can_bus_stream).add_callback(
            PredictionVisualizeOperator.on_can_bus_update)
        input_streams.add_completion_callback(
            PredictionVisualizeOperator.on_notification)
        return []

    def on_prediction_update(self, msg):
        """ The callback function that gets called upon receipt of the
        prediction message.

        Args:
            msg: A message of type `prediction.messages.PredictionMessage` to
                be drawn on the screen.
        """
        with self._lock:
            self._prediction_msgs.append(msg)

    def on_can_bus_update(self, msg):
        """ The callback function that gets called upon receipt of the
        can_bus message.

        Args:
            msg: A message of type `prediction.messages.PredictionMessage` to
                be drawn on the screen.
        """
        with self._lock:
            self._can_bus_msgs.append(msg)

    def on_notification(self, msg):
        prediction_msg = self._prediction_msgs.popleft()
        can_bus_msg = self._can_bus_msgs.popleft()
        transform = can_bus_msg.data.transform

        for prediction in prediction_msg.predictions:
            for location in prediction.trajectory:
                pred_trans = pylot.simulation.utils.Transform(
                    location=pylot.simulation.utils.Location(
                        x=location.x,
                        y=location.y,
                        z=location.z,
                    ),
                    rotation=pylot.simulation.utils.Rotation()
                )
                new_trans = transform * pred_trans  # convert to global from ego
                new_location = new_trans.location
                loc = carla.Location(new_location.x,
                                     new_location.y,
                                     new_location.z + 0.5)

                self._world.debug.draw_point(
                    loc,
                    size=0.2,
                    life_time=DEFAULT_VIS_TIME,
                    color=carla.Color(0, 255, 0)
                )
