from erdos.op import Op
from erdos.utils import setup_logging

import carla
import pylot.utils
import pylot.simulation.carla_utils


class IMUVisualizerOperator(Op):
    """ Subscribes to IMU streams and visualizes forward linear acceleration."""

    def __init__(self, name, flags, log_file_name=None):
        super(IMUVisualizerOperator, self).__init__(name)
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
        input_streams.filter(pylot.utils.is_imu_stream).add_callback(
            IMUVisualizerOperator.on_imu_update)
        return []

    def on_imu_update(self, msg):
        """ The callback function that gets called upon receipt of the
        IMU message to be drawn on the screen. Draws forward linear acceleration.

        Args:
            msg: A message of type `pylot.simulation.messages.IMUMessage` to
                be drawn on the screen.
        """

        transform = msg.transform

        # acceleration measured in ego frame, not global
        # z acceleration not useful for visualization so set to 0
        rotation_transform = carla.Transform(
            location=carla.Location(0, 0, 0),
            rotation=transform.rotation.as_carla_rotation()
        )
        acceleration = msg.acceleration.as_carla_vector()
        rotated_acceleration = rotation_transform.transform(
            carla.Location(acceleration.x, acceleration.y, 0)
        )

        # construct arrow
        loc = transform.location.as_carla_location()
        begin_acc = loc + carla.Location(z=0.5)
        end_acc = begin_acc + carla.Location(rotated_acceleration.x,
                                             rotated_acceleration.y,
                                             0)  # not useful for visualization

        # draw arrow
        self._logger.debug("Acc: {}".format(rotated_acceleration))
        self._world.debug.draw_arrow(begin_acc,
                                     end_acc,
                                     arrow_size=0.1,
                                     life_time=0.1,
                                     color=carla.Color(255, 0, 0))

    def execute(self):
        self.spin()
