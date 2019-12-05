from erdos.op import Op
from erdos.utils import setup_logging

import carla
import pylot.utils
import pylot.simulation.carla_utils


class IMUVisualizerOperator(Op):
    """ Subscribes to IMU streams and visualizes forward linear + angular
    acceleration."""

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
        IMU message to be drawn on the screen. Draws forward linear and angular
        acceleration.

        Args:
            msg: A message of type `pylot.simulation.messages.IMUMessage` to
                be drawn on the screen.
        """
        transform = msg.transform
        acceleration = msg.acceleration
        gyro = msg.gyro

        loc = carla.Location(transform.location.x,
                             transform.location.y,
                             transform.location.z)
        begin_acc = loc + carla.Location(z=0.5)
        end_acc = begin_acc + carla.Location(acceleration.x,
                                             acceleration.y,
                                             acceleration.z)

        begin_gyro = loc + carla.Location(z=0.5)
        end_gyro = begin_gyro + carla.Location(gyro.x,
                                               gyro.y,
                                               gyro.z)

        self._world.debug.draw_arrow(begin_acc,
                                     end_acc,
                                     arrow_size=0.3,
                                     life_time=30.0,
                                     color=carla.Color(255, 0, 0))
        self._world.debug.draw_arrow(begin_gyro,
                                     end_gyro,
                                     arrow_size=0.3,
                                     life_time=30.0,
                                     color=carla.Color(0, 255, 0))

    def execute(self):
        self.spin()
