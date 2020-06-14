"""Implements an operator that visualizes forward linear acceleration."""

import carla
import erdos

import pylot.simulation.utils
import pylot.utils


class IMUVisualizerOperator(erdos.Operator):
    """ Subscribes to IMU stream and visualizes forward linear acceleration.

    Args:
        imu_stream: (:py:class:`erdos.ReadStream`): The stream on which
            :py:class:`~pylot.localization.messages.IMUMessage` are received.
        flags (absl.flags): Object to be used to access absl flags.

    Attributes:
        _logger (:obj:`logging.Logger`): Instance to be used to log messages.
        _flags (absl.flags): Object to be used to access absl flags.
        _world (carla.World): Object used to draw in the Carla world.
    """
    def __init__(self, imu_stream, flags):
        imu_stream.add_callback(self.on_imu_update)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags

    @staticmethod
    def connect(imu_stream):
        return []

    def run(self):
        # Run method is invoked after all operators finished initializing,
        # including the CARLA operator, which reloads the world. Thus, if
        # we get the world here we're sure it is up-to-date.
        _, self._world = pylot.simulation.utils.get_world(
            self._flags.carla_host, self._flags.carla_port,
            self._flags.carla_timeout)
        if self._world is None:
            raise ValueError("Error connecting to the simulator.")

    def on_imu_update(self, msg):
        """Callback that gets called upon receipt of the IMU message.

        Draws forward linear acceleration on the screen.

        Args:
            msg: A message of type `pylot.localization.messages.IMUMessage` to
                be drawn on the screen.
        """
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self.config.name))
        transform = msg.transform
        # Acceleration measured in ego frame, not global
        # z acceleration not useful for visualization so set to 0
        rotation_transform = carla.Transform(
            location=carla.Location(0, 0, 0),
            rotation=transform.rotation.as_carla_rotation())
        acceleration = msg.acceleration.as_carla_vector()
        rotated_acceleration = rotation_transform.transform(
            carla.Location(acceleration.x, acceleration.y, 0))

        # Construct arrow.
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
