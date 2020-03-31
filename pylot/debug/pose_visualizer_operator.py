import erdos

# Pylot specific imports.
import pylot.utils

DEFAULT_VIS_TIME = 30000.0


class PoseVisualizerOperator(erdos.Operator):
    """ PoseVisualizerOperator visualizes the Pose locations.

    This operator listens on the pose stream and draws the locations on the
    world simulation screen.

    Attributes:
        _world: A handle to the world to draw the locations on.
    """
    def __init__(self, pose_stream, flags):
        pose_stream.add_callback(self.on_pose_update)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags

    @staticmethod
    def connect(pose_stream):
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

    def on_pose_update(self, msg):
        """ The callback function that gets called upon receipt of the
        Pose location to be drawn on the screen.

        Args:
            msg: Pose message
        """
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self.config.name))
        vehicle_transform = msg.data.transform
        # Draw position. We add 0.5 to z to ensure that the point is above the
        # road surface.
        loc = (vehicle_transform.location +
               pylot.utils.Location(0, 0, 0.5)).as_carla_location()
        self._world.debug.draw_point(loc, size=0.2, life_time=DEFAULT_VIS_TIME)
