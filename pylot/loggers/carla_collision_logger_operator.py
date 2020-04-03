from collections import deque, defaultdict
import erdos


class CarlaCollisionLoggerOperator(erdos.Operator):
    """ Logs the collision messages published by the collision sensor to a CSV
    file. The format of the log is defined as:
        <location_of_ego_vehicle>, <collided_object>, <impulse>, <intensity>

    Args:
        collision_stream (:py:class:`erdos.ReadStream`): Stream on which the
            operator receives the collision events with respect to the
            ego-vehicle.
        pose_stream (:py:class:`erdos.ReadStream`): Stream on which the
            operator receives the pose of the ego-vehicle.
        flags (absl.flags): Object to be used to access the absl flags.
    """

    def __init__(self, collision_stream, pose_stream, flags):
        # Add the callbacks to both the pose stream and the collision stream.
        pose_stream.add_callback(self.on_pose_update)
        collision_stream.add_callback(self.on_collision_update)

        # Process on each tick and check if we have collision events.
        erdos.add_watermark_callback([collision_stream, pose_stream], [],
                                     self.on_watermark)
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            self.config.name + '-csv', self.config.csv_log_file_name)

        # Data structures that maintain the state necessary for logging.
        self._ego_transform_msg = None
        self._collisions = defaultdict(list)

    @staticmethod
    def connect(collision_stream, pose_stream):
        return []

    def on_pose_update(self, msg):
        """ Invoked when an update to the pose of the ego-vehicle is received.

        Args:
            msg (:py:class:`erdos.Message`): The message contains the latest
                pose of the ego-vehicle.
        """
        self._logger.debug("@{}: received pose update.".format(msg.timestamp))
        self._ego_transform_msg = msg

    def on_collision_update(self, msg):
        """ Invoked when a collision of the ego-vehicle is received.

        Args:
            msg (:py:class:`pylot.simulation.messages.CollisionMessage`): The
                message contains the collision event with respect to the
                ego-vehicle.
        """
        self._logger.debug("@{}: received collision update.".format(
            msg.timestamp))
        self._collisions[msg.timestamp].append(msg)

    def on_watermark(self, timestamp):
        """ Invoked upon receipt of a watermark from the collision events
        stream, and the pose stream.

        Retrieves the set of the collision events from the given timestamp,
        and writes them out to a CSV file.

        Args:
            timestamp (:py:class:`erdos.Timestamp`): The timestamp of the
                watermark.
        """
        self._logger.debug("@{}: received watermark.".format(timestamp))
        if self._ego_transform_msg.timestamp != timestamp:
            raise ValueError(
                "Expected timestamp {} for the pose, got {}".format(
                    timestamp, self._ego_transform_msg.timestamp))
        collision_msgs = self._collisions.pop(timestamp, [])

        for collision in collision_msgs:
            self._csv_logger.info('collision,{},{},{},{}'.format(
                self._ego_transform_msg.data.transform.location,
                collision.collided_actor, collision.impulse,
                collision.intensity))
