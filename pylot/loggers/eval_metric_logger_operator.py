import erdos

from pylot.utils import LaneMarkingColor, LaneType, LaneMarkingType


class EvalMetricLoggerOperator(erdos.Operator):
    """ Logs the evaluation metrics of the ego-vehicle in the CSV file.

    This operator logs the lateral and longitudinal acceleration and jerk,
    along with the traffic infractions, lane invasions and collisions
    committed by the ego vehicle.

    Args:
        collision_stream (:py:class:`erdos.ReadStream`): Stream on which the
            collision events are received.
        lane_invasion_stream (:py:class:`erdos.ReadStream`): Stream on which
            the lane invasion events are received.
        traffic_light_invasion_stream (:py:class:`erdos.ReadStream`): Stream
            on which the traffic light invasion events are received.
        flags (absl.flags): Object to be used to access the absl flags.
    """

    def __init__(self, collision_stream, lane_invasion_stream,
                 traffic_light_invasion_stream, imu_stream, flags):
        # Save the streams.
        self._collision_stream = collision_stream
        self._lane_invasion_stream = lane_invasion_stream
        self._traffic_light_invasion_stream = traffic_light_invasion_stream
        self._imu_stream = imu_stream

        # Register callbacks.
        collision_stream.add_callback(self.on_collision_update)
        lane_invasion_stream.add_callback(self.on_lane_invasion_update)
        traffic_light_invasion_stream.add_callback(
            self.on_traffic_light_invasion_update)
        imu_stream.add_callback(self.on_imu_update)

        # Initialize a logger.
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            self.config.name + '-csv', self.config.csv_log_file_name)

        # Save the last acceleration and timestamp.
        self._last_lateral_acc, self._last_longitudinal_acc = None, None
        self._last_timestamp = None

    @staticmethod
    def connect(collision_stream, lane_invasion_stream,
                traffic_light_invasion_stream, imu_stream):
        return []

    def on_collision_update(self, msg):
        """ Invoked upon receipt of a collision update.

        Logs the collided actor and the intensity of the collision.

        Args:
            msg (:py:class:`pylot.simulation.messages.CollisionMessage`): The
                collision message sent by the sensor.
        """
        self._csv_logger.info('collision,{},{}'.format(
            msg.collided_actor.split('.')[0], msg.intensity))

    def on_lane_invasion_update(self, msg):
        """ Invoked upon receipt of a lane invasion update.

        Logs the type of invasion and the timestamp.

        Args:
            msg (:py:class:`pylot.simulation.messages.LaneInvasionMessage`):
                The lane invasion message sent by the sensor.
        """
        time = msg.timestamp.coordinates[0]
        # We log lane invasion events only if they are illegal.
        if any(map(check_illegal_lane_invasion, msg.lane_markings)):
            self._csv_logger.info('invasion,lane,{}'.format(time))

        if msg.lane_type == LaneType.SIDEWALK:
            self._csv_logger.info('invasion,sidewalk,{}'.format(time))

    def check_illegal_lane_invasion(lane_marking):
        """ Checks if the invaded lane was an illegal lane.

        An illegal lane is either traversed via a Yellow lane marking, or
        by breaking a solid white line.

        Args:
            lane_marking (:py:class:`pylot.utils.LaneMarking`): A lane marking.

        Returns:
            A boolean specifying whether the lane invaded was legal.
        """
        # If we invaded a lane in the opposite direction.
        if lane_marking.marking_color == LaneMarkingColor.YELLOW:
            return True

        # If we invaded a lane when we were not supposed to.
        if lane_marking.marking_type in (LaneMarkingType.SOLID,
                                         LaneMarkingType.SOLIDSOLID,
                                         LaneMarkingType.BROKENSOLID):
            return True

    def on_traffic_light_invasion_update(self, msg):
        """ Invoked upon receipt of a traffic light invasion update.

        Logs the timestamp of the invasion.

        Args:
            msg
            (:py:class:`pylot.simulation.messages.TrafficInfractionMessage`):
                The traffic infraction message sent by the sensor.
        """
        time = msg.timestamp.coordinates[0]
        self._csv_logger.info('invasion,red_light,{}'.format(time))

    def on_imu_update(self, msg):
        """ Invoked upon receipt of an IMU sensor update.

        Logs the lateral and longitudinal acceleration and jerk.

        Args:
            msg (:py:class:`pylot.perception.messages.IMUMessage`): The IMU
                message sent by the sensor.
        """
        lateral_acc, longitudinal_acc = msg.acceleration.y, msg.acceleration.x
        # Log the lateral and the longitudinal acceleration.
        self._csv_logger.info('acceleration,lateral,{}'.format(lateral_acc))
        self._csv_logger.info(
            'acceleration,longitudinal,{}'.format(longitudinal_acc))

        # Calculate the jerk, and log both lateral and longitudinal jerk.
        if self._last_timestamp:
            time_diff = msg.timestamp.coordinates[0] - \
                self._last_timestamp.coordinates[0]
            lateral_jerk = (self._last_lateral_acc - lateral_acc) / time_diff
            longitudinal_jerk = (self._last_longitudinal_acc -
                                 longitudinal_acc) / time_diff
            self._csv_logger.info('jerk,lateral,{}'.format(lateral_jerk))
            self._csv_logger.info(
                'jerk,longitudinal,{}'.format(longitudinal_jerk))

        # Save the new acceleration and timestamp.
        self._last_lateral_acc = lateral_acc
        self._last_longitudinal_acc = longitudinal_acc
        self._last_timestamp = msg.timestamp
