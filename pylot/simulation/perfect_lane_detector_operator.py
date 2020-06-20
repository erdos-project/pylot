import erdos

from pylot.perception.messages import LanesMessage


class PerfectLaneDetectionOperator(erdos.Operator):
    """Operator that uses the Carla world to perfectly detect lanes.

    Args:
        pose_stream (:py:class:`erdos.ReadStream`): Stream on which pose
            info is received.
        open_drive_stream (:py:class:`erdos.ReadStream`): Stream on which open
            drive string representations are received. The operator can
            construct HDMaps out of the open drive strings.
        detected_lane_stream (:py:class:`erdos.WriteStream`): Stream on which
            the operator writes
            :py:class:`~pylot.perception.messages.LanesMessage` messages.
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, pose_stream, open_drive_stream, detected_lane_stream,
                 flags):
        pose_stream.add_callback(self.on_position_update,
                                 [detected_lane_stream])
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)

    @staticmethod
    def connect(pose_stream, open_drive_stream):
        detected_lane_stream = erdos.WriteStream()
        return [detected_lane_stream]

    def run(self):
        # Run method is invoked after all operators finished initializing,
        # including the CARLA operator, which reloads the world. Thus, if
        # we get the world here we're sure it is up-to-date.
        if self._flags.execution_mode == 'simulation':
            from pylot.map.hd_map import HDMap
            from pylot.simulation.utils import get_map
            self._map = HDMap(
                get_map(self._flags.carla_host, self._flags.carla_port,
                        self._flags.carla_timeout))
            from pylot.simulation.utils import get_world
            _, self._world = get_world(self._flags.carla_host,
                                       self._flags.carla_port,
                                       self._flags.carla_timeout)

    def on_opendrive_map(self, msg):
        """Invoked whenever a message is received on the open drive stream.

        Args:
            msg (:py:class:`~erdos.message.Message`): Message that contains
                the open drive string.
        """
        self._logger.debug('@{}: received open drive message'.format(
            msg.timestamp))
        try:
            import carla
        except ImportError:
            raise Exception('Error importing carla.')
        self._logger.info('Initializing HDMap from open drive stream')
        from pylot.map.hd_map import HDMap
        self._map = HDMap(carla.Map('map', msg.data))

    @erdos.profile_method()
    def on_position_update(self, pose_msg, detected_lane_stream):
        """ Invoked on the receipt of an update to the position of the vehicle.

        Uses the position of the vehicle to get future waypoints and draw
        lane markings using those waypoints.

        Args:
            pose_msg: Contains the current location of the ego vehicle.
        """
        self._logger.debug('@{}: received pose message'.format(
            pose_msg.timestamp))
        vehicle_location = pose_msg.data.transform.location
        if self._map:
            lanes = [self._map.get_lane(vehicle_location)]
            lanes[0].draw_on_world(self._world)
        else:
            self._logger.debug('@{}: map is not ready yet'.format(
                pose_msg.timestamp))
            lanes = []
        output_msg = LanesMessage(pose_msg.timestamp, lanes)
        detected_lane_stream.send(output_msg)
