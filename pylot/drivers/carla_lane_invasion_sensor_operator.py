import erdos
import carla

from pylot.utils import LaneMarking, LaneType
from pylot.simulation.messages import LaneInvasionMessage
from pylot.simulation.utils import get_world, get_vehicle_handle


class CarlaLaneInvasionSensorDriverOperator(erdos.Operator):
    """ Publishes lane invasion events of the ego-vehicle on a stream.

    This operator attaches to the LaneInvasionSensor to the ego-vehicle,
    registers callback functions to the lane-invasion events and publishes it
    to downstream operators.

    Note that more than one lane-invasion event may be published for a single
    timestamp, and it is advised to only work upon receiving watermarks from
    this operator.

    Args:
        ground_vehicle_id_stream (:py:class:`erdos.ReadStream`): Stream on
            which the operator receives the ID of the ego-vehicle.
        lane_invasion_stream (:py:class:`erdos.WriteStream`): Stream on which
            the operator sends the lane-invasion events.
        flags (absl.flags): Object to be used to access the absl flags.
    """
    def __init__(self, ground_vehicle_id_stream, lane_invasion_stream, flags):
        self._vehicle_id_stream = ground_vehicle_id_stream
        self._lane_invasion_stream = lane_invasion_stream
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)

        # The hero vehicle actor object we obtain from carla.
        self._vehicle = None
        self._lane_invasion_sensor = None
        self._map = None

        # Keep track of the last timestamp that we need to close.
        self._first_reading = True
        self._time_to_close = None

    @staticmethod
    def connect(ground_vehicle_id_stream):
        lane_invasion_stream = erdos.WriteStream()
        return [lane_invasion_stream]

    def run(self):
        # Read the vehicle ID from the vehicle ID stream.
        vehicle_id_msg = self._vehicle_id_stream.read()
        vehicle_id = vehicle_id_msg.data
        self._logger.debug("@[{}]: Received Vehicle ID: {}".format(
            vehicle_id_msg.timestamp, vehicle_id))

        # Connect to the world.
        _, world = get_world(self._flags.carla_host, self._flags.carla_port,
                             self._flags.carla_timeout)
        if world is None:
            raise ValueError("There was an issue connecting to the simulator.")

        self._vehicle = get_vehicle_handle(world, vehicle_id)
        self._map = world.get_map()

        # Install the lane-invasion sensor.
        lane_invasion_blueprint = world.get_blueprint_library().find(
            'sensor.other.lane_invasion')

        self._logger.debug("Spawning a lane invasion sensor.")
        self._lane_invasion_sensor = world.spawn_actor(lane_invasion_blueprint,
                                                       carla.Transform(),
                                                       attach_to=self._vehicle)

        # Register the callback on the lane-invasion sensor.
        self._lane_invasion_sensor.listen(self.process_lane_invasion)

        # Register an on_tick function to flow watermarks.
        world.on_tick(self.process_timestamp)

    def process_lane_invasion(self, lane_invasion_event):
        """ Invoked when a lane invasion event is received from the simulation.

        Args:
            lane_invasion_event (:py:class:`carla.LaneInvasionEvent`): A lane-
                invasion event that contains the lane marking which was
                invaded by the ego-vehicle.
        """
        game_time = int(lane_invasion_event.timestamp * 1000)
        self._logger.debug(
            "@[{}]: Received a lane-invasion event from the simulator".format(
                game_time))

        # Create the lane markings that were invaded.
        lane_markings = []
        for lane_marking in lane_invasion_event.crossed_lane_markings:
            lane_marking = LaneMarking.from_carla_lane_marking(lane_marking)
            lane_markings.append(lane_marking)

        # Find the type of the lane that was invaded.
        closest_wp = self._map.get_waypoint(
            self._vehicle.get_transform().location,
            project_to_road=False,
            lane_type=carla.LaneType.Any)
        lane_type = LaneType.NONE
        if closest_wp:
            lane_type = LaneType(closest_wp.lane_type)

        # Create a LaneInvasionMessage.
        timestamp = erdos.Timestamp(coordinates=[game_time])
        msg = LaneInvasionMessage(lane_markings, lane_type, timestamp)

        # Send the LaneInvasionMessage
        self._lane_invasion_stream.send(msg)

    def process_timestamp(self, msg):
        """ Invoked upon each tick of the simulator. This callback is used to
        flow watermarks to the downstream operators.

        Since multiple lane-invasion events can be published by Carla for a
        single timestamp, we need to wait till the next tick of the simulator
        to be sure that all the lane-invasion events for the previous timestamp
        have been sent to downstream operators.

        Args:
            msg (:py:class:`carla.WorldSettings`): A snapshot of the world at
            the given tick. We use this to retrieve the timestamp of the
            simulator.
        """
        sim_time = int(msg.elapsed_seconds * 1000)
        if self._flags.carla_localization_frequency == -1:
            if not self._first_reading:
                self._lane_invasion_stream.send(
                    erdos.WatermarkMessage(
                        erdos.Timestamp(coordinates=[self._time_to_close])))
            self._first_reading = False
            self._time_to_close = sim_time
        else:
            # Ensure that the sensor issues watermarks at the same frequency
            # at which pose watermarks are issued. This is needed because
            # the loggers synchronize on both pose and lane invasion info.
            if self._first_reading:
                self._first_reading = False
                self._time_to_close = sim_time
                self._next_time_to_close = sim_time + int(
                    1.0 / self._flags.carla_fps * 1000)
            else:
                self._lane_invasion_stream.send(
                    erdos.WatermarkMessage(
                        erdos.Timestamp(coordinates=[self._time_to_close])))
                self._time_to_close = self._next_time_to_close
                self._next_time_to_close += int(
                    1.0 / self._flags.carla_localization_frequency * 1000)
