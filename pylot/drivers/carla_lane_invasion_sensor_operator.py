"""This module implements an operator that publishes information whenever
the ego vehicle invades a lane on the opposite side of the road.
"""

from carla import LaneType, Transform

import erdos
from erdos.operator import OneInOneOut

import pylot.utils
from pylot.simulation.messages import LaneInvasionMessageTuple
from pylot.simulation.utils import get_vehicle_handle, get_world


class CarlaLaneInvasionSensorDriverOperator(OneInOneOut):
    """Publishes lane invasion events of the ego-vehicle on a stream.

    This operator attaches to the LaneInvasionSensor to the ego-vehicle,
    registers callback functions to the lane-invasion events and publishes it
    to downstream operators.

    Note that more than one lane-invasion event may be published for a single
    timestamp, and it is advised to only work upon receiving watermarks from
    this operator.

    Args:
        flags (absl.flags): Object to be used to access the absl flags.
    """
    def __init__(self, flags):
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        # The hero vehicle actor object we obtain from the simulator.
        self._vehicle = None
        self._lane_invasion_sensor = None
        self._map = None

    def run(self, read_stream, write_stream):
        # Read the vehicle ID from the vehicle ID stream.
        vehicle_id_msg = read_stream.read()
        vehicle_id = vehicle_id_msg.data
        self._logger.debug("@{}: Received Vehicle ID: {}".format(
            vehicle_id_msg.timestamp, vehicle_id))

        # Connect to the world.
        _, world = get_world(self._flags.simulator_host,
                             self._flags.simulator_port,
                             self._flags.simulator_timeout)

        self._vehicle = get_vehicle_handle(world, vehicle_id)
        self._map = world.get_map()

        # Install the lane-invasion sensor.
        lane_invasion_blueprint = world.get_blueprint_library().find(
            'sensor.other.lane_invasion')

        self._logger.debug("Spawning a lane invasion sensor.")
        self._lane_invasion_sensor = world.spawn_actor(lane_invasion_blueprint,
                                                       Transform(),
                                                       attach_to=self._vehicle)

        # Register the callback on the lane-invasion sensor.
        def _process_lane_invasion(lane_invasion_event):
            return self.process_lane_invasion(lane_invasion_event,
                                              write_stream)

        self._lane_invasion_sensor.listen(self.process_lane_invasion)

    def process_lane_invasion(self, lane_invasion_event, write_stream):
        """Invoked when a lane invasion event is received from the simulation.

        The lane-invasion event contains the lane marking which was invaded by
        the ego-vehicle.
        """
        game_time = int(lane_invasion_event.timestamp * 1000)
        self._logger.debug(
            "@[{}]: Received a lane-invasion event from the simulator".format(
                game_time))
        # Create the lane markings that were invaded.
        lane_markings = []
        for lane_marking in lane_invasion_event.crossed_lane_markings:
            lane_marking = pylot.utils.LaneMarking.from_simulator_lane_marking(
                lane_marking)
            lane_markings.append(lane_marking)

        # Find the type of the lane that was invaded.
        closest_wp = self._map.get_waypoint(
            self._vehicle.get_transform().location,
            project_to_road=False,
            lane_type=LaneType.Any)
        lane_type = pylot.utils.LaneType.NONE
        if closest_wp:
            lane_type = pylot.utils.LaneType(closest_wp.lane_type)

        # Create a LaneInvasionMessage.
        timestamp = erdos.Timestamp(coordinates=[game_time])
        msg = LaneInvasionMessageTuple(lane_markings, lane_type)

        # Send the LaneInvasionMessage
        write_stream.send(erdos.Message(timestamp, msg))
        # TODO(ionel): This code will fail if process_lane_invasion is
        # called twice for the same timestamp.
        write_stream.send(erdos.WatermarkMessage(timestamp))
