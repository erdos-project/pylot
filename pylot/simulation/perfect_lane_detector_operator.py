import erdos

from pylot.perception.detection.utils import DetectedLane
from pylot.simulation.messages import DetectedLaneMessage
from pylot.simulation.carla_utils import get_world
import pylot.utils


class PerfectLaneDetectionOperator(erdos.Operator):
    """ Operator that uses the Carla world to perfectly detect lanes."""
    def __init__(self,
                 can_bus_stream,
                 detected_lane_stream,
                 name,
                 flags,
                 log_file_name=None):
        """ Initializes the PerfectLaneDetectionOperator to use the given name
        and output to the given stream.

        Args:
            name: The name to be given to the operator.
            output_stream_name: The name of the output stream.
            flags: The flags to be used while initializing.
            log_file_name: Name of the log file.
        """
        can_bus_stream.add_callback(self.on_position_update,
                                    [detected_lane_stream])
        self._name = name
        self._flags = flags
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._waypoint_precision = 0.05
        _, world = get_world(self._flags.carla_host, self._flags.carla_port,
                             self._flags.carla_timeout)
        self._world_map = world.get_map()

    @staticmethod
    def connect(can_bus_stream):
        detected_lane_stream = erdos.WriteStream()
        return [detected_lane_stream]

    def _lateral_shift(self, transform, shift):
        transform.rotation.yaw += 90
        shifted = transform.location + shift * transform.get_forward_vector()
        return pylot.utils.Location.from_carla_location(shifted)

    def on_position_update(self, can_bus_msg, detected_lane_stream):
        """ Invoked on the receipt of an update to the position of the vehicle.

        Uses the position of the vehicle to get future waypoints and draw
        lane markings using those waypoints.

        Args:
            can_bus_msg: Contains the current location of the ego vehicle.
        """
        self._logger.debug('@{}: received can bus message'.format(
            can_bus_msg.timestamp))
        vehicle_location = can_bus_msg.data.transform.location
        lane_waypoints = []
        next_wp = [
            self._world_map.get_waypoint(vehicle_location.as_carla_location())
        ]

        while len(next_wp) == 1:
            lane_waypoints.append(next_wp[0])
            next_wp = next_wp[0].next(self._waypoint_precision)

        # Get the left and right markings of the lane and send it as a message.
        left_markings = [
            self._lateral_shift(w.transform, -w.lane_width * 0.5)
            for w in lane_waypoints
        ]
        right_markings = [
            self._lateral_shift(w.transform, w.lane_width * 0.5)
            for w in lane_waypoints
        ]

        # Construct the DetectedLaneMessage.
        detected_lanes = [
            DetectedLane(left, right)
            for left, right in zip(left_markings, right_markings)
        ]
        output_msg = DetectedLaneMessage(can_bus_msg.timestamp, detected_lanes)
        detected_lane_stream.send(output_msg)
