import erdos

import pylot.utils
from pylot.perception.detection.utils import DetectedLane
from pylot.perception.messages import DetectedLaneMessage
from pylot.simulation.utils import get_map


class PerfectLaneDetectionOperator(erdos.Operator):
    """Operator that uses the Carla world to perfectly detect lanes.

    Args:
        can_bus_stream (:py:class:`erdos.ReadStream`): Stream on which can bus
            info is received.
        detected_lane_stream (:py:class:`erdos.WriteStream`): Stream on which
            the operator writes
            :py:class:`~pylot.perception.messages.DetectedLaneMessage`
            messages.
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, can_bus_stream, detected_lane_stream, flags):
        can_bus_stream.add_callback(self.on_position_update,
                                    [detected_lane_stream])
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._waypoint_precision = 0.05

    @staticmethod
    def connect(can_bus_stream):
        detected_lane_stream = erdos.WriteStream()
        return [detected_lane_stream]

    def run(self):
        # Run method is invoked after all operators finished initializing,
        # including the CARLA operator, which reloads the world. Thus, if
        # we get the world here we're sure it is up-to-date.
        self._world_map = get_map(self._flags.carla_host,
                                  self._flags.carla_port,
                                  self._flags.carla_timeout)

    def _lateral_shift(self, transform, shift):
        transform.rotation.yaw += 90
        shifted = transform.location + shift * transform.get_forward_vector()
        return pylot.utils.Location.from_carla_location(shifted)

    @erdos.profile_method()
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
