from erdos.op import Op
from erdos.utils import setup_csv_logging, setup_logging

from pylot.utils import create_detected_lane_stream, is_can_bus_stream
from pylot.simulation.utils import DetectedLane, to_pylot_location
from pylot.simulation.messages import DetectedLaneMessage
from pylot.simulation.carla_utils import get_world, to_carla_location


class PerfectLaneDetectionOperator(Op):
    def __init__(self,
                 name,
                 output_stream_name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        """ Initializes the PerfectLaneDetectionOperator to use the given name
        and output to the given stream.

        Args:
            name: The name to be given to the operator.
            output_stream_name: The name of the output stream.
            flags: The flags to be used while initializing.
            log_file_name: Name of the log file.
            csv_file_name: Name of the csv file.
        """
        super(PerfectLaneDetectionOperator, self).__init__(name)
        self._flags = flags
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._world_map = None
        self._output_stream_name = output_stream_name
        self._waypoint_precision = 0.05

    @staticmethod
    def setup_streams(input_streams, output_stream_name):
        # Filter the CanBus message stream to get regular update on the
        # vehicle location.
        input_streams.filter(is_can_bus_stream).add_callback(
            PerfectLaneDetectionOperator.on_position_update)

        # Send the detected lane message.
        return [create_detected_lane_stream(output_stream_name)]

    def lateral_shift(self, transform, shift):
        transform.rotation.yaw += 90
        return transform.location + shift * transform.get_forward_vector()

    def on_position_update(self, can_bus_msg):
        """ Callback to be invoked on the receipt of a new update to the 
        position of the vehicle.

        Uses the position of the vehicle to get future waypoints and draw
        lane markings using those waypoints.

        Args:
            can_bus_msg: Contains the current location of the ego vehicle.
        """
        vehicle_location = can_bus_msg.data.transform.location
        lane_waypoints = []
        next_wp = [
            self._world_map.get_waypoint(to_carla_location(vehicle_location))
        ]

        while len(next_wp) == 1:
            lane_waypoints.append(next_wp[0])
            next_wp = next_wp[0].next(self._waypoint_precision)

        # Get the left and right markings of the lane and send it as a message.
        left_markings = [
            to_pylot_location(
                self.lateral_shift(w.transform, -w.lane_width * 0.5))
            for w in lane_waypoints
        ]
        right_markings = [
            to_pylot_location(
                self.lateral_shift(w.transform, w.lane_width * 0.5))
            for w in lane_waypoints
        ]

        # Construct the DetectedLaneMessage.
        detected_lanes = [
            DetectedLane(left, right)
            for left, right in zip(left_markings, right_markings)
        ]
        output_msg = DetectedLaneMessage(detected_lanes, can_bus_msg.timestamp)
        self.get_output_stream(self._output_stream_name).send(output_msg)

    def execute(self):
        """ Retrieve the world instance from the simulator. """
        _, world = get_world(self._flags.carla_host, self._flags.carla_port,
                             self._flags.carla_timeout)
        self._world_map = world.get_map()
        self.spin()
