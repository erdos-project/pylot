import carla
import pylot.utils
import threading

from collections import deque
from erdos.op import Op
from erdos.utils import setup_csv_logging, setup_logging

from pylot.map.hd_map import HDMap
from pylot.planning.messages import WaypointsMessage
from pylot.simulation.carla_utils import get_map, to_carla_location
from pylot.planning.utils import get_waypoint_vector_and_angle


class RRTStarPlanningOperator(Op):
    """ RRTStar Planning operator for Carla 0.9.x.

    IMPORTANT: Do not use with older Carla versions.
    The operator either receives all the waypoints from the scenario runner
    agent (on the global trajectory stream), or computes waypoints using the
    HD Map.
    """
    def __init__(self,
                 name,
                 flags,
                 goal_location=None,
                 log_file_name=None,
                 csv_file_name=None):
        super(RRTStarPlanningOperator, self).__init__(name)
        self._log_file_name = log_file_name
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._flags = flags

        self._wp_index = 9  # use 10th waypoint for planning
        self._vehicle_transform = None
        self._waypoints = None
        self._map = HDMap(get_map(self._flags.carla_host,
                                  self._flags.carla_port,
                                  self._flags.carla_timeout),
                          log_file_name)
        self._goal_location = carla.Location(*goal_location)

        self._can_bus_msgs = deque()
        self._prediction_msgs = deque()
        self._lock = threading.Lock()

    @staticmethod
    def setup_streams(input_streams):
        input_streams.filter(pylot.utils.is_can_bus_stream).add_callback(
            RRTStarPlanningOperator.on_can_bus_update)
        input_streams.filter(pylot.utils.is_prediction_stream).add_callback(
            RRTStarPlanningOperator.on_prediction_update)
        input_streams.add_completion_callback(
            RRTStarPlanningOperator.on_notification)
        return [pylot.utils.create_waypoints_stream()]

    def on_can_bus_update(self, msg):
        with self._lock:
            self._can_bus_msgs.append(msg)

    def on_prediction_update(self, msg):
        with self._lock:
            self._prediction_msgs.append(msg)

    def on_notification(self, msg):
        can_bus_msg = self._can_bus_msgs.popleft()
        self._vehicle_transform = can_bus_msg.data.transform

        prediction_msg = self._prediction_msgs.popleft()
        self._logger.info("Prediction: {}".format(prediction_msg))

        self.__update_waypoints()
        next_waypoint = self._waypoints[self._wp_index]
        wp_steer_vector, wp_steer_angle = get_waypoint_vector_and_angle(
            next_waypoint, self._vehicle_transform)
        wp_speed_vector, wp_speed_angle = get_waypoint_vector_and_angle(
            next_waypoint, self._vehicle_transform)

        output_msg = WaypointsMessage(
            msg.timestamp,
            waypoints=self._waypoints,
            wp_angle=wp_steer_angle,
            wp_vector=wp_steer_vector,
            wp_angle_speed=wp_speed_angle
        )
        self.get_output_stream('waypoints').send(output_msg)

    def __update_waypoints(self):
        ego_location = to_carla_location(self._vehicle_transform.location)
        self._waypoints = self._map.compute_waypoints(ego_location, self._goal_location)