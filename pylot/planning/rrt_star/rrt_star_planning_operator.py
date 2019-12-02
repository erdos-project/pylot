import carla
import collections
import itertools
import pylot.utils
import threading

from collections import deque
from erdos.op import Op
from erdos.message import WatermarkMessage
from erdos.utils import setup_csv_logging, setup_logging

from pylot.map.hd_map import HDMap
from pylot.planning.messages import WaypointsMessage
from pylot.simulation.utils import Transform, Location, Rotation
from pylot.simulation.carla_utils import get_map, to_carla_location
from pylot.planning.rrt_star.rrt_star import apply_rrt_star
from pylot.planning.rrt_star.utils import start_target_to_space
from pylot.planning.utils import get_waypoint_vector_and_angle
from pylot.utils import is_within_distance_ahead


DEFAULT_OBSTACLE_LENGTH = 3  # 3 meters from front to back
DEFAULT_OBSTACLE_WIDTH = 2  # 2 meters from side to side
DEFAULT_TARGET_LENGTH = 1  # 1.5 meters from front to back
DEFAULT_TARGET_WIDTH = 1  # 1 meters from side to side
DEFAULT_DISTANCE_THRESHOLD = 10  # 10 meters ahead of ego
DEFAULT_NUM_WAYPOINTS = 50  # 50 waypoints to plan for


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

        self._wp_index = 9
        self._vehicle_transform = None
        self._waypoints = None
        self._map = HDMap(get_map(self._flags.carla_host,
                                  self._flags.carla_port,
                                  self._flags.carla_timeout),
                          log_file_name)
        self._goal_location = carla.Location(*goal_location)

        self._can_bus_msgs = deque()
        self._prediction_msgs = deque()
        self.msg_buffers = [self._can_bus_msgs, self._prediction_msgs]

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
        # get ego info
        can_bus_msg = self._can_bus_msgs.popleft()
        self._vehicle_transform = can_bus_msg.data.transform

        # get obstacles
        obstacle_map = {}
        prediction_msg = self._prediction_msgs.popleft()
        for prediction in prediction_msg.predictions:
            location = prediction.trajectory[self._wp_index]
            if is_within_distance_ahead(self._vehicle_transform.location,
                                        location,
                                        self._vehicle_transform.rotation.yaw,
                                        DEFAULT_DISTANCE_THRESHOLD):
                obstacle_origin = ((location.x - DEFAULT_OBSTACLE_LENGTH / 2, location.y - DEFAULT_OBSTACLE_WIDTH / 2),
                                   (DEFAULT_OBSTACLE_LENGTH, DEFAULT_OBSTACLE_WIDTH))
                obstacle_map[prediction.id] = obstacle_origin

        # compute goals
        self.__update_waypoints()
        target_waypoint = self._waypoints[self._wp_index]
        target_location = target_waypoint.location

        # run rrt*
        starting_state = (self._vehicle_transform.location.x, self._vehicle_transform.location.y)
        target_space = ((target_location.x - DEFAULT_TARGET_LENGTH / 2, target_location.y - DEFAULT_TARGET_WIDTH / 2),
                        (DEFAULT_TARGET_LENGTH, DEFAULT_TARGET_WIDTH))
        state_space = start_target_to_space(starting_state, target_space, DEFAULT_TARGET_LENGTH, DEFAULT_TARGET_WIDTH)
        path, cost = apply_rrt_star(state_space=state_space,
                                    starting_state=starting_state,
                                    target_space=target_space,
                                    obstacle_map=obstacle_map)

        # convert to waypoints if path found, else use default waypoints
        if cost is not None:
            waypoints = deque([
                Transform(location=Location(x=p[0], y=p[1], z=0), rotation=Rotation())
                for p in path
            ])
            waypoints.extend(itertools.islice(self._waypoints, self._wp_index, len(self._waypoints)))
        else:
            waypoints = self._waypoints

        # construct waypoints message
        waypoints = collections.deque(itertools.islice(waypoints, 0, DEFAULT_NUM_WAYPOINTS))  # only take 50 meters
        next_waypoint = waypoints[self._wp_index]
        wp_steer_vector, wp_steer_angle = get_waypoint_vector_and_angle(
            next_waypoint, self._vehicle_transform)
        wp_speed_vector, wp_speed_angle = get_waypoint_vector_and_angle(
            next_waypoint, self._vehicle_transform)
        output_msg = WaypointsMessage(
            msg.timestamp,
            waypoints=waypoints,
            wp_angle=wp_steer_angle,
            wp_vector=wp_steer_vector,
            wp_angle_speed=wp_speed_angle
        )

        # log debug info
        self._logger.debug("Starting state: {}".format(starting_state))
        self._logger.debug("Target space: {}".format(target_space))
        self._logger.debug("State space: {}".format(state_space))
        self._logger.debug("Obstacles: {}".format(obstacle_map))
        self._logger.debug("RRT* Path: {}".format(path))
        self._logger.debug("RRT* Cost: {}".format(cost))

        # send waypoints message
        self.get_output_stream('waypoints').send(output_msg)
        self.get_output_stream('waypoints').send(WatermarkMessage(msg.timestamp))

    def __update_waypoints(self):
        ego_location = to_carla_location(self._vehicle_transform.location)
        self._waypoints = self._map.compute_waypoints(ego_location, self._goal_location)
