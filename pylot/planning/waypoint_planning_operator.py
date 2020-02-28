"""Implements an operator that computes waypoints to a goal location."""

from collections import deque
import erdos
import itertools

import pylot.planning.cost_functions
from pylot.planning.messages import WaypointsMessage
from pylot.planning.utils import BehaviorPlannerState
import pylot.utils

DEFAULT_NUM_WAYPOINTS = 50  # 50 waypoints / 50 meters of planning ahead
DEFAULT_TARGET_WAYPOINT = 9  # Use the 10th waypoint for computing speed
WAYPOINT_COMPLETION_THRESHOLD = 0.9


class WaypointPlanningOperator(erdos.Operator):
    """Computes waypoints the ego vehicle must follow.

    If the operator is running in CARLA challenge mode, then it receives all
    the waypoints from the scenario runner agent (on the global trajectory
    stream). Otherwise, it computes waypoints using the HD Map.

    The planner reduces speed/stops whenever it encounters an obstacle,
    and waits for the obstacle to move. It does not implement an obstacle
    avoidance policy.

    Args:
        can_bus_stream (:py:class:`erdos.ReadStream`): Stream on which can bus
            info is received.
        open_drive_stream (:py:class:`erdos.ReadStream`): Stream on which open
            drive string representations are received. The operator can
            construct HDMaps out of the open drive strings.
        global_trajectory_stream (:py:class:`erdos.ReadStream`): Stream on
            which the scenario runner publishes waypoints.
        waypoints_stream (:py:class:`erdos.WriteStream`): Stream on which the
            operator sends waypoints the ego vehicle must follow.
        flags (absl.flags): Object to be used to access absl flags.
        goal_location (:py:class:`~pylot.utils.Location`): The goal location of
            the ego vehicle.
        log_file_name (:obj:`str`, optional): Name of file where log messages
            are written to. If None, then messages are written to stdout.
        csv_file_name (:obj:`str`, optional): Name of file where stats logs are
            written to. If None, then messages are written to stdout.
    """
    def __init__(self,
                 can_bus_stream,
                 open_drive_stream,
                 global_trajectory_stream,
                 obstacles_stream,
                 traffic_lights_stream,
                 waypoints_stream,
                 flags,
                 goal_location=None,
                 log_file_name=None,
                 csv_file_name=None):
        can_bus_stream.add_callback(self.on_can_bus_update)
        open_drive_stream.add_callback(self.on_opendrive_map)
        global_trajectory_stream.add_callback(self.on_global_trajectory)
        obstacles_stream.add_callback(self.on_obstacles_update)
        traffic_lights_stream.add_callback(self.on_traffic_lights_update)
        erdos.add_watermark_callback(
            [can_bus_stream, obstacles_stream, traffic_lights_stream],
            [waypoints_stream], self.on_watermark)

        self._log_file_name = log_file_name
        self._logger = erdos.utils.setup_logging(self.name, log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            self.name + '-csv', csv_file_name)
        self._flags = flags
        # Initialize the state of the behaviour planner.
        # XXX(ionel): The behaviour planner is not ready yet.
        self.__initialize_behaviour_planner()
        # We do not know yet the vehicle's location.
        self._vehicle_transform = None
        self._goal_location = goal_location
        self._map = None
        # Deque of waypoints the vehicle must follow. The waypoints are either
        # received on the global trajectory stream when running using the
        # scenario runner, or computed using the Carla global planner when
        # running in stand-alone mode. The waypoints are Pylot transforms.
        self._waypoints = deque()
        self._can_bus_msgs = deque()
        self._obstacles_msgs = deque()
        self._traffic_light_msgs = deque()

    @staticmethod
    def connect(can_bus_stream, open_drive_stream, global_trajectory_stream,
                obstacles_stream, traffic_lights_stream):
        waypoints_stream = erdos.WriteStream()
        return [waypoints_stream]

    def run(self):
        # Run method is invoked after all operators finished initializing,
        # including the CARLA operator, which reloads the world. Thus, if
        # we get the map here we're sure it is up-to-date.

        # We're not running in challenge mode if no track flag is present.
        # Thus, we can directly get the map from the simulator.
        if not hasattr(self._flags, 'track'):
            from pylot.map.hd_map import HDMap
            from pylot.simulation.utils import get_map
            self._map = HDMap(
                get_map(self._flags.carla_host, self._flags.carla_port,
                        self._flags.carla_timeout), self._log_file_name)
            self._logger.info('Planner running in stand-alone mode')
            # Recompute waypoints upon each run.
            self._recompute_waypoints = True
        else:
            # Do not recompute waypoints upon each run.
            self._recompute_waypoints = False

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
        self._map = HDMap(carla.Map('map', msg.data), self._log_file_name)

    def on_global_trajectory(self, msg):
        """Invoked whenever a message is received on the trajectory stream.

        Args:
            msg (:py:class:`~erdos.message.Message`): Message that contains
                a list of waypoints to the goal location.
        """
        self._logger.debug('@{}: global trajectory has {} waypoints'.format(
            msg.timestamp, len(msg.data)))
        if len(msg.data) > 0:
            # The last waypoint is the goal location.
            self._goal_location = msg.data[-1][0].location
        else:
            # Trajectory does not contain any waypoints. We assume we have
            # arrived at destionation.
            self._goal_location = self._vehicle_transform.location
        assert self._goal_location, 'Planner does not have a goal'
        self._waypoints = deque()
        for waypoint_option in msg.data:
            self._waypoints.append(waypoint_option[0])

    def on_can_bus_update(self, msg):
        """Invoked whenever a message is received on the can bus stream.

        Args:
            msg (:py:class:`~erdos.message.Message`): Message that contains
                info about the ego vehicle.
        """
        self._logger.debug('@{}: received can bus message'.format(
            msg.timestamp))
        self._can_bus_msgs.append(msg)

    def on_obstacles_update(self, msg):
        self._logger.debug('@{}: obstacles update'.format(msg.timestamp))
        self._obstacles_msgs.append(msg)

    def on_traffic_lights_update(self, msg):
        self._logger.debug('@{}: traffic lights update'.format(msg.timestamp))
        self._traffic_light_msgs.append(msg)

    def on_watermark(self, timestamp, waypoints_stream):
        self._logger.debug('@{}: received watermark'.format(timestamp))

        # Get hero vehicle info.
        can_bus_msg = self._can_bus_msgs.popleft()
        self._vehicle_transform = can_bus_msg.data.transform
        tl_msg = self._traffic_light_msgs.popleft()
        obstacles_msg = self._obstacles_msgs.popleft()

        if self._recompute_waypoints:
            self._waypoints = self._map.compute_waypoints(
                self._vehicle_transform.location, self._goal_location)
        self.__remove_completed_waypoints()
        if not self._waypoints or len(self._waypoints) == 0:
            # If waypoints are empty (e.g., reached destination), set waypoint
            # to current vehicle location.
            self._waypoints = deque([self._vehicle_transform])

        wp_vector, wp_angle = \
            pylot.planning.utils.compute_waypoint_vector_and_angle(
                self._vehicle_transform, self._waypoints,
                DEFAULT_TARGET_WAYPOINT)

        speed_factor, _ = pylot.planning.utils.stop_for_agents(
            self._vehicle_transform.location, wp_angle, wp_vector,
            obstacles_msg.obstacles, tl_msg.obstacles, self._flags,
            self._logger, self._map, timestamp)

        target_speed = speed_factor * self._flags.target_speed
        self._logger.debug('@{}: computed speed factor: {}'.format(
            timestamp, speed_factor))
        self._logger.debug('@{}: computed target speed: {}'.format(
            timestamp, target_speed))
        head_waypoints = deque(
            itertools.islice(self._waypoints, 0, DEFAULT_NUM_WAYPOINTS))
        target_speeds = deque(
            [target_speed for _ in range(len(head_waypoints))])
        waypoints_stream.send(
            WaypointsMessage(timestamp, head_waypoints, target_speeds))

    def __remove_completed_waypoints(self):
        """Removes waypoints that the ego vehicle has already completed.

        The method first finds the closest waypoint, removes all waypoints
        that are before the closest waypoint, and finally removes the closest
        waypoint if the ego vehicle is very close to it (i.e., close to
        completion).
        """
        min_dist = 10000000
        min_index = 0
        index = 0
        for waypoint in self._waypoints:
            # XXX(ionel): We only check the first 10 waypoints.
            if index > 10:
                break
            dist = waypoint.location.distance(self._vehicle_transform.location)
            if dist < min_dist:
                min_dist = dist
                min_index = index

        # Remove waypoints that are before the closest waypoint. The ego
        # vehicle already completed them.
        while min_index > 0:
            self._waypoints.popleft()
            min_index -= 1

        # The closest waypoint is almost complete, remove it.
        if min_dist < WAYPOINT_COMPLETION_THRESHOLD:
            self._waypoints.popleft()

    def __initialize_behaviour_planner(self):
        # State the planner is in.
        self._state = BehaviorPlannerState.READY
        # Cost functions. Output between 0 and 1.
        self._cost_functions = [
            pylot.planning.cost_functions.cost_speed,
            pylot.planning.cost_functions.cost_lane_change,
            pylot.planning.cost_functions.cost_inefficiency
        ]
        reach_speed_weight = 10**5
        reach_goal_weight = 10**6
        efficiency_weight = 10**4
        # How important a cost function is.
        self._function_weights = [
            reach_speed_weight, reach_goal_weight, efficiency_weight
        ]

    def __best_transition(self, vehicle_transform, predictions):
        """ Computes most likely state transition from current state."""
        # Get possible next state machine states.
        possible_next_states = self.__successor_states()
        best_next_state = None
        min_state_cost = 10000000
        for state in possible_next_states:
            # Generate trajectory for next state.
            vehicle_info, trajectory_for_state = self.__generate_trajectory(
                state, vehicle_transform, predictions)
            state_cost = 0
            # Compute the cost of the trajectory.
            for i in range(len(self._cost_functions)):
                cost_func = self._cost_functions[i](vehicle_info, predictions,
                                                    trajectory_for_state)
                state_cost += self._function_weights[i] * cost_func
            # Check if it's the best trajectory.
            if best_next_state is None or state_cost < min_state_cost:
                best_next_state = state
                min_state_cost = state_cost
        return best_next_state

    def __generate_trajectory(self, next_state, vehicle_transform,
                              predictions):
        raise NotImplementedError

    def __successor_states(self):
        """ Returns possible state transitions from current state."""
        if self._state == BehaviorPlannerState.READY:
            return [BehaviorPlannerState.READY, BehaviorPlannerState.KEEP_LANE]
        elif self._state == BehaviorPlannerState.KEEP_LANE:
            # 1) keep_lane -> prepare_lane_change_left
            # 2) keep_lane -> prepare_lane_change_right
            return [
                BehaviorPlannerState.KEEP_LANE,
                BehaviorPlannerState.PREPARE_LANE_CHANGE_LEFT,
                BehaviorPlannerState.PREPARE_LANE_CHANGE_RIGHT
            ]
        elif self._state == BehaviorPlannerState.PREPARE_LANE_CHANGE_LEFT:
            # 1) prepare_lane_change_left -> keep_lane
            # 2) prepare_lane_change_left -> lange_change_left
            return [
                BehaviorPlannerState.KEEP_LANE,
                BehaviorPlannerState.PREPARE_LANE_CHANGE_LEFT,
                BehaviorPlannerState.LANE_CHANGE_LEFT
            ]
        elif self._state == BehaviorPlannerState.LANE_CHANGE_LEFT:
            # 1) lange_change_left -> keep_lane
            return [
                BehaviorPlannerState.KEEP_LANE,
                BehaviorPlannerState.LANE_CHANGE_LEFT
            ]
        elif self._state == BehaviorPlannerState.PREPARE_LANE_CHANGE_RIGHT:
            # 1) prepare_lane_change_right -> keep_lane
            # 2) prepare_lane_change_right -> lange_change_right
            return [
                BehaviorPlannerState.KEEP_LANE,
                BehaviorPlannerState.PREPARE_LANE_CHANGE_RIGHT,
                BehaviorPlannerState.LANE_CHANGE_RIGHT
            ]
        elif self._state == BehaviorPlannerState.LANE_CHANGE_RIGHT:
            # 1) lane_change_right -> keep_lane
            return [
                BehaviorPlannerState.KEEP_LANE,
                BehaviorPlannerState.LANE_CHANGE_RIGHT
            ]
        else:
            raise ValueError('Unexpected vehicle state {}'.format(self._state))
