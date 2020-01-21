import collections
import erdos
import itertools

from pylot.map.hd_map import HDMap
import pylot.planning.cost_functions
from pylot.planning.messages import WaypointsMessage
from pylot.planning.utils import get_waypoint_vector_and_angle, \
    BehaviorPlannerState
from pylot.simulation.utils import get_map
import pylot.utils

DEFAULT_NUM_WAYPOINTS = 50
WAYPOINT_COMPLETION_THRESHOLD = 0.9


class WaypointPlanningOperator(erdos.Operator):
    """ Operator that computes waypoints.

    The operator either receives all the waypoints from the scenario runner
    agent (on the global trajectory stream), or computes waypoints using the
    HD Map.
    """
    def __init__(self,
                 can_bus_stream,
                 open_drive_stream,
                 global_trajectory_stream,
                 waypoints_stream,
                 name,
                 flags,
                 goal_location=None,
                 log_file_name=None,
                 csv_file_name=None):
        can_bus_stream.add_callback(self.on_can_bus_update, [waypoints_stream])
        open_drive_stream.add_callback(self.on_opendrive_map)
        global_trajectory_stream.add_callback(self.on_global_trajectory)

        self._log_file_name = log_file_name
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            name + '-csv', csv_file_name)
        self._flags = flags
        # Initialize the state of the behaviour planner.
        # XXX(ionel): The behaviour planner is not ready yet.
        self.__initialize_behaviour_planner()
        # We do not know yet the vehicle's location.
        self._vehicle_transform = None
        # Deque of waypoints the vehicle must follow. The waypoints are either
        # received on the global trajectory stream when running using the
        # scenario runner, or computed using the Carla global planner when
        # running in stand-alone mode. The waypoints are Pylot transforms.
        self._waypoints = collections.deque()
        # The operator picks the wp_num_steer-th waypoint to compute the angle
        # it has to steer by when taking a turn.
        self._wp_num_steer = 9  # use 10th waypoint for steering
        # The operator picks the wp_num_speed-th waypoint to compute the angle
        # it has to steer by when driving straight.
        self._wp_num_speed = 4  # use tth waypoint for speed
        # We're not running in challenge mode if no track flag is present.
        # Thus, we can directly get the map from the simulator.
        if not hasattr(self._flags, 'track'):
            self._map = HDMap(
                get_map(self._flags.carla_host, self._flags.carla_port,
                        self._flags.carla_timeout), log_file_name)
            self._logger.info('Planner running in stand-alone mode')
            assert goal_location, 'Planner has not received a goal location'
            # Transform goal location to carla.Location
            self._goal_location = goal_location
            # Recompute waypoints upon each run.
            self._recompute_waypoints = True
        else:
            # Do not recompute waypoints upon each run.
            self._recompute_waypoints = False
            # TODO(ionel): HACK! In Carla challenge track 1 and 2 the waypoints
            # are coarse grained (30 meters apart). We pick the first waypoint
            # to compute the angles. However, we should instead implement
            # trajectory planning.
            if self._flags.track == 1 or self._flags == 2:
                self._wp_num_steer = 1
                self._wp_num_speed = 1

    @staticmethod
    def connect(can_bus_stream, open_drive_stream, global_trajectory_stream):
        waypoints_stream = erdos.WriteStream()
        return [waypoints_stream]

    def on_opendrive_map(self, msg):
        self._logger.debug('@{}: received open drive message'.format(
            msg.timestamp))
        try:
            import carla
        except ImportError:
            raise Exception('Error importing carla.')
        self._logger.info('Initializing HDMap from open drive stream')
        self._map = HDMap(carla.Map('map', msg.data), self._log_file_name)

    def on_global_trajectory(self, msg):
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
        self._waypoints = collections.deque()
        for waypoint_option in msg.data:
            self._waypoints.append(waypoint_option[0])

    def on_can_bus_update(self, msg, waypoints_stream):
        self._logger.debug('@{}: received can bus message'.format(
            msg.timestamp))
        self._vehicle_transform = msg.data.transform
        self.__update_waypoints()

        next_waypoint_steer = self._waypoints[min(
            len(self._waypoints) - 1, self._wp_num_steer)]
        next_waypoint_speed = self._waypoints[min(
            len(self._waypoints) - 1, self._wp_num_speed)]

        # Get vectors and angles to corresponding speed and steer waypoints.
        wp_steer_vector, wp_steer_angle = get_waypoint_vector_and_angle(
            next_waypoint_steer, self._vehicle_transform)
        wp_speed_vector, wp_speed_angle = get_waypoint_vector_and_angle(
            next_waypoint_speed, self._vehicle_transform)

        head_waypoints = collections.deque(
            itertools.islice(self._waypoints, 0, DEFAULT_NUM_WAYPOINTS))

        output_msg = WaypointsMessage(msg.timestamp,
                                      waypoints=head_waypoints,
                                      target_speed=self._flags.target_speed,
                                      wp_angle=wp_steer_angle,
                                      wp_vector=wp_steer_vector,
                                      wp_angle_speed=wp_speed_angle)
        waypoints_stream.send(output_msg)

    def __update_waypoints(self):
        """ Updates the waypoints.

        Depending on setup, the method either recomputes the waypoints
        between the ego vehicle and the goal location, or just garbage collects
        waypoints that have already been achieved.

        Returns:
            (wp_steer, wp_speed): The waypoints to be used to compute steer and
            speed angles.
        """
        if self._recompute_waypoints:
            self._waypoints = self._map.compute_waypoints(
                self._vehicle_transform.location, self._goal_location)
        self.__remove_completed_waypoints()
        if not self._waypoints or len(self._waypoints) == 0:
            # If waypoints are empty (e.g., reached destination), set waypoint
            # to current vehicle location.
            self._waypoints = collections.deque([self._vehicle_transform])

    def __remove_completed_waypoints(self):
        """ Removes waypoints that the ego vehicle has already completed.

        The method first finds the closest waypoint, removes all waypoints
        that are before the closest waypoint, and finally removes the
        closest waypoint if the ego vehicle is very close to it
        (i.e., close to completion)."""
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
