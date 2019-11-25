from collections import deque

import carla

from erdos.op import Op
from erdos.utils import setup_csv_logging, setup_logging

from pylot.map.hd_map import HDMap
import pylot.planning.cost_functions
from pylot.planning.messages import WaypointsMessage
from pylot.planning.utils import get_distance,\
    get_waypoint_vector_and_angle, BehaviorPlannerState
from pylot.simulation.carla_utils import get_map
import pylot.utils

WAYPOINT_COMPLETION_THRESHOLD = 0.9


class PlanningOperator(Op):
    """ Planning operator for Carla 0.9.x.

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
        super(PlanningOperator, self).__init__(name)
        self._log_file_name = log_file_name
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
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
        self._waypoints = deque()
        # The operator picks the wp_num_steer-th waypoint to compute the angle
        # it has to steer by when taking a turn.
        self._wp_num_steer = 9  # use 9th waypoint for steering
        # The operator picks the wp_num_speed-th waypoint to compute the angle
        # it has to steer by when driving straight.
        self._wp_num_speed = 4  # use 4th waypoint for speed
        # We're not running in challenge mode if no track flag is present.
        # Thus, we can directly get the map from the simulator.
        if not hasattr(self._flags, 'track'):
            self._map = HDMap(get_map(self._flags.carla_host,
                                      self._flags.carla_port,
                                      self._flags.carla_timeout),
                              log_file_name)
            self._logger.info('Planner running in stand-alone mode')
            assert goal_location, 'Planner has not received a goal location'
            # Transform goal location to carla.Location
            self._goal_location = carla.Location(*goal_location)
            # Do not recompute waypoints upon each run.
            self._recompute_waypoints = True
        else:
            # Recompute waypoints upon each run.
            self._recompute_waypoints = False
            # TODO(ionel): HACK! In Carla challenge track 1 and 2 the waypoints
            # are coarse grained (30 meters apart). We pick the first waypoint
            # to compute the angles. However, we should instead implement
            # trajectory planning.
            if self._flags.track == 1 or self._flags == 2:
                self._wp_num_steer = 1
                self._wp_num_speed = 1

    def __initialize_behaviour_planner(self):
        # State the planner is in.
        self._state = BehaviorPlannerState.READY
        # Cost functions. Output between 0 and 1.
        self._cost_functions = [
            pylot.planning.cost_functions.cost_speed,
            pylot.planning.cost_functions.cost_lane_change,
            pylot.planning.cost_functions.cost_inefficiency]
        reach_speed_weight = 10 ** 5
        reach_goal_weight = 10 ** 6
        efficiency_weight = 10 ** 4
        # How important a cost function is.
        self._function_weights = [reach_speed_weight,
                                  reach_goal_weight,
                                  efficiency_weight]

    @staticmethod
    def setup_streams(input_streams):
        input_streams.filter(pylot.utils.is_can_bus_stream).add_callback(
            PlanningOperator.on_can_bus_update)
        input_streams.filter(pylot.utils.is_open_drive_stream).add_callback(
            PlanningOperator.on_opendrive_map)
        input_streams.filter(pylot.utils.is_global_trajectory_stream)\
                     .add_callback(PlanningOperator.on_global_trajectory)
        # input_streams.filter(pylot.utils.is_predictions_stream).add_callback(
        #     PlanningOperator.on_predictions)
        return [pylot.utils.create_waypoints_stream()]

    def on_opendrive_map(self, msg):
        self._logger.info('Planner running in scenario runner mode')
        self._map = HDMap(carla.Map('map', msg.data),
                          self._log_file_name)

    def on_global_trajectory(self, msg):
        self._logger.info('Global trajectory contains {} waypoints'.format(
            len(msg.data)))
        if len(msg.data) > 0:
            # The last waypoint is the goal location.
            self._goal_location = msg.data[-1][0].location.as_carla_location()
        else:
            # Trajectory does not contain any waypoints. We assume we have
            # arrived at destionation.
            goal_loc = self._vehicle_transform.location
            self._goal_location = goal_loc.as_carla_location()
        assert self._goal_location, 'Planner does not have a goal'
        self._waypoints = deque()
        for waypoint_option in msg.data:
            self._waypoints.append(waypoint_option[0])

    def on_can_bus_update(self, msg):
        self._vehicle_transform = msg.data.transform
        (next_waypoint_steer,
         next_waypoint_speed) = self.__update_waypoints()

        # Get vectors and angles to corresponding speed and steer waypoints.
        wp_steer_vector, wp_steer_angle = get_waypoint_vector_and_angle(
            next_waypoint_steer, self._vehicle_transform)
        wp_speed_vector, wp_speed_angle = get_waypoint_vector_and_angle(
            next_waypoint_speed, self._vehicle_transform)

        target_speed = self.__get_target_speed(next_waypoint_steer)

        output_msg = WaypointsMessage(
            msg.timestamp,
            waypoints=[next_waypoint_steer],
            target_speed=target_speed,
            wp_angle=wp_steer_angle,
            wp_vector=wp_steer_vector,
            wp_angle_speed=wp_speed_angle)
        self.get_output_stream('waypoints').send(output_msg)

    def __get_target_speed(self, waypoint):
        if get_distance(waypoint.location,
                        self._vehicle_transform.location) > 0.08:
            target_speed = self._flags.target_speed
        else:
            # We are reaching a waypoint; reduce the speed to half.
            # TODO: Check if this is still necessary.
            target_speed = self._flags.target_speed / 2
        return target_speed

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
            ego_location = self._vehicle_transform.location.as_carla_location()
            self._waypoints = self._map.compute_waypoints(
                ego_location, self._goal_location)
        self.__remove_completed_waypoints()
        if not self._waypoints or len(self._waypoints) == 0:
            # If waypoints are empty (e.g., reached destination), set waypoint
            # to current vehicle location.
            self._waypoints = deque([self._vehicle_transform])

        return (
            self._waypoints[min(len(self._waypoints) - 1, self._wp_num_steer)],
            self._waypoints[min(len(self._waypoints) - 1, self._wp_num_speed)])

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
            dist = get_distance(waypoint.location,
                                self._vehicle_transform.location)
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

    def best_transition(self, vehicle_transform, predictions):
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
                cost_func = self._cost_functions[i](
                    vehicle_info, predictions, trajectory_for_state)
                state_cost += self._function_weights[i] * cost_func
            # Check if it's the best trajectory.
            if best_next_state is None or state_cost < min_state_cost:
                best_next_state = state
                min_state_cost = state_cost
        return best_next_state

    def __generate_trajectory(
            self, next_state, vehicle_transform, predictions):
        # TODO(ionel): Implement.
        pass

    def __successor_states(self):
        """ Returns possible state transitions from current state."""
        if self._state == BehaviorPlannerState.READY:
            return [BehaviorPlannerState.READY, BehaviorPlannerState.KEEP_LANE]
        elif self._state == BehaviorPlannerState.KEEP_LANE:
            # 1) keep_lane -> prepare_lane_change_left
            # 2) keep_lane -> prepare_lane_change_right
            return [BehaviorPlannerState.KEEP_LANE,
                    BehaviorPlannerState.PREPARE_LANE_CHANGE_LEFT,
                    BehaviorPlannerState.PREPARE_LANE_CHANGE_RIGHT]
        elif self._state == BehaviorPlannerState.PREPARE_LANE_CHANGE_LEFT:
            # 1) prepare_lane_change_left -> keep_lane
            # 2) prepare_lane_change_left -> lange_change_left
            return [BehaviorPlannerState.KEEP_LANE,
                    BehaviorPlannerState.PREPARE_LANE_CHANGE_LEFT,
                    BehaviorPlannerState.LANE_CHANGE_LEFT]
        elif self._state == BehaviorPlannerState.LANE_CHANGE_LEFT:
            # 1) lange_change_left -> keep_lane
            return [BehaviorPlannerState.KEEP_LANE,
                    BehaviorPlannerState.LANE_CHANGE_LEFT]
        elif self._state == BehaviorPlannerState.PREPARE_LANE_CHANGE_RIGHT:
            # 1) prepare_lane_change_right -> keep_lane
            # 2) prepare_lane_change_right -> lange_change_right
            return [BehaviorPlannerState.KEEP_LANE,
                    BehaviorPlannerState.PREPARE_LANE_CHANGE_RIGHT,
                    BehaviorPlannerState.LANE_CHANGE_RIGHT]
        elif self._state == BehaviorPlannerState.LANE_CHANGE_RIGHT:
            # 1) lane_change_right -> keep_lane
            return [BehaviorPlannerState.KEEP_LANE,
                    BehaviorPlannerState.LANE_CHANGE_RIGHT]
        else:
            raise ValueError('Unexpected vehicle state {}'.format(self._state))
