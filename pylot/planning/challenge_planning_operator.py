from collections import deque
import math

import carla

from erdos.op import Op
from erdos.utils import setup_csv_logging, setup_logging

import pylot.utils
from pylot.planning.messages import WaypointsMessage
from pylot.planning.utils import get_distance, get_target_speed,\
    BehaviorPlannerState
from pylot.control.utils import get_angle, get_world_vec_dist


class ChallengePlanningOperator(Op):
    def __init__(self, name, flags, log_file_name=None, csv_file_name=None):
        super(ChallengePlanningOperator, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._flags = flags
        self._state = BehaviorPlannerState.READY
        self._sampling_resolution = 1
        self._min_distance = self._sampling_resolution * 0.9
        self._map = None
        self._waypoints = None
        self._vehicle_transform = None
        self._wp_num_steer = 9  # use 9th waypoint for steering
        self._wp_num_speed = 4  # use 4th waypoint for speed
        # Cost functions. Output between 0 and 1.
        self._cost_functions = None
        # How important a cost function is.
        self._function_weights = None

    @staticmethod
    def setup_streams(input_streams):
        input_streams.filter(pylot.utils.is_can_bus_stream).add_callback(
            ChallengePlanningOperator.on_can_bus_update)
        input_streams.filter(pylot.utils.is_open_drive_stream).add_callback(
            ChallengePlanningOperator.on_opendrive_map)
        input_streams.filter(pylot.utils.is_global_trajectory_stream).add_callback(
            ChallengePlanningOperator.on_global_trajectory)
        return [pylot.utils.create_waypoints_stream()]

    def on_can_bus_update(self, msg):
        self._vehicle_transform = msg.data.transform
        next_waypoint_steer, next_waypoint_speed = self.__compute_next_waypoints()

        wp_vector, wp_mag = get_world_vec_dist(
            next_waypoint_steer.location.x,
            next_waypoint_steer.location.y,
            self._vehicle_transform.location.x,
            self._vehicle_transform.location.y)

        if wp_mag > 0:
            wp_angle = get_angle(
                wp_vector,
                [self._vehicle_transform.orientation.x,
                 self._vehicle_transform.orientation.y])
        else:
            wp_angle = 0

        wp_vector_speed, wp_speed_mag = get_world_vec_dist(
            next_waypoint_speed.location.x,
            next_waypoint_speed.location.y,
            self._vehicle_transform.location.x,
            self._vehicle_transform.location.y)

        if wp_speed_mag > 0:
            wp_angle_speed = get_angle(
                wp_vector_speed,
                [self._vehicle_transform.orientation.x,
                 self._vehicle_transform.orientation.y])
        else:
            wp_angle_speed = 0

        target_speed = get_target_speed(
            self._vehicle_transform.location, next_waypoint_steer)

        output_msg = WaypointsMessage(
            msg.timestamp,
            waypoints=[next_waypoint_steer],
            target_speed=target_speed,
            wp_angle=wp_angle,
            wp_vector=wp_vector,
            wp_angle_speed=wp_angle_speed)
        self.get_output_stream('waypoints').send(output_msg)

    def on_opendrive_map(self, msg):
        self._map = carla.Map('test', msg.data)

    def on_global_trajectory(self, msg):
        self._waypoints = deque()
        for waypoint_option in msg.data:
            self._waypoints.append(waypoint_option[0])

    def __compute_next_waypoints(self):
        if self._waypoints is None or len(self._waypoints) == 0:
            return (self._vehicle_transform, self._vehicle_transform)

        dist, index = self.__get_closest_waypoint()
        # Waypoints that are before the closest waypoint are irrelevant now.
        while index > 0:
            self._waypoints.popleft()
            index -= 1

        # The closest waypoint is 90% complete, remove it.
        if dist < self._min_distance:
            self._waypoints.popleft()

        return (
            self._waypoints[min(len(self._waypoints) - 1, self._wp_num_steer)],
            self._waypoints[min(len(self._waypoints) - 1, self._wp_num_speed)])

    def __get_closest_waypoint(self):
        min_dist = 10000000
        min_index = 0
        index = 0
        for waypoint in self._waypoints:
            # We only check the first 10 waypoints.
            if index > 10:
                break
            dist = get_distance(waypoint.location,
                                self._vehicle_transform.location)
            if dist < min_dist:
                min_dist = dist
                min_index = index
        return min_dist, min_index

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

    def generate_trajectory(self, next_state, vehicle_transform, predictions):
        # TODO(ionel): Implement.
        pass

    def best_transition(self, vehicle_transform, predictions):
        """ Computes most likely state transition from current state."""
        # Get possible next states.
        possible_next_states = self.__successor_states()
        best_next_state = None
        min_state_cost = 10000000
        for state in possible_next_states:
            trajectory_for_state = self.__generate_trajectory(
                state, vehicle_transform, predictions)
            state_cost = 0
            for i in range(len(self._cost_functions)):
                cost_func = self._cost_functions[i](
                    trajectory_for_state, predictions)
                state_cost += self._function_weights[i] * cost_func
            if best_next_state is None or state_cost < min_state_cost:
                best_next_state = state
                min_state_cost = state_cost
        return best_next_state

    def cost_speed(self, current_speed, speed_limit):
        # Cost of the car stopping.
        STOP_COST = 0.7
        # How many km/h to drive at bellow speed limit.
        BUFFER_SPEED = 5.0
        target_speed = speed_limit - BUFFER_SPEED
        if current_speed < target_speed:
            return STOP_COST * (target_speed - current_speed) / target_speed
        elif current_speed >= target_speed and current_speed < speed_limit:
            return (current_speed - target_speed) / BUFFER_SPEED
        else:
            return 1

    def cost_lane_change(
            self, goal_lane, intended_lane, final_lane, distance_goal):
        delta_d = 2.0 * goal_lane - intended_lane - final_lane
        cost = 1 - math.exp(-abs(delta_d) / distance_goal)
        return cost

    def cost_inefficiency(
            target_speed, intended_lane, final_lane, lane_speeds):
        # Cost becomes higher for trajectories with intended and final_lane
        # lane that have traffic slower than target_speed.
        speed_intended = lane_speeds[final_lane]
        speed_final = lane_speeds[final_lane]
        cost = (2.0 * target_speed -
                speed_intended - speed_final) / target_speed
        return cost
