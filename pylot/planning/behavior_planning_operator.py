import erdos

import pylot.planning.cost_functions
from pylot.planning.utils import BehaviorPlannerState


class BehaviorPlanningOperator(erdos.Operator):
    def __init__(self, flags):
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags
        # Initialize the state of the behaviour planner.
        self.__initialize_behaviour_planner()

    @staticmethod
    def connect(pose_stream, open_drive_stream, global_trajectory_stream,
                prediction_stream, traffic_lights_stream):
        return []

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

    def __generate_trajectory(self, next_state, vehicle_transform,
                              predictions):
        raise NotImplementedError

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
