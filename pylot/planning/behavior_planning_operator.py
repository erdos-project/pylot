from collections import deque

import erdos

import pylot.planning.cost_functions
from pylot.planning.utils import BehaviorPlannerState


class BehaviorPlanningOperator(erdos.Operator):
    def __init__(self, pose_stream, open_drive_stream, route_stream,
                 trajectory_stream, flags):
        pose_stream.add_callback(self.on_pose_update)
        open_drive_stream.add_callback(self.on_opendrive_map)
        route_stream.add_callback(self.on_route_msg)
        erdos.add_watermark_callback(
            [pose_stream, open_drive_stream, route_stream],
            [trajectory_stream], self.on_watermark)

        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags
        self._goal_location = None
        # Initialize the state of the behaviour planner.
        #self.__initialize_behaviour_planner()
        self._pose_msgs = deque()
        self._route = deque()

    @staticmethod
    def connect(pose_stream, open_drive_stream, route_stream):
        trajectory_stream = erdos.WriteStream()
        return [trajectory_stream]

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
        self._map = HDMap(carla.Map('map', msg.data))

    def on_route_msg(self, msg):
        """Invoked whenever a message is received on the trajectory stream.

        Args:
            msg (:py:class:`~erdos.message.Message`): Message that contains
                a list of waypoints to the goal location.
        """
        self._logger.debug('@{}: global trajectory has {} waypoints'.format(
            msg.timestamp, len(msg.data)))
        for (wp, _) in msg.data:
            self._route.append(wp)

    def on_pose_update(self, msg):
        """Invoked whenever a message is received on the pose stream.

        Args:
            msg (:py:class:`~erdos.message.Message`): Message that contains
                info about the ego vehicle.
        """
        self._logger.debug('@{}: received pose message'.format(msg.timestamp))
        self._pose_msgs.append(msg)

    def on_watermark(self, timestamp, trajectory_stream):
        self._logger.debug('@{}: received watermark'.format(timestamp))
        self._vehicle_transform = self._pose_msgs.popleft().data.transform
        # Remove the waypoint from the route if we're close to it.
        if (len(self._route) > 0 and self._vehicle_transform.location.distance(
                self._route[0].location) < 5):
            self._route.popleft()
        new_goal_location = None
        if len(self._route) > 1:
            new_goal_location = self._route[1].location
        elif len(self._route) == 1:
            new_goal_location = self._route[0].location
        else:
            new_goal_location = self._vehicle_transform.location
        if new_goal_location != self._goal_location:
            self._goal_location = new_goal_location
            waypoints = self._map.compute_waypoints(
                self._vehicle_transform.location, self._goal_location)
            waypoints = [(wp, 0) for wp in waypoints]
            if not waypoints or len(waypoints) == 0:
                # If waypoints are empty (e.g., reached destination), set
                # waypoints to current vehicle location.
                waypoints = deque([[self._vehicle_transform]])
            trajectory_stream.send(erdos.Message(timestamp, waypoints))

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
