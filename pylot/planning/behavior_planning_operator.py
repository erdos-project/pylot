from collections import deque

import erdos

import numpy as np

import pylot.planning.cost_functions
import pylot.utils
from pylot.planning.messages import WaypointsMessage
from pylot.planning.utils import BehaviorPlannerState
from pylot.planning.waypoints import Waypoints


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
        self.__initialize_behaviour_planner()
        self._pose_msgs = deque()
        self._ego_info = EgoInfo()
        self._route = None
        self._map = None

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
        self._map = HDMap(carla.Map('map', msg.data),
                          self.config.log_file_name)

    def on_route_msg(self, msg):
        """Invoked whenever a message is received on the trajectory stream.

        Args:
            msg (:py:class:`~erdos.message.Message`): Message that contains
                a list of waypoints to the goal location.
        """
        self._logger.debug('@{}: global trajectory has {} waypoints'.format(
            msg.timestamp, len(msg.waypoints.waypoints)))
        self._route = msg.waypoints

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
        pose_msg = self._pose_msgs.popleft()
        ego_transform = pose_msg.data.transform
        self._ego_info.update(self._state, pose_msg)
        next_state = self.__best_state_transition(self._ego_info)
        self._logger.debug('@{}: agent transitioned from {} to {}'.format(
            timestamp, self._state, next_state))
        self._state = next_state
        # Remove the waypoint from the route if we're close to it.
        self._route.remove_waypoint_if_close(ego_transform.location, 5)
        new_goal_location = None
        if len(self._route.waypoints) > 1:
            new_goal_location = self._route.waypoints[1].location
        elif len(self._route.waypoints) == 1:
            new_goal_location = self._route.waypoints[0].location
        else:
            new_goal_location = ego_transform.location
        if new_goal_location != self._goal_location:
            self._goal_location = new_goal_location
            if self._map:
                # Use the map to compute more fine-grained waypoints.
                waypoints = self._map.compute_waypoints(
                    ego_transform.location, self._goal_location)
                road_options = deque([
                    pylot.utils.RoadOption.LANE_FOLLOW
                    for _ in range(len(waypoints))
                ])
                waypoints = Waypoints(waypoints, road_options=road_options)
            else:
                # Map is not available, send the route.
                waypoints = self._route
            if not waypoints or waypoints.is_empty():
                # If waypoints are empty (e.g., reached destination), set
                # waypoints to current vehicle location.
                waypoints = Waypoints(
                    deque([ego_transform]),
                    road_options=deque([pylot.utils.RoadOption.LANE_FOLLOW]))
            trajectory_stream.send(
                WaypointsMessage(timestamp, waypoints, self._state))

    def __initialize_behaviour_planner(self):
        # State the planner is in.
        self._state = BehaviorPlannerState.FOLLOW_WAYPOINTS
        # Cost functions. Output between 0 and 1.
        self._cost_functions = [
            pylot.planning.cost_functions.cost_overtake,
        ]
        # How important a cost function is.
        self._function_weights = [1]

    def __successor_states(self):
        """ Returns possible state transitions from current state."""
        if self._state == BehaviorPlannerState.FOLLOW_WAYPOINTS:
            # Can transition to OVERTAKE if the ego vehicle has been stuck
            # behind an obstacle for a while.
            return [
                BehaviorPlannerState.FOLLOW_WAYPOINTS,
                BehaviorPlannerState.OVERTAKE
            ]
        elif self._state == BehaviorPlannerState.OVERTAKE:
            return [
                BehaviorPlannerState.OVERTAKE,
                BehaviorPlannerState.FOLLOW_WAYPOINTS
            ]
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

    def __best_state_transition(self, ego_info):
        """ Computes most likely state transition from current state."""
        # Get possible next state machine states.
        possible_next_states = self.__successor_states()
        best_next_state = None
        min_state_cost = np.infty
        for state in possible_next_states:
            state_cost = 0
            # Compute the cost of the trajectory.
            for i in range(len(self._cost_functions)):
                cost_func = self._cost_functions[i](self._state, state,
                                                    ego_info)
                state_cost += self._function_weights[i] * cost_func
            # Check if it's the best trajectory.
            if state_cost < min_state_cost:
                best_next_state = state
                min_state_cost = state_cost
        return best_next_state


class EgoInfo(object):
    def __init__(self):
        self.last_time_moving = 0
        self.last_time_stopped = 0
        self.current_time = 0

    def update(self, state, pose_msg):
        self.current_time = pose_msg.timestamp.coordinates[0]
        if pose_msg.data.forward_speed >= 0.1:
            self.last_time_moving = self.current_time
        else:
            self.last_time_stopped = self.current_time
