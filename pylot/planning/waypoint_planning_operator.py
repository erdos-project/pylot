"""Implements an operator that computes waypoints to a goal location."""

from collections import deque

import erdos

import pylot.planning.utils
import pylot.utils
from pylot.perception.messages import ObstaclesMessage
from pylot.planning.messages import WaypointsMessage
from pylot.planning.planning_operator import PlanningOperator
from pylot.planning.waypoints import Waypoints
from pylot.prediction.messages import PredictionMessage

# Use a waypoint that is at least 9 meters away.
DEFAULT_MIN_DISTANCE_WAYPOINT = 9
RECOMPUTE_WAYPOINT_EVERY_N_WATERMARKS = 5


class WaypointPlanningOperator(PlanningOperator):
    """Computes waypoints the ego vehicle must follow.

    The planner reduces speed/stops whenever it encounters an obstacle,
    and waits for the obstacle to move. It does not implement an obstacle
    avoidance policy.
    """
    def __init__(self,
                 pose_stream,
                 prediction_stream,
                 traffic_lights_stream,
                 global_trajectory_stream,
                 open_drive_stream,
                 time_to_decision_stream,
                 waypoints_stream,
                 flags,
                 goal_location=None):
        super().__init__(pose_stream, prediction_stream, traffic_lights_stream,
                         global_trajectory_stream, open_drive_stream,
                         time_to_decision_stream, waypoints_stream, flags,
                         goal_location)

    @erdos.profile_method()
    def on_watermark(self, timestamp, waypoints_stream):
        self._logger.debug('@{}: received watermark'.format(timestamp))
        self._watermark_cnt += 1
        ego_transform = self._pose_msgs.popleft().data.transform
        tl_msg = self._traffic_light_msgs.popleft()
        obstacles_msg = self._prediction_msgs.popleft()
        if isinstance(obstacles_msg, ObstaclesMessage):
            obstacles = obstacles_msg.obstacles
        elif isinstance(obstacles_msg, PredictionMessage):
            obstacles = self.predictions_to_world_coordinates(
                ego_transform, obstacles_msg.predictions)
        else:
            raise ValueError('Unexpected obstacles msg type {}'.format(
                type(obstacles_msg)))

        if not self._waypoints:
            if self._map is not None:
                self._waypoints = Waypoints(deque(), deque())
                self._waypoints.recompute_waypoints(self._map,
                                                    ego_transform.location,
                                                    self._goal_location)
            else:
                # Haven't received waypoints from global trajectory stream.
                self._logger.debug(
                    "@{}: Sending target speed 0, haven't"
                    "received global trajectory".format(timestamp))
                waypoints_stream.send(
                    WaypointsMessage(timestamp, Waypoints(deque(), deque())))
                return

        if (self._recompute_waypoints and self._watermark_cnt %
                RECOMPUTE_WAYPOINT_EVERY_N_WATERMARKS == 0):
            self._waypoints.recompute_waypoints(self._map,
                                                ego_transform.location,
                                                self._goal_location)
        self._waypoints.remove_completed(ego_transform.location, ego_transform)

        wp_vector = self._waypoints.get_vector(ego_transform,
                                               DEFAULT_MIN_DISTANCE_WAYPOINT)
        wp_angle = self._waypoints.get_angle(ego_transform,
                                             DEFAULT_MIN_DISTANCE_WAYPOINT)

        speed_factor = pylot.planning.utils.stop_for_agents(
            ego_transform, wp_angle, wp_vector, obstacles, tl_msg.obstacles,
            self._flags, self._logger, self._map, timestamp)
        target_speed = speed_factor * self._flags.target_speed
        self._logger.debug('@{}: speed factor: {}, target speed: {}'.format(
            timestamp, speed_factor, target_speed))
        head_waypoints = self._waypoints.slice_waypoints(
            0, self._flags.num_waypoints_ahead, target_speed)
        waypoints_stream.send(WaypointsMessage(timestamp, head_waypoints))

    def predictions_to_world_coordinates(self, ego_transform,
                                         obstacle_predictions):
        for obstacle_prediction in obstacle_predictions:
            obstacle_prediction.transform = ego_transform * \
                obstacle_prediction.transform
            obstacle_prediction.trajectory = [
                ego_transform * transform
                for transform in obstacle_prediction.trajectory
            ]
        return obstacle_predictions
