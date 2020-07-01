"""Implements an operator that computes waypoints to a goal location."""
import erdos

import pylot.planning.utils
import pylot.utils
from pylot.planning.messages import WaypointsMessage
from pylot.planning.planning_operator import PlanningOperator


class WaypointPlanningOperator(PlanningOperator):
    """Computes waypoints the ego vehicle must follow.

    The planner reduces speed/stops whenever it encounters an obstacle,
    and waits for the obstacle to move. It does not implement an obstacle
    avoidance policy.
    """
    def __init__(self,
                 pose_stream,
                 prediction_stream,
                 static_obstacles_stream,
                 lanes_stream,
                 global_trajectory_stream,
                 open_drive_stream,
                 time_to_decision_stream,
                 waypoints_stream,
                 flags,
                 goal_location=None):
        self._last_stop_ego_location = None
        super().__init__(pose_stream, prediction_stream,
                         static_obstacles_stream, lanes_stream,
                         global_trajectory_stream, open_drive_stream,
                         time_to_decision_stream, waypoints_stream, flags,
                         goal_location)

    @erdos.profile_method()
    def on_watermark(self, timestamp, waypoints_stream):
        self._logger.debug('@{}: received watermark'.format(timestamp))
        pose_msg = self._pose_msgs.popleft().data
        ego_transform = pose_msg.transform
        prediction_msg = self._prediction_msgs.popleft()
        predictions = self.get_predictions(prediction_msg, ego_transform)
        static_obstacles_msg = self._static_obstacles_msgs.popleft()
        if len(self._lanes_msgs) > 0:
            lanes = self._lanes_msgs.popleft().data
        else:
            lanes = None

        # Update the representation of the world.
        self._world.update(timestamp,
                           ego_transform,
                           predictions,
                           static_obstacles_msg.obstacles,
                           hd_map=self._map,
                           lanes=lanes)

        if pose_msg.forward_speed < 0.08:
            distance_since_last_full_stop = 0
            self._last_stop_ego_location = ego_transform.location
        else:
            if self._last_stop_ego_location is not None:
                distance_since_last_full_stop = \
                    ego_transform.location.distance(
                        self._last_stop_ego_location)
            else:
                distance_since_last_full_stop = 0

        self._world.waypoints.remove_completed(ego_transform.location,
                                               ego_transform)

        try:
            wp_vector = self._world.waypoints.get_vector(
                ego_transform, self._flags.min_pid_steer_waypoint_distance)
            wp_angle = self._world.waypoints.get_angle(
                ego_transform, self._flags.min_pid_steer_waypoint_distance)
            speed_factor = pylot.planning.utils.stop_for_agents(
                ego_transform, wp_angle, wp_vector,
                self._world.obstacle_predictions,
                static_obstacles_msg.obstacles, self._flags, self._logger,
                self._map, timestamp, distance_since_last_full_stop)
            target_speed = speed_factor * self._flags.target_speed
            self._logger.debug(
                '@{}: speed factor: {}, target speed: {}'.format(
                    timestamp, speed_factor, target_speed))
        except ValueError:
            # No more waypoints to follow.
            self._logger.debug(
                '@{}: no more waypoints to follow, target speed 0')
            target_speed = 0
        head_waypoints = self._world.waypoints.slice_waypoints(
            0, self._flags.num_waypoints_ahead, target_speed)
        waypoints_stream.send(WaypointsMessage(timestamp, head_waypoints))
        waypoints_stream.send(erdos.WatermarkMessage(timestamp))
