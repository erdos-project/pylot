"""Implements an operator that computes waypoints to a goal location."""
import erdos

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
        self.update_world(timestamp)

        speed_factor = self._world.stop_for_agents(timestamp)
        target_speed = speed_factor * self._flags.target_speed
        self._logger.debug('@{}: speed factor: {}, target speed: {}'.format(
            timestamp, speed_factor, target_speed))
        output_wps = self.follow_waypoints(target_speed)

        waypoints_stream.send(WaypointsMessage(timestamp, output_wps))
        waypoints_stream.send(erdos.WatermarkMessage(timestamp))
