"""
Author: Edward Fang
Email: edward.fang@berkeley.edu
"""
import time
from collections import deque

import erdos

from pylot.planning.messages import WaypointsMessage
from pylot.planning.planning_operator import PlanningOperator
from pylot.planning.waypoints import Waypoints
from pylot.planning.frenet_optimal_trajectory.frenet_optimal_trajectory_planner. \
    FrenetOptimalTrajectory.fot_wrapper import run_fot


class FOTPlanningOperator(PlanningOperator):
    """ Frenet Optimal Trajectory (FOT) Planning operator.

    This planning operator uses a global route and listens for predictions
    to produce a frenet optimal trajectory plan. Details can be found in
    `~pylot.planning.frenet_optimal_trajectory.frenet_optimal_trajectory.py`.

     Args:
        flags(:absl.flags:): Object to be used to access absl flags
        goal_location(:pylot.utils.Location:): Goal location for route planning
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
        super().__init__(pose_stream, prediction_stream,
                         static_obstacles_stream, lanes_stream,
                         global_trajectory_stream, open_drive_stream,
                         time_to_decision_stream, waypoints_stream, flags,
                         goal_location)
        self.s0 = 0.0
        self._hyperparameters = {
            "max_speed": flags.max_speed,
            "max_accel": flags.max_accel,
            "max_curvature": flags.max_curvature,
            "max_road_width_l": flags.max_road_width_l,
            "max_road_width_r": flags.max_road_width_r,
            "d_road_w": flags.d_road_w,
            "dt": flags.dt,
            "maxt": flags.maxt,
            "mint": flags.mint,
            "d_t_s": flags.d_t_s,
            "n_s_sample": flags.n_s_sample,
            "obstacle_clearance": flags.obstacle_clearance_fot,
            "kd": flags.kd,
            "kv": flags.kv,
            "ka": flags.ka,
            "kj": flags.kj,
            "kt": flags.kt,
            "ko": flags.ko,
            "klat": flags.klat,
            "klon": flags.klon
        }

    def fot_parameters_using_99_percentile(self, ttd):
        maxt = self._flags.maxt
        runtimes = [309, 208, 148, 67, 40]
        dts = [0.09, 0.11, 0.13, 0.19, 0.31]
        d_road_ws = [0.3, 0.3, 0.3, 0.5, 0.7]

        for index, runtime in enumerate(runtimes):
            if ttd >= runtime:
                return maxt, dts[index], d_road_ws[index]
        # Not enough time to run the planner.
        self._logger.error(
            'Not enough time to run the planner. Using the fastest version')
        return maxt, dts[-1], d_road_ws[-1]

    def update_hyper_parameters(self, timestamp, ttd):
        """Changes planning hyper parameters depending on time to decision."""
        # Change hyper paramters if static or dynamic deadlines are enabled.
        if self._flags.deadline_enforcement == 'dynamic':
            maxt, dt, d_road_w = self.fot_parameters_using_99_percentile(ttd)
        elif self._flags.deadline_enforcement == 'static':
            maxt, dt, d_road_w = self.fot_parameters_using_99_percentile(
                self._flags.planning_deadline)
        else:
            return
        self._logger.debug(
            '@{}: planner using maxt {}, dt {}, d_road_w {}'.format(
                timestamp, maxt, dt, d_road_w))
        self._hyperparameters['maxt'] = maxt
        self._hyperparameters['dt'] = dt
        self._hyperparameters['d_road_w'] = d_road_w

    @erdos.profile_method()
    def on_watermark(self, timestamp, waypoints_stream):
        self._logger.debug('@{}: received watermark'.format(timestamp))
        pose_msg = self._pose_msgs.popleft()
        ego_transform = pose_msg.data.transform
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
        obstacle_list = self._world.get_obstacle_list()

        if not self._waypoints:
            if self._map is not None and self._goal_location is not None:
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

        self._logger.debug("@{}: Hyperparameters: {}".format(
            timestamp, self._hyperparameters))
        initial_conditions = self._compute_initial_conditions(
            pose_msg, obstacle_list)
        self._logger.debug("@{}: Initial conditions: {}".format(
            timestamp, initial_conditions))
        start = time.time()
        (path_x, path_y, speeds, ix, iy, iyaw, d, s, speeds_x, speeds_y, misc,
         costs, success) = run_fot(initial_conditions, self._hyperparameters)
        fot_runtime = (time.time() - start) * 1000
        self._logger.debug('@{}: Frenet runtime {}'.format(
            timestamp, fot_runtime))
        if success:
            self._logger.debug(
                "@{}: Frenet Optimal Trajectory succeeded.".format(timestamp))
            self._log_output(timestamp, path_x, path_y, speeds, ix, iy, iyaw,
                             d, s, speeds_x, speeds_y, costs)
            output_wps = self.build_output_waypoints(path_x, path_y, speeds)
        else:
            self._logger.debug("@{}: Frenet Optimal Trajectory failed. "
                               "Sending emergency stop.".format(timestamp))
            self._waypoints.remove_completed(ego_transform.location,
                                             ego_transform)
            output_wps = self._waypoints.slice_waypoints(
                0, self._flags.num_waypoints_ahead, 0)

        # update current pose
        self.s0 = misc['s']
        waypoints_stream.send(WaypointsMessage(timestamp, output_wps))

    def _compute_initial_conditions(self, pose_msg, obstacle_list):
        current_index = self._waypoints.closest_waypoint(
            pose_msg.data.transform.location)
        # compute waypoints offset by current location
        wps = self._waypoints.slice_waypoints(
            max(current_index - self._flags.num_waypoints_behind, 0),
            min(current_index + self._flags.num_waypoints_ahead,
                len(self._waypoints.waypoints)))
        initial_conditions = {
            'ps': self.s0,
            'target_speed': self._flags.target_speed,
            'pos': pose_msg.data.transform.location.as_numpy_array_2D(),
            'vel': pose_msg.data.velocity_vector.as_numpy_array_2D(),
            'wp': wps.as_numpy_array_2D().T,
            'obs': obstacle_list,
        }
        return initial_conditions

    def _log_output(self, timestamp, path_x, path_y, speeds, ix, iy, iyaw, d,
                    s, speeds_x, speeds_y, costs):
        self._logger.debug("@{}: Frenet Path X: {}".format(
            timestamp, path_x.tolist()))
        self._logger.debug("@{}: Frenet Path Y: {}".format(
            timestamp, path_y.tolist()))
        self._logger.debug("@{}: Frenet Speeds: {}".format(
            timestamp, speeds.tolist()))
        self._logger.debug("@{}: Frenet IX: {}".format(timestamp, ix.tolist()))
        self._logger.debug("@{}: Frenet IY: {}".format(timestamp, iy.tolist()))
        self._logger.debug("@{}: Frenet IYAW: {}".format(
            timestamp, iyaw.tolist()))
        self._logger.debug("@{}: Frenet D: {}".format(timestamp, d.tolist()))
        self._logger.debug("@{}: Frenet S: {}".format(timestamp, s.tolist()))
        self._logger.debug("@{}: Frenet Speeds X: {}".format(
            timestamp, speeds_x.tolist()))
        self._logger.debug("@{}: Frenet Speeds Y: {}".format(
            timestamp, speeds_y.tolist()))
        self._logger.debug("@{}: Frenet Costs: {}".format(timestamp, costs))
