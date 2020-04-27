"""
Author: Edward Fang
Email: edward.fang@berkeley.edu
"""
import itertools
from collections import deque

import erdos

import numpy as np

from pylot.planning.planning_operator import PlanningOperator
from pylot.planning.frenet_optimal_trajectory.frenet_optimal_trajectory_planner. \
    FrenetOptimalTrajectory.fot_wrapper import run_fot
from pylot.planning.messages import WaypointsMessage
from pylot.utils import Location, Rotation, Transform


class FOTPlanningOperator(PlanningOperator):
    """ Frenet Optimal Trajectory (FOT) Planning operator for Carla 0.9.x.

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
                 global_trajectory_stream,
                 open_drive_stream,
                 time_to_decision_stream,
                 waypoints_stream,
                 flags,
                 goal_location=None,
                 log_file_name=None,
                 csv_file_name=None):
        super().__init__(pose_stream, prediction_stream,
                         global_trajectory_stream, open_drive_stream,
                         time_to_decision_stream, waypoints_stream, flags,
                         goal_location, log_file_name, csv_file_name)
        self._hyperparameters = self.parse_hyperparameters(flags)
        self.s0 = 0.0

    def parse_hyperparameters(self, flags):
        hyperparameters = {
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
        return hyperparameters

    def fot_parameters_using_90_percentile(self, ttd):
        maxt = 4.0
        runtimes = [350, 211, 196, 142, 114, 93, 74, 61, 48, 40, 37, 34, 31]
        dts = [
            0.1, 0.1, 0.15, 0.1, 0.1, 0.15, 0.15, 0.2, 0.2, 0.25, 0.3, 0.35,
            0.45
        ]
        d_road_ws = [
            0.1, 0.3, 0.1, 0.5, 0.7, 0.3, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7, 0.7
        ]
        for index, runtime in enumerate(runtimes):
            if ttd >= runtime:
                return maxt, dts[index], d_road_ws[index]
        # Not enough time to run the planner.
        return None, None, None

    def fot_parameters_using_99_percentile(self, ttd):
        maxt = 4.0
        runtimes = [650, 250, 150, 125, 113, 92, 73, 61, 50, 40, 32]
        dts = [0.1, 0.1, 0.15, 0.15, 0.2, 0.2, 0.2, 0.25, 0.3, 0.35, 0.4]
        d_road_ws = [0.1, 0.3, 0.3, 0.5, 0.3, 0.5, 0.7, 0.7, 0.7, 0.7, 0.7]
        for index, runtime in enumerate(runtimes):
            if ttd >= runtime:
                return maxt, dts[index], d_road_ws[index]
        # Not enough time to run the planner.
        return None, None, None

    def on_time_to_decision(self, msg):
        """Invoked upon the receipt of a time to decision message.

        The method changes planning hyper parameters depending on time to
        decision.
        """
        if self._flags.dynamic_deadlines:
            # Change the hyperparameters if the dynamic deadlines are enabled.
            maxt, dt, d_road_w = self.fot_parameters_using_99_percentile(
                msg.data)
            self._logger.debug(
                '@{}: planner using maxt {}, dt {}, d_road_w {}'.format(
                    msg.timestamp, maxt, dt, d_road_w))
            self._hyperparameters['maxt'] = maxt
            self._hyperparameters['dt'] = dt
            self._hyperparameters['d_road_w'] = d_road_w

    @erdos.profile_method()
    def on_watermark(self, timestamp, waypoints_stream):
        self._logger.debug('@{}: received watermark'.format(timestamp))

        # get ego info
        pose_msg = self._pose_msgs.popleft()
        vehicle_transform = pose_msg.data.transform
        self._vehicle_transform = vehicle_transform

        # get obstacles
        prediction_msg = self._prediction_msgs.popleft()
        obstacle_list = self.build_obstacle_list(vehicle_transform,
                                                 prediction_msg)
        # update waypoints
        if not self._waypoints:
            # running in CARLA
            if self._map is not None:
                self._waypoints = self._map.compute_waypoints(
                    vehicle_transform.location, self._goal_location)
                self._prev_waypoints = self._waypoints
            # haven't received waypoints from global trajectory stream
            else:
                self._logger.debug(
                    "@{}: Sending target speed 0, haven't"
                    "received global trajectory".format(timestamp))
                head_waypoints = deque([vehicle_transform])
                target_speeds = deque([0])
                waypoints_stream.send(
                    WaypointsMessage(timestamp, head_waypoints, target_speeds))
                return

        # compute optimal frenet trajectory
        initial_conditions = self._compute_initial_conditions(
            pose_msg, obstacle_list)

        (path_x, path_y, speeds, ix, iy, iyaw, d, s, speeds_x, speeds_y, misc,
         costs, success) = run_fot(initial_conditions, self._hyperparameters)

        if success:
            self._logger.debug("@{}: Frenet Path X: {}".format(
                timestamp, path_x.tolist()))
            self._logger.debug("@{}: Frenet Path Y: {}".format(
                timestamp, path_y.tolist()))
            self._logger.debug("@{}: Frenet Speeds: {}".format(
                timestamp, speeds.tolist()))
            self._logger.debug("@{}: Frenet IX: {}".format(
                timestamp, ix.tolist()))
            self._logger.debug("@{}: Frenet IY: {}".format(
                timestamp, iy.tolist()))
            self._logger.debug("@{}: Frenet IYAW: {}".format(
                timestamp, iyaw.tolist()))
            self._logger.debug("@{}: Frenet D: {}".format(
                timestamp, d.tolist()))
            self._logger.debug("@{}: Frenet S: {}".format(
                timestamp, s.tolist()))
            self._logger.debug("@{}: Frenet Speeds X: {}".format(
                timestamp, speeds_x.tolist()))
            self._logger.debug("@{}: Frenet Speeds Y: {}".format(
                timestamp, speeds_y.tolist()))
            self._logger.debug("@{}: Frenet Costs: {}".format(
                timestamp, costs))

        # update current pose
        self.s0 = misc['s']

        # log debug
        initial_conditions['pos'] = initial_conditions['pos'].tolist()
        initial_conditions['vel'] = initial_conditions['vel'].tolist()
        initial_conditions['wp'] = initial_conditions['wp'].tolist()
        initial_conditions['obs'] = initial_conditions['obs'].tolist()
        self._logger.debug("@{}: Frenet Initial Conditions: {}".format(
            timestamp, misc))
        self._logger.debug("@{}: Euclidean Initial Conditions: {}".format(
            timestamp, initial_conditions))
        self._logger.debug("@{}: Hyperparameters: {}".format(
            timestamp, self._hyperparameters))

        # construct and send waypoint message
        waypoints_message = self._construct_waypoints(timestamp, pose_msg,
                                                      path_x, path_y, speeds,
                                                      success)
        waypoints_stream.send(waypoints_message)

    def _compute_initial_conditions(self, pose_msg, obstacle_list):
        x = pose_msg.data.transform.location.x
        y = pose_msg.data.transform.location.y
        vx = pose_msg.data.velocity_vector.x
        vy = pose_msg.data.velocity_vector.y
        pos = np.array([x, y])
        vel = np.array([vx, vy])

        # find where the ego vehicle currently is
        current_index = 0
        min_dist = np.infty
        for i, wp in enumerate(self._waypoints):
            dist = np.linalg.norm([wp.location.x - x, wp.location.y - y])
            if dist <= min_dist:
                current_index = i
                min_dist = dist

        # compute waypoints offset by current location
        wx = []
        wy = []

        for wp in itertools.islice(
                self._waypoints,
                max(current_index - self._flags.num_waypoints_behind, 0),
                min(current_index + self._flags.num_waypoints_ahead,
                    len(self._waypoints))):
            wx.append(wp.location.x)
            wy.append(wp.location.y)
        wp = np.array([wx, wy]).T

        initial_conditions = {
            'ps': self.s0,
            'target_speed': self._flags.target_speed,
            'pos': pos,
            'vel': vel,
            'wp': wp,
            'obs': obstacle_list,
        }
        return initial_conditions

    def _construct_waypoints(self, timestamp, pose_msg, path_x, path_y, speeds,
                             success):
        """
        Convert the optimal frenet path into a waypoints message.
        """
        path_transforms = []
        target_speeds = deque()
        if not success:
            self._logger.debug("@{}: Frenet Optimal Trajectory failed. "
                               "Sending emergency stop.".format(timestamp))
            x = pose_msg.data.transform.location.x
            y = pose_msg.data.transform.location.y
            current_index = 0
            min_dist = np.infty
            for i, wp in enumerate(self._prev_waypoints):
                dist = np.linalg.norm([wp.location.x - x, wp.location.y - y])
                if dist <= min_dist:
                    current_index = i
                    min_dist = dist
            for wp in itertools.islice(
                    self._prev_waypoints, current_index,
                    current_index + self._flags.num_waypoints_ahead):
                path_transforms.append(wp)
                target_speeds.append(0)
        else:
            self._logger.debug(
                "@{}: Frenet Optimal Trajectory succeeded.".format(timestamp))
            for point in zip(path_x, path_y, speeds):
                if self._map is not None:
                    p_loc = self._map.get_closest_lane_waypoint(
                        Location(x=point[0], y=point[1], z=0)).location
                else:
                    p_loc = Location(x=point[0], y=point[1], z=0)
                path_transforms.append(
                    Transform(
                        location=Location(x=point[0], y=point[1], z=p_loc.z),
                        rotation=Rotation(),
                    ))
                target_speeds.append(point[2])

        waypoints = deque(path_transforms)
        self._prev_waypoints = waypoints
        return WaypointsMessage(timestamp, waypoints, target_speeds)
