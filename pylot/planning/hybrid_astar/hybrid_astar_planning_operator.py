"""
Author: Edward Fang
Email: edward.fang@berkeley.edu
"""
import itertools
from collections import deque

import erdos

import numpy as np

from pylot.planning.messages import WaypointsMessage
from pylot.planning.planning_operator import PlanningOperator
from pylot.planning.hybrid_astar.hybrid_astar_planner.HybridAStar.hybrid_astar_wrapper \
    import apply_hybrid_astar
from pylot.utils import Location, Rotation, Transform


class HybridAStarPlanningOperator(PlanningOperator):
    """HybridAStar Planning operator for Carla 0.9.x.

    Args:
        flags: Config flags.
        goal_location: Goal pylot.utils.Location for planner to route to.
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
        self._hyperparameters = self.parse_hyperparameters(self._flags)

    def parse_hyperparameters(self, flags):
        hyperparameters = {
            "step_size": flags.step_size_hybrid_astar,
            "max_iterations": flags.max_iterations_hybrid_astar,
            "completion_threshold": flags.completion_threshold,
            "angle_completion_threshold": flags.angle_completion_threshold,
            "rad_step": flags.rad_step,
            "rad_upper_range": flags.rad_upper_range,
            "rad_lower_range": flags.rad_lower_range,
            "obstacle_clearance": flags.obstacle_clearance_hybrid_astar,
            "lane_width": flags.lane_width_hybrid_astar,
            "radius": flags.radius,
            "car_length": flags.car_length,
            "car_width": flags.car_width,
        }
        return hyperparameters

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
            else:
                # haven't received waypoints from global trajectory stream
                self._logger.debug(
                    "@{}: Sending target speed 0, haven't"
                    "received global trajectory".format(timestamp))
                head_waypoints = deque([vehicle_transform])
                target_speeds = deque([0])
                waypoints_stream.send(
                    WaypointsMessage(timestamp, head_waypoints, target_speeds))
                return

        # run hybrid a*
        # hybrid a* does not take into account the driveable region
        # it constructs search space as a top down, minimum bounding rectangle
        # with padding in each dimension
        if (self._flags.track == -1 or self._flags.track == 3) and len(obstacle_list) == 0:
            start = np.array([
                self._vehicle_transform.location.x,
                self._vehicle_transform.location.y
            ])

            # find the closest point to current location
            mindex = self._get_closest_index(start)

            path_x = []
            path_y = []
            path_yaw = []
            for wp in itertools.islice(self._waypoints, mindex,
                                       mindex + self._flags.num_waypoints_ahead):
                path_x.append(wp.location.x)
                path_y.append(wp.location.y)
                path_yaw.append(3.14159)
            path_x = np.array(path_x)
            path_y = np.array(path_y)
            path_yaw = np.array(path_yaw)
            success = 1
        else:
            path_x, path_y, path_yaw, success = self._apply_hybrid_astar(
                obstacle_list, self._hyperparameters, timestamp)

        speeds = [0]
        if success:
            speeds = [self._flags.target_speed] * len(path_x)
            self._logger.debug("@{}: Hybrid A* Path X: {}".format(
                timestamp, path_x.tolist()))
            self._logger.debug("@{}: Hybrid A* Path Y: {}".format(
                timestamp, path_y.tolist()))
            self._logger.debug("@{}: Hybrid A* Path Yaw: {}".format(
                timestamp, path_yaw.tolist()))
            self._logger.debug("@{}: Hybrid A* Speeds: {}".format(
                timestamp, [self._flags.target_speed] * len(path_x)))

        # log debug
        self._logger.debug("@{}: Hyperparameters: {}".format(
            timestamp, self._hyperparameters))

        # construct and send waypoint message
        waypoint_message = self._construct_waypoints(timestamp, pose_msg,
                                                     path_x, path_y,
                                                     speeds, success)
        waypoints_stream.send(waypoint_message)

    def _get_closest_index(self, start):
        min_dist = np.infty
        mindex = 0
        for ind, wp in enumerate(self._waypoints):
            dist = np.linalg.norm(
                [start[0] - wp.location.x, start[1] - wp.location.y])
            if dist <= min_dist:
                mindex = ind
                min_dist = dist
        return mindex

    def _apply_hybrid_astar(self, obstacles, hyperparameters, timestamp):
        start = np.array([
            self._vehicle_transform.location.x,
            self._vehicle_transform.location.y,
            np.deg2rad(self._vehicle_transform.rotation.yaw),
        ])

        # find the closest point to current location
        mindex = self._get_closest_index(start)
        end_ind = min(mindex + self._flags.num_waypoints_ahead,
                      len(self._waypoints) - 1)
        last_wp = self._waypoints[end_ind]
        end_yaw = np.arctan2(
            last_wp.location.x - self._waypoints[end_ind - 1].location.x,
            last_wp.location.y - self._waypoints[end_ind - 1].location.y)

        end = np.array([last_wp.location.x, last_wp.location.y, np.deg2rad(self._vehicle_transform.rotation.yaw)])

        # log initial conditions for debugging
        initial_conditions = {
            "start": start,
            "end": end,
            "obs": obstacles,
        }
        self._logger.debug("@{}: Initial conditions: {}".format(
            timestamp, initial_conditions))
        return apply_hybrid_astar(initial_conditions, hyperparameters)

    def _construct_waypoints(self, timestamp, pose_msg, path_x, path_y, speeds, success):
        """
        Convert the hybrid a* path into a waypoints message.
        """
        path_transforms = []
        target_speeds = deque()
        if not success:
            self._logger.error("@{}: Hybrid A* failed. "
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
            waypoints = deque(path_transforms)
        else:
            self._logger.debug("@{}: Hybrid A* succeeded.".format(timestamp))
            for point in zip(path_x, path_y, speeds):
                if self._map is not None:
                    p_loc = self._map.get_closest_lane_waypoint(
                        Location(x=point[0], y=point[1], z=0)).location
                else:
                    # Hybrid A* does not take into account the driveable region
                    # it constructs search space as a top down, minimum
                    # bounding rectangle with padding in each dimension.
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
