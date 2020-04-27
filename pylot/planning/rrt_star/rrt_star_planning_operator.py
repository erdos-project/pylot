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
from pylot.planning.rrt_star.rrt_star_planning.RRTStar.rrt_star_wrapper \
    import apply_rrt_star
from pylot.utils import Location, Rotation, Transform


class RRTStarPlanningOperator(PlanningOperator):
    """RRTStar Planning operator for Carla 0.9.x.

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
            "step_size": flags.step_size,
            "max_iterations": flags.max_iterations,
            "end_dist_threshold": flags.end_dist_threshold,
            "obstacle_clearance": flags.obstacle_clearance_rrt,
            "lane_width": flags.lane_width,
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

        # if no obstacles, don't use RRT*
        if (self._flags.track == -1 or self._flags.track == 3) and len(obstacle_list) == 0:
            start = np.array([
                self._vehicle_transform.location.x,
                self._vehicle_transform.location.y
            ])

            # find the closest point to current location
            mindex = self._get_closest_index(start)

            path_x = []
            path_y = []
            for wp in itertools.islice(self._waypoints, mindex,
                                       mindex + self._flags.num_waypoints_ahead):
                path_x.append(wp.location.x)
                path_y.append(wp.location.y)
            path_x = np.array(path_x)
            path_y = np.array(path_y)
            success = 1
        else:
            # run rrt star
            # RRT* does not take into account the driveable region
            # it constructs search space as a top down, minimum bounding rectangle
            # with padding in each dimension
            path_x, path_y, success = self._apply_rrt_star(obstacle_list,
                                                           self._hyperparameters,
                                                           timestamp)

        speeds = [0]
        if success:
            speeds = [self._flags.target_speed] * len(path_x)
            self._logger.debug("@{}: RRT* Path X: {}".format(
                timestamp, path_x.tolist()))
            self._logger.debug("@{}: RRT* Path Y: {}".format(
                timestamp, path_y.tolist()))
            self._logger.debug("@{}: RRT* Speeds: {}".format(
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

    def _apply_rrt_star(self, obstacles, hyperparameters, timestamp):
        start = np.array([
            self._vehicle_transform.location.x,
            self._vehicle_transform.location.y
        ])

        # find the closest point to current location
        mindex = self._get_closest_index(start)
        end_ind = min(mindex + self._flags.num_waypoints_ahead,
                      len(self._waypoints) - 1)
        end = np.array([
            self._waypoints[end_ind].location.x,
            self._waypoints[end_ind].location.y
        ])

        # log initial conditions for debugging
        initial_conditions = {
            "start": start,
            "end": end,
            "obs": obstacles,
        }
        self._logger.debug("@{}: Initial conditions: {}".format(
            timestamp, initial_conditions))
        return apply_rrt_star(initial_conditions, hyperparameters)

    def _construct_waypoints(self, timestamp, pose_msg, path_x, path_y, speeds, success):
        """
        Convert the rrt* path into a waypoints message.
        """
        path_transforms = []
        target_speeds = deque()
        if not success:
            self._logger.error("@{}: RRT* failed. "
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
            self._logger.debug("@{}: RRT* succeeded.".format(timestamp))
            for point in zip(path_x, path_y, speeds):
                if self._map is not None:
                    p_loc = self._map.get_closest_lane_waypoint(
                        Location(x=point[0], y=point[1], z=0)).location
                else:
                    # RRT* does not take into account the driveable region it
                    # constructs search space as a top down, minimum bounding
                    # rectangle with padding in each dimension.
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
