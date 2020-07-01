import copy
from collections import deque

import numpy as np

from pylot.planning.waypoints import Waypoints


class World(object):
    """A representation of the world that is used by the planners."""
    def __init__(self, flags, logger):
        self._flags = flags
        self._log_output = logger
        self.static_obstacles = None
        self.obstacle_predictions = None
        self._ego_obstacle_predictions = None
        self.ego_trajectory = deque(maxlen=self._flags.tracking_num_steps)
        self.ego_transform = None
        self.ego_velocity_vector = None
        self._lanes = None
        self._map = None
        self._goal_location = None
        self.waypoints = None
        self.timestamp = None

    def update(self,
               timestamp,
               ego_transform,
               obstacle_predictions,
               static_obstacles,
               hd_map=None,
               lanes=None,
               ego_velocity_vector=None):
        self.timestamp = timestamp
        self.ego_transform = ego_transform
        self.ego_trajectory.append(ego_transform)
        self.obstacle_predictions = obstacle_predictions
        self._ego_obstacle_predictions = copy.deepcopy(obstacle_predictions)
        # Tranform predictions to world frame of reference.
        for obstacle_prediction in self.obstacle_predictions:
            obstacle_prediction.to_world_coordinates(ego_transform)
        # Road signs are in world coordinates.
        self.static_obstacles = static_obstacles
        self._map = hd_map
        self._lanes = lanes
        self.ego_velocity_vector = ego_velocity_vector
        # The waypoints are not received on the global trajectory stream.
        # We need to compute them using the map.
        if not self.waypoints:
            if self._map is not None and self._goal_location is not None:
                self.waypoints = Waypoints(deque(), deque())
                self.waypoints.recompute_waypoints(self._map,
                                                   ego_transform.location,
                                                   self._goal_location)

    def update_waypoints(self, goal_location, waypoints):
        self._goal_location = goal_location
        self.waypoints = waypoints

    def get_obstacle_list(self):
        obstacle_list = []
        for prediction in self.obstacle_predictions:
            # Use all prediction times as potential obstacles.
            previous_origin = None
            for transform in prediction.predicted_trajectory:
                # Ignore predictions that are too close.
                if (previous_origin is None
                        or previous_origin.location.l2_distance(
                            transform.location) >
                        self._flags.obstacle_filtering_distance):
                    previous_origin = transform
                    # Ensure the prediction is nearby.
                    if (self.ego_transform.location.l2_distance(
                            transform.location) <
                            self._flags.distance_threshold):
                        obstacle_corners = \
                            prediction.obstacle_trajectory.obstacle.get_bounding_box_corners(
                                transform, self._flags.obstacle_radius)
                        obstacle_list.append(obstacle_corners)
        if len(obstacle_list) == 0:
            return np.empty((0, 4))
        return np.array(obstacle_list)

    def draw_on_frame(self, frame):
        for obstacle_prediction in self._ego_obstacle_predictions:
            obstacle_prediction.draw_trajectory_on_frame(frame)
        for obstacle in self.static_obstacles:
            if obstacle.is_traffic_light():
                world_transform = obstacle.transform
                # Transform traffic light to ego frame of reference.
                obstacle.transform = (self.ego_transform.inverse_transform() *
                                      obstacle.transform)
                obstacle.draw_on_bird_eye_frame(frame)
                obstacle.transform = world_transform
        if self.waypoints:
            self.waypoints.draw_on_frame(
                frame, self.ego_transform.inverse_transform())
        # TODO: Draw lane markings. We do not draw them currently
        # because we need to transform them from world frame of reference
        # to ego vehicle frame of reference, which is slow to compute.
        # if self._map:
        #     lane = self._map.get_lane(self.ego_transform.location)
        #     lane.draw_on_frame(frame, self.ego_transform.inverse_transform())
        if self._lanes:
            for lane in self._lanes:
                lane.draw_on_frame(frame)
