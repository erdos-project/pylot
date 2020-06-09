import itertools
import math
from collections import deque

import numpy as np

import pylot.utils

DEFAULT_VIS_TIME = 0.1


class Waypoints(object):
    def __init__(self, waypoints, target_speeds=None):
        self.waypoints = waypoints
        if target_speeds is None:
            self.target_speeds = deque([0 for _ in range(len(waypoints))])
        else:
            self.target_speeds = target_speeds

    def as_numpy_array_2D(self):
        wx = []
        wy = []
        for wp in self.waypoints:
            wx.append(wp.location.x)
            wy.append(wp.location.y)
        return np.array([wx, wy])

    def closest_waypoint(self, location):
        """Finds the closest waypoint to the location."""
        min_dist = np.infty
        min_index = 0
        for index, waypoint in enumerate(self.waypoints):
            dist = waypoint.location.distance(location)
            if dist < min_dist:
                min_dist = dist
                min_index = index
        return min_index

    def remove_completed(self, location, ego_transform=None):
        """Removes waypoints that the ego vehicle has already completed.

        The method first finds the closest waypoint to the location,
        and then removes all waypoints that are before the closest waypoint.

        Args:
            location (:py:class:`pylot.utils.Location`): The location
                of the ego-vehicle.
        """
        min_index = self.closest_waypoint(location)
        # Remove waypoints that are before the closest waypoint. The ego
        # vehicle already completed them.
        while min_index > 0:
            self.waypoints.popleft()
            if self.target_speeds:
                self.target_speeds.popleft()
            min_index -= 1

        if self.waypoints is None or len(self.waypoints) == 0:
            if ego_transform is not None:
                self.waypoints = deque([ego_transform])
                self.target_speeds = deque([0])
            else:
                raise Exception('No more waypoints to follow')

    def recompute_waypoints(self, hd_map, ego_location, goal_location):
        self.waypoints = hd_map.compute_waypoints(ego_location, goal_location)
        self.target_speeds = deque([0 for _ in range(len(self.waypoints))])

    def _get_index(self, transform, min_distance):
        min_index = -1
        for index, wp in enumerate(self.waypoints):
            distance = wp.location.distance(transform.location)
            if distance >= min_distance:
                min_index = index
                break
        wp_index = min(len(self.waypoints) - 1, min_index)
        if wp_index < 0:
            raise ValueError('No more waypoints')
        return wp_index

    def get_angle(self, transform, min_distance):
        wp_index = self._get_index(transform, min_distance)
        angle, _ = transform.get_angle_and_magnitude(
            self.waypoints[wp_index].location)
        return angle

    def get_vector(self, transform, min_distance):
        wp_index = self._get_index(transform, min_distance)
        return self.waypoints[wp_index].location.as_vector_2D() - \
            transform.location.as_vector_2D()

    def get_target_speed(self, transform, min_distance):
        wp_index = self._get_index(transform, min_distance)
        return self.target_speeds[wp_index]

    def slice_waypoints(self, start_index, end_index, target_speed=None):
        head_wps = deque(
            itertools.islice(self.waypoints, start_index, end_index))
        if target_speed is not None:
            # Use the specified target speed.
            head_target_speeds = deque(
                [target_speed for _ in range(len(head_wps))])
        else:
            # Otherwise use the already existing target speeds.
            head_target_speeds = deque(
                itertools.islice(self.target_speeds, start_index, end_index))
        return Waypoints(head_wps, head_target_speeds)

    def draw_on_frame(self, bgr_frame):
        extrinsic_matrix = bgr_frame.camera_setup.get_extrinsic_matrix()
        intrinsic_matrix = bgr_frame.camera_setup.get_intrinsic_matrix()
        for wp in self.waypoints:
            pixel_location = wp.location.to_camera_view(
                extrinsic_matrix, intrinsic_matrix)
            bgr_frame.draw_point(pixel_location, [0, 0, 0])

    def draw_on_world(self, world):
        for wp in self.waypoints:
            # Adds 0.5 to z to ensure that the point is above the road surface.
            loc = (wp.location +
                   pylot.utils.Location(0, 0, 0.5)).as_carla_location()
            world.debug.draw_point(loc, size=0.1, life_time=DEFAULT_VIS_TIME)
