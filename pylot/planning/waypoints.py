import csv
import itertools
from collections import deque

import numpy as np

import pylot.utils

DEFAULT_VIS_TIME = 1


class Waypoints(object):
    """Stores waypoints and provides methods to wrangle them."""
    def __init__(self, waypoints, target_speeds=None, road_options=None):
        self.waypoints = waypoints
        if target_speeds is None:
            # Set target speed to 0 if it is not specified.
            self.target_speeds = deque([0 for _ in range(len(waypoints))])
        else:
            self.target_speeds = target_speeds
        self.road_options = road_options

    @classmethod
    def read_from_csv_file(cls, csv_file_name, target_speed):
        """Reads waypoints from a csv file."""
        csv_file = open(csv_file_name)
        csv_reader = csv.reader(csv_file)
        waypoints = []
        for row in csv_reader:
            x = float(row[0])
            y = float(row[1])
            z = float(row[2])
            waypoint = pylot.utils.Transform(pylot.utils.Location(x, y, z),
                                             pylot.utils.Rotation(0, 0, 0))
            waypoints.append(waypoint)
        target_speeds = deque([target_speed for _ in range(len(waypoints))])
        return cls(deque(waypoints), target_speeds),

    def as_numpy_array_2D(self):
        """Returns the waypoints as a numpy array of lists of x and y."""
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
                raise ValueError('No more waypoints to follow')

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
        """Returns the angle between the transform and the first waypoint that
        is at least min_distance away."""
        wp_index = self._get_index(transform, min_distance)
        angle, _ = transform.get_angle_and_magnitude(
            self.waypoints[wp_index].location)
        return angle

    def get_vector(self, transform, min_distance):
        """Returns the vector between the transform and the first waypoint that
        is at least min_distance away."""
        wp_index = self._get_index(transform, min_distance)
        return self.waypoints[wp_index].location.as_vector_2D() - \
            transform.location.as_vector_2D()

    def get_target_speed(self, transform, min_distance):
        """Gets the target speed at the first waypoint that is at least
        min_distance away."""
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
        if self.road_options is not None:
            head_road_options = deque(
                itertools.islice(self.road_options, start_index, end_index))
        else:
            head_road_options = None
        return Waypoints(head_wps, head_target_speeds, head_road_options)

    def draw_on_frame(self, bgr_frame, inverse_transform=None):
        """Draw waypoints on a frame.

        Args:
            bgr_frame: Frame on which to draw the waypoints.
            inverse_transform (optional): To be used to transform the waypoints
                to relative to the ego vehicle.
        """
        extrinsic_matrix = bgr_frame.camera_setup.get_extrinsic_matrix()
        intrinsic_matrix = bgr_frame.camera_setup.get_intrinsic_matrix()
        for index, wp in enumerate(self.waypoints):
            if inverse_transform:
                wp = inverse_transform * wp
            pixel_location = wp.location.to_camera_view(
                extrinsic_matrix, intrinsic_matrix)
            bgr_frame.draw_point(pixel_location, [255, 255, 255])
            waypoint_txt = ''
            if self.target_speeds:
                waypoint_txt = '{:.1f}m/s'.format(self.target_speeds[index])
            if self.road_options:
                waypoint_txt = '{} {}'.format(waypoint_txt,
                                              self.road_options[index])
            if waypoint_txt != '':
                bgr_frame.draw_text(pixel_location, waypoint_txt,
                                    [255, 255, 255])

    def draw_on_world(self, world):
        """Draw waypoints on CARLA world."""
        for index, wp in enumerate(self.waypoints):
            # Adds 0.5 to z to ensure that the point is above the road surface.
            loc = (wp.location +
                   pylot.utils.Location(0, 0, 0.5)).as_carla_location()
            world.debug.draw_point(loc, size=0.1, life_time=DEFAULT_VIS_TIME)
            if self.road_options and index < len(self.road_options):
                world.debug.draw_string(loc,
                                        str(self.road_options[index]),
                                        life_time=DEFAULT_VIS_TIME)
