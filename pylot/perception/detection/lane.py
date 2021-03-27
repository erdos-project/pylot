from collections import deque

import cv2

import numpy as np

from pylot.utils import Location, Rotation, Transform, Vector3D

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class Lane(object):
    """Stores information about a lane.

    Args:
        id (:obj:`int`): The id of the lane (0 for ego lane, negative for
            left lanes, and positive for right lanes).
        left_markings: List of transforms.
        right_markings: List of transforms.
    """
    def __init__(self, id, left_markings, right_markings):
        self.id = id
        self.left_markings = left_markings
        self.right_markings = right_markings
        self._lane_polygon = None
        self._color_map = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                           (125, 125, 0), (0, 125, 125), (125, 0, 125),
                           (50, 100, 50), (100, 50, 100)]

    def draw_on_frame(self, frame, inverse_transform=None, binary_frame=None):
        """Draw lane markings on a frame.

        Args:
            bgr_frame: Frame on which to draw the waypoints.
            inverse_transform (optional): To be used to transform the waypoints
                to relative to the ego vehicle.
        """
        extrinsic_matrix = frame.camera_setup.get_extrinsic_matrix()
        intrinsic_matrix = frame.camera_setup.get_intrinsic_matrix()
        # change color based on lane id
        lane_color_l = self._color_map[self.id % len(self._color_map)]
        lane_color_r = self._color_map[(self.id + 2) % len(self._color_map)]

        for marking in self.left_markings:
            if inverse_transform:
                # marking = inverse_transform * marking
                marking = inverse_transform.transform_points(
                    np.array([marking.as_numpy_array()]))
                marking = Vector3D(marking[0, 0], marking[0, 1], marking[0, 2])
            pixel_location = marking.to_camera_view(extrinsic_matrix,
                                                    intrinsic_matrix)
            if (pixel_location.z >= 0):
                try:
                    frame.draw_point(pixel_location, lane_color_l)
                    if binary_frame:
                        binary_frame.draw_point(pixel_location,
                                                (255, 255, 255))
                except Exception:
                    continue
        for marking in self.right_markings:
            if inverse_transform:
                # marking = inverse_transform * marking
                marking = inverse_transform.transform_points(
                    np.array([marking.as_numpy_array()]))
                marking = Vector3D(marking[0, 0], marking[0, 1], marking[0, 2])
            pixel_location = marking.to_camera_view(extrinsic_matrix,
                                                    intrinsic_matrix)
            if (pixel_location.z >= 0):
                try:
                    frame.draw_point(pixel_location, lane_color_r)
                    if binary_frame:
                        binary_frame.draw_point(pixel_location,
                                                (255, 255, 255))
                except Exception:
                    continue

    def collect_frame_data(self,
                           frame,
                           binary_frame,
                           camera_setup,
                           inverse_transform=None):
        """Draw lane markings on input frames for lane data collection.

        Args:
            frame: Grayscale image on which to draw the waypoints.
            binary_frame: Black and white image on which to draw the waypoints.
            camera_setup: Camera setup used to generate the frame.
            inverse_transform (optional): To be used to transform the waypoints
                to relative to the ego vehicle.
        """
        extrinsic_matrix = camera_setup.get_extrinsic_matrix()
        intrinsic_matrix = camera_setup.get_intrinsic_matrix()
        gray_color_map = [(20, 20), (70, 70), (120, 120), (170, 170),
                          (220, 220), (250, 250)]
        # change color based on lane id
        lane_color_l = gray_color_map[self.id % len(gray_color_map)]
        lane_color_r = gray_color_map[(self.id + 2) % len(gray_color_map)]

        for marking in self.left_markings:
            if inverse_transform:
                # marking = inverse_transform * marking
                marking = inverse_transform.transform_points(
                    np.array([marking.as_numpy_array()]))
                marking = Vector3D(marking[0, 0], marking[0, 1], marking[0, 2])
            pixel_location = marking.to_camera_view(extrinsic_matrix,
                                                    intrinsic_matrix)
            if (pixel_location.z >= 0):
                try:
                    cv2.circle(frame,
                               (int(pixel_location.x), int(pixel_location.y)),
                               3, lane_color_l, -1)
                    cv2.circle(binary_frame,
                               (int(pixel_location.x), int(pixel_location.y)),
                               3, (255, 255), -1)
                except Exception:
                    continue
        for marking in self.right_markings:
            if inverse_transform:
                # marking = inverse_transform * marking
                marking = inverse_transform.transform_points(
                    np.array([marking.as_numpy_array()]))
                marking = Vector3D(marking[0, 0], marking[0, 1], marking[0, 2])
            pixel_location = marking.to_camera_view(extrinsic_matrix,
                                                    intrinsic_matrix)
            if (pixel_location.z >= 0):
                try:
                    cv2.circle(frame,
                               (int(pixel_location.x), int(pixel_location.y)),
                               3, lane_color_r, -1)
                    cv2.circle(binary_frame,
                               (int(pixel_location.x), int(pixel_location.y)),
                               3, (255, 255), -1)
                except Exception:
                    continue

    def draw_on_world(self, world):
        from carla import Color
        for marking in self.left_markings:
            world.debug.draw_point(marking.as_simulator_location(),
                                   size=0.1,
                                   color=Color(255, 255, 0))
        for marking in self.right_markings:
            world.debug.draw_point(marking.as_simulator_location(),
                                   size=0.1,
                                   color=Color(255, 255, 0))

    def get_closest_lane_waypoint(self, location):
        if self.is_on_lane(location):
            return Transform(location, Rotation())
        closest_transform = None
        min_dist = np.infty
        for transform in self.left_markings:
            dist = transform.location.distance(location)
            if dist < min_dist:
                min_dist = dist
                closest_transform = transform
        for transform in self.right_markings:
            dist = transform.location.distance(location)
            if dist < min_dist:
                min_dist = dist
                closest_transform = transform
        return closest_transform

    def get_lane_center_transforms(self):
        if len(self.left_markings) < len(self.right_markings):
            anchor_markings = self.left_markings
            other_markings = self.right_markings
        else:
            anchor_markings = self.right_markings
            other_markings = self.left_markings
        index_other = 0
        center_markings = deque([])
        for transform in anchor_markings:
            dist = transform.location.distance(
                other_markings[index_other].location)
            while (index_other + 1 < len(other_markings)
                   and dist > transform.location.distance(
                       other_markings[index_other + 1].location)):
                index_other += 1
                dist = transform.location.distance(
                    other_markings[index_other].location)
            if index_other < len(other_markings):
                other_loc = other_markings[index_other].location
                center_location = Location(
                    (transform.location.x + other_loc.x) / 2.0,
                    (transform.location.y + other_loc.y) / 2.0,
                    (transform.location.z + other_loc.z) / 2.0)
                center_markings.append(Transform(center_location, Rotation()))
        return center_markings

    def is_on_lane(self, location):
        # We only create the lane polygon if it is necessary.
        if not self._lane_polygon:
            self._create_lane_polygon()
        return self._lane_polygon.contains(Point(location.x, location.y))

    def _create_lane_polygon(self):
        points = [(0, self.left_markings[0].y)]
        for transform in self.left_markings:
            points.append((transform.location.x, transform.location.y))
        for transform in reversed(self.right_markings):
            points.append((transform.location.x, transform.location.y))
        points.append((0, self.right_markings[0].y))
        self._lane_polygon = Polygon(points)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Lane(id: {}, {})'.format(
            self.id, zip(self.left_markings, self.right_markings))
