import numpy as np

import pylot.utils

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

    def draw_on_frame(self, frame, inverse_transform=None):
        """Draw lane markings on a frame.

        Args:
            bgr_frame: Frame on which to draw the waypoints.
            inverse_transform (optional): To be used to transform the waypoints
                to relative to the ego vehicle.
        """
        extrinsic_matrix = frame.camera_setup.get_extrinsic_matrix()
        intrinsic_matrix = frame.camera_setup.get_intrinsic_matrix()
        for marking in self.left_markings:
            if inverse_transform:
                marking = inverse_transform * marking
            pixel_location = marking.location.to_camera_view(
                extrinsic_matrix, intrinsic_matrix)
            frame.draw_point(pixel_location, [255, 255, 0])
        for marking in self.right_markings:
            if inverse_transform:
                marking = inverse_transform * marking
            pixel_location = marking.location.to_camera_view(
                extrinsic_matrix, intrinsic_matrix)
            frame.draw_point(pixel_location, [255, 255, 0])

    def draw_on_world(self, world):
        import carla
        for marking in self.left_markings:
            world.debug.draw_point(marking.as_carla_location(),
                                   size=0.1,
                                   color=carla.Color(255, 255, 0))
        for marking in self.right_markings:
            world.debug.draw_point(marking.as_carla_location(),
                                   size=0.1,
                                   color=carla.Color(255, 255, 0))

    def get_closest_lane_waypoint(self, location):
        if self.is_on_lane(location):
            return pylot.utils.Transform(location, pylot.utils.Rotation())
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
