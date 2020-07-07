import copy
from collections import deque

import erdos

import numpy as np

import pylot.utils
from pylot.utils import Location, Rotation, Transform


class LaneMap(object):
    """Class that provides a map-like interface over lanes."""
    def __init__(self, log_file=None):
        self._logger = erdos.utils.setup_logging('hd_map', log_file)
        self.lanes = []

    def get_closest_lane_waypoint(self, location):
        """Returns the road closest waypoint to location.

        Args:
            location (:py:class:`~pylot.utils.Location`): Location in world
                coordinates.

        Returns:
            :py:class:`~pylot.utils.Transform`: Transform or None if no
            waypoint is found.
        """
        closest_transform = None
        min_dist = np.infty
        for lane in self.lanes:
            transform = lane.get_closest_lane_waypoint(location)
            dist = transform.location.distance(location)
            if dist < min_dist:
                min_dist = dist
                closest_transform = transform
        return closest_transform

    def is_intersection(self, location):
        """Checks if a location is in an intersection.

        Args:
            location (:py:class:`~pylot.utils.Location`): Location in world
                coordinates.

        Returns:
            bool: True if the location is in an intersection.
        """
        lane = self.get_lane(location)
        # Location is likely to be in an intersection if lane is None.
        return lane is None

    def is_on_lane(self, location):
        """Checks if a location is on a lane.

        Args:
            location (:py:class:`~pylot.utils.Location`): Location in world
                coordinates.

        Returns:
            bool: True if the location is on a lane.
        """
        for lane in self.lanes:
            if lane.is_on_lane(location):
                return True
        return False

    def are_on_same_lane(self, location1, location2):
        """Checks if two locations are on the same lane.

        Args:
            location1 (:py:class:`~pylot.utils.Location`): Location in world
                coordinates.
            location2 (:py:class:`~pylot.utils.Location`): Location in world
                coordinates.

        Returns:
            bool: True if the two locations are on the same lane.
        """
        for lane in self.lanes:
            if lane.is_on_lane(location1) and lane.is_on_lane(location2):
                return True
        return False

    def is_on_opposite_lane(self, transform):
        raise NotImplementedError

    def distance_to_intersection(self, location, max_distance_to_check=20):
        assert max_distance_to_check <= 20, 'Cannot check longer distances'
        lane = self.get_lane(location)
        if lane is None:
            return 0
        # TODO: Lanes do not currently stop earlier. Therefore, it's pointless
        # to check the distance to the intersection.
        return max_distance_to_check

    def must_obey_traffic_light(self, ego_location, tl_location):
        """Checks if an ego vehicle must obey a traffic light.

        Args:
            ego_location (:py:class:`~pylot.utils.Location`): Location of the
                ego vehicle in world coordinates.
            tl_location (:py:class:`~pylot.utils.Location`): Location of the
                traffic light in world coordinates.

        Returns:
            bool: True if the ego vehicle must obey the traffic light.
        """
        for lane in self.lanes:
            if lane.id == 0:
                return True
        return False

    def get_lane(self, location):
        """Returns the lane on which location is."""
        for lane in self.lanes:
            if lane.is_on_lane(location):
                return lane
        return None

    def get_left_lane(self, location):
        """Returns the lane to the left of the lane on which location is."""
        for index, lane in enumerate(self.lanes):
            if lane.is_on_lane(location):
                if index > 0:
                    return self.lanes[index - 1]
        return None

    def get_right_lane(self, location):
        """Returns the lane to the right of the lane on which location is."""
        for index, lane in enumerate(self.lanes):
            if lane.is_on_lane(location):
                if index + 1 < len(self.lanes):
                    return self.lanes[index + 1]
        return None

    def compute_waypoints(self, source_loc, destination_loc):
        raise NotImplementedError

    def generate_waypoints(self, ego_transform, waypoint, road_option):
        # Keep on driving in the same lane/straight if the next waypoint is
        # far away.
        if (road_option == pylot.utils.RoadOption.STRAIGHT
                or road_option == pylot.utils.RoadOption.LANE_FOLLOW
                or ego_transform.location.distance(
                    waypoint.location.distance) > 20):
            for lane in self.lanes:
                if lane.id == 0:
                    return lane.get_lane_center_transforms()
            # Lane detector didn't find ego's lane. Keep on driving
            # in the same direction.
            output_wps = deque([])
            for distance in range(1, 20):
                wp = (ego_transform *
                      Transform(Location(x=distance), Rotation()))
                output_wps.append(wp)
            return output_wps
        elif road_option == pylot.utils.RoadOption.LEFT:
            output_wps = deque([])
            wp = copy.deepcopy(ego_transform)
            for distance in range(1, 11):
                wp = wp * Transform(Location(x=1), Rotation(yaw=-9))
                output_wps.append(wp)
            return output_wps
        elif road_option == pylot.utils.RoadOption.RIGHT:
            output_wps = deque([])
            wp = copy.deepcopy(ego_transform)
            for distance in range(1, 11):
                wp = wp * Transform(Location(x=1), Rotation(yaw=9))
                output_wps.append(wp)
            return output_wps
        elif road_option == pylot.utils.RoadOption.CHANGE_LANE_LEFT:
            for lane in self.lanes:
                if lane.id == -1:
                    return lane.get_lane_center_transforms()
            # Lane detector didn't find left lane.
            output_wps = deque([])
            wp = copy.deepcopy(ego_transform)
            for distance in range(1, 11):
                wp = wp * Transform(Location(x=1), Rotation(yaw=-4))
                output_wps.append(wp)
            return output_wps
        elif road_option == pylot.utils.RoadOption.CHANGE_LANE_RIGHT:
            for lane in self.lanes:
                if lane.id == 1:
                    return lane.get_lane_center_transforms()
            # Lane detector didn't find right lane.
            output_wps = deque([])
            wp = copy.deepcopy(ego_transform)
            for distance in range(1, 11):
                wp = wp * Transform(Location(x=1), Rotation(yaw=4))
                output_wps.append(wp)
            return output_wps
        self._logger.debug('Unexpected road option {}'.format(road_option))
        return deque([ego_transform])
