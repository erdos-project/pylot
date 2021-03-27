"""Wrapper module for interacting with the CARLA HD map.

This module implements HDMap class which offers utility methods for interacting
with the CARLA HD map.
"""

from collections import deque

# Import Planner from CARLA codebase
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

from carla import LaneType

import erdos

from pylot.perception.detection.lane import Lane
from pylot.utils import Location, Transform


class HDMap(object):
    """Wrapper class around the CARLA map.

    All Pylot methods should strive to use this class instead of directly
    accessing a CARLA map. This will make it easier to extend the probject
    with support for other types of HD maps in the future.

    Attributes:
        _map: An instance of a CARLA map.
        _grp: An instance of a CARLA global route planner (uses A*).
    """
    def __init__(self, simulator_map, log_file=None):
        self._logger = erdos.utils.setup_logging('hd_map', log_file)
        self._map = simulator_map
        # Setup global planner.
        self._grp = GlobalRoutePlanner(
            GlobalRoutePlannerDAO(
                self._map,
                1.0  # Distance between waypoints
            ))
        self._grp.setup()

    def get_closest_lane_waypoint(self, location: Location) -> Transform:
        """Returns the road closest waypoint to location.

        Args:
            location (:py:class:`~pylot.utils.Location`): Location in world
                coordinates.

        Returns:
            :py:class:`~pylot.utils.Transform`: Transform or None if no
            waypoint is found.
        """
        waypoint = self._get_waypoint(location, project_to_road=True)
        if waypoint:
            return Transform.from_simulator_transform(waypoint.transform)
        else:
            return None

    def is_intersection(self, location: Location) -> bool:
        """Checks if a location is in an intersection.

        Args:
            location (:py:class:`~pylot.utils.Location`): Location in world
                coordinates.

        Returns:
            bool: True if the location is in an intersection.
        """
        waypoint = self._get_waypoint(location)
        if not waypoint:
            # The map didn't return a waypoint because the location not within
            # mapped location.
            return False
        else:
            return self.__is_intersection(waypoint)

    def __is_intersection(self, waypoint) -> bool:
        if waypoint.is_junction:
            return True
        if hasattr(waypoint, 'is_intersection'):
            return waypoint.is_intersection
        return False

    def is_on_lane(self, location: Location) -> bool:
        """Checks if a location is on a lane.

        Args:
            location (:py:class:`~pylot.utils.Location`): Location in world
                coordinates.

        Returns:
            bool: True if the location is on a lane.
        """
        waypoint = self._get_waypoint(location, lane_type=LaneType.Driving)
        if not waypoint:
            # The map didn't return a waypoint because the location not within
            # mapped location.
            return False
        else:
            return True

    def are_on_same_lane(self, location1: Location,
                         location2: Location) -> bool:
        """Checks if two locations are on the same lane.

        Args:
            location1 (:py:class:`~pylot.utils.Location`): Location in world
                coordinates.
            location2 (:py:class:`~pylot.utils.Location`): Location in world
                coordinates.

        Returns:
            bool: True if the two locations are on the same lane.
        """
        waypoint1 = self._get_waypoint(location1, lane_type=LaneType.Driving)
        if not waypoint1:
            # First location is not on a drivable lane.
            return False
        waypoint2 = self._get_waypoint(location2, lane_type=LaneType.Driving)
        if not waypoint2:
            # Second location is not on a drivable lane.
            return False
        if waypoint1.road_id == waypoint2.road_id:
            return waypoint1.lane_id == waypoint2.lane_id
        else:
            # Return False if we're in intersection and the other
            # obstacle isn't.
            if self.__is_intersection(
                    waypoint1) and not self.__is_intersection(waypoint2):
                return False
            if waypoint2.lane_type == LaneType.Driving:
                # This may return True when the lane is different, but in
                # with a different road_id.
                # TODO(ionel): Figure out how lane id map across road id.
                return True
        return False

    def is_on_opposite_lane(self, transform: Transform):
        """Checks if a transform is on an opposite lane.

        Args:
            transform (:py:class:`~pylot.utils.Transform`): Transform in world
                coordinates.

        Returns:
            bool: True if the transform is on the opposite lane.
        """
        waypoint = self._get_waypoint(transform.location,
                                      lane_type=LaneType.Driving)
        if not waypoint:
            return True
        if self.__is_intersection(waypoint):
            return False

        # XXX(ionel): Check logic.
        if (abs(waypoint.transform.rotation.yaw - transform.rotation.yaw) >
                140):
            return True
        else:
            return False

    def is_at_stop(self, location: Location):
        """Checks if a location is close to a stop sign.

        Args:
            location (:py:class:`~pylot.utils.Location`): Location in world
                coordinates.

        Returns:
            bool: True if the location is at a stop sign.
        """
        # TODO(ionel): This method doesn't work yet because the opendrive do
        # not contained waypoints annotated as stops.
        # waypoint = self._get_waypoint(location,
        #                               lane_type=LaneType.Stop)
        raise NotImplementedError

    def distance_to_intersection(self,
                                 location: Location,
                                 max_distance_to_check: float = 30):
        """Computes the distance (in meters) from location to an intersection.

        The method starts from location, moves forward until it reaches an
        intersection or exceeds max_distance_to_check.

        Args:
            location (:py:class:`~pylot.utils.Location`): The starting location
                in world coordinates.
            max_distance_to_check (:obj:`int`): Max distance to move forward
                 (in meters).

        Returns:
            :obj:`int`: The distance in meters, or None if there is no
            intersection within max_distance_to_check.
        """
        waypoint = self._get_waypoint(location)
        if not waypoint:
            return None
        # We're already in an intersection.
        if self.__is_intersection(waypoint):
            return 0
        for i in range(1, max_distance_to_check + 1):
            waypoints = waypoint.next(1)
            if not waypoints or len(waypoints) == 0:
                return None
            for w in waypoints:
                if self.__is_intersection(w):
                    return i
            waypoint = waypoints[0]
        return None

    def is_on_bidirectional_lane(self, location: Location) -> bool:
        """Checks if a location is a bidirectional lane.

        Args:
            location (:py:class:`~pylot.utils.Location`): Location in world
                coordinates.

        Returns:
            bool: True if the location is on a bidirectional lane.
        """
        waypoint = self._get_waypoint(location,
                                      lane_type=LaneType.Bidirectional)
        return not waypoint

    def must_obey_traffic_light(self, ego_location: Location,
                                tl_location: Location) -> bool:
        """Checks if an ego vehicle must obey a traffic light.

        Args:
            ego_location (:py:class:`~pylot.utils.Location`): Location of the
                ego vehicle in world coordinates.
            tl_location (:py:class:`~pylot.utils.Location`): Location of the
                traffic light in world coordinates.

        Returns:
            bool: True if the ego vehicle must obey the traffic light.
        """
        waypoint = self._get_waypoint(ego_location)
        if waypoint and self.__is_intersection(waypoint):
            # Do not obey traffic light if ego is already in the intersection.
            return False

        # TODO(ionel): Implement.

        return True

    def _must_obey_european_traffic_light(self, ego_transform: Transform,
                                          tl_locations,
                                          tl_max_dist_thresh: float) -> bool:
        ego_waypoint = self._get_waypoint(ego_transform.location)
        # We're not on a road, or we're already in the intersection. Carry on.
        if ego_waypoint is None or self.__is_intersection(ego_waypoint):
            return (False, None)
        # Iterate through traffic lights.
        for tl_loc in tl_locations:
            tl_waypoint = self._get_waypoint(tl_loc)
            if (tl_waypoint.road_id != ego_waypoint.road_id
                    or tl_waypoint.lane_id != ego_waypoint.lane_id):
                continue
            if ego_transform.is_within_distance_ahead(tl_loc,
                                                      tl_max_dist_thresh):
                return (True, tl_loc)
        return (False, None)

    def _must_obey_american_traffic_light(self, ego_transform: Transform,
                                          tl_locations,
                                          tl_max_dist_thresh: float) -> bool:
        ego_waypoint = self._get_waypoint(ego_transform.location)
        # We're not on a road, or we're already in the intersection. Carry on.
        if ego_waypoint is None or self.__is_intersection(ego_waypoint):
            return (False, None)

        min_angle = 25.0
        selected_tl_loc = None
        for tl_loc in tl_locations:
            if ego_transform.is_within_distance_ahead(tl_loc,
                                                      tl_max_dist_thresh):
                angle, distance = ego_transform.get_angle_and_magnitude(tl_loc)
                if distance < 60.0 and angle < min(25.0, min_angle):
                    min_angle = angle
                    selected_tl_loc = tl_loc
        if selected_tl_loc is not None:
            return (True, selected_tl_loc)
        else:
            return (False, None)

    def get_lane(self,
                 location: Location,
                 waypoint_precision: float = 0.05,
                 lane_id: int = 0) -> Lane:
        lane_waypoints = []
        # Consider waypoints in opposite direction of camera so we can get
        # lane data for adjacent lanes in opposing directions.
        previous_wp = [self._get_waypoint(location)]

        while len(previous_wp) == 1:
            lane_waypoints.append(previous_wp[0])
            previous_wp = previous_wp[0].previous(waypoint_precision)

        next_wp = [self._get_waypoint(location)]

        while len(next_wp) == 1:
            lane_waypoints.append(next_wp[0])
            next_wp = next_wp[0].next(waypoint_precision)

        # Get the left and right markings of the lane and send it as a message.
        left_markings = [
            self._lateral_shift(w.transform, -w.lane_width * 0.5)
            for w in lane_waypoints
        ]
        right_markings = [
            self._lateral_shift(w.transform, w.lane_width * 0.5)
            for w in lane_waypoints
        ]
        return Lane(lane_id, left_markings, right_markings)

    def get_left_lane(self, location: Location):
        waypoint = self._get_waypoint(location)
        if waypoint:
            left_lane_waypoint = waypoint.get_left_lane()
            if left_lane_waypoint:
                return Transform.from_simulator_transform(
                    left_lane_waypoint.transform)
        return None

    def get_right_lane(self, location: Location):
        waypoint = self._get_waypoint(location)
        if waypoint:
            right_lane_waypoint = waypoint.get_right_lane()
            if right_lane_waypoint:
                return Transform.from_simulator_transform(
                    right_lane_waypoint.transform)
        return None

    def get_all_lanes(self, location: Location):
        lanes = [self.get_lane(location)]

        waypoint = self._get_waypoint(location)
        if waypoint:
            wp_left = waypoint.get_left_lane()
            w_rotation = waypoint.transform.rotation
            while wp_left and wp_left.lane_type == LaneType.Driving:
                left_location = Location.from_simulator_location(
                    wp_left.transform.location)
                lanes.append(
                    self.get_lane(left_location, lane_id=wp_left.lane_id))

                # If left lane is facing the opposite direction, its left
                # lane would point back to the current lane, so we select
                # its right lane to get the left lane relative to current.
                if w_rotation == wp_left.transform.rotation:
                    wp_left = wp_left.get_left_lane()
                else:
                    wp_left = wp_left.get_right_lane()

            wp_right = waypoint.get_right_lane()
            while wp_right and wp_right.lane_type == LaneType.Driving:
                right_location = Location.from_simulator_location(
                    wp_right.transform.location)
                lanes.append(
                    self.get_lane(right_location, lane_id=wp_right.lane_id))

                # Same logic as above. If right lane of current is in
                # opposite direction, move rightwards by selecting it's
                # left lane.
                if w_rotation == wp_right.transform.rotation:
                    wp_right = wp_right.get_right_lane()
                else:
                    wp_right = wp_left.get_left_lane()
        return lanes

    def compute_waypoints(self, source_loc: Location,
                          destination_loc: Location):
        """Computes waypoints between two locations.

        Assumes that the ego vehicle has the same orientation as the lane on
        whch it is on.

        Args:
            source_loc (:py:class:`~pylot.utils.Location`): Source location in
                world coordinates.
            destination_loc (:py:class:`~pylot.utils.Location`): Destination
                location in world coordinates.

        Returns:
            list(:py:class:`~pylot.utils.Transform`): List of waypoint
            transforms.
        """
        start_waypoint = self._get_waypoint(source_loc,
                                            project_to_road=True,
                                            lane_type=LaneType.Driving)
        end_waypoint = self._get_waypoint(destination_loc,
                                          project_to_road=True,
                                          lane_type=LaneType.Driving)
        assert start_waypoint and end_waypoint, 'Map could not find waypoints'
        route = self._grp.trace_route(start_waypoint.transform.location,
                                      end_waypoint.transform.location)
        # TODO(ionel): The planner returns several options in intersections.
        # We always take the first one, but this is not correct.
        return deque([
            Transform.from_simulator_transform(waypoint[0].transform)
            for waypoint in route
        ])

    def _lateral_shift(self, transform, shift) -> Location:
        transform.rotation.yaw += 90
        shifted = transform.location + shift * transform.get_forward_vector()
        return Location.from_simulator_location(shifted)

    def _get_waypoint(self,
                      location: Location,
                      project_to_road: bool = False,
                      lane_type=LaneType.Any):
        try:
            waypoint = self._map.get_waypoint(location.as_simulator_location(),
                                              project_to_road=project_to_road,
                                              lane_type=lane_type)
        except RuntimeError as err:
            self._logger.error('get_waypoint call failed: {}'.format(err))
            waypoint = None
        return waypoint
