"""Wrapper module for interacting with the Carla HD map.

This module implements HDMap class which offers utility methods for interacting
with the carla HD map.
"""

from collections import deque

import carla
# Import Planner from Carla codebase
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

import pylot.utils


class HDMap(object):
    """Wrapper class around the CARLA map.

    All Pylot methods should strive to use this class instead of directly
    accessing a CARLA map. This will make it easier to extend the probject
    with support for other types of HD maps in the future.

    Args:
        carla_map (carla.Map): An instance of a CARLA map.

    Attributes:
        _map (carla.Map): An instance of a CARLA map.
        _grp: An instance of a CARLA global route planner (uses A*).
    """
    def __init__(self, carla_map):
        self._map = carla_map
        # Setup global planner.
        self._grp = GlobalRoutePlanner(
            GlobalRoutePlannerDAO(
                self._map,
                1.0  # Distance between waypoints
            ))
        self._grp.setup()

    def get_closest_lane_waypoint(self, location):
        """Returns the road closest waypoint to location.

        Args:
            location (:py:class:`~pylot.utils.Location`): Location in world
                coordinates.

        Returns:
            :py:class:`~pylot.utils.Transform`: Transform or None if no
            waypoint is found.
        """
        waypoint = self._map.get_waypoint(location.as_carla_location(),
                                          project_to_road=True,
                                          lane_type=carla.LaneType.Any)
        if waypoint:
            return pylot.utils.Transform.from_carla_transform(
                waypoint.transform)
        else:
            return None

    def is_intersection(self, location):
        """Checks if a location is in an intersection.

        Args:
            location (:py:class:`~pylot.utils.Location`): Location in world
                coordinates.

        Returns:
            bool: True if the location is in an intersection.
        """
        waypoint = self._map.get_waypoint(location.as_carla_location(),
                                          project_to_road=False,
                                          lane_type=carla.LaneType.Any)
        if not waypoint:
            # The map didn't return a waypoint because the location not within
            # mapped location.
            return False
        else:
            # XXX(ionel): is_intersection will be deprecated in the future
            # Carla releases.
            return waypoint.is_intersection

    def is_transform_on_lane(self, transform):
        return self.is_on_lane(transform.location)

    def is_on_lane(self, location):
        """Checks if a location is on a lane.

        Args:
            location (:py:class:`~pylot.utils.Location`): Location in world
                coordinates.

        Returns:
            bool: True if the location is on a lane.
        """
        waypoint = self._map.get_waypoint(location.as_carla_location(),
                                          project_to_road=False,
                                          lane_type=carla.LaneType.Driving)
        if not waypoint:
            # The map didn't return a waypoint because the location not within
            # mapped location.
            return False
        else:
            return True

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
        waypoint1 = self._map.get_waypoint(location1.as_carla_location(),
                                           project_to_road=False,
                                           lane_type=carla.LaneType.Driving)
        if not waypoint1:
            # First location is not on a drivable lane.
            return False
        waypoint2 = self._map.get_waypoint(location2.as_carla_location(),
                                           project_to_road=False,
                                           lane_type=carla.LaneType.Driving)
        if not waypoint2:
            # Second location is not on a drivable lane.
            return False
        if waypoint1.road_id == waypoint2.road_id:
            return waypoint1.lane_id == waypoint2.lane_id
        else:
            # Return False if we're in intersection and the other
            # obstacle isn't.
            if waypoint1.is_intersection and not waypoint2.is_intersection:
                return False
            if waypoint2.lane_type == carla.LaneType.Driving:
                # This may return True when the lane is different, but in
                # with a different road_id.
                # TODO(ionel): Figure out how lane id map across road id.
                return True
        return False

    def is_on_opposite_lane(self, transform):
        """Checks if a transform is on an opposite lane.

        Args:
            transform (:py:class:`~pylot.utils.Transform`): Transform in world
                coordinates.

        Returns:
            bool: True if the transform is on the opposite lane.
        """
        waypoint = self._map.get_waypoint(
            transform.location.as_carla_location(),
            project_to_road=False,
            lane_type=carla.LaneType.Driving)
        if not waypoint:
            return True
        if waypoint.is_intersection:
            return False

        # XXX(ionel): Check logic.
        if (abs(waypoint.transform.rotation.yaw - transform.rotation.yaw) >
                140):
            return True
        else:
            return False

    def is_at_stop(self, location):
        """Checks if a location is close to a stop sign.

        Args:
            location (:py:class:`~pylot.utils.Location`): Location in world
                coordinates.

        Returns:
            bool: True if the location is at a stop sign.
        """
        # TODO(ionel): This method doesn't work yet because the opendrive do
        # not contained waypoints annotated as stops.
        waypoint = self._map.get_waypoint(location.as_carla_location(),
                                          project_to_road=False,
                                          lane_type=carla.LaneType.Stop)
        return not waypoint

    def distance_to_intersection(self, location, max_distance_to_check=30):
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
        waypoint = self._map.get_waypoint(location.as_carla_location(),
                                          project_to_road=False,
                                          lane_type=carla.LaneType.Any)
        if not waypoint:
            return None
        # We're already in an intersection.
        if waypoint.is_intersection:
            return 0
        for i in range(1, max_distance_to_check + 1):
            waypoints = waypoint.next(1)
            if not waypoints or len(waypoints) == 0:
                return None
            for w in waypoints:
                if w.is_intersection:
                    return i
            waypoint = waypoints[0]
        return None

    def is_on_bidirectional_lane(self, location):
        """Checks if a location is a bidirectional lane.

        Args:
            location (:py:class:`~pylot.utils.Location`): Location in world
                coordinates.

        Returns:
            bool: True if the location is on a bidirectional lane.
        """
        waypoint = self._map.get_waypoint(
            location.as_carla_location(),
            project_to_road=False,
            lane_type=carla.LaneType.Bidirectional)
        return not waypoint

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
        waypoint = self._map.get_waypoint(ego_location.as_carla_location(),
                                          project_to_road=False,
                                          lane_type=carla.LaneType.Any)
        if waypoint and waypoint.is_intersection:
            # Do not obey traffic light if ego is already in the intersection.
            return False

        # TODO(ionel): Implement.

        # return (math.fabs(
        #   old_carla_map.get_lane_orientation_degrees(
        #     [vehicle_transform.location.x,
        #      vehicle_transform.location.y,
        #      38]) -
        #   old_carla_map.get_lane_orientation_degrees(
        #     [closest_lane_point[0], closest_lane_point[1], 38])) < 1)

        return True

    def _must_obey_european_traffic_light(self, ego_transform, tl_locations,
                                          tl_max_dist_thresh):
        ego_loc = ego_transform.location.as_carla_location()
        ego_waypoint = self._map.get_waypoint(ego_loc,
                                              project_to_road=False,
                                              lane_type=carla.LaneType.Any)
        # We're not on a road, or we're already in the intersection. Carry on.
        if ego_waypoint is None or ego_waypoint.is_intersection:
            return (False, None)
        # Iterate through traffic lights.
        for tl_loc in tl_locations:
            tl_waypoint = self._map.get_waypoint(tl_loc.as_carla_location())
            if (tl_waypoint.road_id != ego_waypoint.road_id
                    or tl_waypoint.lane_id != ego_waypoint.lane_id):
                continue
            if ego_transform.is_within_distance_ahead(tl_loc,
                                                      tl_max_dist_thresh):
                return (True, tl_loc)
        return (False, None)

    def _must_obey_american_traffic_light(self, ego_transform, tl_locations,
                                          tl_max_dist_thresh):
        ego_loc = ego_transform.location.as_carla_location()
        ego_waypoint = self._map.get_waypoint(ego_loc,
                                              project_to_road=False,
                                              lane_type=carla.LaneType.Any)
        # We're not on a road, or we're already in the intersection. Carry on.
        if ego_waypoint is None or ego_waypoint.is_intersection:
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

    def get_freenet_coordinates(self, location):
        """Transforms a location into Freenet coordinates.

        Args:
            location (:py:class:`~pylot.utils.Location`): Location in world
                coordinates.
        Returns:
            s, d for a given Cartesian world location.
        """
        # TODO(ionel): This method assumes that the location has the
        # same orientation as the lanes (i.e., it will always return a
        # positive d).
        waypoint = self._map.get_waypoint(location.as_carla_location(),
                                          project_to_road=False,
                                          lane_type=carla.LaneType.Any)
        current_lane_w = waypoint
        d0_location = None
        while True:
            # Keep on moving left until we're outside the road or on the
            # oposite lane.
            left_lane_w = current_lane_w.get_left_lane()
            if (left_lane_w.lane_type != carla.LaneType.Driving
                    or (current_lane_w.transform.rotation.yaw -
                        left_lane_w.transform.rotation.yaw) > 170):
                # If the left lane is drivable then we've reached the left hand
                # side of a one way road. Alternatively, if the lane is rotated
                # then the lane is on the other side of the road.
                d0_location = current_lane_w
                half_lane = carla.Location(0, -current_lane_w.lane_width / 2.0,
                                           0)
                d0_location = current_lane_w.transform.transform(half_lane)
                break
            current_lane_w = left_lane_w

        # TODO(ionel): Handle the case when the road id changes -> s resets.
        # TODO(ionel): Handle case when the center lane is bidirectional.
        return waypoint.s, location.distance(d0_location)

    def get_left_lane(self, location):
        raise NotImplementedError

    def get_right_lane(self, location):
        raise NotImplementedError

    def compute_waypoints(self, source_loc, destination_loc):
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
        start_waypoint = self._map.get_waypoint(
            source_loc.as_carla_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving)
        end_waypoint = self._map.get_waypoint(
            destination_loc.as_carla_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving)
        assert start_waypoint and end_waypoint, 'Map could not find waypoints'
        route = self._grp.trace_route(start_waypoint.transform.location,
                                      end_waypoint.transform.location)
        # TODO(ionel): The planner returns several options in intersections.
        # We always take the first one, but this is not correct.
        return deque([
            pylot.utils.Transform.from_carla_transform(waypoint[0].transform)
            for waypoint in route
        ])
