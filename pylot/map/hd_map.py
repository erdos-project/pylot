import math

import carla

from erdos.utils import setup_logging

from pylot.simulation.utils import to_erdos_transform


class HDMap(object):
    def __init__(self, carla_map, log_file_name=None):
        self._map = carla_map
        self._logger = setup_logging('hd_map', log_file_name)

    def is_intersection(self, location):
        """ Returns True if the location is in an intersection.

        Args:
            location: Location in world coordinates.
        """
        loc = carla.Location(location.x, location.y, location.z)
        waypoint = self._map.get_waypoint(loc,
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

    def are_on_same_lane(self, location1, location2):
        """ Returns True if the two locations are on the same lane.

        Args:
            location1: Location in world coordinates.
            location1: Location in world coordinates.
        """
        loc1 = carla.Location(location1.x, location1.y, location1.z)
        waypoint1 = self._map.get_waypoint(loc1,
                                           project_to_road=False,
                                           lane_type=carla.LaneType.Driving)
        if not waypoint1:
            # First location is not on a drivable lane.
            return False
        loc2 = carla.Location(location2.x, location2.y, location2.z)
        waypoint2 = self._map.get_waypoint(loc2,
                                           project_to_road=False,
                                           lane_type=carla.LaneType.Driving)
        if not waypoint2:
            # Second location is not on a drivable lane.
            return False
        w_t1 = to_erdos_transform(waypoint1.transform)
        w_t2 = to_erdos_transform(waypoint2.transform)
        self._logger.info('same_lane location1 {} to waypoint1 {}'.format(
            location1, w_t1.location))
        self._logger.info('same_lane location2 {} to waypoint2 {}'.format(
            location2, w_t2.location))
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

    def is_at_stop(self, location):
        """ Returns True if the location is at a stop sign.

        Args:
            location: Location in world coordinates.
        """
        # TODO(ionel): This method doesn't work yet because the opendrive do
        # not contained waypoints annotated as stops.
        loc = carla.Location(location.x, location.y, location.z)
        waypoint = self._map.get_waypoint(loc,
                                          project_to_road=False,
                                          lane_type=carla.LaneType.Stop)
        return not waypoint

    def distance_to_intersection(self, location, max_distance_to_check=30):
        loc = carla.Location(location.x, location.y, location.z)
        waypoint = self._map.get_waypoint(loc,
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
        loc = carla.Location(location.x, location.y, location.z)
        waypoint = self._map.get_waypoint(
            loc,
            project_to_road=False,
            lane_type=carla.LaneType.Bidirectional)
        return not waypoint

    def must_obbey_traffic_light(self, ego_location, tl_location):
        """ Returns True if the ego vehicle must obbey the traffic light.

        Args:
            ego_location: Location of the ego vehicle in world coordinates.
            tl_location: Location of the traffic light in world coordinates.
        """
        loc = carla.Location(ego_location.x, ego_location.y, ego_location.z)
        waypoint = self._map.get_waypoint(loc,
                                          project_to_road=False,
                                          lane_type=carla.LaneType.Any)
        if waypoint and waypoint.is_intersection:
            # Do not obbey traffic light if ego is already in the intersection.
            return False
        # TODO(ionel): Implement.
        return True

    def get_freenet_coordinates(self, location):
        """ Returns s, d for a given Cartesian world location. """
        # TODO(ionel): This method assumes that the location has the
        # same orientation as the lanes (i.e., it will always return a
        # positive d).
        loc = carla.Location(location.x, location.y, location.z)
        waypoint = self._map.get_waypoint(loc,
                                          project_to_road=False,
                                          lane_type=carla.LaneType.Any)
        current_lane_w = waypoint
        d0_location = None
        while True:
            # Keep on moving left until we're outside the road or on the
            # oposite lane.
            left_lane_w = current_lane_w.get_left_lane()
            if (left_lane_w.lane_type != carla.LaneType.Driving or
                (current_lane_w.transform.rotation.yaw -
                 left_lane_w.transform.rotation.yaw) > 170):
                # If the left lane is drivable then we've reached the left hand
                # side of a one way road. Alternatively, if the lane is rotated
                # then the lane is on the other side of the road.
                d0_location = current_lane_w
                half_lane = carla.Location(
                    0, - current_lane_w.lane_width / 2.0, 0)
                d0_location = current_lane_w.transform.transform(half_lane)
                break
            current_lane_w = left_lane_w

        # TODO(ionel): Handle the case when the road id changes -> s resets.
        # TODO(ionel): Handle case when the center lane is bidirectional.
        return waypoint.s, self.__get_distance(location, d0_location)

    def get_left_lane(self, location):
        # TODO(ionel): Implement!
        pass

    def get_right_lane(self, location):
        # TODO(ionel): Implement!
        pass

    def __get_distance(self, location1, location2):
        return math.sqrt((location1.x - location2.x) ** 2 +
                         (location1.y - location2.y) ** 2)
