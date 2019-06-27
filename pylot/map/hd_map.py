import carla


class HDMap(object):
    def __init__(self, carla_map):
        self._map = carla_map

    def is_intersection(self, location):
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
        if waypoint1.road_id == waypoint2.road_id:
            return waypoint1.lane_id == waypoint2.lane_id
        else:
            if waypoint2.lane_type == carla.LaneType.Driving:
                # This may return True when the lane is different, but in
                # with a different road_id.
                # TODO(ionel): Figure out how lane id map across road id.
                return True
        return False

    def get_freenet(self, location):
        loc = carla.Location(location.x, location.y, location.z)
        waypoint = self._map.get_waypoint(loc,
                                          project_to_road=False,
                                          lane_type=carla.LaneType.Any)
        # TODO(ionel): Implement!
        return 0

    def get_left_lane(self, location):
        # TODO(ionel): Implement!
        pass

    def get_right_lane(self, location):
        # TODO(ionel): Implement!
        pass
