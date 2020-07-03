from collections import deque

from pylot.planning.waypoints import Waypoints
from pylot.utils import Location, Rotation, Transform


class Planner(object):
    def __init__(self, world, flags, logger):
        self._flags = flags
        self._logger = logger
        self._world = world
        # TODO: Deal with the map
        self._map = None

    def build_output_waypoints(self, path_x, path_y, speeds):
        wps = deque()
        target_speeds = deque()
        for point in zip(path_x, path_y, speeds):
            if self._map is not None:
                p_loc = self._map.get_closest_lane_waypoint(
                    Location(x=point[0], y=point[1], z=0)).location
            else:
                p_loc = Location(x=point[0], y=point[1], z=0)
            wps.append(
                Transform(
                    location=Location(x=point[0], y=point[1], z=p_loc.z),
                    rotation=Rotation(),
                ))
            target_speeds.append(point[2])
        return Waypoints(wps, target_speeds)

    def run(self, timestamp):
        raise NotImplementedError
