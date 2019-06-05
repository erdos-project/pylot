import time

import carla
# Import Planner from Carla codebase
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

from erdos.op import Op
from erdos.utils import setup_csv_logging, setup_logging

import pylot.utils
from pylot.control.utils import get_angle, get_world_vec_dist
from pylot.planning.messages import WaypointsMessage
from pylot.planning.utils import get_target_speed
from pylot.simulation.carla_utils import get_world
from pylot.simulation.utils import to_erdos_transform


class PlanningOperator(Op):

    def __init__(self,
                 name,
                 goal_location,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        super(PlanningOperator, self).__init__(name)
        self._flags = flags
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)

        # Transform goal location to carla.Location
        self._goal_location = carla.Location(*goal_location)

        _, self._world = get_world(
            self._flags.carla_host, self._flags.carla_port)
        if self._world is None:
            raise ValueError("There was an issue connecting to the simulator.")
        self._map = self._world.get_map()
        # Setup global planner.
        self._hop_resolution = 2.0
        dao = GlobalRoutePlannerDAO(self._map, self._hop_resolution)
        self._grp = GlobalRoutePlanner(dao)
        self._grp.setup()

    @staticmethod
    def setup_streams(input_streams):
        input_streams.filter(pylot.utils.is_can_bus_stream).add_callback(
            PlanningOperator.on_can_bus_update)
        return [pylot.utils.create_waypoints_stream()]

    def on_can_bus_update(self, msg):
        self._vehicle_transform = msg.data.transform
        route = self.__update_waypoints(
            carla.Location(self._vehicle_transform.location.x,
                           self._vehicle_transform.location.y,
                           self._vehicle_transform.location.z),
            self._goal_location)

        if not route or len(route) == 0:
            # If route is empty (e.g., reached destination), set waypoint to
            # current vehicle location.
            next_waypoints = [self._vehicle_transform]
        else:
            # Get the next 9 waypoints
            next_waypoints = route[:min(len(route), 9)]
            next_waypoints = [to_erdos_transform(waypoint[0].transform) for waypoint in next_waypoints]

        # If possible, skip the first two waypoints because they're too close.
        index = min(len(next_waypoints) - 1, 3)

        wp_vector, wp_mag = get_world_vec_dist(next_waypoints[index].location.x,
                                               next_waypoints[index].location.y,
                                               self._vehicle_transform.location.x,
                                               self._vehicle_transform.location.y)

        if wp_mag > 0:
            wp_angle = get_angle(
                wp_vector, [self._vehicle_transform.orientation.x, self._vehicle_transform.orientation.y])
        else:
            wp_angle = 0

        target_speed = get_target_speed(
            self._vehicle_transform.location, next_waypoints[index])
        output_msg = WaypointsMessage(msg.timestamp,
                                      waypoints=next_waypoints,
                                      target_speed=target_speed,
                                      wp_angle=wp_angle,
                                      wp_vector=wp_vector,
                                      wp_angle_speed=wp_angle)
        self.get_output_stream('waypoints').send(output_msg)

    def __update_waypoints(self, source_loc, destination_loc):
        start_waypoint = self._map.get_waypoint(source_loc)
        end_waypoint = self._map.get_waypoint(destination_loc)
        route = self._grp.trace_route(
            start_waypoint.transform.location,
            end_waypoint.transform.location)
        return route
