import carla
import collections
import erdust
import itertools

from pylot.map.hd_map import HDMap
from pylot.planning.messages import WaypointsMessage
from pylot.simulation.carla_utils import get_map
from pylot.planning.utils import get_waypoint_vector_and_angle

DEFAULT_NUM_WAYPOINTS = 50


class WaypointPlanningOperator(erdust.Operator):
    """ Planning operator for Carla 0.9.x.

    IMPORTANT: Do not use with older Carla versions.
    The operator either receives all the waypoints from the scenario runner
    agent (on the global trajectory stream), or computes waypoints using the
    HD Map.
    """
    def __init__(self,
                 can_bus_stream,
                 waypoints_stream,
                 name,
                 flags,
                 goal_location=None,
                 log_file_name=None):
        can_bus_stream.add_callback(self.on_can_bus_update, [waypoints_stream])
        self._log_file_name = log_file_name
        self._flags = flags
        self._wp_index = 4  # use 5th waypoint for steering and speed
        self._map = HDMap(get_map(self._flags.carla_host,
                                  self._flags.carla_port,
                                  self._flags.carla_timeout),
                          log_file_name)
        self._goal_location = carla.Location(*goal_location)

    @staticmethod
    def connect(can_bus_stream):
        waypoints_stream = erdust.WriteStream()
        return [waypoints_stream]

    def on_can_bus_update(self, msg, waypoints_stream):
        """ Recompute path to goal location and send WaypointsMessage."""
        vehicle_transform = msg.data.transform
        waypoints = self._map.compute_waypoints(
            vehicle_transform.location.as_carla_location(),
            self._goal_location)
        next_waypoint = waypoints[self._wp_index]
        wp_steer_vector, wp_steer_angle = get_waypoint_vector_and_angle(
            next_waypoint, vehicle_transform)
        wp_speed_vector, wp_speed_angle = get_waypoint_vector_and_angle(
            next_waypoint, vehicle_transform)

        head_waypoints = collections.deque(
            itertools.islice(waypoints, 0, DEFAULT_NUM_WAYPOINTS)
        )  # only take 50 meters

        output_msg = WaypointsMessage(
            msg.timestamp,
            waypoints=head_waypoints,
            wp_angle=wp_steer_angle,
            wp_vector=wp_steer_vector,
            wp_angle_speed=wp_speed_angle
        )
        waypoints_stream.send(output_msg)
