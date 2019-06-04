from collections import deque
import carla
import math
import numpy as np

from erdos.op import Op
from erdos.utils import frequency, setup_csv_logging, setup_logging

import pylot_utils
from planning.messages import WaypointsMessage
from planning.utils import get_distance, get_target_speed
from control.utils import get_angle, get_world_vec_dist


class ChallengePlanningOperator(Op):
    def __init__(self, name, flags, log_file_name=None, csv_file_name=None):
        super(ChallengePlanningOperator, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._flags = flags
        self._sampling_resolution = 1
        self._min_distance = self._sampling_resolution * 0.9
        self._map = None
        self._waypoints = None
        self._vehicle_transform = None

    @staticmethod
    def setup_streams(input_streams):
        input_streams.filter(pylot_utils.is_can_bus_stream).add_callback(
            ChallengePlanningOperator.on_can_bus_update)
        input_streams.filter(pylot_utils.is_open_drive_stream).add_callback(
            ChallengePlanningOperator.on_opendrive_map)
        input_streams.filter(pylot_utils.is_global_trajectory_stream).add_callback(
            ChallengePlanningOperator.on_global_trajectory)
        return [pylot_utils.create_waypoints_stream()]

    def on_can_bus_update(self, msg):
        self._vehicle_transform = msg.data.transform
        next_waypoint, _ = self.__compute_next_waypoint()

        wp_vector, wp_mag = get_world_vec_dist(
            next_waypoint.location.x,
            next_waypoint.location.y,
            self._vehicle_transform.location.x,
            self._vehicle_transform.location.y)

        if wp_mag > 0:
            wp_angle = get_angle(
                wp_vector,
                [self._vehicle_transform.orientation.x,
                 self._vehicle_transform.orientation.y])
        else:
            wp_angle = 0

        target_speed = get_target_speed(
            self._vehicle_transform.location, next_waypoint)

        output_msg = WaypointsMessage(
            msg.timestamp,
            waypoints=[next_waypoint],
            target_speed=target_speed,
            wp_angle=wp_angle,
            wp_vector=wp_vector,
            wp_angle_speed=wp_angle)
        self.get_output_stream('waypoints').send(output_msg)

    def on_opendrive_map(self, msg):
        assert self._map is None, 'Already receveid opendrive map'
        self._map = carla.Map('test', msg.data)

    def on_global_trajectory(self, msg):
        assert self._waypoints is None, 'Already received global trajectory'
        self._waypoints = deque()
        for waypoint_option in msg.data:
            self._waypoints.append(waypoint_option)

    def __compute_next_waypoint(self):
        if self._waypoints is None or len(self._waypoints) == 0:
            return self._vehicle_transform, None

        dist, index = self.__get_closest_waypoint()
        # Waypoints that are before the closest waypoint are irrelevant now.
        while index > 0:
            self._waypoints.popleft()
            index -= 1

        # The closest waypoint is 90% complete, remove it.
        if dist < self._min_distance:
            self._waypoints.popleft()

        return self._waypoints[min(len(self._waypoints) - 1, 3)]

    def __get_closest_waypoint(self):
        min_dist = 10000000
        min_index = 0
        index = 0
        for waypoint in self._waypoints:
            # We only check the first 10 waypoints.
            if index > 10:
                break
            dist = get_distance(waypoint[0].location,
                                self._vehicle_transform.location)
            if dist < min_dist:
                min_dist = dist
                min_index = index
        return min_dist, min_index
