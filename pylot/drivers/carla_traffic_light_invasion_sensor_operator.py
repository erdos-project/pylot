"""This module implements an operator that publishes traffic lights infractions
committed by the ego vehicle.
"""

import time

from carla import Location, TrafficLightState

import erdos

import numpy as np

import pylot.utils
from pylot.simulation.messages import TrafficInfractionMessage
from pylot.simulation.utils import TrafficInfractionType, get_vehicle_handle, \
    get_world
from pylot.utils import Vector3D

from shapely.geometry import LineString


class CarlaTrafficLightInvasionSensorOperator(erdos.Operator):
    def __init__(self, ground_vehicle_id_stream: erdos.ReadStream,
                 pose_stream: erdos.ReadStream,
                 traffic_light_invasion_stream: erdos.WriteStream, flags):
        # Save the streams.
        self._vehicle_id_stream = ground_vehicle_id_stream
        self._pose_stream = pose_stream
        self._traffic_light_invasion_stream = traffic_light_invasion_stream

        # Register a callback on the pose stream to check if the ego-vehicle
        # is invading a traffic light.
        pose_stream.add_callback(self.on_pose_update)

        # The hero vehicle object we obtain from the simulator.
        self._vehicle = None

        # A list of all the traffic lights and their corresponding effected
        # waypoints.
        self._traffic_lights = []
        self._last_red_light_id = None
        self._world, self._map = None, None

        # Initialize a logger.
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)

        # Distance from the light to trigger the check at.
        self.DISTANCE_LIGHT = 10

    @staticmethod
    def connect(ground_vehicle_id_stream: erdos.ReadStream,
                pose_stream: erdos.ReadStream):
        traffic_light_invasion_stream = erdos.WriteStream()
        return [traffic_light_invasion_stream]

    def is_vehicle_crossing_line(self, seg1, seg2):
        """Checks if vehicle crosses a line segment."""
        line1 = LineString([(seg1[0].x, seg1[0].y), (seg1[1].x, seg1[1].y)])
        line2 = LineString([(seg2[0].x, seg2[0].y), (seg2[1].x, seg2[1].y)])
        inter = line1.intersection(line2)

        return not inter.is_empty

    def on_pose_update(self, msg):
        self._logger.debug("@{}: pose update.".format(msg.timestamp))

        transform = msg.data.transform
        location = Location(transform.location.x, transform.location.y,
                            transform.location.z)

        veh_extent = self._vehicle.bounding_box.extent.x

        tail_close_pt = Vector3D(-0.8 * veh_extent, 0.0, location.z).rotate(
            transform.rotation.yaw).as_simulator_vector()
        tail_close_pt = location + Location(tail_close_pt)

        tail_far_pt = Vector3D(-veh_extent - 1, 0.0, location.z).rotate(
            transform.rotation.yaw).as_simulator_vector()
        tail_far_pt = location + Location(tail_far_pt)

        for traffic_light, center, waypoints in self._traffic_lights:
            center_loc = Location(center)

            if self._last_red_light_id and \
                    self._last_red_light_id == traffic_light.id:
                continue
            if center_loc.distance(location) > self.DISTANCE_LIGHT:
                continue
            if traffic_light.state != TrafficLightState.Red:
                continue

            for wp in waypoints:
                tail_wp = self._map.get_waypoint(tail_far_pt)

                # Calculate the dot product
                ve_dir = transform.forward_vector
                wp_dir = wp.transform.get_forward_vector()
                dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + \
                    ve_dir.z * wp_dir.z

                # Check the lane until all the "tail" has passed
                if tail_wp.road_id == wp.road_id and \
                        tail_wp.lane_id == wp.lane_id and dot_ve_wp > 0:
                    # This light is red and is affecting our lane
                    yaw_wp = wp.transform.rotation.yaw
                    lane_width = wp.lane_width
                    location_wp = wp.transform.location

                    lft_lane_wp = Vector3D(
                        0.4 * lane_width, 0.0,
                        location_wp.z).rotate(yaw_wp +
                                              90).as_simulator_vector()
                    lft_lane_wp = location_wp + Location(lft_lane_wp)
                    rgt_lane_wp = Vector3D(
                        0.4 * lane_width, 0.0,
                        location_wp.z).rotate(yaw_wp -
                                              90).as_simulator_vector()
                    rgt_lane_wp = location_wp + Location(rgt_lane_wp)

                    # Is the vehicle traversing the stop line?
                    seg1 = (tail_close_pt, tail_far_pt)
                    seg2 = (lft_lane_wp, rgt_lane_wp)
                    if self.is_vehicle_crossing_line(seg1, seg2):
                        location = traffic_light.get_transform().location
                        message = TrafficInfractionMessage(
                            TrafficInfractionType.RED_LIGHT_INVASION,
                            pylot.utils.Location.from_simulator_location(
                                location), msg.timestamp)
                        self._traffic_light_invasion_stream.send(message)
                        self._last_red_light_id = traffic_light.id
                        break

    def run(self):
        # Read the vehicle ID from the vehicle ID stream.
        vehicle_id_msg = self._vehicle_id_stream.read()
        vehicle_id = vehicle_id_msg.data
        self._logger.debug("@{}: Received Vehicle ID: {}".format(
            vehicle_id_msg.timestamp, vehicle_id))

        # Connect to the world.
        _, self._world = get_world(self._flags.simulator_host,
                                   self._flags.simulator_port,
                                   self._flags.simulator_timeout)
        self._map = self._world.get_map()

        # Retrieve all the traffic lights from the world.
        while len(self._world.get_actors()) == 0:
            time.sleep(1)
        for _actor in self._world.get_actors():
            if 'traffic_light' in _actor.type_id:
                center, waypoints = self.get_traffic_light_waypoints(_actor)
                self._traffic_lights.append((_actor, center, waypoints))

        # Retrieve the vehicle.
        self._vehicle = get_vehicle_handle(self._world, vehicle_id)

    def get_traffic_light_waypoints(self, traffic_light):
        """
        get area of a given traffic light
        """
        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(
            traffic_light.trigger_volume.location)

        # Discretize the trigger box into points
        area_ext = traffic_light.trigger_volume.extent
        x_values = np.arange(-0.9 * area_ext.x, 0.9 * area_ext.x,
                             1.0)  # 0.9 to avoid crossing to adjacent lanes

        area = []
        for x in x_values:

            point = Vector3D(
                x, 0, area_ext.z).rotate(base_rot).as_simulator_vector()
            point_location = area_loc + Location(x=point.x, y=point.y)
            area.append(point_location)

        # Get the waypoints of these points, removing duplicates
        ini_wps = []
        for pt in area:
            wpx = self._map.get_waypoint(pt)
            # As x_values are arranged in order, only the last one has
            # to be checked
            if not ini_wps or ini_wps[-1].road_id != wpx.road_id or ini_wps[
                    -1].lane_id != wpx.lane_id:
                ini_wps.append(wpx)

        # Advance them until the intersection
        wps = []
        for wpx in ini_wps:
            while not wpx.is_intersection:
                next_wp = wpx.next(0.5)[0]
                if next_wp and not next_wp.is_intersection:
                    wpx = next_wp
                else:
                    break
            wps.append(wpx)

        return area_loc, wps
