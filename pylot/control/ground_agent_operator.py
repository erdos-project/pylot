from collections import deque
import erdust
import math
from pid_controller.pid import PID

from pylot.control.messages import ControlMessage
import pylot.control.utils
from pylot.map.hd_map import HDMap
from pylot.simulation.carla_utils import get_map
import pylot.utils


class GroundAgentOperator(erdust.Operator):
    def __init__(self,
                 can_bus_stream,
                 ground_obstacles_stream,
                 ground_traffic_lights_stream,
                 waypoints_stream,
                 control_stream,
                 name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        can_bus_stream.add_callback(self.on_can_bus_update)
        ground_obstacles_stream.add_callback(self.on_obstacles_update)
        ground_traffic_lights_stream.add_callback(
            self.on_traffic_lights_update)
        waypoints_stream.add_callback(self.on_waypoints_update)
        erdust.add_watermark_callback(
            [can_bus_stream,
             ground_obstacles_stream,
             ground_traffic_lights_stream,
             waypoints_stream],
            [control_stream],
            self.on_watermark)
        self._name = name
        self._logger = erdust.utils.setup_logging(name, log_file_name)
        self._csv_logger = erdust.utils.setup_csv_logging(
            name + '-csv', csv_file_name)
        self._flags = flags
        self._map = HDMap(get_map(self._flags.carla_host,
                                  self._flags.carla_port,
                                  self._flags.carla_timeout),
                          log_file_name)
        self._pid = PID(p=flags.pid_p, i=flags.pid_i, d=flags.pid_d)
        self._can_bus_msgs = deque()
        self._obstacle_msgs = deque()
        self._traffic_light_msgs = deque()
        self._waypoint_msgs = deque()

    @staticmethod
    def connect(can_bus_stream,
                ground_obstacles_stream,
                ground_traffic_lights_stream,
                waypoints_stream):
        control_stream = erdust.WriteStream()
        return [control_stream]

    def on_watermark(self, timestamp, control_stream):
        self._logger.debug('Received watermark {}'.format(timestamp))
        # Get hero vehicle info.
        can_bus_msg = self._can_bus_msgs.popleft()
        vehicle_transform = can_bus_msg.data.transform
        vehicle_speed = can_bus_msg.data.forward_speed
        # Get waypoints.
        waypoint_msg = self._waypoint_msgs.popleft()
        wp_angle = waypoint_msg.wp_angle
        wp_vector = waypoint_msg.wp_vector
        wp_angle_speed = waypoint_msg.wp_angle_speed
        # Get ground obstacle info.
        obstacles = self._obstacle_msgs.popleft().obstacles
        # Get ground traffic lights info.
        traffic_lights = self._traffic_light_msgs.popleft().traffic_lights

        speed_factor, state = self.stop_for_agents(vehicle_transform.location,
                                                   wp_angle,
                                                   wp_vector,
                                                   obstacles,
                                                   traffic_lights)
        control_stream.send(
            self.get_control_message(
                wp_angle,
                wp_angle_speed,
                speed_factor,
                vehicle_speed,
                timestamp))

    def on_waypoints_update(self, msg):
        self._waypoint_msgs.append(msg)

    def on_can_bus_update(self, msg):
        self._can_bus_msgs.append(msg)

    def on_obstacles_update(self, msg):
        self._obstacle_msgs.append(msg)

    def on_traffic_lights_update(self, msg):
        self._traffic_light_msgs.append(msg)

    def stop_for_agents(self,
                        ego_vehicle_location,
                        wp_angle,
                        wp_vector,
                        obstacles,
                        traffic_lights):
        speed_factor = 1
        speed_factor_tl = 1
        speed_factor_p = 1
        speed_factor_v = 1

        for obstacle in obstacles:
            if obstacle.label == 'vehicle' and self._flags.stop_for_vehicles:
                # Only brake for vehicles that are in ego vehicle's lane.
                if self._map.are_on_same_lane(
                        ego_vehicle_location,
                        obstacle.transform.location):
                    new_speed_factor_v = pylot.control.utils.stop_vehicle(
                        ego_vehicle_location,
                        obstacle.transform.location,
                        wp_vector,
                        speed_factor_v,
                        self._flags)
                    speed_factor_v = min(speed_factor_v, new_speed_factor_v)
            if obstacle.label == 'pedestrian' and \
               self._flags.stop_for_pedestrians:
                # Only brake for pedestrians that are on the road.
                if self._map.is_on_lane(obstacle.transform.location):
                    new_speed_factor_p = pylot.control.utils.stop_pedestrian(
                        ego_vehicle_location,
                        obstacle.transform.location,
                        wp_vector,
                        speed_factor_p,
                        self._flags)
                    speed_factor_p = min(speed_factor_p, new_speed_factor_p)

        if self._flags.stop_for_traffic_lights:
            for tl in traffic_lights:
                if (self._map.must_obbey_traffic_light(
                        ego_vehicle_location, tl.transform.location) and
                    self._is_traffic_light_visible(
                        ego_vehicle_location, tl.transform.location)):
                    new_speed_factor_tl = pylot.control.utils.stop_traffic_light(
                        ego_vehicle_location,
                        tl.transform.location,
                        tl.state,
                        wp_vector,
                        wp_angle,
                        speed_factor_tl,
                        self._flags)
                    speed_factor_tl = min(speed_factor_tl, new_speed_factor_tl)

        speed_factor = min(speed_factor_tl, speed_factor_p, speed_factor_v)
        state = {
            'stop_pedestrian': speed_factor_p,
            'stop_vehicle': speed_factor_v,
            'stop_traffic_lights': speed_factor_tl
        }

        return speed_factor, state

    def get_control_message(self, wp_angle, wp_angle_speed, speed_factor,
                            current_speed, timestamp):
        current_speed = max(current_speed, 0)
        steer = self._flags.steer_gain * wp_angle
        if steer > 0:
            steer = min(steer, 1)
        else:
            steer = max(steer, -1)

        # Don't go to fast around corners
        if math.fabs(wp_angle_speed) < 0.1:
            target_speed_adjusted = self._flags.target_speed * speed_factor / 2
        else:
            target_speed_adjusted = self._flags.target_speed * speed_factor

        self._pid.target = target_speed_adjusted
        pid_gain = self._pid(feedback=current_speed)
        throttle = min(
            max(self._flags.default_throttle - 1.3 * pid_gain, 0),
            self._flags.throttle_max)

        if pid_gain > 0.5:
            brake = min(0.35 * pid_gain * self._flags.brake_strength, 1)
        else:
            brake = 0

        return ControlMessage(steer, throttle, brake, False, False, timestamp)

    def _is_traffic_light_visible(self, ego_vehicle_location, tl_location):
        _, tl_dist = pylot.control.utils.get_world_vec_dist(
            ego_vehicle_location.x,
            ego_vehicle_location.y,
            tl_location.x,
            tl_location.y)
        return tl_dist > self._flags.traffic_light_min_dist_thres
