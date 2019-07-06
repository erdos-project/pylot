from collections import deque
import math
import threading
import time
from pid_controller.pid import PID

from erdos.op import Op
from erdos.utils import setup_csv_logging, setup_logging, time_epoch_ms

from pylot.control.messages import ControlMessage
import pylot.control.utils
from pylot.simulation.planner.map import CarlaMap
import pylot.utils


class GroundAgentOperator(Op):
    def __init__(self,
                 name,
                 city_name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        super(GroundAgentOperator, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._map = CarlaMap(city_name)
        self._flags = flags
        self._pid = PID(p=flags.pid_p, i=flags.pid_i, d=flags.pid_d)
        self._can_bus_msgs = deque()
        self._pedestrian_msgs = deque()
        self._vehicle_msgs = deque()
        self._traffic_light_msgs = deque()
        self._speed_limit_sign_msgs = deque()
        self._waypoint_msgs = deque()
        self._lock = threading.Lock()

    @staticmethod
    def setup_streams(input_streams):
        input_streams.filter(pylot.utils.is_can_bus_stream).add_callback(
            GroundAgentOperator.on_can_bus_update)
        input_streams.filter(
            pylot.utils.is_ground_pedestrians_stream).add_callback(
                GroundAgentOperator.on_pedestrians_update)
        input_streams.filter(
            pylot.utils.is_ground_vehicles_stream).add_callback(
                GroundAgentOperator.on_vehicles_update)
        input_streams.filter(
            pylot.utils.is_ground_traffic_lights_stream).add_callback(
                GroundAgentOperator.on_traffic_lights_update)
        input_streams.filter(
            pylot.utils.is_ground_speed_limit_signs_stream).add_callback(
                GroundAgentOperator.on_speed_limit_signs_update)
        input_streams.filter(pylot.utils.is_waypoints_stream).add_callback(
            GroundAgentOperator.on_waypoints_update)
        input_streams.add_completion_callback(
            GroundAgentOperator.on_notification)
        # Set no watermark on the output stream so that we do not
        # close the watermark loop with the carla operator.
        return [pylot.utils.create_control_stream()]

    def on_notification(self, msg):
        # Get hero vehicle info.
        can_bus_msg = self._can_bus_msgs.popleft()
        vehicle_transform = can_bus_msg.data.transform
        vehicle_speed = can_bus_msg.data.forward_speed
        # Get waypoints.
        waypoint_msg = self._waypoint_msgs.popleft()
        wp_angle = waypoint_msg.wp_angle
        wp_vector = waypoint_msg.wp_vector
        wp_angle_speed = waypoint_msg.wp_angle_speed
        # Get ground pedestrian info.
        pedestrians_msg = self._pedestrian_msgs.popleft()
        pedestrians = pedestrians_msg.pedestrians
        # Get ground vehicle info.
        vehicles_msg = self._vehicle_msgs.popleft()
        vehicles = vehicles_msg.vehicles
        # Get ground traffic lights info.
        traffic_lights_msg = self._traffic_light_msgs.popleft()
        traffic_lights = traffic_lights_msg.traffic_lights
        # Get ground traffic signs info.
        speed_limit_signs_msg = self._speed_limit_sign_msgs.popleft()
        speed_limit_signs = speed_limit_signs_msg.speed_signs
        # TODO(ionel): Use traffic signs info as well.

        speed_factor, state = self.stop_for_agents(vehicle_transform,
                                                   wp_angle,
                                                   wp_vector,
                                                   vehicles,
                                                   pedestrians,
                                                   traffic_lights)
        control_msg = self.get_control_message(
            wp_angle, wp_angle_speed, speed_factor,
            vehicle_speed, msg.timestamp)
        self.get_output_stream('control_stream').send(control_msg)

    def on_waypoints_update(self, msg):
        with self._lock:
            self._waypoint_msgs.append(msg)

    def on_can_bus_update(self, msg):
        with self._lock:
            self._can_bus_msgs.append(msg)

    def on_pedestrians_update(self, msg):
        with self._lock:
            self._pedestrian_msgs.append(msg)

    def on_vehicles_update(self, msg):
        with self._lock:
            self._vehicle_msgs.append(msg)

    def on_traffic_lights_update(self, msg):
        with self._lock:
            self._traffic_light_msgs.append(msg)

    def on_speed_limit_signs_update(self, msg):
        with self._lock:
            self._speed_limit_sign_msgs.append(msg)

    def stop_for_agents(self,
                        vehicle_transform,
                        wp_angle,
                        wp_vector,
                        vehicles,
                        pedestrians,
                        traffic_lights):
        speed_factor = 1
        speed_factor_tl = 1
        speed_factor_p = 1
        speed_factor_v = 1

        if self._flags.stop_for_vehicles:
            for obs_vehicle in vehicles:
                if self._is_vehicle_on_same_lane(
                        vehicle_transform,
                        obs_vehicle.transform):
                    new_speed_factor_v = pylot.control.utils.stop_vehicle(
                        vehicle_transform,
                        obs_vehicle.transform.location,
                        wp_vector,
                        speed_factor_v,
                        self._flags)
                    speed_factor_v = min(speed_factor_v, new_speed_factor_v)

        if self._flags.stop_for_pedestrians:
            for pedestrian in pedestrians:
                if self._is_pedestrian_hitable(pedestrian.transform):
                    new_speed_factor_p = pylot.control.utils.stop_pedestrian(
                        vehicle_transform,
                        pedestrian.transform.location,
                        wp_vector,
                        speed_factor_p,
                        self._flags)
                    speed_factor_p = min(speed_factor_p, new_speed_factor_p)

        if self._flags.stop_for_traffic_lights:
            for tl in traffic_lights:
                if (self._is_traffic_light_active(
                        vehicle_transform, tl.transform) and
                    self._is_traffic_light_visible(
                        vehicle_transform, tl.transform)):
                    new_speed_factor_tl = pylot.control.utils.stop_traffic_light(
                        vehicle_transform,
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

    def execute(self):
        self.spin()

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
            target_speed_adjusted = self._flags.target_speed * speed_factor
        elif math.fabs(wp_angle_speed) < 0.5:
            target_speed_adjusted = 20 * speed_factor
        else:
            target_speed_adjusted = 15 * speed_factor

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

    def _is_pedestrian_hitable(self, ped_transform):
        return self._map.is_point_on_lane(
            [ped_transform.location.x, ped_transform.location.y, 38])

    def _is_vehicle_on_same_lane(self, ego_transform, obs_vehicle_transform):
        if self._map.is_point_on_intersection(
                [obs_vehicle_transform.location.x,
                 obs_vehicle_transform.location.y,
                 38]):
            return True
        return (math.fabs(
            self._map.get_lane_orientation_degrees(
                [ego_transform.location.x,
                 ego_transform.location.y,
                 38]) -
            self._map.get_lane_orientation_degrees(
                [obs_vehicle_transform.location.x,
                 obs_vehicle_transform.location.y,
                 38])) < 1)

    def _is_traffic_light_active(self, vehicle_transform, tl_transform):
        def search_closest_lane_point(x_agent, y_agent, depth):
            step_size = 4
            if depth > 1:
                return None
            try:
                degrees = self._map.get_lane_orientation_degrees(
                    [x_agent, y_agent, 38])
            except:
                return None
            if not self._map.is_point_on_lane([x_agent, y_agent, 38]):
                result = search_closest_lane_point(x_agent + step_size,
                                                   y_agent, depth + 1)
                if result is not None:
                    return result
                result = search_closest_lane_point(
                    x_agent, y_agent + step_size, depth + 1)
                if result is not None:
                    return result
                result = search_closest_lane_point(
                    x_agent + step_size, y_agent + step_size, depth + 1)
                if result is not None:
                    return result
                result = search_closest_lane_point(
                    x_agent + step_size, y_agent - step_size, depth + 1)
                if result is not None:
                    return result
                result = search_closest_lane_point(
                    x_agent - step_size, y_agent + step_size, depth + 1)
                if result is not None:
                    return result
                result = search_closest_lane_point(x_agent - step_size,
                                                   y_agent, depth + 1)
                if result is not None:
                    return result
                result = search_closest_lane_point(
                    x_agent, y_agent - step_size, depth + 1)
                if result is not None:
                    return result
                result = search_closest_lane_point(
                    x_agent - step_size, y_agent - step_size, depth + 1)
                if result is not None:
                    return result
            else:
                if degrees < 6:
                    return [x_agent, y_agent]
                else:
                    return None

        closest_lane_point = search_closest_lane_point(
            tl_transform.location.x, tl_transform.location.y, 0)

        if closest_lane_point is not None:
            return (math.fabs(
                self._map.get_lane_orientation_degrees(
                    [vehicle_transform.location.x,
                     vehicle_transform.location.y,
                     38]) -
                self._map.get_lane_orientation_degrees(
                    [closest_lane_point[0], closest_lane_point[1], 38])) < 1)
        else:
            return None

    def _is_traffic_light_visible(self, vehicle_transform, tl_transform):
        _, tl_dist = pylot.control.utils.get_world_vec_dist(
            vehicle_transform.location.x,
            vehicle_transform.location.y,
            tl_transform.location.x,
            tl_transform.location.y)
        return tl_dist > self._flags.traffic_light_min_dist_thres
