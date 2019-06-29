from collections import deque
import math
from pid_controller.pid import PID
import threading
import time

import carla

# ERDOS imports
from erdos.op import Op
from erdos.utils import setup_csv_logging, setup_logging, time_epoch_ms

# Pylot imports
from pylot.control.messages import ControlMessage
import pylot.control.utils
from pylot.map.hd_map import HDMap
import pylot.simulation.utils
from pylot.simulation.utils import get_3d_world_position_with_point_cloud
import pylot.utils


class LidarERDOSAgentOperator(Op):
    def __init__(self,
                 name,
                 flags,
                 bgr_camera_setup,
                 log_file_name=None,
                 csv_file_name=None):
        super(LidarERDOSAgentOperator, self).__init__(name)
        self._flags = flags
        self._log_file_name = log_file_name
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._bgr_camera_setup = bgr_camera_setup
        self._map = None
        self._pid = PID(p=self._flags.pid_p,
                        i=self._flags.pid_i,
                        d=self._flags.pid_d)
        self._waypoint_msgs = deque()
        self._can_bus_msgs = deque()
        self._traffic_lights = deque()
        self._obstacles = deque()
        self._point_clouds = deque()
        self._vehicle_labels = {'car', 'bicycle', 'motorcycle', 'bus', 'truck'}
        self._lock = threading.Lock()
        self._last_traffic_light_game_time = -100000
        self._last_moving_time = 0

    @staticmethod
    def setup_streams(input_streams):
        input_streams.filter(
            pylot.utils.is_can_bus_stream).add_callback(
                LidarERDOSAgentOperator.on_can_bus_update)
        input_streams.filter(
            pylot.utils.is_waypoints_stream).add_callback(
                LidarERDOSAgentOperator.on_waypoints_update)
        input_streams.filter(
            pylot.utils.is_traffic_lights_stream).add_callback(
                LidarERDOSAgentOperator.on_traffic_lights_update)
        input_streams.filter(
            pylot.utils.is_obstacles_stream).add_callback(
                LidarERDOSAgentOperator.on_obstacles_update)
        input_streams.filter(
            pylot.utils.is_lidar_stream).add_callback(
                LidarERDOSAgentOperator.on_lidar_update)
        input_streams.filter(
            pylot.utils.is_open_drive_stream).add_callback(
                LidarERDOSAgentOperator.on_opendrive_map)

        input_streams.add_completion_callback(
            LidarERDOSAgentOperator.on_notification)

        # Set no watermark on the output stream so that we do not
        # close the watermark loop with the carla operator.
        return [pylot.utils.create_control_stream()]

    def on_notification(self, msg):
        start_time = time.time()
        with self._lock:
            can_bus_msg = self._can_bus_msgs.popleft()
            waypoint_msg = self._waypoint_msgs.popleft()
            pc_msg = self._point_clouds.popleft()
            tl_output = self._traffic_lights.popleft()
            obstacles = self._obstacles.popleft()

        self._logger.info("Timestamps {} {} {} {} {}".format(
            can_bus_msg.timestamp, waypoint_msg.timestamp, pc_msg.timestamp,
            tl_output.timestamp, obstacles.timestamp))
        assert (can_bus_msg.timestamp == waypoint_msg.timestamp ==
                pc_msg.timestamp == tl_output.timestamp == obstacles.timestamp)

        vehicle_transform = can_bus_msg.data.transform
        vehicle_speed = can_bus_msg.data.forward_speed
        wp_angle = waypoint_msg.wp_angle
        wp_vector = waypoint_msg.wp_vector
        wp_angle_speed = waypoint_msg.wp_angle_speed
        target_speed = waypoint_msg.target_speed
        # Transform point cloud to camera coordinates.
        point_cloud = pylot.simulation.utils.lidar_point_cloud_to_camera_coordinates(
            pc_msg.point_cloud)

        traffic_lights = self.__transform_tl_output(
            tl_output, point_cloud, vehicle_transform)
        game_time = msg.timestamp.coordinates[0]
        if len(traffic_lights) > 0:
            self._last_traffic_light_game_time = game_time
        (pedestrians, vehicles) = self.__transform_detector_output(
            obstacles, point_cloud, vehicle_transform)

        # if self._map.is_on_opposite_lane(vehicle_transform):
        #     # Ignore obstacles
        #     self._logger.info('Ego-vehicle {} on opposite lange'.format(
        #         vehicle_transform))
        #     pedestrians = []
        #     vehicles = []
        #     traffic_lights = []

        self._logger.info('{} Current speed {} and location {}'.format(
            msg.timestamp, vehicle_speed, vehicle_transform))
        self._logger.info('{} Pedestrians {}'.format(
            msg.timestamp, pedestrians))
        self._logger.info('{} Vehicles {}'.format(
            msg.timestamp, vehicles))

        speed_factor, _ = self.__stop_for_agents(
            vehicle_transform,
            wp_angle,
            wp_vector,
            vehicles,
            pedestrians,
            traffic_lights,
            msg.timestamp)

        new_target_speed = self.reduce_speed_when_approaching_intersection(
            vehicle_transform,
            vehicle_speed,
            target_speed,
            game_time)
        if new_target_speed != target_speed:
            self._logger.info('Proximity to intersection, reducing speed from {} to {}'.format(
                target_speed, new_target_speed))
            target_speed = new_target_speed

        self._logger.info('{} Current speed factor {}'.format(
            msg.timestamp, speed_factor))

        control_msg = self.get_control_message(
            wp_angle, wp_angle_speed, speed_factor,
            vehicle_speed, target_speed, msg.timestamp)

        if control_msg.throttle > 0.001:
            self._last_moving_time = game_time
        # Overide control message if we haven't been moving for a while.
        # Might be stuck because of a faulty detector.
        if game_time - self._last_moving_time > 30000:
            control_msg = ControlMessage(
                0, 0.5, 0, False, False, msg.timestamp)

        # Get runtime in ms.
        runtime = (time.time() - start_time) * 1000
        self._csv_logger.info('{},{},"{}",{}'.format(
            time_epoch_ms(), self.name, msg.timestamp, runtime))

        self.get_output_stream('control_stream').send(control_msg)

    def on_waypoints_update(self, msg):
        self._logger.info("Waypoints update at {}".format(msg.timestamp))
        with self._lock:
            self._waypoint_msgs.append(msg)

    def on_can_bus_update(self, msg):
        self._logger.info("Can bus update at {}".format(msg.timestamp))
        with self._lock:
            self._can_bus_msgs.append(msg)

    def on_traffic_lights_update(self, msg):
        self._logger.info("Traffic light update at {}".format(msg.timestamp))
        with self._lock:
            self._traffic_lights.append(msg)

    def on_obstacles_update(self, msg):
        self._logger.info("Obstacle update at {}".format(msg.timestamp))
        with self._lock:
            self._obstacles.append(msg)

    def on_lidar_update(self, msg):
        self._logger.info("Lidar update at {}".format(msg.timestamp))
        with self._lock:
            self._point_clouds.append(msg)

    def on_opendrive_map(self, msg):
        self._map = HDMap(carla.Map('challenge', msg.data),
                          self._log_file_name)

    def execute(self):
        self.spin()

    def reduce_speed_when_approaching_intersection(
            self,
            vehicle_transform,
            vehicle_speed,
            target_speed,
            game_time):
        intersection_dist = self._map.distance_to_intersection(
            vehicle_transform.location,
            max_distance_to_check=30)
        if not intersection_dist or intersection_dist < 4:
            # We are not close to an intersection or we're already
            # too close.
            return target_speed

        # Reduce the speed because we're getting close to an intersection.
        # In this way, we can stop even if we detect the traffic light
        # very late.
        if intersection_dist < 30:
            target_speed = min(target_speed, 5)

        # We assume that we are at a stop sign.
        if (intersection_dist < 10 and
            game_time - self._last_traffic_light_game_time > 4000):
            if vehicle_speed < 0.09:
                # We've already stopped at the intersection.
                target_speed = min(target_speed, 12)
            else:
                # Stop at the intersection.
                target_speed = min(target_speed, 0)

        return target_speed

    def __transform_to_3d(self, x, y, point_cloud, vehicle_transform):
        pos = get_3d_world_position_with_point_cloud(
            x, y, point_cloud,
            vehicle_transform * self._bgr_camera_setup.transform,
            self._bgr_camera_setup.width,
            self._bgr_camera_setup.height,
            self._bgr_camera_setup.fov)
        if pos is None:
            self._logger.error(
                'Could not find lidar point for {} {}'.format(x, y))
        return pos

    def __transform_tl_output(self, tls, point_cloud, vehicle_transform):
        traffic_lights = []
        for tl in tls.detected_objects:
            x = (tl.corners[0] + tl.corners[1]) / 2
            y = (tl.corners[2] + tl.corners[3]) / 2
            pos = self.__transform_to_3d(x, y, point_cloud, vehicle_transform)
            if pos:
                traffic_lights.append((pos, tl.label))
        return traffic_lights

    def __transform_detector_output(
            self, obstacles, point_cloud, vehicle_transform):
        vehicles = []
        pedestrians = []
        for detected_obj in obstacles.detected_objects:
            x = (detected_obj.corners[0] + detected_obj.corners[1]) / 2
            y = (detected_obj.corners[2] + detected_obj.corners[3]) / 2
            if detected_obj.label == 'person':
                pos = self.__transform_to_3d(
                    x, y, point_cloud, vehicle_transform)
                if pos:
                    pedestrians.append(pos)
            elif (detected_obj.label in self._vehicle_labels):
                pos = self.__transform_to_3d(
                    x, y, point_cloud, vehicle_transform)
                if pos:
                    vehicles.append(pos)
        return (pedestrians, vehicles)

    def __stop_for_agents(self,
                          vehicle_transform,
                          wp_angle,
                          wp_vector,
                          vehicles,
                          pedestrians,
                          traffic_lights,
                          timestamp):
        speed_factor = 1
        speed_factor_tl = 1
        speed_factor_p = 1
        speed_factor_v = 1

        for obs_vehicle_loc in vehicles:
            if self._map.are_on_same_lane(vehicle_transform.location,
                                          obs_vehicle_loc):
                self._logger.info('Ego {} and vehicle {} are on the same lane'.format(
                    vehicle_transform.location, obs_vehicle_loc))
                new_speed_factor_v = pylot.control.utils.stop_vehicle(
                    vehicle_transform, obs_vehicle_loc, wp_vector,
                    speed_factor_v, self._flags)
                if new_speed_factor_v < speed_factor_v:
                    speed_factor_v = new_speed_factor_v
                    self._logger.info('Vehicle {} reduced speed factor to {}'.format(
                        obs_vehicle_loc, speed_factor_v))

        for obs_ped_loc in pedestrians:
            if self._map.are_on_same_lane(vehicle_transform.location,
                                          obs_ped_loc):
                self._logger.info('Ego {} and pedestrian {} are on the same lane'.format(
                    vehicle_transform.location, obs_ped_loc))
                new_speed_factor_p = pylot.control.utils.stop_pedestrian(
                    vehicle_transform,
                    obs_ped_loc,
                    wp_vector,
                    speed_factor_p,
                    self._flags)
                if new_speed_factor_p < speed_factor_p:
                    speed_factor_p = new_speed_factor_p
                    self._logger.info('Pedestrian {} reduced speed factor to {}'.format(
                        obs_ped_loc, speed_factor_p))

        for tl in traffic_lights:
            if self._map.must_obbey_traffic_light(vehicle_transform.location,
                                                  tl[0]):
                self._logger.info('Ego is obbeying traffic light {}'.format(
                    vehicle_transform.location, tl[0]))
                tl_state = tl[1]
                new_speed_factor_tl = pylot.control.utils.stop_traffic_light(
                    vehicle_transform,
                    tl[0],
                    tl_state,
                    wp_vector,
                    wp_angle,
                    speed_factor_tl,
                    self._flags)
                if new_speed_factor_tl < speed_factor_tl:
                    speed_factor_tl = new_speed_factor_tl
                    self._logger.info('Traffic light {} reduced speed factor to {}'.format(
                        tl[0], speed_factor_tl))

        speed_factor = min(speed_factor_tl, speed_factor_p, speed_factor_v)
        state = {
            'stop_pedestrian': speed_factor_p,
            'stop_vehicle': speed_factor_v,
            'stop_traffic_lights': speed_factor_tl
        }
        self._logger.info('{}: Agent speed factors {}'.format(
            timestamp, state))
        return speed_factor, state

    def __get_steer(self, wp_angle):
        steer = self._flags.steer_gain * wp_angle
        if steer > 0:
            steer = min(steer, 1)
        else:
            steer = max(steer, -1)
        return steer

    def __get_throttle_brake_without_factor(self, current_speed, target_speed):
        self._pid.target = target_speed
        pid_gain = self._pid(feedback=current_speed)
        throttle = min(max(self._flags.default_throttle - 1.3 * pid_gain, 0),
                       self._flags.throttle_max)
        if pid_gain > 0.5:
            brake = min(0.35 * pid_gain * self._flags.brake_strength, 1)
        else:
            brake = 0
        return throttle, brake

    def __get_throttle_brake(self,
                             current_speed,
                             target_speed,
                             wp_angle_speed,
                             speed_factor):
        # TODO(ionel): DO NOT HARDCODE VALUES!
        # Don't go to fast around corners
        if math.fabs(wp_angle_speed) < 0.1:
            target_speed_adjusted = target_speed * speed_factor
        elif math.fabs(wp_angle_speed) < 0.5:
            target_speed_adjusted = 6 * speed_factor
        else:
            target_speed_adjusted = 3 * speed_factor
        self._pid.target = target_speed_adjusted
        pid_gain = self._pid(feedback=current_speed)
        throttle = min(
            max(self._flags.default_throttle - 1.3 * pid_gain, 0),
            self._flags.throttle_max)

        if pid_gain > 0.5:
            brake = min(0.35 * pid_gain * self._flags.brake_strength, 1)
        else:
            brake = 0
        return throttle, brake

    def get_control_message(self, wp_angle, wp_angle_speed, speed_factor,
                            current_speed, target_speed, timestamp):
        current_speed = max(current_speed, 0)
        steer = self.__get_steer(wp_angle)
        throttle, brake = self.__get_throttle_brake(
            current_speed, target_speed, wp_angle_speed, speed_factor)
        return ControlMessage(steer, throttle, brake, False, False, timestamp)
