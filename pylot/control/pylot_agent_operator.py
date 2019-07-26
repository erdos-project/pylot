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
import pylot.simulation.utils
from pylot.simulation.utils import get_3d_world_position_with_point_cloud,\
    get_3d_world_position_with_depth_map
import pylot.utils


class PylotAgentOperator(Op):
    def __init__(self,
                 name,
                 flags,
                 bgr_camera_setup,
                 log_file_name=None,
                 csv_file_name=None):
        super(PylotAgentOperator, self).__init__(name)
        self._flags = flags
        self._log_file_name = log_file_name
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._bgr_camera_setup = bgr_camera_setup
        self._map = None
        if '0.9' in self._flags.carla_version:
            from pylot.map.hd_map import HDMap
            from pylot.simulation.carla_utils import get_map
            if not hasattr(self._flags, 'track'):
                self._map = HDMap(get_map(self._flags.carla_host,
                                          self._flags.carla_port,
                                          self._flags.carla_timeout),
                                  log_file_name)
                self._logger.info('Agent running using map')
        elif hasattr(self._flags, 'track'):
            from pylot.map.hd_map import HDMap
        self._pid = PID(p=self._flags.pid_p,
                        i=self._flags.pid_i,
                        d=self._flags.pid_d)
        self._waypoint_msgs = deque()
        self._can_bus_msgs = deque()
        self._traffic_lights_msgs = deque()
        self._obstacles_msgs = deque()
        self._point_clouds = deque()
        self._depth_camera_msgs = deque()
        self._vehicle_labels = {'car', 'bicycle', 'motorcycle', 'bus', 'truck'}
        self._lock = threading.Lock()
        self._last_traffic_light_game_time = -100000
        self._last_moving_time = 0
        # Num of control commands to override to ensure the agent doesn't get
        # stuck.
        self._num_control_override = 0

    @staticmethod
    def setup_streams(input_streams):
        input_streams.filter(
            pylot.utils.is_can_bus_stream).add_callback(
                PylotAgentOperator.on_can_bus_update)
        input_streams.filter(
            pylot.utils.is_waypoints_stream).add_callback(
                PylotAgentOperator.on_waypoints_update)
        input_streams.filter(
            pylot.utils.is_traffic_lights_stream).add_callback(
                PylotAgentOperator.on_traffic_lights_update)
        input_streams.filter(
            pylot.utils.is_obstacles_stream).add_callback(
                PylotAgentOperator.on_obstacles_update)
        input_streams.filter(
            pylot.utils.is_lidar_stream).add_callback(
                PylotAgentOperator.on_lidar_update)
        input_streams.filter(
            pylot.utils.is_open_drive_stream).add_callback(
                PylotAgentOperator.on_opendrive_map)
        input_streams.filter(
            pylot.utils.is_depth_camera_stream).add_callback(
                PylotAgentOperator.on_depth_camera_update)
        input_streams.filter(
            pylot.utils.is_segmented_camera_stream).add_callback(
                PylotAgentOperator.on_segmented_frame)
        input_streams.filter(
            pylot.utils.is_detected_lane_stream).add_callback(
                PylotAgentOperator.on_detected_lane_update)

        # Set no watermark on the output stream so that we do not
        # close the watermark loop with the carla operator.
        return [pylot.utils.create_control_stream()]

    def compute_command(self,
                        can_bus_msg,
                        waypoint_msg,
                        tl_msg,
                        obstacles_msg,
                        pc_msg,
                        depth_msg,
                        timestamp):
        start_time = time.time()
        vehicle_transform = can_bus_msg.data.transform
        vehicle_speed = can_bus_msg.data.forward_speed
        wp_angle = waypoint_msg.wp_angle
        wp_vector = waypoint_msg.wp_vector
        wp_angle_speed = waypoint_msg.wp_angle_speed
        target_speed = waypoint_msg.target_speed
        # Transform point cloud to camera coordinates.
        point_cloud = None
        if pc_msg:
            point_cloud = pylot.simulation.utils.lidar_point_cloud_to_camera_coordinates(
                pc_msg.point_cloud)
        depth_frame = None
        if depth_msg:
            depth_frame = depth_msg.frame

        traffic_lights = self.__transform_tl_output(
            tl_msg, vehicle_transform, point_cloud, depth_frame)
        game_time = timestamp.coordinates[0]
        if len(traffic_lights) > 0:
            self._last_traffic_light_game_time = game_time
        (pedestrians, vehicles) = self.__transform_detector_output(
            obstacles_msg, vehicle_transform, point_cloud, depth_frame)

        # if self._map.is_on_opposite_lane(vehicle_transform):
        #     # Ignore obstacles
        #     self._logger.info('Ego-vehicle {} on opposite lange'.format(
        #         vehicle_transform))
        #     pedestrians = []
        #     vehicles = []
        #     traffic_lights = []

        self._logger.info('{} Current speed {} and location {}'.format(
            timestamp, vehicle_speed, vehicle_transform))
        self._logger.info('{} Pedestrians {}'.format(
            timestamp, pedestrians))
        self._logger.info('{} Vehicles {}'.format(
            timestamp, vehicles))

        speed_factor, _ = self.__stop_for_agents(
            vehicle_transform,
            wp_angle,
            wp_vector,
            vehicles,
            pedestrians,
            traffic_lights,
            timestamp)

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
            timestamp, speed_factor))

        control_msg = self.get_control_message(
            wp_angle, wp_angle_speed, speed_factor,
            vehicle_speed, target_speed, timestamp)

        if control_msg.throttle > 0.001:
            self._last_moving_time = game_time
            self._num_control_override = 0

        if self._num_control_override > 0:
            self._num_control_override -= 1
            control_msg.throttle = 0.75

        # Might be stuck because of a faulty detector.
        # Override control message if we haven't been moving for a while.
        if game_time - self._last_moving_time > 30000:
            self._num_control_override = 6
            control_msg = ControlMessage(
                0, 0.75, 0, False, False, timestamp)

        # Get runtime in ms.
        runtime = (time.time() - start_time) * 1000
        self._csv_logger.info('{},{},"{}",{}'.format(
            time_epoch_ms(), self.name, timestamp, runtime))

        self.get_output_stream('control_stream').send(control_msg)

    def synchronize_msg_buffers(self, timestamp, buffers):
        for buffer in buffers:
            while (len(buffer) > 0 and buffer[0].timestamp < timestamp):
                buffer.popleft()
            if len(buffer) == 0:
                return False
            assert buffer[0].timestamp == timestamp
        return True

    def run_if_you_can(self):
        streams = [self._can_bus_msgs, self._waypoint_msgs,
                   self._traffic_lights_msgs, self._obstacles_msgs]
        for stream in streams:
            if len(stream) == 0:
                return
        can_bus_msg = self._can_bus_msgs.popleft()
        waypoint_msg = self._waypoint_msgs.popleft()
        tl_msg = self._traffic_lights_msgs.popleft()
        obstacles_msg = self._obstacles_msgs.popleft()

        self._logger.info('Timestamps {} {} {} {}'.format(
            can_bus_msg.timestamp, waypoint_msg.timestamp,
            tl_msg.timestamp, obstacles_msg.timestamp))
        assert (can_bus_msg.timestamp == waypoint_msg.timestamp ==
                tl_msg.timestamp == obstacles_msg.timestamp)

        if len(self._point_clouds) == 0 and len(self._depth_camera_msgs) == 0:
            # No point clouds or depth frame msgs available.
            return
        pc_msg = None
        if len(self._point_clouds) > 0:
            pc_msg = self._point_clouds.popleft()
        depth_msg = None
        if len(self._depth_camera_msgs) > 0:
            depth_msg = self._depth_camera_msgs.popleft()

        self.compute_command(can_bus_msg,
                             waypoint_msg,
                             tl_msg,
                             obstacles_msg,
                             pc_msg,
                             depth_msg,
                             can_bus_msg.timestamp)

    def on_waypoints_update(self, msg):
        self._logger.info('Waypoints update at {}'.format(msg.timestamp))
        with self._lock:
            self._waypoint_msgs.append(msg)
            self.run_if_you_can()

    def on_can_bus_update(self, msg):
        self._logger.info('Can bus update at {}'.format(msg.timestamp))
        with self._lock:
            self._can_bus_msgs.append(msg)
            self.run_if_you_can()

    def on_traffic_lights_update(self, msg):
        self._logger.info('Traffic light update at {}'.format(msg.timestamp))
        with self._lock:
            self._traffic_lights_msgs.append(msg)
            self.run_if_you_can()

    def on_obstacles_update(self, msg):
        self._logger.info('Obstacle update at {}'.format(msg.timestamp))
        with self._lock:
            self._obstacles_msgs.append(msg)
            self.run_if_you_can()

    def on_lidar_update(self, msg):
        self._logger.info('Lidar update at {}'.format(msg.timestamp))
        with self._lock:
            self._point_clouds.append(msg)
            self.run_if_you_can()

    def on_opendrive_map(self, msg):
        self._map = HDMap(carla.Map('challenge', msg.data),
                          self._log_file_name)

    def on_depth_camera_update(self, msg):
        self._logger.info('Depth camera frame at {}'.format(msg.timestamp))
        with self._lock:
            self._depth_camera_msgs.append(msg)
            self.run_if_you_can()

    def on_segmented_frame(self, msg):
        self._logger.info('Received segmented frame at {}'.format(
            msg.timestamp))
        # TODO(ionel): Implement!

    def on_detected_lane_update(self, msg):
        # TODO(ionel): Implement!
        pass

    def execute(self):
        self.spin()

    def reduce_speed_when_approaching_intersection(
            self,
            vehicle_transform,
            vehicle_speed,
            target_speed,
            game_time):
        if not self._map:
            return target_speed
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

    def __transform_to_3d(
            self, x, y, vehicle_transform, point_cloud, depth_frame):
        pos = None
        if depth_frame:
            pos = get_3d_world_position_with_depth_map(
                x, y, depth_frame, self._bgr_camera_setup.width,
                self._bgr_camera_setup.height, self._bgr_camera_setup.fov,
                vehicle_transform * self._bgr_camera_setup.transform)
        elif point_cloud is not None:
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

    def __transform_tl_output(
            self, tls, vehicle_transform, point_cloud, depth_frame):
        traffic_lights = []
        for tl in tls.detected_objects:
            x = (tl.corners[0] + tl.corners[1]) / 2
            y = (tl.corners[2] + tl.corners[3]) / 2
            pos = self.__transform_to_3d(
                x, y, vehicle_transform, point_cloud, depth_frame)
            if pos:
                traffic_lights.append((pos, tl.label))
        return traffic_lights

    def __transform_detector_output(
            self, obstacles_msg, vehicle_transform, point_cloud, depth_frame):
        vehicles = []
        pedestrians = []
        for detected_obj in obstacles_msg.detected_objects:
            x = (detected_obj.corners[0] + detected_obj.corners[1]) / 2
            y = (detected_obj.corners[2] + detected_obj.corners[3]) / 2
            if detected_obj.label == 'person':
                pos = self.__transform_to_3d(
                    x, y, vehicle_transform, point_cloud, depth_frame)
                if pos:
                    pedestrians.append(pos)
            elif (detected_obj.label in self._vehicle_labels):
                pos = self.__transform_to_3d(
                    x, y, vehicle_transform, point_cloud, depth_frame)
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
            if (not self._map or
                self._map.are_on_same_lane(vehicle_transform.location,
                                           obs_vehicle_loc)):
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
            if (not self._map or
                self._map.are_on_same_lane(vehicle_transform.location,
                                           obs_ped_loc)):
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
            if (not self._map or
                self._map.must_obbey_traffic_light(vehicle_transform.location,
                                                   tl[0])):
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
