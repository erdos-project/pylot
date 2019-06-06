from collections import deque
import math
import numpy as np
from pid_controller.pid import PID
import threading

from erdos.op import Op
from erdos.utils import setup_csv_logging, setup_logging

from pylot.control.messages import ControlMessage
import pylot.control.utils
import pylot.simulation.utils
from pylot.simulation.utils import get_3d_world_position_with_point_cloud
import pylot.utils


class LidarERDOSAgentOperator(Op):
    def __init__(self,
                 name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        super(LidarERDOSAgentOperator, self).__init__(name)
        self._flags = flags
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._pid = PID(p=self._flags.pid_p,
                        i=self._flags.pid_i,
                        d=self._flags.pid_d)
        self._waypoint_msgs = deque()
        self._can_bus_msgs = deque()
        self._traffic_lights = deque()
        self._obstacles = deque()
        self._point_clouds = deque()
        # TODO(ionel): DANGEROUS! DO NOT HARDCODE!
        loc = pylot.simulation.utils.Location(2.0, 0.0, 1.40)
        rot = pylot.simulation.utils.Rotation(0, 0, 0)
        self._camera_transform = pylot.simulation.utils.Transform(loc, rot)
        self._camera_width = 800
        self._camera_height = 600
        self._camera_fov = 100
        self._lock = threading.Lock()

    @staticmethod
    def setup_streams(input_streams):
        input_streams.filter(pylot.utils.is_can_bus_stream).add_callback(
            LidarERDOSAgentOperator.on_can_bus_update)
        input_streams.filter(pylot.utils.is_waypoints_stream).add_callback(
            LidarERDOSAgentOperator.on_waypoints_update)
        input_streams.filter(pylot.utils.is_traffic_lights_stream).add_callback(
            LidarERDOSAgentOperator.on_traffic_lights_update)
        input_streams.filter(pylot.utils.is_obstacles_stream).add_callback(
            LidarERDOSAgentOperator.on_obstacles_update)
        input_streams.filter(pylot.utils.is_lidar_stream).add_callback(
            LidarERDOSAgentOperator.on_lidar_update)

        input_streams.add_completion_callback(
            LidarERDOSAgentOperator.on_notification)

        # Set no watermark on the output stream so that we do not
        # close the watermark loop with the carla operator.
        return [pylot.utils.create_control_stream()]

    def on_notification(self, msg):
        can_bus_msg = None
        waypoint_msg = None
        pc_msg = None
        tl_output = None
        obstacles = None

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
        #point_cloud = self.__point_cloud_to_world_coordinates(pc_msg)
        point_cloud = pc_msg.point_cloud.tolist()

        traffic_lights = self.__transform_tl_output(tl_output, point_cloud)
        (pedestrians, vehicles) = self.__transform_detector_output(
            obstacles, point_cloud)

        self._logger.info('Current location {}'.format(vehicle_transform))
        self._logger.info('Pedestrians {}'.format(pedestrians))
        self._logger.info('Vehicles {}'.format(vehicles))

        speed_factor, _ = self.__stop_for_agents(
            vehicle_transform,
            wp_angle,
            wp_vector,
            vehicles,
            pedestrians,
            traffic_lights)

        self._logger.info('Current speed factor {}'.format(speed_factor))

        control_msg = self.get_control_message(
            wp_angle, wp_angle_speed, speed_factor,
            vehicle_speed, target_speed, msg.timestamp)
        self.get_output_stream('control_stream').send(control_msg)

    def __point_cloud_to_world_coordinates(self, point_cloud_msg):
        transform = pylot.simulation.utils.lidar_to_unreal_transform(
            point_cloud_msg.transform)
        return transform.transform_points(point_cloud_msg.point_cloud).tolist()

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

    def execute(self):
        self.spin()

    def __transform_to_3d(self, x, y, point_cloud):
        pos = get_3d_world_position_with_point_cloud(x,
                                                     y,
                                                     point_cloud,
                                                     self._camera_transform,
                                                     self._camera_width,
                                                     self._camera_height,
                                                     self._camera_fov)
        if pos is None:
            self._logger.error(
                'Could not find lidar point for {} {}'.format(x, y))
        return pos

    def __transform_tl_output(self, tls, point_cloud):
        traffic_lights = []
        for tl in tls.detected_objects:
            x = (tl.corners[0] + tl.corners[1]) / 2
            y = (tl.corners[2] + tl.corners[3]) / 2
            pos = self.__transform_to_3d(x, y, point_cloud)
            if pos:
                state = 0
                if tl.label is not 'Green':
                    state = 1
                traffic_lights.append((pos, state))
        return traffic_lights

    def __transform_detector_output(self, obstacles, point_cloud):
        vehicles = []
        pedestrians = []
        for detected_obj in obstacles.detected_objects:
            x = (detected_obj.corners[0] + detected_obj.corners[1]) / 2
            y = (detected_obj.corners[2] + detected_obj.corners[3]) / 2
            if detected_obj.label == 'person':
                pos = self.__transform_to_3d(x, y, point_cloud)
                if pos:
                    pedestrians.append(pos)
            elif (detected_obj.label == 'car' or
                  detected_obj.label == 'bicycle' or
                  detected_obj.label == 'motorcycle' or
                  detected_obj.label == 'bus' or
                  detected_obj.label == 'truck'):
                pos = self.__transform_to_3d(x, y, point_cloud)
                if pos:
                    vehicles.append(pos)
        return (pedestrians, vehicles)

    def __stop_for_agents(self,
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

        for obs_vehicle_pos in vehicles:
            if pylot.control.utils.is_vehicle_on_same_lane(
                    vehicle_transform, obs_vehicle_pos):
                new_speed_factor_v = pylot.control.utils.stop_vehicle(
                    vehicle_transform, obs_vehicle_pos, wp_vector,
                    speed_factor_v, self._flags)
                speed_factor_v = min(speed_factor_v, new_speed_factor_v)

        for obs_ped_pos in pedestrians:
            if pylot.control.utils.is_pedestrian_hitable(obs_ped_pos):
                new_speed_factor_p = pylot.control.utils.stop_pedestrian(
                    vehicle_transform,
                    obs_ped_pos,
                    wp_vector,
                    speed_factor_p,
                    self._flags)
                speed_factor_p = min(speed_factor_p, new_speed_factor_p)

        for tl in traffic_lights:
            if (pylot.control.utils.is_traffic_light_active(
                    vehicle_transform, tl[0]) and
                pylot.control.utils.is_traffic_light_visible(
                    vehicle_transform, tl[0], self._flags)):
                tl_state = tl[1]
                new_speed_factor_tl = pylot.control.utils.stop_traffic_light(
                    vehicle_transform,
                    tl[0],
                    tl_state,
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
        self._logger.info('Aggent speed factors {}'.format(state))
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
