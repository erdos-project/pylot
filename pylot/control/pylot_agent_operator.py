import carla
from collections import deque
import erdos
import math
from pid_controller.pid import PID
import time

# Pylot imports
from pylot.control.messages import ControlMessage
import pylot.control.utils
import pylot.simulation.utils
from pylot.map.hd_map import HDMap
from pylot.simulation.carla_utils import get_map
from pylot.utils import time_epoch_ms

INTERSECTION_SPEED_M_PER_SEC = 5


class PylotAgentOperator(erdos.Operator):
    def __init__(self,
                 can_bus_stream,
                 waypoints_stream,
                 traffic_lights_stream,
                 obstacles_stream,
                 lidar_stream,
                 open_drive_stream,
                 depth_camera_stream,
                 control_stream,
                 name,
                 flags,
                 bgr_camera_setup,
                 log_file_name=None,
                 csv_file_name=None):
        can_bus_stream.add_callback(self.on_can_bus_update)
        waypoints_stream.add_callback(self.on_waypoints_update)
        traffic_lights_stream.add_callback(self.on_traffic_lights_update)
        obstacles_stream.add_callback(self.on_obstacles_update)
        lidar_stream.add_callback(self.on_lidar_update)
        open_drive_stream.add_callback(self.on_opendrive_map)
        depth_camera_stream.add_callback(self.on_depth_camera_update)
        erdos.add_watermark_callback([
            can_bus_stream, waypoints_stream, traffic_lights_stream,
            obstacles_stream
        ], [control_stream], self.on_watermark)
        self._name = name
        self._flags = flags
        self._log_file_name = log_file_name
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            name + '-csv', csv_file_name)
        self._bgr_camera_setup = bgr_camera_setup
        self._map = None
        if not hasattr(self._flags, 'track'):
            self._map = HDMap(
                get_map(self._flags.carla_host, self._flags.carla_port,
                        self._flags.carla_timeout), log_file_name)
            self._logger.debug('Agent running using map')
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
        self._last_traffic_light_game_time = -100000
        self._last_moving_time = 0
        # Num of control commands to override to ensure the agent doesn't get
        # stuck.
        self._num_control_override = 0

    @staticmethod
    def connect(can_bus_stream, waypoints_stream, traffic_lights_stream,
                obstacles_stream, lidar_stream, open_drive_stream,
                depth_camera_stream):
        control_stream = erdos.WriteStream()
        return [control_stream]

    def compute_command(self, can_bus_msg, waypoint_msg, tl_msg, obstacles_msg,
                        pc_msg, depth_msg, timestamp):
        start_time = time.time()
        vehicle_transform = can_bus_msg.data.transform
        # Vehicle sped in m/s
        vehicle_speed = can_bus_msg.data.forward_speed
        wp_angle = waypoint_msg.wp_angle
        wp_vector = waypoint_msg.wp_vector
        wp_angle_speed = waypoint_msg.wp_angle_speed
        target_speed = waypoint_msg.target_speed
        # Transform point cloud to camera coordinates.
        point_cloud = None
        if pc_msg is not None:
            point_cloud = pc_msg.point_cloud
        depth_frame = None
        if depth_msg is not None:
            depth_frame = depth_msg.frame
            # We need to transform the static setup of the camera relative to
            # the position of the vehicle.
            depth_frame.camera_setup.set_transform(
                vehicle_transform * depth_frame.camera_setup.transform)

        traffic_lights = self.__transform_tl_output(tl_msg, point_cloud,
                                                    depth_frame)
        assert len(timestamp.coordinates) == 1
        game_time = timestamp.coordinates[0]
        if len(traffic_lights) > 0:
            self._last_traffic_light_game_time = game_time
        (pedestrians,
         vehicles) = self.__transform_detector_output(obstacles_msg,
                                                      point_cloud, depth_frame)

        self._logger.debug('@{}: speed {} and location {}'.format(
            timestamp, vehicle_speed, vehicle_transform))
        self._logger.debug('@{}: pedestrians {}'.format(
            timestamp, pedestrians))
        self._logger.debug('@{}: vehicles {}'.format(timestamp, vehicles))

        speed_factor, _ = self.__stop_for_agents(vehicle_transform.location,
                                                 wp_angle, wp_vector, vehicles,
                                                 pedestrians, traffic_lights,
                                                 timestamp)

        control_msg = self.get_control_message(wp_angle, wp_angle_speed,
                                               speed_factor, vehicle_speed,
                                               target_speed, timestamp)

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
            control_msg = ControlMessage(0, 0.75, 0, False, False, timestamp)

        # Get runtime in ms.
        runtime = (time.time() - start_time) * 1000
        self._csv_logger.info('{},{},"{}",{}'.format(time_epoch_ms(),
                                                     self._name, timestamp,
                                                     runtime))

        return control_msg

    def on_watermark(self, timestamp, control_stream):
        self._logger.debug('@{}: received watermark'.format(timestamp))
        can_bus_msg = self._can_bus_msgs.popleft()
        waypoint_msg = self._waypoint_msgs.popleft()
        tl_msg = self._traffic_lights_msgs.popleft()
        obstacles_msg = self._obstacles_msgs.popleft()
        pc_msg = None
        if len(self._point_clouds) == 0 and len(self._depth_camera_msgs) == 0:
            self._logger.fatal('No point clouds or depth frame msgs available')
            return
        elif len(self._point_clouds) > 0:
            pc_msg = self._point_clouds.popleft()
        depth_msg = None
        if len(self._depth_camera_msgs) > 0:
            depth_msg = self._depth_camera_msgs.popleft()

        control_command_msg = self.compute_command(can_bus_msg, waypoint_msg,
                                                   tl_msg, obstacles_msg,
                                                   pc_msg, depth_msg,
                                                   timestamp)
        control_stream.send(control_command_msg)

    def on_waypoints_update(self, msg):
        self._logger.debug('@{}: waypoints update'.format(msg.timestamp))
        self._waypoint_msgs.append(msg)

    def on_can_bus_update(self, msg):
        self._logger.debug('@{}: can bus update'.format(msg.timestamp))
        self._can_bus_msgs.append(msg)

    def on_traffic_lights_update(self, msg):
        self._logger.debug('@{}: traffic lights update'.format(msg.timestamp))
        self._traffic_lights_msgs.append(msg)

    def on_obstacles_update(self, msg):
        self._logger.debug('@{}: obstacles update'.format(msg.timestamp))
        self._obstacles_msgs.append(msg)

    def on_lidar_update(self, msg):
        self._logger.debug('@{}: lidar update'.format(msg.timestamp))
        self._point_clouds.append(msg)

    def on_opendrive_map(self, msg):
        self._logger.debug('@{}: open drive update'.format(msg.timestamp))
        self._map = HDMap(carla.Map('challenge', msg.data),
                          self._log_file_name)

    def on_depth_camera_update(self, msg):
        self._logger.debug('@{}: depth camera update'.format(msg.timestamp))
        self._depth_camera_msgs.append(msg)

    def __transform_to_3d(self, pixel, point_cloud, depth_frame):
        """ Transforms a camera view pixel location to 3d world location.

        Args:
            x: The x-axis pixel.
            y: The y-axis pixel.
            point_cloud: A lidar point cloud.
            depth_frame: A pylot.perception.depth_frame.DepthFrame.

        Note: It is sufficient to pass either a point cloud or a depth frame.

        Returns:
            The location in 3D world coordinates.
        """
        location = None
        if depth_frame is not None:
            location = depth_frame.get_pixel_locations([pixel])[0]
        elif point_cloud is not None:
            location = point_cloud.get_pixel_location(pixel,
                                                      depth_frame.camera_setup)
        if location is None:
            self._logger.error(
                'Could not find lidar point for {}'.format(pixel))
        return location

    def __transform_tl_output(self, tls, point_cloud, depth_frame):
        """ Transforms traffic light bounding boxes to world coordinates.

        Args:
            tls: A list of traffic light detected obstacles.
            point_cloud: The Lidar point cloud. Must be taken captured at the
                         same time as the frame on which the traffic lights
                         were detected.
            depth_frame: A pylot.perception.depth_frame.DepthFrame recorded at
                the same time as the RGB frame used in detection.

        Returns:
            A list of traffic light locations.
        """
        traffic_lights = []
        for tl in tls.obstacles:
            location = self.__transform_to_3d(tl.get_center_point(),
                                              point_cloud, depth_frame)
            traffic_lights.append((location, tl.label))
        return traffic_lights

    def __transform_detector_output(self, obstacles_msg, point_cloud,
                                    depth_frame):
        """ Transforms detected obstacles to world coordinates.

        Args:
            obstacles_msg: A list of detected obstacles.
            point_cloud: The Lidar point cloud. Must be taken captured at the
                         same time as the frame on which the obstacles were
                         detected.
            depth_frame: A pylot.perception.depth_frame.DepthFrame recorded at
                the same time as the RGB frame used in detection.

        Returns:
            A list of 3D world locations.
        """
        vehicles = []
        pedestrians = []
        for obstacle in obstacles_msg.obstacles:
            if obstacle.label == 'person':
                location = self.__transform_to_3d(obstacle.get_center_point(),
                                                  point_cloud, depth_frame)
                pedestrians.append(location)
            elif (obstacle.label in self._vehicle_labels):
                location = self.__transform_to_3d(obstacle.get_center_point(),
                                                  point_cloud, depth_frame)
                vehicles.append(location)
        return (pedestrians, vehicles)

    def __stop_for_agents(self, ego_vehicle_location, wp_angle, wp_vector,
                          vehicles, pedestrians, traffic_lights, timestamp):
        speed_factor = 1
        speed_factor_tl = 1
        speed_factor_p = 1
        speed_factor_v = 1

        for obs_vehicle_loc in vehicles:
            if (not self._map or self._map.are_on_same_lane(
                    ego_vehicle_location, obs_vehicle_loc)):
                self._logger.debug(
                    '@{}: ego {} and vehicle {} are on the same lane'.format(
                        timestamp, ego_vehicle_location, obs_vehicle_loc))
                new_speed_factor_v = pylot.control.utils.stop_vehicle(
                    ego_vehicle_location, obs_vehicle_loc, wp_vector,
                    speed_factor_v, self._flags)
                if new_speed_factor_v < speed_factor_v:
                    speed_factor_v = new_speed_factor_v
                    self._logger.debug(
                        '@{}: vehicle {} reduced speed factor to {}'.format(
                            timestamp, obs_vehicle_loc, speed_factor_v))

        for obs_ped_loc in pedestrians:
            if (not self._map or self._map.are_on_same_lane(
                    ego_vehicle_location, obs_ped_loc)):
                self._logger.debug(
                    '@{}: ego {} and pedestrian {} are on the same lane'.
                    format(timestamp, ego_vehicle_location, obs_ped_loc))
                new_speed_factor_p = pylot.control.utils.stop_pedestrian(
                    ego_vehicle_location, obs_ped_loc, wp_vector,
                    speed_factor_p, self._flags)
                if new_speed_factor_p < speed_factor_p:
                    speed_factor_p = new_speed_factor_p
                    self._logger.debug(
                        '@{}: pedestrian {} reduced speed factor to {}'.format(
                            timestamp, obs_ped_loc, speed_factor_p))

        for tl in traffic_lights:
            if (not self._map or self._map.must_obbey_traffic_light(
                    ego_vehicle_location, tl[0])):
                self._logger.debug(
                    '@{}: ego is obbeying traffic light {}'.format(
                        timestamp, ego_vehicle_location, tl[0]))
                tl_state = tl[1]
                new_speed_factor_tl = pylot.control.utils.stop_traffic_light(
                    ego_vehicle_location, tl[0], tl_state, wp_vector, wp_angle,
                    speed_factor_tl, self._flags)
                if new_speed_factor_tl < speed_factor_tl:
                    speed_factor_tl = new_speed_factor_tl
                    self._logger.debug(
                        '@{}: traffic light {} reduced speed factor to {}'.
                        format(timestamp, tl[0], speed_factor_tl))

        speed_factor = min(speed_factor_tl, speed_factor_p, speed_factor_v)
        state = {
            'stop_pedestrian': speed_factor_p,
            'stop_vehicle': speed_factor_v,
            'stop_traffic_lights': speed_factor_tl
        }
        self._logger.debug('@{}: agent speed factors {}'.format(
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

    def __get_throttle_brake(self, current_speed, target_speed, wp_angle_speed,
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
        throttle = min(max(self._flags.default_throttle - 1.3 * pid_gain, 0),
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
        throttle, brake = self.__get_throttle_brake(current_speed,
                                                    target_speed,
                                                    wp_angle_speed,
                                                    speed_factor)
        return ControlMessage(steer, throttle, brake, False, False, timestamp)
