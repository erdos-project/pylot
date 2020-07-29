import logging
from collections import deque

from absl import flags

import carla

import erdos

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, \
    Track

import numpy as np

import pylot.flags
import pylot.component_creator  # noqa: I100
import pylot.operator_creator
import pylot.perception.messages
import pylot.simulation.utils
import pylot.utils
from pylot.drivers.sensor_setup import LidarSetup, RGBCameraSetup
from pylot.localization.messages import GNSSMessage, IMUMessage
from pylot.perception.camera_frame import CameraFrame
from pylot.perception.messages import ObstaclesMessage, TrafficLightsMessage
from pylot.perception.point_cloud import PointCloud
from pylot.planning.messages import WaypointsMessage
from pylot.planning.waypoints import Waypoints

FLAGS = flags.FLAGS

flags.DEFINE_bool('carla_localization', False,
                  'True to use perfect localization')


# Certain visualizations are not supported when running in challenge mode.
def unsupported_visualizations_validator(flags_dict):
    return not (flags_dict['visualize_depth_camera']
                or flags_dict['visualize_imu'] or flags_dict['visualize_pose']
                or flags_dict['visualize_prediction'])


flags.register_multi_flags_validator(
    [
        'visualize_depth_camera', 'visualize_imu', 'visualize_pose',
        'visualize_prediction'
    ],
    unsupported_visualizations_validator,
    message='Trying to visualize unsupported_visualization')

CENTER_CAMERA_LOCATION = pylot.utils.Location(0.0, 0.0, 2.0)
CENTER_CAMERA_NAME = 'center_camera'
LANE_CAMERA_LOCATION = pylot.utils.Location(1.3, 0.0, 1.8)
LANE_CAMERA_NAME = 'lane_camera'
TL_CAMERA_NAME = 'traffic_lights_camera'
LEFT_CAMERA_NAME = 'left_camera'
RIGHT_CAMERA_NAME = 'right_camera'


def get_entry_point():
    return 'ERDOSAgent'


def using_perfect_component():
    """Returns True if the agent uses any perfect component."""
    return (FLAGS.carla_obstacle_detection
            or FLAGS.carla_traffic_light_detection
            or FLAGS.perfect_obstacle_tracking or FLAGS.carla_localization)


def using_lidar():
    """Returns True if Lidar is required for the setup."""
    return not (FLAGS.carla_obstacle_detection
                and FLAGS.carla_traffic_light_detection)


class ERDOSAgent(AutonomousAgent):
    """Agent class that interacts with the CARLA challenge scenario runner.

    Attributes:
        track: Track the agent is running in.
        _camera_setups: Mapping between camera names and
            :py:class:`~pylot.drivers.sensor_setup.CameraSetup`.
        _lidar_transform (:py:class:`~pylot.utils.Transform`): Transform of
            the Lidar relative to the ego vehicle.
        _waypoints (list(:py:class:`~pylot.utils.Transform`)): List of
            waypoints the agent receives from the challenge planner.
    """
    def setup(self, path_to_conf_file):
        """Setup phase. Invoked by the scenario runner."""
        # Disable Tensorflow logging.
        pylot.utils.set_tf_loglevel(logging.ERROR)
        # Parse the flag file.
        flags.FLAGS([__file__, '--flagfile={}'.format(path_to_conf_file)])
        self._logger = erdos.utils.setup_logging('erdos_agent',
                                                 FLAGS.log_file_name)
        enable_logging()
        self.track = get_track()
        self._camera_setups = create_camera_setups()
        # Set the lidar in the same position as the center camera.
        self._lidar_transform = pylot.utils.Transform(CENTER_CAMERA_LOCATION,
                                                      pylot.utils.Rotation())
        self._lidar_setup = LidarSetup('lidar',
                                       'sensor.lidar.ray_cast',
                                       self._lidar_transform,
                                       range=8500)
        self._last_point_cloud = None
        self._last_yaw = 0
        # Stores the waypoints we get from the challenge planner.
        self._waypoints = None
        # Stores the open drive string we get when we run in track 3.
        self._open_drive_data = None
        (camera_streams, pose_stream, route_stream, global_trajectory_stream,
         open_drive_stream, point_cloud_stream, imu_stream, gnss_stream,
         control_stream, control_display_stream, ground_obstacles_stream,
         ground_traffic_lights_stream, vehicle_id_stream,
         streams_to_send_top_on) = create_data_flow()
        self._camera_streams = camera_streams
        self._pose_stream = pose_stream
        self._route_stream = route_stream
        self._sent_initial_pose = False
        self._global_trajectory_stream = global_trajectory_stream
        self._open_drive_stream = open_drive_stream
        self._sent_open_drive = False
        self._point_cloud_stream = point_cloud_stream
        self._imu_stream = imu_stream
        self._gnss_stream = gnss_stream
        self._control_stream = control_stream
        self._control_display_stream = control_display_stream
        self._ground_obstacles_stream = ground_obstacles_stream
        self._ground_traffic_lights_stream = ground_traffic_lights_stream
        self._vehicle_id_stream = vehicle_id_stream
        self._ego_vehicle = None
        self._sent_vehicle_id = False
        # Execute the dataflow.
        self._node_handle = erdos.run_async()
        for stream in streams_to_send_top_on:
            stream.send(erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))
        if using_perfect_component():
            # The agent is using a perfect component. It must directly connect
            # to the simulator to send perfect data to the data-flow.
            _, self._world = pylot.simulation.utils.get_world(
                FLAGS.carla_host, FLAGS.carla_port, FLAGS.carla_timeout)

    def destroy(self):
        """Clean-up the agent. Invoked between different runs."""
        self._logger.info('ERDOSAgent destroy method invoked')
        self._node_handle.shutdown()
        erdos.reset()

    def sensors(self):
        """
        Defines the sensor suite required by the agent.
        """
        opendrive_map_sensors = []
        if self.track == Track.MAP:
            opendrive_map_sensors = [{
                'type': 'sensor.opendrive_map',
                'reading_frequency': 20,
                'id': 'opendrive'
            }]

        gnss_sensors = [{
            'type': 'sensor.other.gnss',
            'x': 0.0,
            'y': 0.0,
            'z': 0.0,
            'id': 'gnss'
        }]

        imu_sensors = [{
            'type': 'sensor.other.imu',
            'x': 0.0,
            'y': 0.0,
            'z': 0.0,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'id': 'imu'
        }]

        speed_sensors = [{
            'type': 'sensor.speedometer',
            'reading_frequency': 20,
            'id': 'speed'
        }]

        if using_lidar():
            lidar_sensors = [{
                'type': 'sensor.lidar.ray_cast',
                'x': self._lidar_transform.location.x,
                'y': self._lidar_transform.location.y,
                'z': self._lidar_transform.location.z,
                'roll': self._lidar_transform.rotation.roll,
                'pitch': self._lidar_transform.rotation.pitch,
                'yaw': self._lidar_transform.rotation.yaw,
                'id': 'LIDAR'
            }]
        else:
            lidar_sensors = []

        camera_sensors = []
        for cs in self._camera_setups.values():
            camera_sensor = {
                'type': cs.camera_type,
                'x': cs.transform.location.x,
                'y': cs.transform.location.y,
                'z': cs.transform.location.z,
                'roll': cs.transform.rotation.roll,
                'pitch': cs.transform.rotation.pitch,
                'yaw': cs.transform.rotation.yaw,
                'width': cs.width,
                'height': cs.height,
                'fov': cs.fov,
                'id': cs.name
            }
            camera_sensors.append(camera_sensor)

        return (gnss_sensors + speed_sensors + imu_sensors +
                opendrive_map_sensors + camera_sensors + lidar_sensors)

    def run_step(self, input_data, timestamp):
        game_time = int(timestamp * 1000)
        self._logger.debug("Current game time {}".format(game_time))
        erdos_timestamp = erdos.Timestamp(coordinates=[game_time])

        if not self._sent_open_drive and self.track != Track.MAP:
            # We do not have access to the open drive map. Send top watermark.
            self._sent_open_drive = True
            self._open_drive_stream.send(
                erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))
        # Send the waypoints.
        self.send_waypoints_msg(erdos_timestamp)

        speed_data = None
        imu_data = None
        gnss_data = None
        carla_pc = None
        for key, val in input_data.items():
            # val is a tuple of (data timestamp, data).
            # print("Sensor {} at {}".format(key, val[0]))
            if key in self._camera_streams:
                self._camera_streams[key].send(
                    pylot.perception.messages.FrameMessage(
                        erdos_timestamp,
                        CameraFrame(val[1][:, :, :3], 'BGR',
                                    self._camera_setups[key])))
                self._camera_streams[key].send(
                    erdos.WatermarkMessage(erdos_timestamp))
            elif key == 'imu':
                imu_data = val[1]
            elif key == 'speed':
                speed_data = val[1]
            elif key == 'gnss':
                gnss_data = val[1]
            elif key == 'opendrive':
                self.send_opendrive_map_msg(val[1], erdos_timestamp)
            elif key == 'LIDAR':
                carla_pc = val[1]
            else:
                self._logger.warning("Sensor {} not used".format(key))

        self.send_vehicle_id_msg()
        self.send_ground_data(erdos_timestamp)

        if FLAGS.localization:
            self.send_imu_msg(imu_data, erdos_timestamp)
            self.send_gnss_msg(gnss_data, erdos_timestamp)
            self.send_initial_pose_msg(erdos_timestamp)
        elif FLAGS.carla_localization:
            self.send_perfect_pose_msg(erdos_timestamp)
        else:
            self.send_pose_msg(speed_data, imu_data, gnss_data,
                               erdos_timestamp)

        if using_lidar():
            self.send_lidar_msg(carla_pc, self._lidar_transform,
                                erdos_timestamp)

        if pylot.flags.must_visualize():
            import pygame
            from pygame.locals import K_n
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.KEYUP:
                    if event.key == K_n:
                        self._control_display_stream.send(
                            erdos.Message(erdos.Timestamp(coordinates=[0]),
                                          event.key))

        # Wait until the control is set.
        while True:
            control_msg = self._control_stream.read()
            if not isinstance(control_msg, erdos.WatermarkMessage):
                output_control = carla.VehicleControl()
                output_control.throttle = control_msg.throttle
                output_control.brake = control_msg.brake
                output_control.steer = control_msg.steer
                output_control.reverse = control_msg.reverse
                output_control.hand_brake = control_msg.hand_brake
                output_control.manual_gear_shift = False
                return output_control

    def send_gnss_msg(self, gnss_data, timestamp):
        latitude = gnss_data[0]
        longitude = gnss_data[1]
        altitude = gnss_data[2]
        location = pylot.utils.Location.from_gps(latitude, longitude, altitude)
        transform = pylot.utils.Transform(location, pylot.utils.Rotation())
        msg = GNSSMessage(timestamp, transform, altitude, latitude, longitude)
        self._gnss_stream.send(msg)
        self._gnss_stream.send(erdos.WatermarkMessage(timestamp))

    def send_imu_msg(self, imu_data, timestamp):
        accelerometer = pylot.utils.Vector3D(imu_data[0], imu_data[1],
                                             imu_data[2])
        gyroscope = pylot.utils.Vector3D(imu_data[3], imu_data[4], imu_data[5])
        compass = imu_data[6]
        msg = IMUMessage(timestamp, None, accelerometer, gyroscope, compass)
        self._imu_stream.send(msg)
        self._imu_stream.send(erdos.WatermarkMessage(timestamp))

    def send_opendrive_map_msg(self, data, timestamp):
        # Sending once opendrive data
        if self._open_drive_data is None:
            self._open_drive_data = data['opendrive']
            self._open_drive_stream.send(
                erdos.Message(timestamp, self._open_drive_data))
            self._open_drive_stream.send(
                erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))
        else:
            self._logger.warning(
                'Agent did not sent open drive data for {}'.format(timestamp))

    def send_pose_msg(self, speed_data, imu_data, gnss_data, timestamp):
        forward_speed = speed_data['speed']
        # vehicle_transform = pylot.utils.Transform.from_carla_transform(
        #     speed_data['transform'])
        latitude = gnss_data[0]
        longitude = gnss_data[1]
        altitude = gnss_data[2]
        location = pylot.utils.Location.from_gps(latitude, longitude, altitude)
        if np.isnan(imu_data[6]):
            yaw = self._last_yaw
        else:
            compass = np.degrees(imu_data[6])
            if compass < 270:
                yaw = compass - 90
            else:
                yaw = compass - 450
            self._last_yaw = yaw
        vehicle_transform = pylot.utils.Transform(
            location, pylot.utils.Rotation(yaw=yaw))
        velocity_vector = pylot.utils.Vector3D(forward_speed * np.cos(yaw),
                                               forward_speed * np.sin(yaw), 0)
        current_pose = pylot.utils.Pose(vehicle_transform, forward_speed,
                                        velocity_vector,
                                        timestamp.coordinates[0])
        self._logger.debug("@{}: Predicted pose: {}".format(
            timestamp, current_pose))
        self._pose_stream.send(erdos.Message(timestamp, current_pose))
        self._pose_stream.send(erdos.WatermarkMessage(timestamp))

    def send_initial_pose_msg(self, timestamp):
        if not self._sent_initial_pose:
            self._sent_initial_pose = True
            initial_transform = self._global_plan_world_coord[0][0]
            initial_pose = pylot.utils.Pose(
                pylot.utils.Transform.from_carla_transform(initial_transform),
                0, pylot.utils.Vector3D(), timestamp.coordinates[0])
            self._route_stream.send(erdos.Message(timestamp, initial_pose))
            self._route_stream.send(
                erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))

    def send_lidar_msg(self,
                       carla_pc,
                       transform,
                       timestamp,
                       ego_transform=None):
        # Remove the intensity component of the point cloud.
        # carla_pc = carla_pc[:, [1, 0, 2]]
        point_cloud = PointCloud(carla_pc, self._lidar_setup)
        if self._last_point_cloud is not None:
            # TODO(ionel): Should offset the last point cloud wrt to the
            # current location.
            # self._last_point_cloud.global_points = \
            #     ego_transform.transform_points(
            #         self._last_point_cloud.global_points)
            # self._last_point_cloud.points = \
            #     self._last_point_cloud._to_camera_coordinates(
            #         self._last_point_cloud.global_points)
            point_cloud.merge(self._last_point_cloud)
        self._point_cloud_stream.send(
            pylot.perception.messages.PointCloudMessage(
                timestamp, point_cloud))
        self._point_cloud_stream.send(erdos.WatermarkMessage(timestamp))
        # global_pc = ego_transform.inverse_transform_points(carla_pc)
        self._last_point_cloud = PointCloud(carla_pc, self._lidar_setup)

    def send_waypoints_msg(self, timestamp):
        # Send once the global waypoints.
        if self._waypoints is None:
            # Gets global waypoints from the agent.
            waypoints = deque([])
            road_options = deque([])
            for (transform, road_option) in self._global_plan_world_coord:
                waypoints.append(
                    pylot.utils.Transform.from_carla_transform(transform))
                road_options.append(pylot.utils.RoadOption(road_option.value))
            self._waypoints = Waypoints(waypoints, road_options=road_options)
            self._global_trajectory_stream.send(
                WaypointsMessage(timestamp, self._waypoints))
            self._global_trajectory_stream.send(
                erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))

    def send_vehicle_id_msg(self):
        if ((FLAGS.carla_localization or FLAGS.perfect_obstacle_tracking)
                and not self._sent_vehicle_id):
            actor_list = self._world.get_actors()
            vec_actors = actor_list.filter('vehicle.*')
            for actor in vec_actors:
                if ('role_name' in actor.attributes
                        and actor.attributes['role_name'] == 'hero'):
                    self._ego_vehicle = actor
                    break
            if self._ego_vehicle:
                self._sent_vehicle_id = True
                self._vehicle_id_stream.send(
                    erdos.Message(erdos.Timestamp(coordinates=[0]),
                                  self._ego_vehicle.id))
                self._vehicle_id_stream.send(
                    erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))

    def send_perfect_pose_msg(self, timestamp):
        vec_transform = pylot.utils.Transform.from_carla_transform(
            self._ego_vehicle.get_transform())
        velocity_vector = pylot.utils.Vector3D.from_carla_vector(
            self._ego_vehicle.get_velocity())
        forward_speed = velocity_vector.magnitude()
        pose = pylot.utils.Pose(vec_transform, forward_speed, velocity_vector,
                                timestamp.coordinates[0])
        self._pose_stream.send(erdos.Message(timestamp, pose))
        self._pose_stream.send(erdos.WatermarkMessage(timestamp))

    def send_ground_data(self, timestamp):
        if not (FLAGS.carla_obstacle_detection
                or FLAGS.carla_traffic_light_detection):
            return
        actor_list = self._world.get_actors()
        (vehicles, people, traffic_lights, _,
         _) = pylot.simulation.utils.extract_data_in_pylot_format(actor_list)
        if FLAGS.carla_obstacle_detection:
            self._ground_obstacles_stream.send(
                ObstaclesMessage(timestamp, vehicles + people))
            self._ground_obstacles_stream.send(
                erdos.WatermarkMessage(timestamp))
        if FLAGS.carla_traffic_light_detection:
            self._ground_traffic_lights_stream.send(
                TrafficLightsMessage(timestamp, traffic_lights))
            self._ground_traffic_lights_stream.send(
                erdos.WatermarkMessage(timestamp))


def get_track():
    track = None
    if FLAGS.execution_mode == 'challenge-sensors':
        track = Track.SENSORS
    elif FLAGS.execution_mode == 'challenge-map':
        track = Track.MAP
    else:
        raise ValueError('Unexpected track {}'.format(FLAGS.execution_mode))
    return track


def create_data_flow():
    """ Create the challenge data-flow graph."""
    streams_to_send_top_on = []
    time_to_decision_loop_stream = erdos.LoopStream()
    camera_setups = create_camera_setups()
    camera_streams = {}
    for name in camera_setups:
        camera_streams[name] = erdos.IngestStream()
    global_trajectory_stream = erdos.IngestStream()
    open_drive_stream = erdos.IngestStream()
    point_cloud_stream = erdos.IngestStream()
    imu_stream = erdos.IngestStream()
    gnss_stream = erdos.IngestStream()
    route_stream = erdos.IngestStream()

    if FLAGS.localization:
        pose_stream = pylot.operator_creator.add_localization(
            imu_stream, gnss_stream, route_stream)
    else:
        pose_stream = erdos.IngestStream()

    ground_obstacles_stream = erdos.IngestStream()
    ground_traffic_lights_stream = erdos.IngestStream()
    vehicle_id_stream = erdos.IngestStream()
    if not (FLAGS.perfect_obstacle_tracking or FLAGS.carla_localization):
        streams_to_send_top_on.append(vehicle_id_stream)

    if FLAGS.carla_obstacle_detection:
        obstacles_stream = ground_obstacles_stream
    elif any('efficientdet' in model
             for model in FLAGS.obstacle_detection_model_names):
        obstacles_stream = pylot.operator_creator.\
            add_efficientdet_obstacle_detection(
                camera_streams[CENTER_CAMERA_NAME],
                time_to_decision_loop_stream)[0]
        streams_to_send_top_on.append(ground_obstacles_stream)
    else:
        obstacles_stream = pylot.operator_creator.add_obstacle_detection(
            camera_streams[CENTER_CAMERA_NAME],
            time_to_decision_loop_stream)[0]
        streams_to_send_top_on.append(ground_obstacles_stream)

    if FLAGS.carla_traffic_light_detection:
        traffic_lights_stream = ground_traffic_lights_stream
        camera_streams[TL_CAMERA_NAME] = erdos.IngestStream()
        streams_to_send_top_on.append(camera_streams[TL_CAMERA_NAME])
    else:
        traffic_lights_stream = \
            pylot.operator_creator.add_traffic_light_detector(
                camera_streams[TL_CAMERA_NAME])
        # Adds an operator that finds the world location of the traffic lights.
        traffic_lights_stream = \
            pylot.operator_creator.add_obstacle_location_finder(
                traffic_lights_stream, point_cloud_stream, pose_stream,
                camera_setups[TL_CAMERA_NAME])
        streams_to_send_top_on.append(ground_traffic_lights_stream)

    obstacles_tracking_stream = pylot.component_creator.add_obstacle_tracking(
        camera_streams[CENTER_CAMERA_NAME],
        camera_setups[CENTER_CAMERA_NAME],
        obstacles_stream,
        depth_stream=point_cloud_stream,
        vehicle_id_stream=vehicle_id_stream,
        pose_stream=pose_stream,
        ground_obstacles_stream=ground_obstacles_stream,
        time_to_decision_stream=time_to_decision_loop_stream)

    if FLAGS.execution_mode == 'challenge-sensors':
        lanes_stream = pylot.operator_creator.add_lanenet_detection(
            camera_streams[LANE_CAMERA_NAME])
    else:
        lanes_stream = erdos.IngestStream()
        streams_to_send_top_on.append(lanes_stream)

    prediction_stream = pylot.operator_creator.add_linear_prediction(
        obstacles_tracking_stream)

    waypoints_stream = pylot.component_creator.add_planning(
        None, pose_stream, prediction_stream, traffic_lights_stream,
        lanes_stream, open_drive_stream, global_trajectory_stream,
        time_to_decision_loop_stream)

    if pylot.flags.must_visualize():
        control_display_stream, ingest_streams = \
            pylot.operator_creator.add_visualizer(
                pose_stream=pose_stream,
                camera_stream=camera_streams[CENTER_CAMERA_NAME],
                tl_camera_stream=camera_streams[TL_CAMERA_NAME],
                point_cloud_stream=point_cloud_stream,
                obstacles_stream=obstacles_stream,
                traffic_lights_stream=traffic_lights_stream,
                tracked_obstacles_stream=obstacles_tracking_stream,
                waypoints_stream=waypoints_stream,
                lane_detection_stream=lanes_stream,
                prediction_stream=prediction_stream)
        streams_to_send_top_on += ingest_streams
    else:
        control_display_stream = None

    control_stream = pylot.operator_creator.add_pid_control(
        pose_stream, waypoints_stream)
    extract_control_stream = erdos.ExtractStream(control_stream)

    time_to_decision_stream = pylot.operator_creator.add_time_to_decision(
        pose_stream, obstacles_stream)
    time_to_decision_loop_stream.set(time_to_decision_stream)

    return (camera_streams, pose_stream, route_stream,
            global_trajectory_stream, open_drive_stream, point_cloud_stream,
            imu_stream, gnss_stream, extract_control_stream,
            control_display_stream, ground_obstacles_stream,
            ground_traffic_lights_stream, vehicle_id_stream,
            streams_to_send_top_on)


def create_camera_setups():
    camera_setups = {}
    transform = pylot.utils.Transform(CENTER_CAMERA_LOCATION,
                                      pylot.utils.Rotation())
    center_camera_setup = RGBCameraSetup(CENTER_CAMERA_NAME,
                                         FLAGS.camera_image_width,
                                         FLAGS.camera_image_height, transform,
                                         90)
    camera_setups[CENTER_CAMERA_NAME] = center_camera_setup
    if not FLAGS.carla_traffic_light_detection:
        tl_camera_setup = RGBCameraSetup(TL_CAMERA_NAME,
                                         FLAGS.camera_image_width,
                                         FLAGS.camera_image_height, transform,
                                         45)
        camera_setups[TL_CAMERA_NAME] = tl_camera_setup
    if FLAGS.execution_mode == 'challenge-sensors':
        # Add camera for lane detection.
        lane_transform = pylot.utils.Transform(LANE_CAMERA_LOCATION,
                                               pylot.utils.Rotation(pitch=-15))
        lane_camera_setup = RGBCameraSetup(LANE_CAMERA_NAME, 1280, 720,
                                           lane_transform, 90)
        camera_setups[LANE_CAMERA_NAME] = lane_camera_setup

    # left_transform = pylot.utils.Transform(CENTER_CAMERA_LOCATION,
    #                                        pylot.utils.Rotation(yaw=-45))
    # left_camera_setup = RGBCameraSetup(LEFT_CAMERA_NAME,
    #                                    FLAGS.camera_image_width,
    #                                    FLAGS.camera_image_height,
    #                                    left_transform, 90)
    # camera_setups[LEFT_CAMERA_NAME] = left_camera_setup
    # right_transform = pylot.utils.Transform(CENTER_CAMERA_LOCATION,
    #                                         pylot.utils.Rotation(yaw=45))
    # right_camera_setup = RGBCameraSetup(RIGHT_CAMERA_NAME,
    #                                     FLAGS.camera_image_width,
    #                                     FLAGS.camera_image_height,
    #                                     right_transform, 90)
    # camera_setups[RIGHT_CAMERA_NAME] = right_camera_setup
    return camera_setups


def enable_logging():
    """Overwrites logging config so that loggers can control verbosity.

    This method is required because the challenge evaluator overwrites
    verbosity, which causes Pylot log messages to be discarded.
    """
    import logging
    logging.root.setLevel(logging.NOTSET)
