import logging
import math

from absl import flags

import carla

import erdos

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, \
    Track

import numpy as np

import pylot.flags
import pylot.operator_creator
import pylot.perception.messages
import pylot.utils
from pylot.drivers.sensor_setup import LidarSetup, RGBCameraSetup
from pylot.perception.camera_frame import CameraFrame
from pylot.perception.point_cloud import PointCloud

FLAGS = flags.FLAGS

CENTER_CAMERA_LOCATION = pylot.utils.Location(1.5, 0.0, 1.4)
CENTER_CAMERA_NAME = 'center_camera'
TL_CAMERA_NAME = 'traffic_lights_camera'
LEFT_CAMERA_NAME = 'left_camera'
RIGHT_CAMERA_NAME = 'right_camera'


def get_entry_point():
    return 'ERDOSAgent'


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
        # Setup the pygame window.
        if FLAGS.visualizer_backend == 'pygame':
            import pygame
            pygame.init()
            pylot.utils.create_pygame_display(FLAGS.carla_camera_image_width,
                                              FLAGS.carla_camera_image_height)
        self._logger = erdos.utils.setup_logging('erdos_agent',
                                                 FLAGS.log_file_name)
        enable_logging()
        self.track = get_track()
        self._camera_setups = create_camera_setups()
        # Set the lidar in the same position as the center camera.
        self._lidar_transform = pylot.utils.Transform(CENTER_CAMERA_LOCATION,
                                                      pylot.utils.Rotation())
        self._lidar_setup = LidarSetup('lidar', 'sensor.lidar.ray_cast',
                                       self._lidar_transform)
        # Stores the waypoints we get from the challenge planner.
        self._waypoints = None
        # Stores the open drive string we get when we run in track 3.
        self._open_drive_data = None
        (camera_streams, pose_stream, global_trajectory_stream,
         open_drive_stream, point_cloud_stream,
         control_stream) = create_data_flow()
        self._camera_streams = camera_streams
        self._pose_stream = pose_stream
        self._global_trajectory_stream = global_trajectory_stream
        self._open_drive_stream = open_drive_stream
        self._sent_open_drive = False
        self._point_cloud_stream = point_cloud_stream
        self._control_stream = control_stream
        # Execute the data-flow.
        erdos.run_async()

    def destroy(self):
        """Clean-up the agent. Invoked between different runs."""
        self._logger.info('ERDOSAgent destroy method invoked')

    def sensors(self):
        """
        Defines the sensor suite required by the agent.
        """
        opendrive_map_sensors = []
        if self.track == Track.MAP:
            opendrive_map_sensors = [{
                'type': 'sensor.opendrive_map',
                'reading_frequency': 10,
                'id': 'opendrive'
            }]

        gnss_sensors = [{
            'type': 'sensor.other.gnss',
            'x': 1.5,
            'y': 0.0,
            'z': 1.4,
            'reading_frequency': 10,
            'id': 'gnss'
        }]

        imu_sensors = [{
            'type': 'sensor.other.imu',
            'x': 1.5,
            'y': 0.0,
            'z': 1.40,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'reading_frequency': 10,
            'id': 'imu'
        }]

        speed_sensors = [{
            'type': 'sensor.speedometer',
            'reading_frequency': 10,
            'id': 'speed'
        }]

        lidar_sensors = [{
            'type': 'sensor.lidar.ray_cast',
            'x': self._lidar_transform.location.x,
            'y': self._lidar_transform.location.y,
            'z': self._lidar_transform.location.z,
            'roll': self._lidar_transform.rotation.roll,
            'pitch': self._lidar_transform.rotation.pitch,
            'yaw': self._lidar_transform.rotation.yaw,
            'reading_frequency': 10,
            'id': 'LIDAR'
        }]

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
                'reading_frequency': 10,
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
        for key, val in input_data.items():
            # val is a tuple of (data timestamp, data).
            print("Sensor {} at {}".format(key, val[0]))
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
                self.send_lidar_msg(val[1], self._lidar_transform,
                                    erdos_timestamp)
            else:
                self._logger.warning("Sensor {} not used".format(key))

        self.send_pose_msg(speed_data, imu_data, gnss_data, erdos_timestamp)

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
        # TODO(ionel): Remove the patch that gives us the perfect transform.
        vehicle_transform = pylot.utils.Transform.from_carla_transform(
            speed_data['transform'])
        # latitude = gnss_data[0]
        # longitude = gnss_data[1]
        # altitude = gnss_data[2]
        # location = pylot.utils.Location.from_gps(latitude, longitude, altitude)
        # vehicle_transform = pylot.utils.Transform(
        #     location, pylot.utils.Rotation(yaw=-90))
        #        carla_north_vector = np.array([0.0, -1.0, 0.0])
        # compass = imu_data[6]
        # fwd_y = -math.cos(compass)
        yaw = vehicle_transform.rotation.yaw
        velocity_vector = pylot.utils.Vector3D(forward_speed * np.cos(yaw),
                                               forward_speed * np.sin(yaw), 0)
        self._pose_stream.send(
            erdos.Message(
                timestamp,
                pylot.utils.Pose(vehicle_transform, forward_speed,
                                 velocity_vector)))
        self._pose_stream.send(erdos.WatermarkMessage(timestamp))

    def send_lidar_msg(self, carla_pc, transform, timestamp):
        msg = pylot.perception.messages.PointCloudMessage(
            timestamp, PointCloud(carla_pc, self._lidar_setup))
        self._point_cloud_stream.send(msg)
        self._point_cloud_stream.send(erdos.WatermarkMessage(timestamp))

    def send_waypoints_msg(self, timestamp):
        # Send once the global waypoints.
        if self._waypoints is None:
            # Gets global waypoints from the agent.
            self._waypoints = self._global_plan_world_coord
            data = [(pylot.utils.Transform.from_carla_transform(transform),
                     road_option)
                    for (transform, road_option) in self._waypoints]
            self._global_trajectory_stream.send(erdos.Message(timestamp, data))
            self._global_trajectory_stream.send(
                erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))


def get_track():
    track = None
    if FLAGS.track == 1:
        track = Track.SENSORS
    elif FLAGS.track == 3:
        track = Track.MAP
    else:
        raise ValueError('Unexpected track {}'.format(FLAGS.track))
    return track


def create_data_flow():
    """ Create the challenge data-flow graph."""
    time_to_decision_loop_stream = erdos.LoopStream()
    camera_setups = create_camera_setups()
    camera_streams = {}
    for name in camera_setups:
        camera_streams[name] = erdos.IngestStream()
    pose_stream = erdos.IngestStream()
    global_trajectory_stream = erdos.IngestStream()
    open_drive_stream = erdos.IngestStream()
    point_cloud_stream = erdos.IngestStream()

    obstacles_stream = pylot.operator_creator.add_obstacle_detection(
        camera_streams[CENTER_CAMERA_NAME], time_to_decision_loop_stream)[0]
    # Adds an operator that finds the world locations of the obstacles.
    obstacles_stream = pylot.operator_creator.add_obstacle_location_finder(
        obstacles_stream, point_cloud_stream, pose_stream,
        camera_setups[CENTER_CAMERA_NAME])

    traffic_lights_stream = pylot.operator_creator.add_traffic_light_detector(
        camera_streams[TL_CAMERA_NAME])
    # Adds an operator that finds the world locations of the traffic lights.
    traffic_lights_stream = \
        pylot.operator_creator.add_obstacle_location_finder(
            traffic_lights_stream, point_cloud_stream, pose_stream,
            camera_setups[TL_CAMERA_NAME])

    waypoints_stream = pylot.operator_creator.add_waypoint_planning(
        pose_stream, open_drive_stream, global_trajectory_stream,
        obstacles_stream, traffic_lights_stream, None)

    if FLAGS.visualize_rgb_camera:
        pylot.operator_creator.add_camera_visualizer(
            camera_streams[CENTER_CAMERA_NAME], CENTER_CAMERA_NAME)

    control_stream = pylot.operator_creator.add_pid_agent(
        pose_stream, waypoints_stream)
    extract_control_stream = erdos.ExtractStream(control_stream)

    time_to_decision_stream = pylot.operator_creator.add_time_to_decision(
        pose_stream, obstacles_stream)
    time_to_decision_loop_stream.set(time_to_decision_stream)

    return (camera_streams, pose_stream, global_trajectory_stream,
            open_drive_stream, point_cloud_stream, extract_control_stream)


def create_camera_setups():
    camera_setups = {}
    transform = pylot.utils.Transform(CENTER_CAMERA_LOCATION,
                                      pylot.utils.Rotation())
    center_camera_setup = RGBCameraSetup(CENTER_CAMERA_NAME,
                                         FLAGS.carla_camera_image_width,
                                         FLAGS.carla_camera_image_height,
                                         transform, 90)
    camera_setups[CENTER_CAMERA_NAME] = center_camera_setup
    tl_camera_setup = RGBCameraSetup(TL_CAMERA_NAME,
                                     FLAGS.carla_camera_image_width,
                                     FLAGS.carla_camera_image_height,
                                     transform, 45)
    camera_setups[TL_CAMERA_NAME] = tl_camera_setup
    return camera_setups


def enable_logging():
    """Overwrites logging config so that loggers can control verbosity.

    This method is required because the challenge evaluator overwrites
    verbosity, which causes Pylot log messages to be discarded.
    """
    import logging
    logging.root.setLevel(logging.NOTSET)
