import logging
from collections import deque

from absl import flags

from carla import VehicleControl

import erdos

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, \
    Track

import numpy as np

import pylot.utils
from pylot.localization.messages import GNSSMessage, IMUMessage
from pylot.perception.messages import ObstaclesMessage, TrafficLightsMessage
from pylot.perception.point_cloud import PointCloud
from pylot.planning.messages import WaypointsMessage
from pylot.planning.waypoints import Waypoints

FLAGS = flags.FLAGS


class ERDOSBaseAgent(AutonomousAgent):
    """ERDOSBaseAgent class.

    Attributes:
        track: Track the agent is running in.
        logger: A handle to a logger.
    """
    def setup(self, path_to_conf_file):
        """Setup phase. Invoked by the scenario runner."""
        # Disable Tensorflow logging.
        pylot.utils.set_tf_loglevel(logging.ERROR)
        # Parse the flag file. Users can use the different flags defined
        # across the Pylot directory.
        flags.FLAGS([__file__, '--flagfile={}'.format(path_to_conf_file)])
        self.logger = erdos.utils.setup_logging('erdos_agent',
                                                FLAGS.log_file_name)
        self.csv_logger = erdos.utils.setup_csv_logging(
            'erdos_agent_csv', FLAGS.csv_log_file_name)
        enable_logging()
        self.track = get_track()
        # Town name is only used when the agent is directly receiving
        # traffic lights from the simulator.
        self._town_name = None
        # Stores a simulator handle to the ego vehicle. This handle is only
        # used when the agent is using a perfect localization or perception.
        self._ego_vehicle = None
        # Stores ego-vehicle's yaw from last game time. This is used in the
        # naive localization solution.
        self._last_yaw = 0
        # Stores the point cloud from the previous sensor reading.
        self._last_point_cloud = None
        if using_perfect_component():
            from pylot.simulation.utils import get_world
            # The agent is using a perfect component. It must directly connect
            # to the simulator to send perfect data to the data-flow.
            _, self._world = get_world(FLAGS.simulator_host,
                                       FLAGS.simulator_port,
                                       FLAGS.simulator_timeout)

    def sensors(self):
        """Defines the sensor suite required by the agent."""
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
                'x': self._lidar_setup.transform.location.x,
                'y': self._lidar_setup.transform.location.y,
                'z': self._lidar_setup.transform.location.z,
                'roll': self._lidar_setup.transform.rotation.roll,
                'pitch': self._lidar_setup.transform.rotation.pitch,
                'yaw': self._lidar_setup.transform.rotation.yaw,
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

    def compute_pose(self, speed_data, imu_data, gnss_data, timestamp):
        """Computes the pose of the ego-vehicle.

        This method implements a naive localization that transforms the
        noisy gnss readings into locations, and the noisy IMU readings into
        rotations.
        """
        forward_speed = speed_data['speed']
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
        return current_pose

    def send_perfect_detections(self, perfect_obstacles_stream,
                                perfect_traffic_lights_stream, timestamp,
                                tl_camera_location):
        """Send perfect detections for agents and traffic lights.

        This method first connects to the simulator to extract all the
        agents and traffic light in a scenario. Next, it transforms them into
        the types Pylot expects, and sends them on the streams for perfect
        detections.

        Note: This is only used when executing using a perfect perception
        component.
        """
        if not (FLAGS.simulator_obstacle_detection
                or FLAGS.simulator_traffic_light_detection
                or FLAGS.evaluate_obstacle_detection
                or FLAGS.evaluate_obstacle_tracking):
            return
        from pylot.simulation.utils import extract_data_in_pylot_format
        actor_list = self._world.get_actors()
        (vehicles, people, traffic_lights, _,
         _) = extract_data_in_pylot_format(actor_list)
        if (FLAGS.simulator_obstacle_detection
                or FLAGS.evaluate_obstacle_detection
                or FLAGS.evaluate_obstacle_tracking):
            perfect_obstacles_stream.send(
                ObstaclesMessage(timestamp, vehicles + people))
            perfect_obstacles_stream.send(erdos.WatermarkMessage(timestamp))
        if FLAGS.simulator_traffic_light_detection:
            vec_transform = pylot.utils.Transform.from_simulator_transform(
                self._ego_vehicle.get_transform())
            tl_camera_transform = pylot.utils.Transform(
                tl_camera_location, pylot.utils.Rotation())
            visible_tls = []
            if self._town_name is None:
                self._town_name = self._world.get_map().name
            for tl in traffic_lights:
                if tl.is_traffic_light_visible(
                        vec_transform * tl_camera_transform,
                        self._town_name,
                        distance_threshold=FLAGS.
                        static_obstacle_distance_threshold):
                    if self._town_name not in ['Town01', 'Town02']:
                        delta_y = -5
                        if self._town_name == 'Town04':
                            delta_y = -2
                        # Move the traffic light location to the road.
                        tl.transform = tl.transform * pylot.utils.Transform(
                            pylot.utils.Location(delta_y, 0, 5),
                            pylot.utils.Rotation())
                    visible_tls.append(tl)
            perfect_traffic_lights_stream.send(
                TrafficLightsMessage(timestamp, visible_tls))
            perfect_traffic_lights_stream.send(
                erdos.WatermarkMessage(timestamp))

    def send_global_trajectory_msg(self, global_trajectory_stream, timestamp):
        """Sends the route the agent must follow."""
        # Send once the global waypoints.
        if not global_trajectory_stream.is_closed():
            # Gets global waypoints from the agent.
            waypoints = deque([])
            road_options = deque([])
            for (transform, road_option) in self._global_plan_world_coord:
                waypoints.append(
                    pylot.utils.Transform.from_simulator_transform(transform))
                road_options.append(pylot.utils.RoadOption(road_option.value))
            waypoints = Waypoints(waypoints, road_options=road_options)
            global_trajectory_stream.send(
                WaypointsMessage(timestamp, waypoints))
            global_trajectory_stream.send(
                erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))

    def send_gnss_msg(self, gnss_stream, gnss_data, timestamp):
        """Sends the GNSS data on the Pylot stream."""
        latitude = gnss_data[0]
        longitude = gnss_data[1]
        altitude = gnss_data[2]
        location = pylot.utils.Location.from_gps(latitude, longitude, altitude)
        transform = pylot.utils.Transform(location, pylot.utils.Rotation())
        # Build a Pylot GNSSMessage out of the challenge GNSS data.
        msg = GNSSMessage(timestamp, transform, altitude, latitude, longitude)
        gnss_stream.send(msg)
        gnss_stream.send(erdos.WatermarkMessage(timestamp))

    def send_imu_msg(self, imu_stream, imu_data, timestamp):
        """Sends the IMU data on the Pylot stream."""
        accelerometer = pylot.utils.Vector3D(imu_data[0], imu_data[1],
                                             imu_data[2])
        gyroscope = pylot.utils.Vector3D(imu_data[3], imu_data[4], imu_data[5])
        compass = imu_data[6]
        # Build a Pylot IMUMessage out of the challenge IMU sensor data.
        msg = IMUMessage(timestamp, None, accelerometer, gyroscope, compass)
        imu_stream.send(msg)
        imu_stream.send(erdos.WatermarkMessage(timestamp))

    def send_lidar_msg(self,
                       point_cloud_stream,
                       simulator_pc,
                       timestamp,
                       lidar_setup,
                       ego_transform=None):
        """Transforms and sends a point cloud reading.

        This method is transforms a point cloud from the challenge format
        to the type Pylot uses, and it sends it on the point cloud stream.
        """
        # Remove the intensity component of the point cloud.
        simulator_pc = simulator_pc[:, :3]
        point_cloud = PointCloud(simulator_pc, lidar_setup)
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
        point_cloud_stream.send(
            pylot.perception.messages.PointCloudMessage(
                timestamp, point_cloud))
        point_cloud_stream.send(erdos.WatermarkMessage(timestamp))
        # global_pc = ego_transform.inverse_transform_points(simulator_pc)
        self._last_point_cloud = PointCloud(simulator_pc, lidar_setup)

    def send_perfect_pose_msg(self, pose_stream, timestamp):
        """Sends the perfectly accurate location of the ego-vehicle.

        The perfect ego-vehicle pose is directly fetched from the simulator.
        This method is only used when the agent is running with perfect
        localization. It is meant to be used mostly for debugging
        and testing.
        """
        vec_transform = pylot.utils.Transform.from_simulator_transform(
            self._ego_vehicle.get_transform())
        velocity_vector = pylot.utils.Vector3D.from_simulator_vector(
            self._ego_vehicle.get_velocity())
        forward_speed = velocity_vector.magnitude()
        pose = pylot.utils.Pose(vec_transform, forward_speed, velocity_vector,
                                timestamp.coordinates[0])
        pose_stream.send(erdos.Message(timestamp, pose))
        pose_stream.send(erdos.WatermarkMessage(timestamp))

    def send_vehicle_id_msg(self, vehicle_id_stream):
        """Sends the simulator actor id of the ego-vehicle.

        This method is only used when the agent is running with perfect
        localization or with perfect obstacle trajectory tracking.
        """
        if ((FLAGS.perfect_localization or FLAGS.perfect_obstacle_tracking
             or FLAGS.simulator_traffic_light_detection)
                and not self._ego_vehicle):
            actor_list = self._world.get_actors()
            vec_actors = actor_list.filter('vehicle.*')
            for actor in vec_actors:
                if ('role_name' in actor.attributes
                        and actor.attributes['role_name'] == 'hero'):
                    self._ego_vehicle = actor
                    break
            if not vehicle_id_stream.is_closed():
                vehicle_id_stream.send(
                    erdos.Message(erdos.Timestamp(coordinates=[0]),
                                  self._ego_vehicle.id))
                vehicle_id_stream.send(
                    erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))


def get_track():
    track = None
    if FLAGS.execution_mode == 'challenge-sensors':
        track = Track.SENSORS
    elif FLAGS.execution_mode == 'challenge-map':
        track = Track.MAP
    else:
        raise ValueError('Unexpected track {}'.format(FLAGS.execution_mode))
    return track


def enable_logging():
    """Overwrites logging config so that loggers can control verbosity.

    This method is required because the challenge evaluator overwrites
    verbosity, which causes Pylot log messages to be discarded.
    """
    import logging
    logging.root.setLevel(logging.NOTSET)


def process_visualization_events(control_display_stream):
    if pylot.flags.must_visualize():
        import pygame
        from pygame.locals import K_n
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYUP:
                # Change the visualization view when n is pressed.
                if event.key == K_n:
                    control_display_stream.send(
                        erdos.Message(erdos.Timestamp(coordinates=[0]),
                                      event.key))


def read_control_command(control_stream):
    # Wait until the control is set.
    while True:
        # Read the control command from the control stream.
        control_msg = control_stream.read()
        if not isinstance(control_msg, erdos.WatermarkMessage):
            # We have read a control message. Return the command
            # so that the leaderboard can tick the simulator.
            output_control = VehicleControl()
            output_control.throttle = control_msg.throttle
            output_control.brake = control_msg.brake
            output_control.steer = control_msg.steer
            output_control.reverse = control_msg.reverse
            output_control.hand_brake = control_msg.hand_brake
            output_control.manual_gear_shift = False
            return output_control


def using_lidar():
    """Returns True if Lidar is required for the setup."""
    return not (FLAGS.simulator_obstacle_detection
                and FLAGS.simulator_traffic_light_detection)


def using_perfect_component():
    """Returns True if the agent uses any perfect component."""
    return (FLAGS.simulator_obstacle_detection
            or FLAGS.simulator_traffic_light_detection
            or FLAGS.perfect_obstacle_tracking or FLAGS.perfect_localization)
