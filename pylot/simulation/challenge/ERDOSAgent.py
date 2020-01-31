from absl import flags
import carla
import erdos
import sys

import pylot.flags
import pylot.operator_creator
import pylot.perception.messages
from pylot.perception.camera_frame import CameraFrame
from pylot.perception.point_cloud import PointCloud
from pylot.simulation.sensor_setup import RGBCameraSetup
import pylot.utils

from srunner.challenge.autoagents.autonomous_agent import AutonomousAgent,\
    Track

FLAGS = flags.FLAGS

flags.DEFINE_integer('track', 3, 'Track to execute')

CENTER_CAMERA_LOCATION = pylot.utils.Location(1.5, 0.0, 1.4)
CENTER_CAMERA_NAME = 'center_camera'
TL_CAMERA_NAME = 'traffic_lights_camera'
LEFT_CAMERA_NAME = 'left_camera'
RIGHT_CAMERA_NAME = 'right_camera'


class ERDOSAgent(AutonomousAgent):
    """Agent class that interacts with the CARLA challenge scenario runner.

    Attributes:
        track: Track the agent is running in.
        _camera_setups: Mapping between camera names and
            :py:class:`~pylot.simulation.sensor_setup.CameraSetup`.
        _lidar_transform (:py:class:`~pylot.utils.Transform`): Transform of
            the Lidar relative to the ego vehicle.
        _waypoints (list(:py:class:`~pylot.utils.Transform`)): List of
            waypoints the agent receives from the challenge planner.
    """
    def __init_attributes(self, path_to_conf_file):
        flags.FLAGS([__file__, '--flagfile={}'.format(path_to_conf_file)])
        self._logger = erdos.utils.setup_logging('erdos_agent',
                                                 FLAGS.log_file_name)
        enable_logging()
        self.track = get_track()
        self._camera_setups = create_camera_setups(self.track)
        # Set the lidar in the same position as the center camera.
        self._lidar_transform = pylot.utils.Transform(CENTER_CAMERA_LOCATION,
                                                      pylot.utils.Rotation())
        # Stores the waypoints we get from the challenge planner.
        self._waypoints = None
        # Stores the open drive string we get when we run in track 3.
        self._open_drive_data = None
        (camera_streams, can_bus_stream, global_trajectory_stream,
         open_drive_stream, point_cloud_stream,
         control_stream) = erdos.run_async(create_data_flow)
        self._camera_streams = camera_streams
        self._can_bus_stream = can_bus_stream
        self._global_trajectory_stream = global_trajectory_stream
        self._open_drive_stream = open_drive_stream
        self._point_cloud_stream = point_cloud_stream
        self._control_stream = control_stream

    def setup(self, path_to_conf_file):
        """ Setup phase. Invoked by the scenario runner."""
        self.__init_attributes(path_to_conf_file)

    def destroy(self):
        """ Clean-up the agent. Invoked between different runs."""
        self._logger.info('ERDOSAgent destroy method invoked')

    def sensors(self):
        """
        Defines the sensor suite required by the agent.
        """
        can_sensors = [{
            'type': 'sensor.can_bus',
            'reading_frequency': 20,
            'id': 'can_bus'
        }]

        hd_map_sensors = []
        if self.track == Track.ALL_SENSORS_HDMAP_WAYPOINTS:
            hd_map_sensors = [{
                'type': 'sensor.hd_map',
                'reading_frequency': 20,
                'id': 'hdmap'
            }]

        gps_sensors = []
        lidar_sensors = []
        if (self.track == Track.ALL_SENSORS_HDMAP_WAYPOINTS
                or self.track == Track.ALL_SENSORS):
            gps_sensors = [{
                'type': 'sensor.other.gnss',
                'x': 0.7,
                'y': -0.4,
                'z': 1.60,
                'id': 'GPS'
            }]
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

        return (can_sensors + gps_sensors + hd_map_sensors + camera_sensors +
                lidar_sensors)

    def run_step(self, input_data, timestamp):
        game_time = int(timestamp * 1000)
        self._logger.debug("Current game time {}".format(game_time))
        erdos_timestamp = erdos.Timestamp(coordinates=[game_time])

        self.send_waypoints_msg(erdos_timestamp)

        for key, val in input_data.items():
            # print("{} {} {}".format(key, val[0], type(val[1])))
            if key in self._camera_streams:

                self._camera_streams[key].send(
                    pylot.perception.messages.FrameMessage(
                        erdos_timestamp,
                        CameraFrame(val[1][:, :, :3], 'BGR',
                                    self._camera_setups[key])))
                self._camera_streams[key].send(
                    erdos.WatermarkMessage(erdos_timestamp))
            elif key == 'can_bus':
                self.send_can_bus_msg(val[1], erdos_timestamp)
            elif key == 'hdmap':
                self.send_hd_map_msg(val[1], erdos_timestamp)
            elif key == 'LIDAR':
                self.send_lidar_msg(val[1], self._lidar_transform,
                                    erdos_timestamp)
            else:
                self._logger.warning("Sensor {} not used".format(key))

        # Wait until the control is set.
        while True:
            control_msg = self._control_stream.read()
            if isinstance(control_msg, erdos.Message):
                output_control = carla.VehicleControl()
                output_control.throttle = control_msg.throttle
                output_control.brake = control_msg.brake
                output_control.steer = control_msg.steer
                output_control.reverse = control_msg.reverse
                output_control.hand_brake = control_msg.hand_brake
                output_control.manual_gear_shift = False
                return output_control

    def send_hd_map_msg(self, data, timestamp):
        # Sending once opendrive data
        if self._open_drive_data is None:
            self._open_drive_data = data['opendrive']
            self._open_drive_stream.send(
                erdos.Message(timestamp, self._open_drive_data))
            self._open_drive_stream.send(
                erdos.WatermarkMessage(
                    erdos.Timestamp(coordinates=[sys.maxsize])))
        # TODO: Send point cloud data.
        # pc_file = data['map_file']

    def send_can_bus_msg(self, data, timestamp):
        # The can bus dict contains other fields as well, but we don't use
        # them yet.
        vehicle_transform = pylot.utils.Transform.from_carla_transform(
            data['transform'])
        forward_speed = data['speed']
        self._can_bus_stream.send(
            erdos.Message(timestamp,
                          pylot.utils.CanBus(vehicle_transform,
                                             forward_speed)))
        self._can_bus_stream.send(erdos.WatermarkMessage(timestamp))

    def send_lidar_msg(self, carla_pc, transform, timestamp):
        points = [pylot.utils.Location(x, y, z) for x, y, z in carla_pc]
        msg = pylot.perception.messages.PointCloudMessage(
            timestamp, PointCloud(points, transform))
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
                erdos.WatermarkMessage(
                    erdos.Timestamp(coordinates=[sys.maxsize])))


def get_track():
    track = None
    if FLAGS.track == 1:
        track = Track.ALL_SENSORS
    elif FLAGS.track == 2:
        track = Track.CAMERAS
    elif FLAGS.track == 3:
        track = Track.ALL_SENSORS_HDMAP_WAYPOINTS
    elif FLAGS.track == 4:
        track = Track.SCENE_LAYOUT
    else:
        raise ValueError('Unexpected track {}'.format(FLAGS.track))
    return track


def create_data_flow():
    """ Create the challenge data-flow graph."""
    track = get_track()
    camera_setups = create_camera_setups(track)
    camera_streams = {}
    for name in camera_setups:
        camera_streams[name] = erdos.IngestStream()
    can_bus_stream = erdos.IngestStream()
    global_trajectory_stream = erdos.IngestStream()
    open_drive_stream = erdos.IngestStream()
    if track != Track.ALL_SENSORS_HDMAP_WAYPOINTS:
        # We do not have access to the open drive map. Send top watermark.
        open_drive_stream.send(
            erdos.WatermarkMessage(erdos.Timestamp(coordinates=[sys.maxsize])))

    if (track == Track.ALL_SENSORS
            or track == Track.ALL_SENSORS_HDMAP_WAYPOINTS):
        point_cloud_stream = erdos.IngestStream()
    else:
        point_cloud_stream = pylot.operator_creator.add_depth_estimation(
            camera_streams[LEFT_CAMERA_NAME],
            camera_streams[RIGHT_CAMERA_NAME],
            camera_setups[CENTER_CAMERA_NAME])

    obstacles_stream = pylot.operator_creator.add_obstacle_detection(
        camera_streams[CENTER_CAMERA_NAME])[0]
    # Adds an operator that finds the world locations of the obstacles.
    obstacles_stream = pylot.operator_creator.add_obstacle_location_finder(
        obstacles_stream, point_cloud_stream, can_bus_stream,
        camera_setups[CENTER_CAMERA_NAME])

    traffic_lights_stream = pylot.operator_creator.add_traffic_light_detector(
        camera_streams[TL_CAMERA_NAME])
    # Adds an operator that finds the world locations of the traffic lights.
    traffic_lights_stream = \
        pylot.operator_creator.add_obstacle_location_finder(
            traffic_lights_stream, point_cloud_stream, can_bus_stream,
            camera_setups[TL_CAMERA_NAME])

    waypoints_stream = pylot.operator_creator.add_waypoint_planning(
        can_bus_stream, open_drive_stream, global_trajectory_stream, None)

    if FLAGS.visualize_rgb_camera:
        pylot.operator_creator.add_camera_visualizer(
            camera_streams[CENTER_CAMERA_NAME], CENTER_CAMERA_NAME)

    control_stream = pylot.operator_creator.add_pylot_agent(
        can_bus_stream, waypoints_stream, traffic_lights_stream,
        obstacles_stream, open_drive_stream)
    extract_control_stream = erdos.ExtractStream(control_stream)
    return (camera_streams, can_bus_stream, global_trajectory_stream,
            open_drive_stream, point_cloud_stream, extract_control_stream)


def create_camera_setups(track):
    """Creates different camera setups depending on the track."""
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
    left_camera_setup = None
    right_camera_setup = None
    # Add left and right cameras if we don't have access to lidar.
    if track == Track.CAMERAS:
        left_location = CENTER_CAMERA_LOCATION + pylot.utils.Location(
            0, -FLAGS.offset_left_right_cameras, 0)
        left_camera_setup = RGBCameraSetup(
            LEFT_CAMERA_NAME, FLAGS.carla_camera_image_width,
            FLAGS.carla_camera_image_height,
            pylot.utils.Transform(left_location, pylot.utils.Rotation()), 90)
        camera_setups[LEFT_CAMERA_NAME] = left_camera_setup
        right_location = CENTER_CAMERA_LOCATION + pylot.utils.Location(
            0, FLAGS.offset_left_right_cameras, 0)
        right_camera_setup = RGBCameraSetup(
            RIGHT_CAMERA_NAME, FLAGS.carla_camera_image_width,
            FLAGS.carla_camera_image_height,
            pylot.utils.Transform(right_location, pylot.utils.Rotation()), 90)
        camera_setups[RIGHT_CAMERA_NAME] = right_camera_setup
    return camera_setups


def enable_logging():
    """Overwrites logging config so that loggers can control verbosity.

    This method is required because the challenge evaluator overwrites
    verbosity, which causes Pylot log messages to be discarded.
    """
    import logging
    logging.root.setLevel(logging.NOTSET)
