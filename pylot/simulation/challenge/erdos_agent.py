from absl import flags
import carla
import erdos
import sys

import pylot.flags
import pylot.operator_creator
import pylot.perception.messages
from pylot.simulation.sensor_setup import RGBCameraSetup
import pylot.utils
from pylot.perception.point_cloud import PointCloud

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
    """Agent class that interacts with the scenario runner."""
    def __init_attributes(self, path_to_conf_file):
        flags.FLAGS([__file__, '--flagfile={}'.format(path_to_conf_file)])
        if FLAGS.track == 1:
            self.track = Track.ALL_SENSORS
        elif FLAGS.track == 2:
            self.track = Track.CAMERAS
        elif FLAGS.track == 3:
            self.track = Track.ALL_SENSORS_HDMAP_WAYPOINTS
        elif FLAGS.track == 4:
            self.track = Track.SCENE_LAYOUT
        else:
            raise ValueError('Unexpected track {}'.format(FLAGS.track))
        self._logger = erdos.utils.setup_logging('erdos_agent',
                                                 FLAGS.log_file_name)
        self._camera_setups = create_camera_setups(self.track)
        # Set the lidar in the same position as the center camera.
        self._lidar_transform = pylot.utils.Transform(CENTER_CAMERA_LOCATION,
                                                      pylot.utils.Rotation())
        self._waypoints = None
        self._sent_open_drive_data = False
        self._open_drive_data = None
        # Declare the data-flow streams.
        self._camera_streams = {}
        self._can_bus_stream = None
        self._global_trajectory_stream = None
        self._open_drive_stream = None
        self._point_cloud_stream = None
        self._obstacles_stream = None
        self._traffic_lights_stream = None
        self._waypoints_stream = None
        self._point_cloud_stream = None
        self._control_stream = None
        self._extract_control_stream = None

    def setup(self, path_to_conf_file):
        """ Setup phase. Invoked by the scenario runner."""
        self.__init_attributes(path_to_conf_file)
        self.__create_data_flow()

    def destroy(self):
        """ Clean-up the agent. Invoked between different runs."""
        pass

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
        for cs in self._camera_setups:
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
        self._logger.debug("Current game time {}".format(timestamp))
        erdos_timestamp = erdos.Timestamp(coordinates=[timestamp])

        self.send_waypoints_msg(erdos_timestamp)

        for key, val in input_data.items():
            # print("{} {} {}".format(key, val[0], type(val[1])))
            if key in self._camera_streams:
                self._camera_streams[key].send(
                    pylot.perception.messages.FrameMessage(
                        val[1], erdos_timestamp))
                self._camera_streams[key].send(
                    erdos.WatermarkMessage(erdos_timestamp))
            elif key == 'can_bus':
                self.send_can_bus_msg(val[1], erdos_timestamp)
            elif key == 'GPS':
                # gps = LocationGeo(val[1][0], val[1][1], val[1][2])
                pass
            elif key == 'hdmap':
                self.send_hd_map_msg(val[1], erdos_timestamp)
            elif key == 'LIDAR':
                self.send_lidar_msg(val[1], self._lidar_transform,
                                    erdos_timestamp)
            else:
                self._logger.warning("Sensor {} not used".format(key))

        # Wait until the control is set.
        control_msg = self._extract_control_stream.read()
        output_control = carla.VehicleControl()
        output_control.throttle = control_msg.throttle
        output_control.brake = control_msg.brake
        output_control.steer = control_msg.steer
        output_control.reverse = control_msg.reverse
        output_control.hand_brake = control_msg.hand_brake
        output_control.manual_gear_shift = False
        return output_control

    def send_hd_map_reading(self, data, timestamp):
        # Sending once opendrive data
        if not self._sent_open_drive_data:
            self._open_drive_data = data['opendrive']
            self._sent_open_drive_data = True
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

    def send_lidar_msg(self, data, transform, timestamp):
        msg = pylot.perception.messages.PointCloudMessage(PointCloud(
            data, transform),
                                                          timestamp=timestamp)
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

    def __create_data_flow(self):
        """ Create the challenge data-flow graph."""
        self._camera_streams = {}
        for name in self._camera_setups:
            self._camera_streams[name] = erdos.IngestStream()
        self._can_bus_stream = erdos.IngestStream()
        self._global_trajectory_stream = erdos.IngestStream()
        self._open_drive_stream = erdos.IngestStream()
        if self.track != Track.ALL_SENSORS_HDMAP_WAYPOINTS:
            # We do not have access to the open drive map. Send top watermark.
            if not self._sent_open_drive_data:
                self._sent_open_drive_data = True
                self._open_drive_stream.send(
                    erdos.WatermarkMessage(
                        erdos.Timestamp(coordinates=[sys.maxsize])))

        if (self.track == Track.ALL_SENSORS
                or self.track == Track.ALL_SENSORS_HDMAP_WAYPOINTS):
            self._point_cloud_stream = erdos.IngestStream()
        else:
            self._point_cloud_stream = \
                pylot.operator_creator.add_depth_estimation(
                    self._camera_streams[LEFT_CAMERA_NAME],
                    self._camera_streams[RIGHT_CAMERA_NAME],
                    self._camera_setups[CENTER_CAMERA_NAME])

        self._obstacles_stream = pylot.operator_creator.add_obstacle_detection(
            self._camera_streams[CENTER_CAMERA_NAME])
        self._traffic_lights_stream = \
            pylot.operator_creator.add_traffic_light_detector(
                self._camera_streams[TL_CAMERA_NAME])

        self._waypoints_stream = pylot.operator_creator.add_waypoint_planning(
            self._can_bus_stream, self._open_drive_stream,
            self._global_trajectory_stream, None)

        if FLAGS.visualize_rgb_camera:
            pylot.operator_creator.add_camera_visualizer(
                self._camera_streams[CENTER_CAMERA_NAME], CENTER_CAMERA_NAME)

        self._control_stream = pylot.operator_creator.add_pylot_agent(
            self._can_bus_stream, self._waypoints_stream,
            self._traffic_lights_stream, self._obstacles_stream,
            self._point_cloud_stream, self._open_drive_stream,
            depth_camera_stream, self._camera_setups[CENTER_CAMERA_NAME])
        self._extract_control_stream = erdos.ExtractStream(
            self._control_stream)


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
