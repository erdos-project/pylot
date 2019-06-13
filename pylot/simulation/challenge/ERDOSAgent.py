from absl import flags
import pickle
import rospy
from std_msgs.msg import String
import sys
import time
import threading

import carla

from srunner.challenge.autoagents.autonomous_agent import AutonomousAgent, Track

from erdos.data_stream import DataStream
import erdos.graph
from erdos.message import Message, WatermarkMessage
from erdos.operators import NoopOp
from erdos.ros.ros_output_data_stream import ROSOutputDataStream
from erdos.timestamp import Timestamp

import pylot.config
from pylot.control.lidar_erdos_agent_operator import LidarERDOSAgentOperator
import pylot.operator_creator
from pylot.planning.challenge_planning_operator import ChallengePlanningOperator
from pylot.utils import bgra_to_bgr
import pylot.simulation.messages
from pylot.simulation.utils import to_erdos_transform
import pylot.simulation.utils


FLAGS = flags.FLAGS
CENTER_CAMERA_NAME = 'front_center_camera'
LEFT_CAMERA_NAME = 'front_left_camera'
RIGHT_CAMERA_NAME = 'front_right_camera'


flags.DEFINE_integer('track', 3, 'Track to execute')


def add_visualization_operators(graph, rgb_camera_name):
    visualization_ops = []
    if FLAGS.visualize_rgb_camera:
        camera_video_op = pylot.operator_creator.create_camera_video_op(
            graph, rgb_camera_name, rgb_camera_name)
        visualization_ops.append(camera_video_op)
    if FLAGS.visualize_segmentation:
        # Segmented camera. The stream comes from CARLA.
        segmented_video_op = pylot.operator_creator.create_segmented_video_op(graph)
        visualization_ops.append(segmented_video_op)
    return visualization_ops


def create_planning_op(graph):
    planning_op = graph.add(
        ChallengePlanningOperator,
        name='planning',
        init_args={
            'flags': FLAGS,
            'log_file_name': FLAGS.log_file_name,
            'csv_file_name': FLAGS.csv_log_file_name
        })
    return planning_op


def create_agent_op(graph):
    agent_op = graph.add(
        LidarERDOSAgentOperator,
        name='lidar_erdos_agent',
        init_args={
            'flags': FLAGS,
            'log_file_name': FLAGS.log_file_name,
            'csv_file_name': FLAGS.csv_log_file_name
        })
    return agent_op


def create_detection_ops(graph):
    obj_det_ops = []
    tracker_ops = []
    if FLAGS.obj_detection:
        obj_det_ops = pylot.operator_creator.create_detector_ops(graph)
        if FLAGS.obj_tracking:
            tracker_op = pylot.operator_creator.create_object_tracking_op(
                graph)
            tracker_ops.append(tracker_op)

    traffic_light_det_ops = []
    if FLAGS.traffic_light_det:
        traffic_light_det_ops.append(
            pylot.operator_creator.create_traffic_light_op(graph))

    lane_det_ops = []
    if FLAGS.lane_detection:
        lane_det_ops.append(
            pylot.operator_creator.create_lane_detection_op(graph))

    return (obj_det_ops, tracker_ops, traffic_light_det_ops, lane_det_ops)


def create_segmentation_ops(graph):
    segmentation_ops = []
    if FLAGS.segmentation_drn:
        segmentation_op = pylot.operator_creator.create_segmentation_drn_op(
            graph)
        segmentation_ops.append(segmentation_op)

    if FLAGS.segmentation_dla:
        segmentation_op = pylot.operator_creator.create_segmentation_dla_op(
            graph)
        segmentation_ops.append(segmentation_op)
    return segmentation_ops


def add_depth_estimation_op(graph, scenario_input_op):
    if FLAGS.depth_estimation:
        left_ops = add_visualization_operators(graph, LEFT_CAMERA_NAME)
        right_ops = add_visualization_operators(graph, RIGHT_CAMERA_NAME)
        graph.connect([scenario_input_op], left_ops + right_ops)
        depth_estimation_op = pylot.operator_creator.create_depth_estimation_op(
            graph, LEFT_CAMERA_NAME, RIGHT_CAMERA_NAME)
        graph.connect([scenario_input_op], [depth_estimation_op])


class ERDOSAgent(AutonomousAgent):

    def setup(self, path_to_conf_file):
        flags.FLAGS([__file__, '--flagfile={}'.format(path_to_conf_file)])
        if FLAGS.track == 1:
            self.track = Track.ALL_SENSORS
        elif FLAGS.track == 2:
            self.track = Track.CAMERAS
        elif FLAGS.track == 3:
            self.track = Track.ALL_SENSORS_HDMAP_WAYPOINTS
        else:
            print('Unexpected track {}'.format(FLAGS.track))

        self.__setup_sensors()
        self._camera_streams = {}
        self._lock = threading.Lock()
        # Planning related attributes
        self._waypoints = None
        self._sent_open_drive_data = False
        self._open_drive_data = None

        # TODO(ionel): We should have a top watermark.
        self._top_watermark = WatermarkMessage(
            Timestamp(coordinates=[sys.maxint]))

        # Set up graph
        self.graph = erdos.graph.get_current_graph()

        # Create an operator to which we connect all the input streams we
        # publish data from this script.
        (scenario_input_op, input_streams) = self.__create_scenario_input_op()

        visualization_ops = add_visualization_operators(
            self.graph, CENTER_CAMERA_NAME)

        add_depth_estimation_op(self.graph, scenario_input_op)

        segmentation_ops = create_segmentation_ops(self.graph)

        (obj_det_ops,
         tracker_ops,
         traffic_light_det_ops,
         lane_det_ops) = create_detection_ops(self.graph)

        perception_ops = (segmentation_ops + obj_det_ops + tracker_ops +
                          traffic_light_det_ops + lane_det_ops)

        planning_ops = [create_planning_op(self.graph)]

        agent_op = create_agent_op(self.graph)

        self.graph.connect(
            [scenario_input_op],
            perception_ops + planning_ops + visualization_ops + [agent_op])

        self.graph.connect(perception_ops + planning_ops, [agent_op])

        # Execute graph. Do not block on execute so that the script can
        # input data into the data-flow graph.
        self.graph.execute(FLAGS.framework, blocking=False)

        # Initialize the driver script as a ROS node so that we can receive
        # data back from the data-flow.
        rospy.init_node('erdos_driver', anonymous=True)
        # Subscribe to the control stream
        rospy.Subscriber('default/lidar_erdos_agent/control_stream',
                         String,
                         callback=self.__on_control_msg,
                         queue_size=None)
        # Setup all the input streams.
        for input_stream in input_streams:
            input_stream.setup()

    def sensors(self):
        """
        Define the sensor suite required by the agent.
        """
        can_sensors = [{'type': 'sensor.can_bus',
                        'reading_frequency': 20,
                        'id': 'can_bus'}]

        hd_map_sensors = []
        if self.track == Track.ALL_SENSORS_HDMAP_WAYPOINTS:
            hd_map_sensors = [{'type': 'sensor.hd_map',
                               'reading_frequency': 20,
                               'id': 'hdmap'}]

        gps_sensors = []
        lidar_sensors = []
        if (self.track == Track.ALL_SENSORS_HDMAP_WAYPOINTS or
            self.track == Track.ALL_SENSORS):
            gps_sensors = [{'type': 'sensor.other.gnss',
                            'x': 0.7,
                            'y': -0.4,
                            'z': 1.60,
                            'id': 'GPS'}]
            lidar_sensors = [{'type': 'sensor.lidar.ray_cast',
                              'x': self._lidar_transform.location.x,
                              'y': self._lidar_transform.location.y,
                              'z': self._lidar_transform.location.z,
                              'roll': self._lidar_transform.rotation.roll,
                              'pitch': self._lidar_transform.rotation.pitch,
                              'yaw': self._lidar_transform.rotation.yaw,
                              'id': 'LIDAR'}]

        camera_sensors = self.__define_camera_sensors()

        return (can_sensors +
                gps_sensors +
                hd_map_sensors +
                camera_sensors +
                lidar_sensors)

    def run_step(self, input_data, timestamp):
        with self._lock:
            self._control = None
            self._control_timestamp = None
        game_time = int(timestamp * 1000)
        erdos_timestamp = Timestamp(coordinates=[game_time])
        watermark = WatermarkMessage(erdos_timestamp)

        self.send_waypoints(erdos_timestamp)

        for key, val in input_data.items():
            #print("{} {} {}".format(key, val[0], type(val[1])))
            if key in self._camera_names:
                # Send camera frames.
                self._camera_streams[key].send(
                    pylot.simulation.messages.FrameMessage(
                        bgra_to_bgr(val[1]), erdos_timestamp))
                self._camera_streams[key].send(watermark)
            elif key == 'can_bus':
                self.send_can_bus_reading(val[1], erdos_timestamp, watermark)
            elif key == 'GPS':
                gps = pylot.simulation.utils.LocationGeo(
                    val[1][0], val[1][1], val[1][2])
            elif key == 'hdmap':
                self.send_hd_map_reading(val[1], erdos_timestamp)
            elif key == 'LIDAR':
                self.send_lidar_reading(val[1], erdos_timestamp, watermark)

        # Wait until the control is set.
        while (self._control_timestamp is None or
               self._control_timestamp < erdos_timestamp):
            time.sleep(0.01)

        return self._control

    def send_waypoints(self, timestamp):
        # Send once the global waypoints.
        if self._waypoints is None:
            self._waypoints = self._global_plan_world_coord
            data = [(to_erdos_transform(transform), road_option)
                    for (transform, road_option) in self._waypoints]
            self._global_trajectory_stream.send(Message(data, timestamp))
            self._global_trajectory_stream.send(self._top_watermark)
        assert self._waypoints == self._global_plan_world_coord,\
            'Global plan has been updated.'

    def send_can_bus_reading(self, data, timestamp, watermark_msg):
        # The can bus dict contains other fields as well, but we don't
        # curently use them.
        vehicle_transform = to_erdos_transform(data['transform'])
        # TODO(ionel): Scenario runner computes speed differently from
        # the way we do it in the CARLA operator. This affects
        # agent stopping constants. Check!
        forward_speed = data['speed']
        can_bus = pylot.simulation.utils.CanBus(
            vehicle_transform, forward_speed)
        self._can_bus_stream.send(Message(can_bus, timestamp))
        self._can_bus_stream.send(watermark_msg)

    def send_lidar_reading(self, data, timestamp, watermark_msg):
        msg = pylot.simulation.messages.PointCloudMessage(
            point_cloud=data,
            transform=self._lidar_transform,
            timestamp=timestamp)
        self._point_cloud_stream.send(msg)
        self._point_cloud_stream.send(watermark_msg)

    def send_hd_map_reading(self, data, timestamp):
        # Sending once opendrive data
        if not self._sent_open_drive_data:
            self._open_drive_data = data['opendrive']
            self._sent_open_drive_data = True
            self._open_drive_stream.send(
                Message(self._open_drive_data, timestamp))
            self._open_drive_stream.send(self._top_watermark)
            assert self._open_drive_data == data['opendrive'],\
                'Opendrive data changed.'
        # TODO(ionel): Send point cloud data.
        pc_file = data['map_file']

    def __on_control_msg(self, msg):
        # Unpickle the data sent over the ROS topic.
        msg = pickle.loads(msg.data)
        if not isinstance(msg, WatermarkMessage):
            with self._lock:
                print('Received control message {}'.format(msg))
                self._control_timestamp = msg.timestamp
                self._control = carla.VehicleControl()
                self._control.throttle = msg.throttle
                self._control.brake = msg.brake
                self._control.steer = msg.steer
                self._control.reverse = msg.reverse
                self._control.hand_brake = msg.hand_brake
                self._control.manual_gear_shift = False

    def __define_camera_sensors(self):
        camera_sensors = [{'type': 'sensor.camera.rgb',
                           'x': self._camera_transform.location.x,
                           'y': self._camera_transform.location.y,
                           'z': self._camera_transform.location.z,
                           'roll': self._camera_transform.rotation.roll,
                           'pitch': self._camera_transform.rotation.pitch,
                           'yaw': self._camera_transform.rotation.yaw,
                           'width': 800,
                           'height': 600,
                           'fov': 100,
                           'id': CENTER_CAMERA_NAME}]
        if self.track == Track.CAMERAS:
            left_camera_sensor = {'type': 'sensor.camera.rgb',
                                  'x': 2.0,
                                  'y': -0.4,
                                  'z': 1.40,
                                  'roll': 0,
                                  'pitch': 0,
                                  'yaw': 0,
                                  'width': 800,
                                  'height': 600,
                                  'fov': 100,
                                  'id': LEFT_CAMERA_NAME}
            camera_sensors.append(left_camera_sensor)
            right_camera_sensor = {'type': 'sensor.camera.rgb',
                                   'x': 2.0,
                                   'y': 0.4,
                                   'z': 1.40,
                                   'roll': 0,
                                   'pitch': 0,
                                   'yaw': 0,
                                   'width': 800,
                                   'height': 600,
                                   'fov': 100,
                                   'id': RIGHT_CAMERA_NAME}
            camera_sensors.append(right_camera_sensor)
        return camera_sensors

    def __setup_sensors(self):
        loc = pylot.simulation.utils.Location(2.0, 0.0, 1.40)
        rot = pylot.simulation.utils.Rotation(0, 0, 0)
        self._camera_transform = pylot.simulation.utils.Transform(loc, rot)
        self._lidar_transform = pylot.simulation.utils.Transform(loc, rot)
        self._camera_names = {CENTER_CAMERA_NAME}
        if FLAGS.depth_estimation:
            self._camera_names.add(LEFT_CAMERA_NAME)
            self._camera_names.add(RIGHT_CAMERA_NAME)

    def __create_scenario_input_op(self):
        for name in self._camera_names:
            stream = ROSOutputDataStream(
                DataStream(name=name,
                           uid=name,
                           labels={'sensor_type': 'camera',
                                   'camera_type': 'sensor.camera.rgb'}))
            self._camera_streams[name] = stream

        self._can_bus_stream = ROSOutputDataStream(
            DataStream(name='can_bus', uid='can_bus'))

        # Stream on which we send the global trajectory.
        self._global_trajectory_stream = ROSOutputDataStream(
            DataStream(name='global_trajectory_stream',
                       uid='global_trajectory_stream',
                       labels={'global': 'true',
                               'waypoints': 'true'}))

        input_streams = (self._camera_streams.values() +
                         [self._global_trajectory_stream,
                          self._can_bus_stream])

        if self.track == Track.ALL_SENSORS_HDMAP_WAYPOINTS:
            # Stream on which we send the opendrive map.
            self._open_drive_stream = ROSOutputDataStream(
                DataStream(name='open_drive_stream',
                           uid='open_drive_stream'))
            input_streams.append(self._open_drive_stream)

        if (self.track == Track.ALL_SENSORS or
            self.track == Track.ALL_SENSORS_HDMAP_WAYPOINTS):
            self._point_cloud_stream = ROSOutputDataStream(
                DataStream(name='lidar',
                           uid='lidar',
                           labels={'sensor_type': 'sensor.lidar.ray_cast'}))
            input_streams.append(self._point_cloud_stream)

        input_op = self.graph.add(NoopOp,
                                  name='scenario_input',
                                  input_streams=input_streams)
        return (input_op, input_streams)
