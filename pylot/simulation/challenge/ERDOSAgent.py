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
from pylot.control.pylot_agent_operator import PylotAgentOperator
import pylot.operator_creator
from pylot.planning.planning_operator import PlanningOperator
from pylot.utils import bgra_to_bgr
import pylot.simulation.messages
from pylot.simulation.utils import to_pylot_transform, Location, Rotation, Transform
import pylot.simulation.utils


FLAGS = flags.FLAGS
CENTER_CAMERA_NAME = 'front_center_camera'
LEFT_CAMERA_NAME = 'front_left_camera'
RIGHT_CAMERA_NAME = 'front_right_camera'

# XXX(ionel): Hacks to work around issues with ROS init_node().
# Variable used to check if the Agent ROS node has already been initialized.
ROS_NODE_INITIALIZED = False
# Variable use to denote top watermark. It is increased by one before each run.
TOP_TIME = sys.maxint - 10000
GAME_TIME = 0

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


def create_agent_op(graph, bgr_camera_setup):
    agent_op = graph.add(
        PylotAgentOperator,
        name='pylot_agent',
        init_args={
            'flags': FLAGS,
            'bgr_camera_setup': bgr_camera_setup,
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


def add_depth_estimation_op(graph, scenario_input_op, center_transform):
    if FLAGS.depth_estimation:
        left_ops = add_visualization_operators(graph, LEFT_CAMERA_NAME)
        right_ops = add_visualization_operators(graph, RIGHT_CAMERA_NAME)
        graph.connect([scenario_input_op], left_ops + right_ops)
        depth_estimation_op = pylot.operator_creator.create_depth_estimation_op(
            graph, center_transform, LEFT_CAMERA_NAME, RIGHT_CAMERA_NAME)
        graph.connect([scenario_input_op], [depth_estimation_op])


class ERDOSAgent(AutonomousAgent):

    def __initialize_data_flow(self, input_streams, bgr_camera_setup):
        # Create an operator to which we connect all the input streams we
        # publish data from this script.
        scenario_input_op = self.__create_scenario_input_op(input_streams)

        visualization_ops = add_visualization_operators(
            self.graph, CENTER_CAMERA_NAME)

        add_depth_estimation_op(
            self.graph, scenario_input_op, bgr_camera_setup.transform)

        segmentation_ops = create_segmentation_ops(self.graph)

        (obj_det_ops,
         tracker_ops,
         traffic_light_det_ops,
         lane_det_ops) = create_detection_ops(self.graph)

        perception_ops = (segmentation_ops + obj_det_ops + tracker_ops +
                          traffic_light_det_ops + lane_det_ops)

        planning_ops = [
            pylot.operator_creator.create_planning_op(self.graph, None)]

        agent_op = create_agent_op(self.graph, bgr_camera_setup)

        self.graph.connect(
            [scenario_input_op],
            perception_ops + planning_ops + visualization_ops + [agent_op])

        self.graph.connect(perception_ops + planning_ops, [agent_op])

        # Execute graph. Do not block on execute so that the script can
        # input data into the data-flow graph.
        self.graph.execute(FLAGS.framework, blocking=False)

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

        (bgr_camera_setup, all_camera_setups) = self.__create_camera_setups()
        self._camera_setups = all_camera_setups
        self._lidar_transform = Transform(
            Location(1.5, 0.0, 1.40), Rotation(0, 0, 0))
        self._camera_names = {CENTER_CAMERA_NAME}
        if FLAGS.depth_estimation or self.track == Track.CAMERAS:
            self._camera_names.add(LEFT_CAMERA_NAME)
            self._camera_names.add(RIGHT_CAMERA_NAME)

        self._camera_streams = {}
        self._lock = threading.Lock()
        # Planning related attributes
        self._waypoints = None
        self._sent_open_drive_data = False
        self._open_drive_data = None

        # TODO(ionel): We should have a top watermark.
        global TOP_TIME
        self._top_watermark = WatermarkMessage(
            Timestamp(coordinates=[TOP_TIME]))
        TOP_TIME += 1

        # Set up graph
        self.graph = erdos.graph.get_current_graph()

        # The publishers must be re-created every time the Agent object
        # is constructed.
        self._input_streams = self.__create_input_streams()

        global ROS_NODE_INITIALIZED
        # We only initialize the data-flow once. On the first run.
        # We do not re-initialize it before each run because ROS does not
        # support several init_node calls from the same process or from
        # a process started with Python multiprocess.
        if not ROS_NODE_INITIALIZED:
            self.__initialize_data_flow(self._input_streams, bgr_camera_setup)
            ROS_NODE_INITIALIZED = True

        # Initialize the driver script as a ROS node so that we can receive
        # data back from the data-flow.
        rospy.init_node('erdos_driver', anonymous=True)

        # Subscribe to the control stream.
        self._subscriber = rospy.Subscriber(
            'default/pylot_agent/control_stream',
            String,
            callback=self.__on_control_msg,
            queue_size=None)
        # Setup all the input streams.
        for input_stream in self._input_streams:
            input_stream.setup()
        time.sleep(5)

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
                'id': cs.name}
            camera_sensors.append(camera_sensor)

        return (can_sensors +
                gps_sensors +
                hd_map_sensors +
                camera_sensors +
                lidar_sensors)

    def run_step(self, input_data, timestamp):
        with self._lock:
            self._control = None
            self._control_timestamp = None
        global GAME_TIME
        GAME_TIME += 1
        print("Current game time {}".format(GAME_TIME))
        erdos_timestamp = Timestamp(coordinates=[GAME_TIME])
        watermark = WatermarkMessage(erdos_timestamp)

        if self.track != Track.ALL_SENSORS_HDMAP_WAYPOINTS:
            if not self._sent_open_drive_data:
                self._sent_open_drive_data = True
                #self._open_drive_stream.send(self._top_watermark)

        self.send_waypoints(erdos_timestamp)

        for key, val in input_data.items():
            #print("{} {} {}".format(key, val[0], type(val[1])))
            if key in self._camera_names:
                # Send camera frames.
                self._camera_streams[key].send(
                    pylot.simulation.messages.FrameMessage(
                        bgra_to_bgr(val[1]), erdos_timestamp))
                #self._camera_streams[key].send(watermark)
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
        output_control = None
        with self._lock:
            # Ensure that control is not reset by another run_step invocation
            # from another thread when a new scenario is loaded.
            while (self._control_timestamp is None or
                   self._control_timestamp < erdos_timestamp):
                time.sleep(0.01)
            # Create output message. We create the VehicleControl while we
            # held the lock to ensure that control does not get changed.
            output_control = carla.VehicleControl()
            output_control.throttle = self._control.throttle
            output_control.brake = self._control.brake
            output_control.steer = self._control.steer
            output_control.reverse = self._control.reverse
            output_control.hand_brake = self._control.hand_brake
            output_control.manual_gear_shift = False

        return output_control

    def destroy(self):
        """
        Destroy (clean-up) the agent
        :return:
        """
        # Unregister the subscriber. We will register it again when
        # a new Agent object is created.
        self._subscriber.unregister()
        # We do not kill the agent driver ROS node.
        # rospy.signal_shutdown("Completed track")
        time.sleep(5)

    def send_waypoints(self, timestamp):
        # Send once the global waypoints.
        if self._waypoints is None:
            self._waypoints = self._global_plan_world_coord
            data = [(to_pylot_transform(transform), road_option)
                    for (transform, road_option) in self._waypoints]
            self._global_trajectory_stream.send(Message(data, timestamp))
            #self._global_trajectory_stream.send(self._top_watermark)
        assert self._waypoints == self._global_plan_world_coord,\
            'Global plan has been updated.'

    def send_can_bus_reading(self, data, timestamp, watermark_msg):
        # The can bus dict contains other fields as well, but we don't
        # curently use them.
        vehicle_transform = to_pylot_transform(data['transform'])
        # TODO(ionel): Scenario runner computes speed differently from
        # the way we do it in the CARLA operator. This affects
        # agent stopping constants. Check!
        forward_speed = data['speed']
        can_bus = pylot.simulation.utils.CanBus(
            vehicle_transform, forward_speed)
        self._can_bus_stream.send(Message(can_bus, timestamp))
        #self._can_bus_stream.send(watermark_msg)

    def send_lidar_reading(self, data, timestamp, watermark_msg):
        msg = pylot.simulation.messages.PointCloudMessage(
            point_cloud=data,
            transform=self._lidar_transform,
            timestamp=timestamp)
        self._point_cloud_stream.send(msg)
        #self._point_cloud_stream.send(watermark_msg)

    def send_hd_map_reading(self, data, timestamp):
        # Sending once opendrive data
        if not self._sent_open_drive_data:
            self._open_drive_data = data['opendrive']
            self._sent_open_drive_data = True
            self._open_drive_stream.send(
                Message(self._open_drive_data, timestamp))
            #self._open_drive_stream.send(self._top_watermark)
            assert self._open_drive_data == data['opendrive'],\
                'Opendrive data changed.'
        # TODO(ionel): Send point cloud data.
        pc_file = data['map_file']

    def __on_control_msg(self, msg):
        # Unpickle the data sent over the ROS topic.
        msg = pickle.loads(msg.data)
        if not isinstance(msg, WatermarkMessage):
            print('Received control message {}'.format(msg))
            self._control_timestamp = msg.timestamp
            self._control = msg

    def __create_camera_setups(self):
        location = Location(1.5, 0.0, 1.4)
        rotation = Rotation(0, 0, 0)
        transform = Transform(location, rotation)
        bgr_camera_setup = pylot.simulation.utils.CameraSetup(
            CENTER_CAMERA_NAME,
            'sensor.camera.rgb',
            FLAGS.carla_camera_image_width,
            FLAGS.carla_camera_image_height,
            transform,
            100)
        camera_setups = [bgr_camera_setup]
        if self.track == Track.CAMERAS:
            left_loc = Location(1.5, -0.3, 1.4)
            left_transform = Transform(left_loc, rotation)
            left_camera_setup = pylot.simulation.utils.CameraSetup(
                LEFT_CAMERA_NAME,
                'sensor.camera.rgb',
                FLAGS.carla_camera_image_width,
                FLAGS.carla_camera_image_height,
                left_transform,
                100)
            camera_setups.append(left_camera_setup)
            right_loc = Location(1.5, 0.3, 1.4)
            right_transform = Transform(right_loc, rotation)
            right_camera_setup = pylot.simulation.utils.CameraSetup(
                RIGHT_CAMERA_NAME,
                'sensor.camera.rgb',
                FLAGS.carla_camera_image_width,
                FLAGS.carla_camera_image_height,
                right_transform,
                100)
            camera_setups.append(right_camera_setup)

        return (bgr_camera_setup, camera_setups)

    def __create_input_streams(self):
        for name in self._camera_names:
            stream = ROSOutputDataStream(
                DataStream(name=name,
                           uid=name,
                           labels={'sensor_type': 'camera',
                                   'camera_type': 'sensor.camera.rgb',
                                   'ground': 'true'}))
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

#        if self.track == Track.ALL_SENSORS_HDMAP_WAYPOINTS:
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
        return input_streams

    def __create_scenario_input_op(self, input_streams):
        input_op = self.graph.add(NoopOp,
                                  name='scenario_input',
                                  input_streams=input_streams)
        return input_op
