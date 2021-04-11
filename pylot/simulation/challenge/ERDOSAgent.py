import time

from absl import flags

import erdos

from leaderboard.autoagents.autonomous_agent import Track

import pylot.flags
import pylot.component_creator  # noqa: I100
import pylot.operator_creator
import pylot.perception.messages
import pylot.utils
from pylot.drivers.sensor_setup import LidarSetup, RGBCameraSetup
from pylot.perception.camera_frame import CameraFrame
from pylot.simulation.challenge.ERDOSBaseAgent import ERDOSBaseAgent, \
    process_visualization_events, read_control_command

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    'perfect_localization', False,
    'Set to True to receive ego-vehicle locations from the simulator')


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

# Location of the center camera relative to the ego-vehicle.
CENTER_CAMERA_LOCATION = pylot.utils.Location(0.0, 0.0, 2.0)
CENTER_CAMERA_NAME = 'center_camera'
# Location of the camera used for lane detection.
LANE_CAMERA_LOCATION = pylot.utils.Location(1.3, 0.0, 1.8)
LANE_CAMERA_NAME = 'lane_camera'
TL_CAMERA_NAME = 'traffic_lights_camera'


def get_entry_point():
    return 'ERDOSAgent'


class ERDOSAgent(ERDOSBaseAgent):
    """Agent class that interacts with the challenge leaderboard.

    Attributes:
        _camera_setups: Mapping between camera names and
            :py:class:`~pylot.drivers.sensor_setup.CameraSetup`.
        _lidar_setup (:py:class:`~pylot.drivers.sensor_setup.LidarSetup`):
            Setup of the Lidar sensor.
    """
    def setup(self, path_to_conf_file):
        super(ERDOSAgent, self).setup(path_to_conf_file)
        self._camera_setups = create_camera_setups()
        self._lidar_setup = create_lidar_setup()

        self._sent_open_drive = False

        # Create the dataflow of AV components. Change the method
        # to add your operators.
        (self._camera_streams, self._pose_stream, self._route_stream,
         self._global_trajectory_stream, self._open_drive_stream,
         self._point_cloud_stream, self._imu_stream, self._gnss_stream,
         self._control_stream, self._control_display_stream,
         self._perfect_obstacles_stream, self._perfect_traffic_lights_stream,
         self._vehicle_id_stream, streams_to_send_top_on) = create_data_flow()
        # Execute the dataflow.
        self._node_handle = erdos.run_async()
        # Close the streams that are not used (i.e., send top watermark).
        for stream in streams_to_send_top_on:
            stream.send(erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))

    def destroy(self):
        """Clean-up the agent. Invoked between different runs."""
        self.logger.info('ERDOSAgent destroy method invoked')
        # Stop the ERDOS node.
        self._node_handle.shutdown()
        # Reset the ERDOS dataflow graph.
        erdos.reset()

    def run_step(self, input_data, timestamp):
        start_time = time.time()
        game_time = int(timestamp * 1000)
        erdos_timestamp = erdos.Timestamp(coordinates=[game_time])

        # Parse the sensor data the agent receives from the leaderboard.
        speed_data = None
        imu_data = None
        gnss_data = None
        for key, val in input_data.items():
            # val is a tuple of (timestamp, data).
            if key in self._camera_streams:
                # The data is for one of the cameras. Transform it to a Pylot
                # CameraFrame, and send it on the camera's corresponding
                # stream.
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
                if not self._open_drive_stream.is_closed():
                    # The data is only sent once because it does not change
                    # throught the duration of a scenario.
                    self._open_drive_stream.send(
                        erdos.Message(erdos_timestamp, val[1]['opendrive']))
                    # Inform the operators that read the open drive stream that
                    # they will not receive any other messages on this stream.
                    self._open_drive_stream.send(
                        erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))
            elif key == 'LIDAR':
                # Send the LiDAR point cloud.
                self.send_lidar_msg(self._point_cloud_stream, val[1],
                                    erdos_timestamp, self._lidar_setup)
            else:
                self.logger.warning("Sensor {} not used".format(key))

        # Send the route the vehicle must follow.
        self.send_global_trajectory_msg(self._global_trajectory_stream,
                                        erdos_timestamp)

        # The following two methods are only relevant when the agent
        # is using perfect perception.
        self.send_vehicle_id_msg(self._vehicle_id_stream)
        self.send_perfect_detections(self._perfect_obstacles_stream,
                                     self._perfect_traffic_lights_stream,
                                     erdos_timestamp, CENTER_CAMERA_LOCATION)

        # Send localization data.
        self.send_localization(erdos_timestamp, imu_data, gnss_data,
                               speed_data)

        # Ensure that the open drive stream is closed when the agent
        # is not running on the MAP track.
        if not self._open_drive_stream.is_closed() and self.track != Track.MAP:
            # We do not have access to the open drive map. Send top watermark.
            self._open_drive_stream.send(
                erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))

        sensor_send_runtime = (time.time() - start_time) * 1000
        self.csv_logger.info('{},{},sensor_send_runtime,{:.4f}'.format(
            pylot.utils.time_epoch_ms(), game_time, sensor_send_runtime))

        process_visualization_events(self._control_display_stream)

        # Return the control command received on the control stream.
        command = read_control_command(self._control_stream)
        e2e_runtime = (time.time() - start_time) * 1000
        self.csv_logger.info('{},{},e2e_runtime,{:.4f}'.format(
            pylot.utils.time_epoch_ms(), game_time, e2e_runtime))
        if FLAGS.simulator_mode == 'synchronous':
            return command
        elif FLAGS.simulator_mode == 'pseudo-asynchronous':
            return command, int(e2e_runtime - sensor_send_runtime)
        else:
            raise ValueError('Unexpected simulator_mode {}'.format(
                FLAGS.simulator_mode))

    def send_localization(self, timestamp, imu_data, gnss_data, speed_data):
        if FLAGS.localization:
            # The agent uses our localization. We need to send data on the
            # IMU and GNSS streams, and the initial position of the ego-vehicle
            # on the route stream.
            self.send_imu_msg(self._imu_stream, imu_data, timestamp)
            self.send_gnss_msg(self._gnss_stream, gnss_data, timestamp)
            # Naively compute the pose of the ego-vehicle, and send it on
            # the route stream. Pylot's localization operator will refine this
            # pose using the GNSS and IMU data.
            pose = self.compute_pose(speed_data, imu_data, gnss_data,
                                     timestamp)
            self._route_stream.send(erdos.Message(timestamp, pose))
            self._route_stream.send(erdos.WatermarkMessage(timestamp))
        elif FLAGS.perfect_localization:
            self.send_perfect_pose_msg(self._pose_stream, timestamp)
        else:
            # In this configuration, the agent is not using a localization
            # operator. It is driving using the noisy localization it receives
            # from the leaderboard.
            pose = self.compute_pose(speed_data, imu_data, gnss_data,
                                     timestamp)
            self._pose_stream.send(erdos.Message(timestamp, pose))
            self._pose_stream.send(erdos.WatermarkMessage(timestamp))


def create_data_flow():
    """Creates a dataflow graph of operators.

    This is the place to add other operators (e.g., other detectors,
    behavior planning).
    """
    streams_to_send_top_on = []
    camera_setups = create_camera_setups()
    # Creates a dataflow stream for each camera.
    camera_streams = {}
    for name in camera_setups:
        camera_streams[name] = erdos.IngestStream()
    # Creates a stream on which the agent sends the high-level route the
    # agent must follow in the challenge.
    global_trajectory_stream = erdos.IngestStream()
    # Creates a stream on which the agent sends the open drive stream it
    # receives when it executes in the MAP track.
    open_drive_stream = erdos.IngestStream()
    # Creates a stream on which the agent sends point cloud messages it
    # receives from the LiDAR sensor.
    point_cloud_stream = erdos.IngestStream()
    imu_stream = erdos.IngestStream()
    gnss_stream = erdos.IngestStream()
    route_stream = erdos.IngestStream()
    time_to_decision_loop_stream = erdos.LoopStream()

    if FLAGS.localization:
        # Pylot localization is enabled. Add the localization operator to
        # the dataflow. The operator receives GNSS and IMU messages, and
        # uses an Extended Kalman Filter to compute poses.
        pose_stream = pylot.operator_creator.add_localization(
            imu_stream, gnss_stream, route_stream)
    else:
        # The agent either directly forwards the poses it receives from
        # the challenge, which are noisy, or the perfect poses if the
        # --perfect_localization flag is set.
        pose_stream = erdos.IngestStream()

    # Stream on which the obstacles are sent when the agent is using perfect
    # detection.
    perfect_obstacles_stream = erdos.IngestStream()
    if FLAGS.simulator_obstacle_detection:
        # Execute with perfect perception. In this configuration, the agent
        # directly gets the location of the other agents from the simulator.
        # This configuration is meant for testing and debugging.
        obstacles_stream = perfect_obstacles_stream
    elif any('efficientdet' in model
             for model in FLAGS.obstacle_detection_model_names):
        # Add an operator that runs EfficientDet to detect obstacles.
        obstacles_stream = pylot.operator_creator.\
            add_efficientdet_obstacle_detection(
                camera_streams[CENTER_CAMERA_NAME],
                time_to_decision_loop_stream)[0]
        if not (FLAGS.evaluate_obstacle_detection
                or FLAGS.evaluate_obstacle_tracking):
            streams_to_send_top_on.append(perfect_obstacles_stream)
    else:
        # Add an operator that uses one of the Tensorflow model zoo
        # detection models. By default, we use FasterRCNN.
        obstacles_stream = pylot.operator_creator.add_obstacle_detection(
            camera_streams[CENTER_CAMERA_NAME],
            time_to_decision_loop_stream)[0]
        if not (FLAGS.evaluate_obstacle_detection
                or FLAGS.evaluate_obstacle_tracking):
            streams_to_send_top_on.append(perfect_obstacles_stream)

    # Stream on which the traffic lights are sent when the agent is
    # using perfect traffic light detection.
    perfect_traffic_lights_stream = erdos.IngestStream()
    if FLAGS.simulator_traffic_light_detection:
        # In this debug configuration, the agent is using perfectly located
        # traffic lights it receives directly from the simulator. Therefore,
        # there's no need to a traffic light detector.
        traffic_lights_stream = perfect_traffic_lights_stream
        camera_streams[TL_CAMERA_NAME] = erdos.IngestStream()
        streams_to_send_top_on.append(camera_streams[TL_CAMERA_NAME])
    else:
        # Adds a traffic light detector operator, which uses the camera with
        # the small fov.
        traffic_lights_stream = \
            pylot.operator_creator.add_traffic_light_detector(
                camera_streams[TL_CAMERA_NAME], time_to_decision_loop_stream)
        # Adds an operator that finds the world location of the traffic lights.
        # The operator synchronizes LiDAR point cloud readings with camera
        # frames, and uses them to compute the depth to traffic light bounding
        # boxes.
        traffic_lights_stream = \
            pylot.operator_creator.add_obstacle_location_finder(
                traffic_lights_stream, point_cloud_stream, pose_stream,
                camera_setups[TL_CAMERA_NAME])
        # We do not send perfectly located traffic lights in this
        # configuration. Therefore, ensure that the stream is "closed"
        # (i.e., send a top watermark)
        streams_to_send_top_on.append(perfect_traffic_lights_stream)

    vehicle_id_stream = erdos.IngestStream()
    if not (FLAGS.perfect_obstacle_tracking or FLAGS.perfect_localization):
        # The vehicle_id_stream is only used when perfect localization
        # or perfect obstacle tracking are enabled.
        streams_to_send_top_on.append(vehicle_id_stream)

    # Adds an operator for tracking detected agents. The operator uses the
    # frames from the center camera, and the bounding boxes found by the
    # obstacle detector operator.
    obstacles_tracking_stream = pylot.component_creator.add_obstacle_tracking(
        camera_streams[CENTER_CAMERA_NAME],
        camera_setups[CENTER_CAMERA_NAME],
        obstacles_stream,
        depth_stream=point_cloud_stream,
        vehicle_id_stream=vehicle_id_stream,
        pose_stream=pose_stream,
        ground_obstacles_stream=perfect_obstacles_stream,
        time_to_decision_stream=time_to_decision_loop_stream)

    if FLAGS.execution_mode == 'challenge-sensors':
        # The agent is running is sensors-only track. Therefore, we need
        # to add a lane detector because the agent does not have access to
        # the OpenDrive map.
        lanes_stream = pylot.operator_creator.add_lanenet_detection(
            camera_streams[LANE_CAMERA_NAME])
    else:
        # The lanes stream is not used when running in the Map track.
        # We add the stream to the list of streams that are not used, and
        # must be manually "closed" (i.e., send a top watermark).
        lanes_stream = erdos.IngestStream()
        streams_to_send_top_on.append(lanes_stream)

    # The agent uses a linear predictor to compute future trajectories
    # of the other agents.
    prediction_stream, _, _ = pylot.component_creator.add_prediction(
        obstacles_tracking_stream,
        vehicle_id_stream,
        time_to_decision_loop_stream,
        pose_stream=pose_stream)

    # Adds a planner to the agent. The planner receives the pose of
    # the ego-vehicle, detected traffic lights, predictions for other
    # agents, the route the agent must follow, and the open drive data if
    # the agent is executing in the Map track, or detected lanes if it is
    # executing in the Sensors-only track.
    waypoints_stream = pylot.component_creator.add_planning(
        None, pose_stream, prediction_stream, traffic_lights_stream,
        lanes_stream, open_drive_stream, global_trajectory_stream,
        time_to_decision_loop_stream)

    if pylot.flags.must_visualize():
        # Adds a visualization dataflow operator if any of the
        # --visualize_* is enabled. The operator creates a pygame window
        # in which the different sensors, detections can be visualized.
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

    # Adds a controller which tries to follow the waypoints computed
    # by the planner.
    control_stream = pylot.component_creator.add_control(
        pose_stream, waypoints_stream)
    # The controller returns a stream of commands (i.e., throttle, steer)
    # from which the agent can read the command it must return to the
    # challenge.
    extract_control_stream = erdos.ExtractStream(control_stream)

    pylot.component_creator.add_evaluation(vehicle_id_stream, pose_stream,
                                           imu_stream)

    # Operator that computes how much time each component gets to execute.
    # This is needed in Pylot, but can be ignored when running in challenge
    # mode.
    time_to_decision_stream = pylot.operator_creator.add_time_to_decision(
        pose_stream, obstacles_stream)
    time_to_decision_loop_stream.set(time_to_decision_stream)

    return (camera_streams, pose_stream, route_stream,
            global_trajectory_stream, open_drive_stream, point_cloud_stream,
            imu_stream, gnss_stream, extract_control_stream,
            control_display_stream, perfect_obstacles_stream,
            perfect_traffic_lights_stream, vehicle_id_stream,
            streams_to_send_top_on)


def create_camera_setups():
    """Creates RGBCameraSetups for the cameras deployed on the car.

    Note: Change this method if your agent requires more cameras, or if
    you want to change camera properties.

    It returns a dict from camera name to
    :py:class:`~pylot.drivers.sensor_setup.RGBCameraSetup`.
    """
    camera_setups = {}
    # Add a center front camera.
    transform = pylot.utils.Transform(CENTER_CAMERA_LOCATION,
                                      pylot.utils.Rotation())
    center_camera_setup = RGBCameraSetup(CENTER_CAMERA_NAME,
                                         FLAGS.camera_image_width,
                                         FLAGS.camera_image_height, transform,
                                         90)
    camera_setups[CENTER_CAMERA_NAME] = center_camera_setup

    if not FLAGS.simulator_traffic_light_detection:
        # Add a camera with a narrow field of view. The traffic light
        # camera is added in the same position as the center camera.
        # We use this camera for traffic light detection.
        tl_camera_setup = RGBCameraSetup(TL_CAMERA_NAME,
                                         FLAGS.camera_image_width,
                                         FLAGS.camera_image_height, transform,
                                         45)
        camera_setups[TL_CAMERA_NAME] = tl_camera_setup

    if FLAGS.execution_mode == 'challenge-sensors':
        # The agent in executed in the sensors-only track.
        # We add camera, which we use for lane detection because we do not
        # have access to the OpenDrive map in this track.
        lane_transform = pylot.utils.Transform(LANE_CAMERA_LOCATION,
                                               pylot.utils.Rotation(pitch=-15))
        lane_camera_setup = RGBCameraSetup(LANE_CAMERA_NAME, 1280, 720,
                                           lane_transform, 90)
        camera_setups[LANE_CAMERA_NAME] = lane_camera_setup
    return camera_setups


def create_lidar_setup():
    """Creates a setup for the LiDAR deployed on the car."""
    # Set the lidar in the same position as the center camera.
    # Pylot uses the LiDAR point clouds to compute the distance to
    # the obstacles detected using the camera frames.
    lidar_transform = pylot.utils.Transform(CENTER_CAMERA_LOCATION,
                                            pylot.utils.Rotation())
    lidar_setup = LidarSetup(
        'lidar',
        'sensor.lidar.ray_cast',
        lidar_transform,
        range=8500,  # Range is only used for visualization.
        legacy=False)
    return lidar_setup
