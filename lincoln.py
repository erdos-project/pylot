import signal
import time

from absl import app, flags

import erdos

import pylot.component_creator
import pylot.drivers.sensor_setup
import pylot.flags
import pylot.operator_creator
import pylot.planning.utils
import pylot.utils
from pylot.drivers.drive_by_wire_operator import DriveByWireOperator
from pylot.drivers.grasshopper3_driver_operator import \
    Grasshopper3DriverOperator
from pylot.drivers.velodyne_driver_operator import VelodyneDriverOperator
from pylot.localization.ndt_autoware_operator import NDTAutowareOperator
from pylot.perception.messages import ObstacleTrajectoriesMessage, \
    ObstaclesMessage, TrafficLightsMessage
from pylot.planning.messages import WaypointsMessage
from pylot.planning.waypoints import Waypoints

FLAGS = flags.FLAGS

flags.DEFINE_string('waypoints_csv_file', '',
                    'Path to the file storing the waypoints csv file')
flags.DEFINE_bool('drive_by_wire', False,
                  'True to enable sending commands to the car')
flags.DEFINE_integer('sensor_frequency', 10,
                     'Frequency at which to process sensors')

# The location of the center camera relative to the ego-vehicle.
LEFT_CAMERA_LOCATION = pylot.utils.Location(1.5, 0.0, 1.4)
RIGHT_CAMERA_LOCATION = pylot.utils.Location(1.5, 0.0, 1.4)
VELODYNE_LOCATION = pylot.utils.Location(1.47, 0, 1.8)


def add_grasshopper3_camera(transform,
                            name='grasshopper3',
                            topic_name='/pg_0/image_color',
                            fov=72):
    camera_setup = pylot.drivers.sensor_setup.RGBCameraSetup(
        name, FLAGS.camera_image_width, FLAGS.camera_image_height, transform,
        fov)
    op_config = erdos.OperatorConfig(name=camera_setup.get_name() +
                                     '_operator',
                                     flow_watermarks=False,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [camera_stream] = erdos.connect(Grasshopper3DriverOperator, op_config, [],
                                    camera_setup, topic_name, FLAGS)
    return (camera_stream, camera_setup)


def add_velodyne_lidar(transform, name='velodyne', topic_name='/points_raw'):
    lidar_setup = pylot.drivers.sensor_setup.LidarSetup(name=name,
                                                        lidar_type='velodyne',
                                                        transform=transform)
    op_config = erdos.OperatorConfig(name=lidar_setup.get_name() + '_operator',
                                     flow_watermarks=False,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [point_cloud_stream] = erdos.connect(VelodyneDriverOperator, op_config, [],
                                         lidar_setup, topic_name, FLAGS)
    return (point_cloud_stream, lidar_setup)


def add_localization():
    op_config = erdos.OperatorConfig(name='ndt_localizer_operator',
                                     flow_watermarks=False,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [pose_stream] = erdos.connect(NDTAutowareOperator, op_config, [], FLAGS)
    return pose_stream


def add_drive_by_wire_operator(control_stream):
    op_config = erdos.OperatorConfig(name='drive_by_wire_operator',
                                     flow_watermarks=False,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(DriveByWireOperator, op_config, [control_stream], FLAGS)


def create_data_flow():
    left_camera_transform = pylot.utils.Transform(LEFT_CAMERA_LOCATION,
                                                  pylot.utils.Rotation())
    right_camera_transform = pylot.utils.Transform(RIGHT_CAMERA_LOCATION,
                                                   pylot.utils.Rotation())
    velodyne_transform = pylot.utils.Transform(VELODYNE_LOCATION,
                                               pylot.utils.Rotation())

    streams_to_send_top_on = []
    time_to_decision_loop_stream = erdos.LoopStream()

    (left_camera_stream, left_camera_setup) = add_grasshopper3_camera(
        left_camera_transform,
        name='left_grasshopper',
        topic_name='/pg_0/image_color')

    (right_camera_stream, right_camera_setup) = add_grasshopper3_camera(
        right_camera_transform,
        name='right_grasshopper',
        topic_name='/pg_1/image_color')

    (point_cloud_stream,
     lidar_setup) = add_velodyne_lidar(velodyne_transform,
                                       topic_name='/points_raw')

    pose_stream = add_localization()

    if FLAGS.obstacle_detection:
        obstacles_streams, _ = pylot.operator_creator.add_obstacle_detection(
            left_camera_stream, time_to_decision_loop_stream)
        obstacles_stream = obstacles_streams[0]
        # Adds an operator that finds the world locations of the obstacles.
        obstacles_stream = pylot.operator_creator.add_obstacle_location_finder(
            obstacles_stream, point_cloud_stream, pose_stream,
            left_camera_stream, left_camera_setup)
    else:
        obstacles_stream = erdos.IngestStream()

    if FLAGS.traffic_light_detection:
        # The right camera is more likely to contain the traffic lights.
        traffic_lights_stream = \
            pylot.operator_creator.add_traffic_light_detector(
                right_camera_stream, time_to_decision_loop_stream)
        # Adds operator that finds the world locations of the traffic lights.
        traffic_lights_stream = \
            pylot.operator_creator.add_obstacle_location_finder(
                traffic_lights_stream, point_cloud_stream, pose_stream,
                right_camera_stream, right_camera_setup)
    else:
        traffic_lights_stream = erdos.IngestStream()

    if FLAGS.lane_detection:
        lane_detection_stream = pylot.component_creator.add_lane_detection(
            left_camera_stream)
    else:
        lane_detection_stream = erdos.IngestStream()
        streams_to_send_top_on.append(lane_detection_stream)

    if FLAGS.obstacle_tracking:
        obstacles_wo_history_tracking_stream = \
            pylot.operator_creator.add_obstacle_tracking(
                obstacles_stream,
                left_camera_stream,
                time_to_decision_loop_stream)
        obstacles_tracking_stream = \
            pylot.operator_creator.add_obstacle_location_history(
                obstacles_wo_history_tracking_stream, point_cloud_stream,
                pose_stream, left_camera_stream, left_camera_setup)
    else:
        obstacles_tracking_stream = erdos.IngestStream()

    if FLAGS.prediction:
        prediction_stream = pylot.operator_creator.add_linear_prediction(
            obstacles_tracking_stream, time_to_decision_loop_stream)
    else:
        prediction_stream = obstacles_stream

    open_drive_stream = erdos.IngestStream()
    global_trajectory_stream = erdos.IngestStream()

    waypoints_stream = pylot.component_creator.add_planning(
        None, pose_stream, prediction_stream, traffic_lights_stream,
        lane_detection_stream, open_drive_stream, global_trajectory_stream,
        time_to_decision_loop_stream)

    if FLAGS.control == 'pid':
        control_stream = pylot.operator_creator.add_pid_control(
            pose_stream, waypoints_stream)
    else:
        raise ValueError('Only PID control is currently supported')

    if FLAGS.drive_by_wire:
        add_drive_by_wire_operator(control_stream)

    control_display_stream = None
    if pylot.flags.must_visualize():
        control_display_stream, ingest_streams = \
            pylot.operator_creator.add_visualizer(
                pose_stream, camera_stream=left_camera_stream,
                tl_camera_stream=right_camera_stream,
                point_cloud_stream=point_cloud_stream,
                obstacles_stream=obstacles_stream,
                traffic_lights_stream=traffic_lights_stream,
                tracked_obstacles_stream=obstacles_tracking_stream,
                lane_detection_stream=lane_detection_stream,
                waypoints_stream=waypoints_stream)
        streams_to_send_top_on += ingest_streams

    time_to_decision_stream = pylot.operator_creator.add_time_to_decision(
        pose_stream, obstacles_stream)
    time_to_decision_loop_stream.set(time_to_decision_stream)

    return (obstacles_stream, traffic_lights_stream, obstacles_tracking_stream,
            open_drive_stream, global_trajectory_stream,
            control_display_stream, streams_to_send_top_on)


def main(argv):
    (obstacles_stream, traffic_lights_stream, obstacles_tracking_stream,
     open_drive_stream, global_trajectory_stream, control_display_stream,
     streams_to_send_top_on) = create_data_flow()
    # Run the data-flow.
    node_handle = erdos.run_async()
    signal.signal(signal.SIGINT, shutdown)

    # Send waypoints.
    waypoints = Waypoints.read_from_csv_file(FLAGS.waypoints_csv_file,
                                             FLAGS.target_speed)
    global_trajectory_stream.send(
        WaypointsMessage(
            erdos.Timestamp(coordinates=[0]), waypoints,
            pylot.planning.utils.BehaviorPlannerState.FOLLOW_WAYPOINTS))

    # Send top watermark on all streams that require it.
    top_msg = erdos.WatermarkMessage(erdos.Timestamp(is_top=True))
    open_drive_stream.send(top_msg)
    global_trajectory_stream.send(top_msg)
    for stream in streams_to_send_top_on:
        stream.send(top_msg)

    time_to_sleep = 1.0 / FLAGS.sensor_frequency
    count = 0
    try:
        while True:
            timestamp = erdos.Timestamp(coordinates=[count])
            if not FLAGS.obstacle_detection:
                obstacles_stream.send(ObstaclesMessage(timestamp, []))
                obstacles_stream.send(erdos.WatermarkMessage(timestamp))
            if not FLAGS.traffic_light_detection:
                traffic_lights_stream.send(TrafficLightsMessage(timestamp, []))
                traffic_lights_stream.send(erdos.WatermarkMessage(timestamp))
            if not FLAGS.obstacle_tracking:
                obstacles_tracking_stream.send(
                    ObstacleTrajectoriesMessage(timestamp, []))
                obstacles_tracking_stream.send(
                    erdos.WatermarkMessage(timestamp))
            count += 1
            if pylot.flags.must_visualize():
                import pygame
                from pygame.locals import K_n
                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.KEYUP:
                        if event.key == K_n:
                            control_display_stream.send(
                                erdos.Message(erdos.Timestamp(coordinates=[0]),
                                              event.key))
                    elif event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                    elif event.type == pygame.KEYDOWN:
                        if (event.key == pygame.K_c
                                and pygame.key.get_mods() & pygame.KMOD_CTRL):
                            raise KeyboardInterrupt

            # NOTE: We should offset sleep time by the time it takes to send
            # the messages.
            time.sleep(time_to_sleep)
    except KeyboardInterrupt:
        node_handle.shutdown()
        if pylot.flags.must_visualize():
            import pygame
            pygame.quit()


def shutdown(sig, frame):
    raise KeyboardInterrupt


if __name__ == '__main__':
    app.run(main)
