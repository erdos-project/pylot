from absl import app, flags
import csv
import erdos
import sys
import time

from pylot.drivers.drive_by_wire_operator import DriveByWireOperator
from pylot.drivers.grasshopper3_driver_operator import \
    Grasshopper3DriverOperator
import pylot.drivers.sensor_setup
from pylot.drivers.velodyne_driver_operator import VelodyneDriverOperator
import pylot.flags
from pylot.localization.ndt_autoware_operator import NDTAutowareOperator
import pylot.operator_creator
from pylot.perception.messages import ObstaclesMessage, \
    ObstacleTrajectoriesMessage, TrafficLightsMessage
import pylot.utils

FLAGS = flags.FLAGS

flags.DEFINE_integer('camera_image_width', 2048, 'Camera image width')
flags.DEFINE_integer('camera_image_height', 2048, 'Camera image height')
flags.DEFINE_integer('track', 3, 'Track to execute')
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
                            fov=90):
    camera_setup = pylot.drivers.sensor_setup.RGBCameraSetup(
        name, FLAGS.camera_image_width, FLAGS.camera_image_height, transform,
        fov)
    [camera_stream] = erdos.connect(Grasshopper3DriverOperator, [],
                                    False,
                                    camera_setup.get_name() + '_operator',
                                    camera_setup,
                                    topic_name,
                                    FLAGS,
                                    log_file_name=FLAGS.log_file_name,
                                    csv_file_name=FLAGS.csv_log_file_name)
    return (camera_stream, camera_setup)


def add_velodyne_lidar(transform, name='velodyne', topic_name='/points_raw'):
    lidar_setup = pylot.drivers.sensor_setup.LidarSetup(name=name,
                                                        lidar_type='velodyne',
                                                        transform=transform)
    [point_cloud_stream] = erdos.connect(VelodyneDriverOperator, [],
                                         False,
                                         lidar_setup.get_name() + '_operator',
                                         lidar_setup,
                                         topic_name,
                                         FLAGS,
                                         log_file_name=FLAGS.log_file_name,
                                         csv_file_name=FLAGS.csv_log_file_name)
    return (point_cloud_stream, lidar_setup)


def add_localization():
    [can_bus_stream] = erdos.connect(NDTAutowareOperator, [],
                                     False,
                                     'ndt_localizer_operator',
                                     FLAGS,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_file_name=FLAGS.csv_log_file_name)
    return can_bus_stream


def add_drive_by_wire_operator(control_stream):
    erdos.connect(DriveByWireOperator, [control_stream],
                  False,
                  'drive_by_wire_operator',
                  FLAGS,
                  log_file_name=FLAGS.log_file_name,
                  csv_file_name=FLAGS.csv_log_file_name)


def create_data_flow():
    left_camera_transform = pylot.utils.Transform(LEFT_CAMERA_LOCATION,
                                                  pylot.utils.Rotation())
    right_camera_transform = pylot.utils.Transform(RIGHT_CAMERA_LOCATION,
                                                   pylot.utils.Rotation())
    velodyne_transform = pylot.utils.Transform(VELODYNE_LOCATION,
                                               pylot.utils.Rotation())

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

    can_bus_stream = add_localization()

    if FLAGS.obstacle_detection:
        obstacles_streams = pylot.operator_creator.add_obstacle_detection(
            left_camera_stream)
        obstacles_stream = obstacles_streams[0]
        # Adds an operator that finds the world locations of the obstacles.
        obstacles_stream = pylot.operator_creator.add_obstacle_location_finder(
            obstacles_stream, point_cloud_stream, can_bus_stream,
            left_camera_setup)
    else:
        obstacles_stream = erdos.IngestStream()

    if FLAGS.traffic_light_detection:
        # The right camera is more likely to contain the traffic lights.
        traffic_lights_stream = pylot.operator_creator.add_traffic_light_detector(
            right_camera_stream)
        # Adds operator that finds the world locations of the traffic lights.
        traffic_lights_stream = \
            pylot.operator_creator.add_obstacle_location_finder(
                traffic_lights_stream, point_cloud_stream, can_bus_stream,
                right_camera_setup)
    else:
        traffic_lights_stream = erdos.IngestStream()

    if FLAGS.lane_detection:
        lane_detection = pylot.operator_creator.add_canny_edge_lane_detection(
            left_camera_stream)

    if FLAGS.obstacle_tracking:
        obstacles_tracking_stream = pylot.operator_creator.add_obstacle_tracking(
            obstacles_stream, left_camera_stream)
    else:
        obstacles_tracking_stream = erdos.IngestStream()

    if FLAGS.prediction:
        prediction_stream = pylot.operator_creator.add_linear_prediction(
            obstacles_tracking_stream)

    open_drive_stream = erdos.IngestStream()
    global_trajectory_stream = erdos.IngestStream()

    if FLAGS.planning_type == 'waypoint':
        waypoints_stream = pylot.operator_creator.add_waypoint_planning(
            can_bus_stream, open_drive_stream, global_trajectory_stream,
            obstacles_stream, traffic_lights_stream, None)
    else:
        raise ValueError('Only waypoint planning is currently supported')

    if FLAGS.control_agent == 'pid':
        control_stream = pylot.operator_creator.add_pid_agent(
            can_bus_stream, waypoints_stream)
    else:
        raise ValueError('Only PID control is currently supported')

    if FLAGS.drive_by_wire:
        add_drive_by_wire_operator(control_stream)

    # Add visualizers.
    if FLAGS.visualize_rgb_camera:
        pylot.operator_creator.add_camera_visualizer(
            left_camera_stream, 'left_grasshopper3_camera')
    if FLAGS.visualize_waypoints:
        pylot.operator_creator.add_waypoint_visualizer(waypoints_stream,
                                                       left_camera_stream,
                                                       can_bus_stream)

    return (obstacles_stream, traffic_lights_stream, obstacles_tracking_stream,
            open_drive_stream, global_trajectory_stream)


def read_waypoints():
    csv_file = open(FLAGS.waypoints_csv_file)
    csv_reader = csv.reader(csv_file)
    waypoints = []
    for row in csv_reader:
        x = float(row[0])
        y = float(row[1])
        z = float(row[2])
        waypoint = pylot.utils.Transform(pylot.utils.Location(x, y, z),
                                         pylot.utils.Rotation(0, 0, 0))
        waypoints.append(waypoint)
    return waypoints


def main(argv):
    (obstacles_stream, traffic_lights_stream, obstacles_tracking_stream,
     open_drive_stream, global_trajectory_stream) = create_data_flow()
    # Run the data-flow.
    erdos.run_async()

    top_timestamp = erdos.Timestamp(coordinates=[sys.maxsize])
    open_drive_stream.send(erdos.WatermarkMessage(top_timestamp))

    waypoints = [[waypoint] for waypoint in read_waypoints()]
    global_trajectory_stream.send(
        erdos.Message(erdos.Timestamp(coordinates=[0]), waypoints))
    global_trajectory_stream.send(erdos.WatermarkMessage(top_timestamp))

    time_to_sleep = 1.0 / FLAGS.sensor_frequency
    count = 0
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
            obstacles_tracking_stream.send(erdos.WatermarkMessage(timestamp))
        count += 1
        # NOTE: We should offset sleep time by the time it takes to send the
        # messages.
        time.sleep(time_to_sleep)


if __name__ == '__main__':
    app.run(main)
