from absl import app, flags
import erdos

from pylot.drivers.grasshopper3_driver_operator import \
    Grasshopper3DriverOperator
from pylot.drivers.velodyne_driver_operator import VelodyneDriverOperator
import pylot.flags
import pylot.operator_creator
import pylot.simulation.sensor_setup
import pylot.utils

FLAGS = flags.FLAGS

flags.DEFINE_integer('camera_image_width', 2736, 'Camera image width')
flags.DEFINE_integer('camera_image_height', 2192, 'Camera image height')

# The location of the center camera relative to the ego-vehicle.
CENTER_CAMERA_LOCATION = pylot.utils.Location(1.5, 0.0, 1.4)


def add_grasshopper3_camera(transform, name='grasshopper3', fov=90):
    camera_setup = pylot.simulation.sensor_setup.RGBCameraSetup(
        name, FLAGS.camera_image_width, FLAGS.camera_image_height, transform,
        fov)
    [camera_stream] = erdos.connect(Grasshopper3DriverOperator, [],
                                    False,
                                    camera_setup.get_name() + '_operator',
                                    camera_setup,
                                    FLAGS,
                                    log_file_name=FLAGS.log_file_name)
    return (camera_stream, camera_setup)


def add_velodyne_lidar(transform, name='velodyne'):
    lidar_setup = pylot.simulation.sensor_setup.LidarSetup(
        name=name, lidar_type='velodyne', transform=transform)
    [point_cloud_stream] = erdos.connect(VelodyneDriverOperator, [],
                                         False,
                                         lidar_setup.get_name() + '_operator',
                                         lidar_setup,
                                         FLAGS,
                                         log_file_name=FLAGS.log_file_name)
    return (point_cloud_stream, lidar_setup)


def driver():
    # TODO: Set the correct camera locations.
    transform = pylot.utils.Transform(CENTER_CAMERA_LOCATION,
                                      pylot.utils.Rotation())

    (left_camera_stream,
     left_camera_setup) = add_grasshopper3_camera(transform,
                                                  'left_grasshopper')

    (right_camera_stream,
     right_camera_setup) = add_grasshopper3_camera(transform,
                                                   'right_grasshopper')

    # TODO: Set the correct lidar location.
    (point_cloud_stream, lidar_setup) = add_velodyne_lidar(transform)

    obstacles_streams = pylot.operator_creator.add_obstacle_detection(
        left_camera_stream)
    obstacles_stream = obstacles_streams[0]

    # The right camera is more likely to contain the traffic lights.
    traffic_lights_stream = pylot.operator_creator.add_traffic_light_detector(
        right_camera_stream)

    lane_detection = pylot.operator_creator.add_canny_edge_lane_detection(
        left_camera_stream)

    obstacles_tracking_stream = pylot.operator_creator.add_obstacle_tracking(
        obstacles_stream, left_camera_stream)

    prediction_stream = pylot.operator_creator.add_linear_prediction(
        obstacles_tracking_stream)

    # waypoints_stream = pylot.operator_creator.add_fot_planning(
    #     can_bus_stream, prediction_stream, goal_location)

    # Add visualizers.
    pylot.operator_creator.add_camera_visualizer(left_camera_stream,
                                                 'left_grasshopper3_camera')


def main(argv):
    erdos.run(driver)


if __name__ == '__main__':
    app.run(main)
