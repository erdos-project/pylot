from absl import app
from absl import flags

import erdust

import pylot.config
import pylot.operator_creator
from pylot.simulation.sensor_setup import DepthCameraSetup, RGBCameraSetup, \
    SegmentedCameraSetup
import pylot.simulation.utils

FLAGS = flags.FLAGS

CENTER_CAMERA_LOCATION = pylot.simulation.utils.Location(1.5, 0.0, 1.4)


def add_prediction(obstacle_tracking_stream):
    if FLAGS.prediction_type == 'linear':
        prediction_stream = pylot.operator_creator.add_linear_prediction(
            obstacle_tracking_stream)
    else:
        raise ValueError('Unexpected prediction_type {}'.format(
            FLAGS.prediction_type))
    return prediction_stream


def add_planning(can_bus_stream,
                 open_drive_stream,
                 global_trajectory_stream,
                 goal_location):
    if FLAGS.waypoint_planning_operator:
        waypoints_stream = pylot.operator_creator.add_waypoint_planning(
            can_bus_stream,
            goal_location)
    else:
        waypoints_stream = pylot.operator_creator.add_planning(
            can_bus_stream,
            open_drive_stream,
            global_trajectory_stream,
            goal_location)
    if FLAGS.visualize_waypoints:
        pylot.operator_creator.add_waypoint_visualizer(waypoints_stream)
    return waypoints_stream


def add_rgb_camera(transform, vehicle_id_stream, name='center_rgb_camera'):
    rgb_camera_setup = RGBCameraSetup(name,
                                      FLAGS.carla_camera_image_width,
                                      FLAGS.carla_camera_image_height,
                                      transform)
    camera_stream = pylot.operator_creator.add_camera_driver(
        vehicle_id_stream, rgb_camera_setup)
    return (camera_stream, rgb_camera_setup)


def add_depth_camera(transform, vehicle_id_stream, name='center_depth_camera'):
    depth_camera_setup = DepthCameraSetup(name,
                                          FLAGS.carla_camera_image_width,
                                          FLAGS.carla_camera_image_height,
                                          transform)
    ground_depth_camera_stream = pylot.operator_creator.add_camera_driver(
        vehicle_id_stream, depth_camera_setup)
    return (ground_depth_camera_stream, depth_camera_setup)


def add_segmented_camera(transform,
                         vehicle_id_stream,
                         name='center_segmented_camera'):
    segmented_camera_setup = SegmentedCameraSetup(
        name,
        FLAGS.carla_camera_image_width,
        FLAGS.carla_camera_image_height,
        transform)
    ground_segmented_camera_stream = pylot.operator_creator.add_camera_driver(
        vehicle_id_stream, segmented_camera_setup)
    return (ground_segmented_camera_stream, segmented_camera_setup)


def add_traffic_light_camera(transform,
                             vehicle_id_stream,
                             name='center_traffic_light_rgb_camera'):
    traffic_light_camera_setup = RGBCameraSetup(
        name,
        FLAGS.carla_camera_image_width,
        FLAGS.carla_camera_image_height,
        transform,
        fov=45)
    traffic_light_camera_stream = pylot.operator_creator.add_camera_driver(
        vehicle_id_stream, traffic_light_camera_setup)
    return (traffic_light_camera_stream, traffic_light_camera_setup)


def add_lidar(vehicle_id_stream):
    # Place Lidar sensor in the same location as the center camera.
    lidar_setup = pylot.simulation.sensor_setup.create_center_lidar_setup(
        CENTER_CAMERA_LOCATION)
    point_cloud_stream = pylot.operator_creator.add_lidar_driver(
        vehicle_id_stream, lidar_setup)
    return (point_cloud_stream, lidar_setup)


def add_top_down_segmented_camera(vehicle_id_stream):
    top_down_segmented_camera_setup = \
        pylot.simulation.sensor_setup.create_top_down_segmented_camera_setup(
            'top_down',
            CENTER_CAMERA_LOCATION,
            FLAGS.carla_camera_image_width,
            FLAGS.carla_camera_image_height,
            FLAGS.top_down_lateral_view)
    top_down_segmented_stream = pylot.operator_creator.add_camera_driver(
        vehicle_id_stream, top_down_segmented_camera_setup)
    return (top_down_segmented_stream, top_down_segmented_camera_setup)

    
def driver():
    transform = pylot.simulation.utils.Transform(
        CENTER_CAMERA_LOCATION, pylot.simulation.utils.Rotation(0, 0, 0))

    # Create carla operator.
    (can_bus_stream,
     ground_traffic_lights_stream,
     ground_vehicles_stream,
     ground_pedestrians_stream,
     ground_speed_limit_signs_stream,
     ground_stop_signs_stream,
     vehicle_id_stream) = pylot.operator_creator.add_carla_bridge()

    # Add sensors.
    (center_camera_stream,
     rgb_camera_setup) = add_rgb_camera(transform, vehicle_id_stream)
    (depth_stream,
     depth_camera_setup) = add_depth_camera(transform, vehicle_id_stream)
    (segmented_stream, _) = add_segmented_camera(transform, vehicle_id_stream)
    (point_cloud_stream, lidar_setup) = add_lidar(vehicle_id_stream)
    (traffic_light_camera_stream,
     traffic_light_camera_setup) = add_traffic_light_camera(
         transform, vehicle_id_stream)
    # TODO(ionel): Check when we need to add top_down.
    (top_down_segmented_stream,
     top_down_camera_setup) = add_top_down_segmented_camera(vehicle_id_stream)


def main(argv):
    erdust.run(driver)


if __name__ == '__main__':
    app.run(main)
