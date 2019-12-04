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


def add_depth_estimation(vehicle_id_stream, center_camera_setup):
    (left_camera_setup, right_camera_setup) = \
        pylot.simulation.sensor_setup.create_left_right_camera_setups(
            'camera',
            CENTER_CAMERA_LOCATION,
            FLAGS.carla_camera_image_width,
            FLAGS.carla_camera_image_height)
    left_camera_stream = pylot.operator_creator.add_camera_driver(
        vehicle_id_stream, left_camera_setup)
    right_camera_stream = pylot.operator_creator.add_camera_driver(
        vehicle_id_stream, right_camera_setup)
    depth_estimation_stream = pylot.operator_creator.add_depth_estimation(
        left_camera_stream, right_camera_stream, center_camera_setup)
    return depth_estimation_stream


def add_perception(center_camera_stream, traffic_light_camera_stream):

    if FLAGS.obj_detection:
        obstacles_stream = pylot.operator_creator.add_obstacle_detection(
            center_camera_stream)
        if FLAGS.evaluate_obj_detection:
            pylot.operator_creator.add_detection_evaluation(
                obstacles_stream,
                ground_obstacles_stream)

    if FLAGS.traffic_light_det:
        traffic_lights_stream = \
            pylot.operator_creator.add_traffic_light_detector(
                traffic_light_camera_stream)

    if FLAGS.lane_detection:
        lane_detection_stream = pylot.operator_creator.add_lane_detection(
            center_camera_stream)

    if FLAGS.segmentation:
        segmented_stream = pylot.operator_creator.add_segmentation(
            center_camera_stream)
        if FLAGS.evaluate_segmentation:
            pylot.operator_creator.add_segmentation_evaluation(
                ground_segmented_stream, segmented_stream)

    if FLAGS.obj_tracking:
        obstacles_tracking_stream = \
            pylot.operator_creator.add_obstacle_tracking(
                obstacles_stream,
                center_camera_stream)

    if FLAGS.fusion:
        if FLAGS.evaluate_fusion:
            pylot.operator_creator.add_fusion(
                can_bus_stream,
                obstacles_stream,
                depth_stream,
                ground_vehicles_stream)
        else:
            pylot.operator_creator.add_fusion(
                can_bus_stream,
                obstacles_stream,
                depth_stream)

    return (obstacles_stream,
            obstacles_tracking_stream,
            traffic_lights_stream,
            lane_detection_stream,
            segmented_stream)


def add_perfect_perception(center_camera_stream,
                           depth_camera_stream,
                           segmented_camera_stream,
                           can_bus_stream,
                           ground_pedestrians_stream,
                           ground_vehicles_stream,
                           ground_traffic_lights_stream,
                           ground_speed_limit_signs_stream,
                           ground_stop_signs_stream,
                           center_camera_setup):
    obstacles_stream = pylot.operator_creator.add_perfect_detector(
        depth_camera_stream,
        center_camera_stream,
        segmented_camera_stream,
        can_bus_stream,
        ground_pedestrians_stream,
        ground_vehicles_stream,
        ground_speed_limit_signs_stream,
        ground_stop_signs_stream,
        center_camera_setup)

    # The perfect traffic light detector uses the regular camera because
    # it is not sensitive to fov settings.
    traffic_lights_stream = \
        pylot.operator_creator.add_perfect_traffic_light_detector(
            ground_traffic_lights_stream,
            center_camera_stream,
            depth_camera_stream,
            segmented_camera_stream,
            can_bus_stream)
    
    lane_detection_stream = pylot.operator_creator.add_perfect_lane_detector(
        can_bus_stream)

    # TODO: The perfect tracker returns ego-vehicle (x,y,z) coordinates, while
    # our existing trackers use camera coordinates. Fix!
    obstacles_tracking_stream = pylot.operator_creator.add_perfect_tracking(
        ground_vehicles_stream, ground_pedestrians_stream, can_bus_stream)

    return (obstacles_stream,
            obstacles_tracking_stream,
            traffic_lights_stream,
            lane_detection_stream,
            segmented_camera_stream)


def add_prediction(obstacles_tracking_stream):
    if FLAGS.prediction_type == 'linear':
        prediction_stream = pylot.operator_creator.add_linear_prediction(
            obstacles_tracking_stream)
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


def add_control(can_bus_stream,
                obstacles_stream,
                traffic_lights_stream,
                waypoints_stream):
    if FLAGS.control_agent_operator == 'pylot':
        control_stream = pylot.operator_creator.add_pylot_agent(
            can_bus_stream,
            waypoints_stream,
            traffic_lights_stream,
            obstacles_stream,
            lidar_stream,
            open_drive_stream,
            depth_camera_stream,
            camera_setup)
    elif FLAGS.control_agent_operator == 'mpc':
        control_stream = pylot.operator_creator.add_mpc_agent(
            can_bus_stream,
            ground_pedestrians_stream,
            ground_vehicles_stream,
            traffic_lights_stream,
            ground_speed_limit_signs_stream,
            waypoints_stream)
    elif FLAGS.control_agent_operator == 'ground':
        control_stream = pylot.operator_creator.add_ground_agent(
            can_bus_stream,
            ground_pedestrians_stream,
            ground_vehicles_stream,
            traffic_lights_stream,
            ground_speed_limit_signs_stream,
            waypoints_stream)
    else:
        raise ValueError('Unexpected control_agent_operator {}'.format(
            FLAGS.control_agent_operator))
    return control_stream


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
    (depth_camera_stream,
     depth_camera_setup) = add_depth_camera(transform, vehicle_id_stream)
    (segmented_stream, _) = add_segmented_camera(transform, vehicle_id_stream)
    (point_cloud_stream, lidar_setup) = add_lidar(vehicle_id_stream)
    (traffic_light_camera_stream,
     traffic_light_camera_setup) = add_traffic_light_camera(
         transform, vehicle_id_stream)

    if FLAGS.depth_estimation:
        add_depth_estimation(vehicle_id_stream, rgb_camera_setup)

    if FLAGS.visualize_top_down_segmentation:
        (top_down_segmented_stream,
         top_down_segmetned_camera_setup) = add_top_down_segmented_camera(
             vehicle_id_stream)

    if FLAGS.use_perfect_perception:
        (obstacles_stream,
         obstacles_tracking_stream,
         traffic_lights_stream,
         lane_detection_stream,
         segmented_stream) = add_perfect_perception(
             center_camera_stream,
             depth_camera_stream,
             segmented_stream,
             can_bus_stream,
             ground_pedestrians_stream,
             ground_vehicles_stream,
             ground_traffic_lights_stream,
             ground_speed_limit_signs_stream,
             ground_stop_signs_stream,
             rgb_camera_setup)
    else:
        (obstacles_stream,
         obstacles_tracking_stream,
         traffic_lights_stream,
         lane_detection_stream,
         segmented_stream) = add_perception(
             center_camera_stream, traffic_light_camera_stream)

    if FLAGS.prediction:
        prediction_stream = add_prediction(obstacles_tracking_stream)

    # Add planning operators.
    goal_location = (234.269989014, 59.3300170898, 39.4306259155)
    waypoints_stream = add_planning(can_bus_stream,
                                    open_drive_stream,
                                    global_trajectory_stream,
                                    goal_location)

    # Add the behaviour planning and control operator.
    control_stream = add_control(can_bus_stream,
                                 obstacles_stream,
                                 traffic_lights_stream,
                                 waypoints_stream)

    pylot.operator_creator.add_visualizers(
        center_camera_stream,
        depth_camera_stream,
        point_cloud_stream,
        segmented_stream,
        top_down_segmented_stream,
        obstacles_tracking_stream,
        prediction_stream,
        top_down_segmetned_camera_setup)


def main(argv):
    erdust.run(driver)


if __name__ == '__main__':
    app.run(main)
