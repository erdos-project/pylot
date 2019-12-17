from absl import app
from absl import flags

import erdust

import pylot.config
import pylot.operator_creator
import pylot.simulation.utils

FLAGS = flags.FLAGS

CENTER_CAMERA_LOCATION = pylot.simulation.utils.Location(1.5, 0.0, 1.4)


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
                           tl_camera_stream,
                           tl_depth_camera_stream,
                           tl_segmented_camera_stream,
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
            tl_camera_stream,
            tl_depth_camera_stream,
            tl_segmented_camera_stream,
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
                 prediction_stream,
                 open_drive_stream,
                 global_trajectory_stream,
                 goal_location):
    if FLAGS.planning_type == 'single_waypoint':
        waypoints_stream = pylot.operator_creator.add_planning(
            can_bus_stream,
            open_drive_stream,
            global_trajectory_stream,
            goal_location)
    elif FLAGS.planning_type == 'multiple_waypoints':
        waypoints_stream = pylot.operator_creator.add_waypoint_planning(
            can_bus_stream,
            goal_location)
    elif FLAGS.planning_type == 'rrt_star':
        waypoints_stream = pylot.operator_creator.add_rrt_start_planning(
            can_bus_stream, prediction_stream, goal_location)
    else:
        raise ValueError('Unexpected planning_type {}'.format(
            FLAGS.planning_type))
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
     rgb_camera_setup) = pylot.operator_creator.add_rgb_camera(
         transform, vehicle_id_stream)
    (depth_camera_stream,
     depth_camera_setup) = pylot.operator_creator.add_depth_camera(
         transform, vehicle_id_stream)
    (segmented_stream, _) = pylot.operator_creator.add_segmented_camera(
        transform, vehicle_id_stream)
    # Place Lidar sensor in the same location as the center camera.
    (point_cloud_stream, lidar_setup) = pylot.operator_creator.add_lidar(
        transform, vehicle_id_stream)
    (tl_camera_stream,
     tl_camera_setup) = pylot.operator_creator.add_rgb_camera(
         transform, vehicle_id_stream, 'traffic_light_camera', 45)
    (imu_stream, _) = pylot.operator_creator.add_imu(
        transform, vehicle_id_stream)

    if FLAGS.depth_estimation:
        (left_camera_stream,
         right_camera_stream) = pylot.operator_creator.add_left_right_cameras(
             transform, vehicle_id_stream)
        depth_estimation_stream = pylot.operator_creator.add_depth_estimation(
            left_camera_stream, right_camera_stream, rgb_camera_setup)

    if FLAGS.use_perfect_perception:
        # Add segmented and depth cameras with fov 45. These cameras are needed
        # by the perfect traffic light detector.
        (tl_depth_camera_stream, _) = pylot.operator_creator.add_depth_camera(
            transform, vehicle_id_stream, 'traffic_light_depth_camera', 45)
        (tl_segmented_camera_stream, _) = \
            pylot.operator_creator.add_segmented_camera(
                transform,
                vehicle_id_stream,
                'traffic_light_segmented_camera',
                45)
        (obstacles_stream,
         obstacles_tracking_stream,
         traffic_lights_stream,
         lane_detection_stream,
         segmented_stream) = add_perfect_perception(
             center_camera_stream,
             depth_camera_stream,
             segmented_stream,
             tl_camera_stream,
             tl_depth_camera_stream,
             tl_segmented_camera_stream,
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

    prediction_stream = None
    if FLAGS.prediction:
        prediction_stream = add_prediction(obstacles_tracking_stream)

    # Add planning operators.
    goal_location = (234.269989014, 59.3300170898, 39.4306259155)
    waypoints_stream = add_planning(can_bus_stream,
                                    prediction_stream,
                                    open_drive_stream,
                                    global_trajectory_stream,
                                    goal_location)

    # Add the behaviour planning and control operator.
    control_stream = add_control(can_bus_stream,
                                 obstacles_stream,
                                 traffic_lights_stream,
                                 waypoints_stream)

    if FLAGS.visualize_top_down_segmentation:
        top_down_transform = pylot.simulation.utils.get_top_down_transform(
            transform, FLAGS.top_down_lateral_view)
        (top_down_segmented_stream,
         top_down_segmetned_camera_setup) = \
            pylot.operator_creator.add_segmented_camera(
                top_down_transform,
                vehicle_id_stream,
                name='top_down_segmented_camera',
                fov=90)

    pylot.operator_creator.add_visualizers(
        center_camera_stream,
        depth_camera_stream,
        point_cloud_stream,
        segmented_stream,
        imu_stream,
        can_bus_stream,
        top_down_segmented_stream,
        obstacles_tracking_stream,
        prediction_stream,
        top_down_segmetned_camera_setup)


def main(argv):
    erdust.run(driver)


if __name__ == '__main__':
    app.run(main)
