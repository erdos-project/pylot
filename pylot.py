from absl import app, flags
import carla
import erdos

import pylot.flags
import pylot.operator_creator
import pylot.simulation.utils
import pylot.utils

FLAGS = flags.FLAGS

flags.DEFINE_list('goal_location', '234, 59, 39', 'Ego-vehicle goal location')

# The location of the center camera relative to the ego-vehicle.
CENTER_CAMERA_LOCATION = pylot.utils.Location(1.5, 0.0, 1.4)


def add_obstacle_detection(center_camera_stream,
                           can_bus_stream=None,
                           depth_camera_stream=None,
                           segmented_camera_stream=None,
                           ground_obstacles_stream=None,
                           ground_speed_limit_signs_stream=None,
                           ground_stop_signs_stream=None):
    """ Adds operators for obstacle detection.

    Depending on how flags are set, the operators are either perfect or using
    trained models.

    Args:
        center_camera_stream: A stream on which BGR frames are published.
    Returns:
        A stream on which obstacles are published.
    """
    obstacles_stream = None
    if FLAGS.obstacle_detection:
        # TODO: Only returns the first obstacles stream.
        obstacles_streams = pylot.operator_creator.add_obstacle_detection(
            center_camera_stream)
        obstacles_stream = obstacles_streams[0]
    if FLAGS.perfect_obstacle_detection or FLAGS.evaluate_obstacle_detection:
        assert (can_bus_stream is not None and depth_camera_stream is not None
                and segmented_camera_stream is not None
                and ground_obstacles_stream is not None
                and ground_speed_limit_signs_stream is not None
                and ground_stop_signs_stream is not None)
        perfect_obstacles_stream = pylot.operator_creator.add_perfect_detector(
            depth_camera_stream, center_camera_stream, segmented_camera_stream,
            can_bus_stream, ground_obstacles_stream,
            ground_speed_limit_signs_stream, ground_stop_signs_stream)
        if FLAGS.evaluate_obstacle_detection:
            pylot.operator_creator.add_detection_evaluation(
                obstacles_stream, perfect_obstacles_stream)
        else:
            obstacles_stream = perfect_obstacles_stream
    return obstacles_stream


def add_traffic_light_detection(tl_transform,
                                vehicle_id_stream,
                                can_bus_stream=None,
                                ground_traffic_lights_stream=None):
    """ Adds traffic light detection operators.

    Depending on how flags are set, the function adds a perfect traffic light
    detector or a detector that uses a trained model. The detector uses a
    camera with a narrow fov.

    Args:
        tl_transform: A pylot.utils.Transform of the camera relative to
             the ego vehicle.
        vehicle_id_stream: A stream on which the Carla ego-vehicle id is
             published.
    Returns:
        A stream on which traffic lights are published.
    """
    if FLAGS.traffic_light_detection or FLAGS.perfect_traffic_light_detection:
        # Only add the TL camera if traffic light detection is enabled.
        (tl_camera_stream,
         tl_camera_setup) = pylot.operator_creator.add_rgb_camera(
             tl_transform, vehicle_id_stream, 'traffic_light_camera', 45)

    traffic_lights_stream = None
    if FLAGS.traffic_light_detection:
        traffic_lights_stream = \
            pylot.operator_creator.add_traffic_light_detector(tl_camera_stream)
    elif FLAGS.perfect_traffic_light_detection:
        assert (can_bus_stream is not None
                and ground_traffic_lights_stream is not None)
        # Add segmented and depth cameras with fov 45. These cameras are needed
        # by the perfect traffic light detector.
        (tl_depth_camera_stream, _) = pylot.operator_creator.add_depth_camera(
            tl_transform, vehicle_id_stream, 'traffic_light_depth_camera', 45)
        (tl_segmented_camera_stream, _) = \
            pylot.operator_creator.add_segmented_camera(
                tl_transform,
                vehicle_id_stream,
                'traffic_light_segmented_camera',
                45)

        traffic_lights_stream = \
            pylot.operator_creator.add_perfect_traffic_light_detector(
                ground_traffic_lights_stream,
                tl_camera_stream,
                tl_depth_camera_stream,
                tl_segmented_camera_stream,
                can_bus_stream)
    return traffic_lights_stream


def add_lane_detection(center_camera_stream, can_bus_stream=None):
    """ Adds operators for lane detection.

    Depending on how flags are set, the operators are using ground information
    (i.e., perfect) or are using only sensor data.

    Args:
        A stream on which BGR frames are published.
    Returns:
        A stream on which lanes are published.
    """
    lane_detection_stream = None
    if FLAGS.lane_detection:
        lane_detection_stream = pylot.operator_creator.add_lane_detection(
            center_camera_stream)
    elif FLAGS.perfect_lane_detection:
        assert can_bus_stream is not None
        lane_detection_stream = \
            pylot.operator_creator.add_perfect_lane_detector(can_bus_stream)
    return lane_detection_stream


def add_obstacle_tracking(center_camera_stream,
                          obstacles_stream,
                          can_bus_stream=None,
                          ground_obstacles_stream=None):
    """ Adds operators for obstacle tracking.

    Depending on how flags are set, the operators either use ground information
    or sensor data and outputs from obstacle detection operators.

    Args:
        center_camera_stream: A stream on which BGR frames are published.
        obstacles_stream: A stream on which obstacles are published.
    Returns:
        A stream on which tracked obstacles are published.
    """
    obstacles_tracking_stream = None
    if FLAGS.obstacle_tracking:
        obstacles_tracking_stream = \
            pylot.operator_creator.add_obstacle_tracking(
                obstacles_stream,
                center_camera_stream)
    elif FLAGS.perfect_obstacle_tracking:
        assert (can_bus_stream is not None
                and ground_obstacles_stream is not None)
        # TODO: The perfect tracker returns ego-vehicle (x,y,z) coordinates,
        # while our existing trackers use camera coordinates. Fix!
        obstacles_tracking_stream = \
            pylot.operator_creator.add_perfect_tracking(
                ground_obstacles_stream, can_bus_stream)
    return obstacles_tracking_stream


def add_segmentation(center_camera_stream, ground_segmented_stream=None):
    """ Adds operators for pixel semantic segmentation.

    Depending on how flags are set, the returned stream is either the perfectly
    sematically segmented ground camera, or a stream output from an operator
    that uses a trained model.

    Args:
        center_camera_stream: A stream on which BGR frames are published.
    Returns:
        A stream on which semanically segmented frames are published
    """
    segmented_stream = None
    if FLAGS.segmentation:
        segmented_stream = pylot.operator_creator.add_segmentation(
            center_camera_stream)
        if FLAGS.evaluate_segmentation:
            assert ground_segmented_stream is not None
            pylot.operator_creator.add_segmentation_evaluation(
                ground_segmented_stream, segmented_stream)
    elif FLAGS.perfect_segmentation:
        assert ground_segmented_stream is not None
        return ground_segmented_stream
    return segmented_stream


def add_depth(transform, vehicle_id_stream, center_camera_setup,
              depth_camera_stream):
    depth_stream = None
    if FLAGS.depth_estimation:
        (left_camera_stream,
         right_camera_stream) = pylot.operator_creator.add_left_right_cameras(
             transform, vehicle_id_stream)
        depth_stream = pylot.operator_creator.add_depth_estimation(
            left_camera_stream, right_camera_stream, center_camera_setup)
    elif FLAGS.perfect_depth_estimation:
        depth_stream = depth_camera_stream
    return depth_stream


def add_prediction(obstacles_tracking_stream,
                   vehicle_id_stream,
                   camera_transform,
                   can_bus_stream=None):
    """ Adds prediction operators.

    Args:
        obstacles_tracking_stream: A stream of tracked obstacles.
    Returns:
        A stream on which obstacle predictions are published.
    """
    prediction_stream = None
    if FLAGS.prediction:
        if FLAGS.prediction_type == 'linear':
            prediction_stream = pylot.operator_creator.add_linear_prediction(
                obstacles_tracking_stream)
        else:
            raise ValueError('Unexpected prediction_type {}'.format(
                FLAGS.prediction_type))
        if FLAGS.evaluate_prediction:
            assert can_bus_stream is not None
            pylot.operator_creator.add_prediction_evaluation(
                can_bus_stream, obstacles_tracking_stream, prediction_stream)
        if FLAGS.visualize_prediction:
            pylot.operator_creator.add_prediction_visualizer(
                obstacles_tracking_stream, prediction_stream,
                vehicle_id_stream, camera_transform)
    return prediction_stream


def add_planning(goal_location,
                 can_bus_stream,
                 prediction_stream,
                 camera_stream,
                 open_drive_stream=None,
                 global_trajectory_stream=None):
    """ Adds planning operators.

    Args:
        goal_location: A carla.Location representing the destination.
        can_bus_stream: A stream of ego-vehicle CanBus messages.
        prediction_stream: A stream of obstacles prediction messages.
        open_drive_stream: A stream on which open drive string representations
            are published. Operators can construct HDMaps out of the open drive
            strings.
    Returns:
        A stream on which waypoints are published.
    """
    if FLAGS.planning_type == 'waypoint':
        assert (open_drive_stream is not None
                and global_trajectory_stream is not None)
        waypoints_stream = pylot.operator_creator.add_waypoint_planning(
            can_bus_stream, open_drive_stream, global_trajectory_stream,
            goal_location)
    elif FLAGS.planning_type == 'rrt_star':
        waypoints_stream = pylot.operator_creator.add_rrt_start_planning(
            can_bus_stream, prediction_stream, goal_location)
    else:
        raise ValueError('Unexpected planning_type {}'.format(
            FLAGS.planning_type))
    if FLAGS.visualize_waypoints:
        pylot.operator_creator.add_waypoint_visualizer(waypoints_stream,
                                                       camera_stream,
                                                       can_bus_stream)
    return waypoints_stream


def add_control(can_bus_stream, obstacles_stream, traffic_lights_stream,
                waypoints_stream, open_drive_stream, point_cloud_stream,
                ground_obstacles_stream, ground_traffic_lights_stream,
                camera_setup):
    if FLAGS.control_agent == 'pylot':
        control_stream = pylot.operator_creator.add_pylot_agent(
            can_bus_stream, waypoints_stream, traffic_lights_stream,
            obstacles_stream, point_cloud_stream, open_drive_stream,
            camera_setup)
    elif FLAGS.control_agent == 'mpc':
        control_stream = pylot.operator_creator.add_mpc_agent(
            can_bus_stream, ground_obstacles_stream,
            ground_traffic_lights_stream, waypoints_stream)
    elif FLAGS.control_agent == 'ground':
        control_stream = pylot.operator_creator.add_ground_agent(
            can_bus_stream, ground_obstacles_stream,
            ground_traffic_lights_stream, waypoints_stream)
    elif FLAGS.control_agent == 'carla_auto_pilot':
        # TODO: Hack! We synchronize on a single stream, based on a
        # guesestimate of which stream is slowest.
        stream_to_sync_on = ground_obstacles_stream
        if obstacles_stream is not None:
            stream_to_sync_on = obstacles_stream
        elif traffic_lights_stream is not None:
            stream_to_sync_on = traffic_lights_stream
        elif waypoints_stream is not None:
            stream_to_sync_on = waypoints_stream
        control_stream = pylot.operator_creator.add_synchronizer(
            stream_to_sync_on)
    else:
        raise ValueError('Unexpected control_agent {}'.format(
            FLAGS.control_agent))
    return control_stream


def driver():
    transform = pylot.utils.Transform(CENTER_CAMERA_LOCATION,
                                      pylot.utils.Rotation())

    control_loop_stream = erdos.LoopStream()
    # Create carla operator.
    (can_bus_stream, ground_traffic_lights_stream, ground_obstacles_stream,
     ground_speed_limit_signs_stream, ground_stop_signs_stream,
     vehicle_id_stream, open_drive_stream, global_trajectory_stream
     ) = pylot.operator_creator.add_carla_bridge(control_loop_stream)

    # Add sensors.
    (center_camera_stream,
     rgb_camera_setup) = pylot.operator_creator.add_rgb_camera(
         transform, vehicle_id_stream)
    (depth_camera_stream,
     depth_camera_setup) = pylot.operator_creator.add_depth_camera(
         transform, vehicle_id_stream)
    (ground_segmented_stream,
     _) = pylot.operator_creator.add_segmented_camera(transform,
                                                      vehicle_id_stream)
    # Place Lidar sensor in the same location as the center camera.
    (point_cloud_stream,
     lidar_setup) = pylot.operator_creator.add_lidar(transform,
                                                     vehicle_id_stream)

    imu_stream = None
    if FLAGS.imu:
        (imu_stream,
         _) = pylot.operator_creator.add_imu(transform, vehicle_id_stream)

    obstacles_stream = add_obstacle_detection(
        center_camera_stream, can_bus_stream, depth_camera_stream,
        ground_segmented_stream, ground_obstacles_stream,
        ground_speed_limit_signs_stream, ground_stop_signs_stream)
    traffic_lights_stream = add_traffic_light_detection(
        transform, vehicle_id_stream, can_bus_stream,
        ground_traffic_lights_stream)

    lane_detection_stream = add_lane_detection(center_camera_stream,
                                               can_bus_stream)

    obstacles_tracking_stream = add_obstacle_tracking(center_camera_stream,
                                                      obstacles_stream,
                                                      can_bus_stream,
                                                      ground_obstacles_stream)

    segmented_stream = add_segmentation(center_camera_stream,
                                        ground_segmented_stream)

    depth_stream = add_depth(transform, vehicle_id_stream, rgb_camera_setup,
                             depth_camera_stream)

    if FLAGS.fusion:
        pylot.operator_creator.add_fusion(can_bus_stream, obstacles_stream,
                                          depth_stream,
                                          ground_obstacles_stream)

    prediction_stream = add_prediction(obstacles_tracking_stream,
                                       vehicle_id_stream, transform,
                                       can_bus_stream)

    # Add planning operators.
    goal_location = carla.Location(float(FLAGS.goal_location[0]),
                                   float(FLAGS.goal_location[1]),
                                   float(FLAGS.goal_location[2]))
    waypoints_stream = add_planning(goal_location, can_bus_stream,
                                    prediction_stream, center_camera_stream,
                                    open_drive_stream,
                                    global_trajectory_stream)

    # TODO: Merge depth camera stream and point cloud stream.
    # Add the behaviour planning and control operator.
    control_stream = add_control(can_bus_stream, obstacles_stream,
                                 traffic_lights_stream, waypoints_stream,
                                 open_drive_stream, point_cloud_stream,
                                 ground_obstacles_stream,
                                 ground_traffic_lights_stream,
                                 rgb_camera_setup)
    control_loop_stream.set(control_stream)

    pylot.operator_creator.add_sensor_visualizers(center_camera_stream,
                                                  depth_camera_stream,
                                                  point_cloud_stream,
                                                  segmented_stream, imu_stream,
                                                  can_bus_stream)


def main(argv):
    erdos.run(driver)


if __name__ == '__main__':
    app.run(main)
