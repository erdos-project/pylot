from absl import flags

import copy
import erdos

import pylot.utils

FLAGS = flags.FLAGS


def add_carla_bridge(control_stream, sensor_ready_stream):
    from pylot.simulation.carla_operator import CarlaOperator
    op_config = erdos.OperatorConfig(name='carla_operator',
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    return erdos.connect(CarlaOperator, op_config,
                         [control_stream, sensor_ready_stream], FLAGS)


def add_efficientdet_obstacle_detection(camera_stream,
                                        time_to_decision_stream,
                                        csv_file_name=None):
    from pylot.perception.detection.efficientdet_operator import \
            EfficientDetOperator
    if csv_file_name is None:
        csv_file_name = FLAGS.csv_log_file_name
    op_config = erdos.OperatorConfig(name='efficientdet_operator',
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=csv_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    obstacles_streams = erdos.connect(EfficientDetOperator, op_config,
                                      [camera_stream, time_to_decision_stream],
                                      FLAGS.obstacle_detection_model_names,
                                      FLAGS.obstacle_detection_model_paths,
                                      FLAGS)
    return obstacles_streams


def add_obstacle_detection(camera_stream,
                           time_to_decision_stream,
                           csv_file_name=None):
    from pylot.perception.detection.detection_operator import DetectionOperator
    obstacles_streams = []
    if csv_file_name is None:
        csv_file_name = FLAGS.csv_log_file_name
    for i in range(0, len(FLAGS.obstacle_detection_model_paths)):
        op_config = erdos.OperatorConfig(
            name=FLAGS.obstacle_detection_model_names[i],
            log_file_name=FLAGS.log_file_name,
            csv_log_file_name=csv_file_name,
            profile_file_name=FLAGS.profile_file_name)
        obstacles_streams += erdos.connect(
            DetectionOperator, op_config,
            [camera_stream, time_to_decision_stream],
            FLAGS.obstacle_detection_model_paths[i], FLAGS)
    return obstacles_streams


def add_obstacle_location_finder(obstacles_stream, depth_stream, pose_stream,
                                 camera_stream, camera_setup):
    """Adds an operator that finds the world locations of the obstacles.

    Args:
        obstacles_stream (:py:class:`erdos.ReadStream`): Stream on which
            detected obstacles are received.
        depth_stream (:py:class:`erdos.ReadStream`): Stream on which
            either point cloud messages or depth frames are received. The
            message type differs dependening on how data-flow operators are
            connected.
        pose_stream (:py:class:`erdos.ReadStream`, optional): Stream on
            which pose info is received.
        camera_stream (:py:class:`erdos.ReadStream`): The stream on which
            camera frames are received.
        camera_setup (:py:class:`~pylot.drivers.sensor_setup.CameraSetup`):
            The setup of the center camera.

    Returns:
        :py:class:`erdos.ReadStream`: Stream on which
        :py:class:`~pylot.perception.messages.ObstaclesMessage` messages with
        world locations are published.
    """
    from pylot.perception.detection.obstacle_location_finder_operator import \
        ObstacleLocationFinderOperator
    op_config = erdos.OperatorConfig(name=camera_setup.get_name() +
                                     '_location_finder_operator',
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [obstacles_with_loc_stream] = erdos.connect(
        ObstacleLocationFinderOperator, op_config,
        [obstacles_stream, depth_stream, pose_stream, camera_stream], FLAGS,
        camera_setup)
    return obstacles_with_loc_stream


def add_obstacle_location_history(obstacles_stream, depth_stream, pose_stream,
                                  camera_stream, camera_setup):
    """Adds an operator that finds obstacle trajectories in world coordinates.

    Args:
        obstacles_stream (:py:class:`erdos.ReadStream`): Stream on which
            detected obstacles are received.
        depth_stream (:py:class:`erdos.ReadStream`): Stream on which
            either point cloud messages or depth frames are received. The
            message type differs dependening on how data-flow operators are
            connected.
        pose_stream (:py:class:`erdos.ReadStream`, optional): Stream on
            which pose info is received.
        camera_stream (:py:class:`erdos.ReadStream`): The stream on which
            camera frames are received.
        camera_setup (:py:class:`~pylot.drivers.sensor_setup.CameraSetup`):
            The setup of the center camera.

    Returns:
        :py:class:`erdos.ReadStream`: Stream on which
        :py:class:`~pylot.perception.messages.ObstacleTrajectoriesMessage`
        messages are published.
    """
    from pylot.perception.tracking.obstacle_location_history_operator import \
        ObstacleLocationHistoryOperator
    op_config = erdos.OperatorConfig(name=camera_setup.get_name() +
                                     '_location_finder_history_operator',
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [tracked_obstacles] = erdos.connect(
        ObstacleLocationHistoryOperator, op_config,
        [obstacles_stream, depth_stream, pose_stream, camera_stream], FLAGS,
        camera_setup)
    return tracked_obstacles


def add_detection_decay(ground_obstacles_stream):
    from pylot.perception.detection.detection_decay_operator import \
        DetectionDecayOperator
    op_config = erdos.OperatorConfig(name='detection_decay_operator',
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [map_stream] = erdos.connect(DetectionDecayOperator, op_config,
                                 [ground_obstacles_stream], FLAGS)
    return map_stream


def add_detection_evaluation(obstacles_stream,
                             ground_obstacles_stream,
                             name='detection_eval_operator'):
    from pylot.perception.detection.detection_eval_operator import \
        DetectionEvalOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(DetectionEvalOperator, op_config,
                  [obstacles_stream, ground_obstacles_stream], FLAGS)


def add_control_evaluation(pose_stream,
                           waypoints_stream,
                           name='control_eval_operator'):
    from pylot.control.control_eval_operator import ControlEvalOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(ControlEvalOperator, op_config,
                  [pose_stream, waypoints_stream], FLAGS)


def add_traffic_light_detector(traffic_light_camera_stream):
    from pylot.perception.detection.traffic_light_det_operator import \
        TrafficLightDetOperator
    op_config = erdos.OperatorConfig(name='traffic_light_detector_operator',
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [traffic_lights_stream] = erdos.connect(TrafficLightDetOperator, op_config,
                                            [traffic_light_camera_stream],
                                            FLAGS)
    return traffic_lights_stream


def add_traffic_light_invasion_sensor(ground_vehicle_id_stream, pose_stream):
    from pylot.simulation.traffic_light_invasion_sensor_operator import \
            TrafficLightInvasionSensorOperator
    op_config = erdos.OperatorConfig(
        name='traffic_light_invasion_sensor_operator',
        log_file_name=FLAGS.log_file_name,
        csv_log_file_name=FLAGS.csv_log_file_name,
        profile_file_name=FLAGS.profile_file_name)
    [traffic_light_invasion_stream
     ] = erdos.connect(TrafficLightInvasionSensorOperator, op_config,
                       [ground_vehicle_id_stream, pose_stream], FLAGS)
    return traffic_light_invasion_stream


def add_canny_edge_lane_detection(bgr_camera_stream,
                                  name='canny_edge_lane_detection'):
    from pylot.perception.detection.lane_detection_canny_operator import \
        CannyEdgeLaneDetectionOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [lane_detection_stream] = erdos.connect(CannyEdgeLaneDetectionOperator,
                                            op_config, [bgr_camera_stream],
                                            FLAGS)
    return lane_detection_stream


def add_obstacle_tracking(obstacles_stream,
                          bgr_camera_stream,
                          time_to_decision_stream,
                          name_prefix='tracker_'):
    from pylot.perception.tracking.object_tracker_operator import \
        ObjectTrackerOperator
    op_config = erdos.OperatorConfig(name=name_prefix + FLAGS.tracker_type,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)

    [obstacle_tracking_stream] = erdos.connect(
        ObjectTrackerOperator, op_config,
        [obstacles_stream, bgr_camera_stream, time_to_decision_stream],
        FLAGS.tracker_type, FLAGS)
    return obstacle_tracking_stream


def add_tracking_evaluation(obstacle_tracking_stream,
                            ground_obstacles_stream,
                            name='tracking_eval_operator'):
    from pylot.perception.tracking.tracking_eval_operator import \
        TrackingEvalOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(TrackingEvalOperator, op_config,
                  [obstacle_tracking_stream, ground_obstacles_stream], FLAGS)


def add_depth_estimation(left_camera_stream,
                         right_camera_stream,
                         center_camera_setup,
                         name='depth_estimation_operator'):
    try:
        from pylot.perception.depth_estimation.depth_estimation_operator\
            import DepthEstimationOperator
    except ImportError:
        raise Exception("Error importing AnyNet depth estimation.")
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [depth_estimation_stream
     ] = erdos.connect(DepthEstimationOperator, op_config,
                       [left_camera_stream, right_camera_stream],
                       center_camera_setup.get_transform(),
                       center_camera_setup.get_fov(), FLAGS)
    return depth_estimation_stream


def add_segmentation(bgr_camera_stream, name='drn_segmentation_operator'):
    from pylot.perception.segmentation.segmentation_drn_operator import\
        SegmentationDRNOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [segmented_stream] = erdos.connect(SegmentationDRNOperator, op_config,
                                       [bgr_camera_stream], FLAGS)
    return segmented_stream


def add_segmentation_evaluation(ground_segmented_stream,
                                segmented_stream,
                                name='segmentation_evaluation_operator'):
    from pylot.perception.segmentation.segmentation_eval_operator import \
        SegmentationEvalOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(SegmentationEvalOperator, op_config,
                  [ground_segmented_stream, segmented_stream], FLAGS)


def add_segmentation_decay(ground_segmented_stream,
                           name='segmentation_decay_operator'):
    from pylot.perception.segmentation.segmentation_decay_operator import \
        SegmentationDecayOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [iou_stream] = erdos.connect(SegmentationDecayOperator, op_config,
                                 [ground_segmented_stream], FLAGS)
    return iou_stream


def add_linear_prediction(tracking_stream):
    from pylot.prediction.linear_predictor_operator import \
            LinearPredictorOperator
    op_config = erdos.OperatorConfig(name='linear_prediction_operator',
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [prediction_stream] = erdos.connect(LinearPredictorOperator, op_config,
                                        [tracking_stream], FLAGS)
    return prediction_stream


def add_r2p2_prediction(pose_stream, point_cloud_stream,
                        obstacles_tracking_stream, vehicle_id_stream,
                        lidar_setup):
    from pylot.prediction.r2p2_predictor_operator import \
            R2P2PredictorOperator
    op_config = erdos.OperatorConfig(name='r2p2_prediction_operator',
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [prediction_stream] = erdos.connect(R2P2PredictorOperator, op_config, [
        pose_stream, point_cloud_stream, obstacles_tracking_stream,
        vehicle_id_stream
    ], FLAGS, lidar_setup)
    return prediction_stream


def add_prediction_evaluation(pose_stream,
                              tracking_stream,
                              prediction_stream,
                              name='prediction_eval_operator'):
    from pylot.prediction.prediction_eval_operator import \
            PredictionEvalOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(PredictionEvalOperator, op_config,
                  [pose_stream, tracking_stream, prediction_stream], FLAGS)


def add_fot_planning(pose_stream,
                     prediction_stream,
                     global_trajectory_stream,
                     open_drive_stream,
                     time_to_decision_stream,
                     goal_location,
                     name='fot_planning_operator'):
    from pylot.planning.frenet_optimal_trajectory.fot_planning_operator \
        import FOTPlanningOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [waypoints_stream] = erdos.connect(FOTPlanningOperator, op_config, [
        pose_stream, prediction_stream, global_trajectory_stream,
        open_drive_stream, time_to_decision_stream
    ], FLAGS, goal_location)
    return waypoints_stream


def add_rrt_star_planning(pose_stream,
                          prediction_stream,
                          global_trajectory_stream,
                          open_drive_stream,
                          time_to_decision_stream,
                          goal_location,
                          name='rrt_star_planning_operator'):
    from pylot.planning.rrt_star.rrt_star_planning_operator import \
        RRTStarPlanningOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [waypoints_stream] = erdos.connect(RRTStarPlanningOperator, op_config, [
        pose_stream, prediction_stream, global_trajectory_stream,
        open_drive_stream, time_to_decision_stream
    ], FLAGS, goal_location)
    return waypoints_stream


def add_hybrid_astar_planning(pose_stream,
                              prediction_stream,
                              global_trajectory_stream,
                              open_drive_stream,
                              time_to_decision_stream,
                              goal_location,
                              name='hybrid_astar_planning_operator'):
    from pylot.planning.hybrid_astar.hybrid_astar_planning_operator import \
        HybridAStarPlanningOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [waypoints_stream
     ] = erdos.connect(HybridAStarPlanningOperator, op_config, [
         pose_stream, prediction_stream, global_trajectory_stream,
         open_drive_stream, time_to_decision_stream
     ], FLAGS, goal_location)
    return waypoints_stream


def add_waypoint_planning(pose_stream,
                          open_drive_stream,
                          global_trajectory_stream,
                          obstacles_stream,
                          traffic_lights_stream,
                          goal_location,
                          name='waypoint_planning_operator'):
    from pylot.planning.waypoint_planning_operator import \
            WaypointPlanningOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [waypoints_stream] = erdos.connect(WaypointPlanningOperator, op_config, [
        pose_stream, open_drive_stream, global_trajectory_stream,
        obstacles_stream, traffic_lights_stream
    ], FLAGS, goal_location)
    return waypoints_stream


def add_rgb_camera(transform,
                   vehicle_id_stream,
                   release_sensor_stream,
                   name='center_rgb_camera',
                   fov=90):
    from pylot.drivers.sensor_setup import RGBCameraSetup
    rgb_camera_setup = RGBCameraSetup(name, FLAGS.carla_camera_image_width,
                                      FLAGS.carla_camera_image_height,
                                      transform, fov)
    camera_stream, notify_reading_stream = _add_camera_driver(
        vehicle_id_stream, release_sensor_stream, rgb_camera_setup)
    return (camera_stream, notify_reading_stream, rgb_camera_setup)


def add_depth_camera(transform,
                     vehicle_id_stream,
                     release_sensor_stream,
                     name='center_depth_camera',
                     fov=90):
    from pylot.drivers.sensor_setup import DepthCameraSetup
    depth_camera_setup = DepthCameraSetup(name, FLAGS.carla_camera_image_width,
                                          FLAGS.carla_camera_image_height,
                                          transform, fov)
    ground_depth_camera_stream, notify_reading_stream = _add_camera_driver(
        vehicle_id_stream, release_sensor_stream, depth_camera_setup)
    return (ground_depth_camera_stream, notify_reading_stream,
            depth_camera_setup)


def add_segmented_camera(transform,
                         vehicle_id_stream,
                         release_sensor_stream,
                         name='center_segmented_camera',
                         fov=90):
    from pylot.drivers.sensor_setup import SegmentedCameraSetup
    segmented_camera_setup = SegmentedCameraSetup(
        name, FLAGS.carla_camera_image_width, FLAGS.carla_camera_image_height,
        transform, fov)
    ground_segmented_camera_stream, notify_reading_stream = _add_camera_driver(
        vehicle_id_stream, release_sensor_stream, segmented_camera_setup)
    return (ground_segmented_camera_stream, notify_reading_stream,
            segmented_camera_setup)


def add_left_right_cameras(transform,
                           vehicle_id_stream,
                           release_sensor_stream,
                           fov=90):
    (left_camera_setup, right_camera_setup) = \
        pylot.drivers.sensor_setup.create_left_right_camera_setups(
            'camera',
            transform.location,
            FLAGS.carla_camera_image_width,
            FLAGS.carla_camera_image_height,
            FLAGS.offset_left_right_cameras,
            fov)
    left_camera_stream, notify_left_stream = _add_camera_driver(
        vehicle_id_stream, release_sensor_stream, left_camera_setup)
    right_camera_stream, notify_right_stream = _add_camera_driver(
        vehicle_id_stream, release_sensor_stream, right_camera_setup)
    return (left_camera_stream, right_camera_stream, notify_left_stream,
            notify_right_stream)


def add_collision_sensor(vehicle_id_stream):
    """ Adds a collision sensor to the pipeline.

    Args:
        vehicle_id_stream (:py:class:`erdos.ReadStream`): Stream on which the
            ID of the ego-vehicle is received.

    Returns:
        :py:class:`erdos.ReadStream`: Stream on which
        :py:class:`~pylot.simulation.messages.CollisionMessage` messages with
        collision events are published.
    """
    from pylot.drivers.carla_collision_sensor_operator import \
        CarlaCollisionSensorDriverOperator
    op_config = erdos.OperatorConfig(name='carla_collision_sensor_operator',
                                     flow_watermarks=False,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [collision_stream] = erdos.connect(CarlaCollisionSensorDriverOperator,
                                       op_config, [vehicle_id_stream], FLAGS)
    return collision_stream


def add_lane_invasion_sensor(vehicle_id_stream):
    """ Adds a lane invasion sensor to the pipeline.

    Args:
        vehicle_id_stream (:py:class:`erdos.ReadStream`): Stream on which the
            ID of the ego-vehicle is received.

    Returns:
        :py:class:`erdos.ReadStream`: Stream on which
        :py:class:`~pylot.simulation.messages.LaneInvasionMessage` messages
        with lane invasion events are published.
    """
    from pylot.drivers.carla_lane_invasion_sensor_operator import \
        CarlaLaneInvasionSensorDriverOperator
    op_config = erdos.OperatorConfig(
        name='carla_lane_invasion_sensor_operator',
        flow_watermarks=False,
        log_file_name=FLAGS.log_file_name,
        csv_log_file_name=FLAGS.csv_log_file_name,
        profile_file_name=FLAGS.profile_file_name)
    [lane_invasion_stream
     ] = erdos.connect(CarlaLaneInvasionSensorDriverOperator, op_config,
                       [vehicle_id_stream], FLAGS)
    return lane_invasion_stream


def _add_camera_driver(vehicle_id_stream, release_sensor_stream, camera_setup):
    from pylot.drivers.carla_camera_driver_operator import \
        CarlaCameraDriverOperator
    op_config = erdos.OperatorConfig(name=camera_setup.get_name() +
                                     '_operator',
                                     flow_watermarks=False,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    return erdos.connect(CarlaCameraDriverOperator, op_config,
                         [vehicle_id_stream, release_sensor_stream],
                         camera_setup, FLAGS)


def add_lidar(transform,
              vehicle_id_stream,
              release_sensor_stream,
              name='center_lidar'):
    # Ensure that each lidar reading offers a 360 degree view.
    rotation_frequency = FLAGS.carla_lidar_frequency
    if rotation_frequency == -1:
        # If no lidar reading frequency is specified, set the
        # rotation frequency to the tick frequency.
        rotation_frequency = FLAGS.carla_fps
    lidar_setup = pylot.drivers.sensor_setup.create_center_lidar_setup(
        transform.location, rotation_frequency)
    point_cloud_stream, notify_reading_stream = _add_lidar_driver(
        vehicle_id_stream, release_sensor_stream, lidar_setup)
    return (point_cloud_stream, notify_reading_stream, lidar_setup)


def _add_lidar_driver(vehicle_id_stream, release_sensor_stream, lidar_setup):
    from pylot.drivers.carla_lidar_driver_operator import \
        CarlaLidarDriverOperator
    op_config = erdos.OperatorConfig(name=lidar_setup.get_name() + '_operator',
                                     flow_watermarks=False,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    return erdos.connect(CarlaLidarDriverOperator, op_config,
                         [vehicle_id_stream, release_sensor_stream],
                         lidar_setup, FLAGS)


def add_imu(transform, vehicle_id_stream, name='imu'):
    from pylot.drivers.carla_imu_driver_operator import CarlaIMUDriverOperator
    imu_setup = pylot.drivers.sensor_setup.IMUSetup(name, transform)
    op_config = erdos.OperatorConfig(name=imu_setup.get_name() + '_operator',
                                     flow_watermarks=False,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [imu_stream] = erdos.connect(CarlaIMUDriverOperator, op_config,
                                 [vehicle_id_stream], imu_setup, FLAGS)
    return (imu_stream, imu_setup)


def add_fusion(pose_stream, obstacles_stream, depth_stream,
               ground_obstacles_stream):
    from pylot.perception.fusion.fusion_operator import FusionOperator
    from pylot.perception.fusion.fusion_verification_operator import \
        FusionVerificationOperator
    op_config = erdos.OperatorConfig(name='fusion_operator',
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [obstacle_pos_stream
     ] = erdos.connect(FusionOperator, op_config,
                       [pose_stream, obstacles_stream, depth_stream], FLAGS)

    if FLAGS.evaluate_fusion:
        eval_op_config = erdos.OperatorConfig(
            name='fusion_verification_operator',
            log_file_name=FLAGS.log_file_name,
            csv_log_file_name=FLAGS.csv_log_file_name,
            profile_file_name=FLAGS.profile_file_name)
        erdos.connect(FusionVerificationOperator, eval_op_config,
                      [ground_obstacles_stream, obstacle_pos_stream])
    return obstacle_pos_stream


def add_mpc_agent(pose_stream, waypoints_stream):
    from pylot.control.mpc.mpc_agent_operator import MPCAgentOperator
    op_config = erdos.OperatorConfig(name='mpc_agent_operator',
                                     flow_watermarks=False,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [control_stream] = erdos.connect(MPCAgentOperator, op_config,
                                     [pose_stream, waypoints_stream], FLAGS)
    return control_stream


def add_pid_agent(pose_stream, waypoints_stream):
    from pylot.control.pid_agent_operator import PIDAgentOperator
    op_config = erdos.OperatorConfig(name='pid_agent_operator',
                                     flow_watermarks=False,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [control_stream] = erdos.connect(PIDAgentOperator, op_config,
                                     [pose_stream, waypoints_stream], FLAGS)
    return control_stream


def add_synchronizer(stream_to_sync_on):
    from pylot.simulation.synchronizer_operator import SynchronizerOperator
    op_config = erdos.OperatorConfig(name='synchronizer_operator',
                                     flow_watermarks=False,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    (control_stream, ) = erdos.connect(SynchronizerOperator, op_config,
                                       [stream_to_sync_on], FLAGS)
    return control_stream


def add_planning_pose_synchronizer(waypoint_stream, pose_stream,
                                   localization_stream, notify_stream1,
                                   notify_stream2):
    from pylot.simulation.planning_pose_synchronizer_operator import \
        PlanningPoseSynchronizerOperator
    op_config = erdos.OperatorConfig(
        name='planning_pose_synchronizer_operator',
        flow_watermarks=False,
        log_file_name=FLAGS.log_file_name,
        csv_log_file_name=FLAGS.csv_log_file_name,
        profile_file_name=FLAGS.profile_file_name)
    return erdos.connect(PlanningPoseSynchronizerOperator, op_config, [
        waypoint_stream, pose_stream, localization_stream, notify_stream1,
        notify_stream2
    ], FLAGS)


def add_bounding_box_logging(obstacles_stream,
                             name='bounding_box_logger_operator'):
    from pylot.loggers.bounding_box_logger_operator import \
            BoundingBoxLoggerOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(BoundingBoxLoggerOperator, op_config, [obstacles_stream],
                  FLAGS)


def add_camera_logging(stream, name, filename_prefix):
    from pylot.loggers.camera_logger_operator import CameraLoggerOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(CameraLoggerOperator, op_config, [stream], FLAGS,
                  filename_prefix)


def add_chauffeur_logging(vehicle_id_stream, pose_stream,
                          obstacle_tracking_stream, top_down_camera_stream,
                          top_down_segmentation_stream, top_down_camera_setup):
    from pylot.loggers.chauffeur_logger_operator import ChauffeurLoggerOperator
    op_config = erdos.OperatorConfig(name='chauffeur_logger_operator',
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)

    erdos.connect(ChauffeurLoggerOperator, op_config, [
        vehicle_id_stream, pose_stream, obstacle_tracking_stream,
        top_down_camera_stream, top_down_segmentation_stream
    ], FLAGS, top_down_camera_setup)


def add_eval_metric_logging(collision_stream, lane_invasion_stream,
                            traffic_light_invasion_stream, imu_stream,
                            pose_stream, obstacle_stream):
    """ Adds an evaluation metric logging operator to the pipeline.

    Args:
        collision_stream (:py:class:`erdos.ReadStream`): Stream on which the
            collision events are received.
        lane_invasion_stream (:py:class:`erdos.ReadStream`): Stream on which
            the lane invasion events are received.
        traffic_light_invasion_stream (:py:class:`erdos.ReadStream`): Stream on
            which the traffic light invasion events are received.
        imu_stream (:py:class:`erdos.ReadStream`): Stream on which the IMU
            messages are received.
    """
    from pylot.loggers.eval_metric_logger_operator import \
            EvalMetricLoggerOperator
    op_config = erdos.OperatorConfig(name='eval_metric_logger_operator',
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(EvalMetricLoggerOperator, op_config, [
        collision_stream, lane_invasion_stream, traffic_light_invasion_stream,
        imu_stream, pose_stream, obstacle_stream
    ], FLAGS)


def add_imu_logging(imu_stream, name='imu_logger_operator'):
    from pylot.loggers.imu_logger_operator import IMULoggerOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(IMULoggerOperator, op_config, [imu_stream], FLAGS)


def add_lidar_logging(point_cloud_stream,
                      name='lidar_logger_operator',
                      filename_prefix='carla-lidar-'):
    from pylot.loggers.lidar_logger_operator import LidarLoggerOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(LidarLoggerOperator, op_config, [point_cloud_stream], FLAGS,
                  filename_prefix)


def add_multiple_object_tracker_logging(
    obstacles_stream, name='multiple_object_tracker_logger_operator'):
    from pylot.loggers.multiple_object_tracker_logger_operator import \
        MultipleObjectTrackerLoggerOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(MultipleObjectTrackerLoggerOperator, op_config,
                  [obstacles_stream], FLAGS)


def add_trajectory_logging(obstacles_tracking_stream,
                           name='trajectory_logger_operator'):
    from pylot.loggers.trajectory_logger_operator import \
            TrajectoryLoggerOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(TrajectoryLoggerOperator, op_config,
                  [obstacles_tracking_stream], FLAGS)


def add_sensor_visualizers(camera_stream, depth_camera_stream,
                           point_cloud_stream, segmented_stream, imu_stream,
                           pose_stream):
    """ Adds operators for visualizing sensors streams. """
    if FLAGS.visualize_rgb_camera:
        add_camera_visualizer(camera_stream, 'rgb_camera')
    if FLAGS.visualize_depth_camera:
        add_camera_visualizer(depth_camera_stream, 'depth_camera')
    if FLAGS.visualize_imu:
        add_imu_visualizer(imu_stream)
    if FLAGS.visualize_pose:
        add_pose_visualizer(pose_stream)
    if FLAGS.visualize_lidar:
        add_lidar_visualizer(point_cloud_stream)
    if FLAGS.visualize_segmentation:
        add_camera_visualizer(segmented_stream, 'segmented_camera')


def add_lidar_visualizer(point_cloud_stream, name='lidar_visualizer_operator'):
    from pylot.debug.lidar_visualizer_operator import LidarVisualizerOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(LidarVisualizerOperator, op_config, [point_cloud_stream],
                  FLAGS)


def add_camera_visualizer(camera_stream, name):
    from pylot.debug.camera_visualizer_operator import CameraVisualizerOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(CameraVisualizerOperator, op_config, [camera_stream])


def add_imu_visualizer(imu_stream, name='imu_visualizer_operator'):
    from pylot.debug.imu_visualizer_operator import IMUVisualizerOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(IMUVisualizerOperator, op_config, [imu_stream], FLAGS)


def add_pose_visualizer(pose_stream, name='pose_visualizer_operator'):
    from pylot.debug.pose_visualizer_operator import PoseVisualizerOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(PoseVisualizerOperator, op_config, [pose_stream], FLAGS)


def add_prediction_visualizer(obstacle_tracking_stream,
                              prediction_stream,
                              vehicle_id_stream,
                              camera_transform,
                              release_sensor_stream,
                              name='top_down_tracking_visualizer_operator'):
    from pylot.debug.track_visualizer_operator import TrackVisualizerOperator
    top_down_transform = pylot.utils.get_top_down_transform(
        camera_transform, FLAGS.top_down_lateral_view)
    (top_down_segmented_camera_stream,
     notify_reading_stream,
     top_down_segmented_camera_setup) = \
        pylot.operator_creator.add_segmented_camera(
            top_down_transform,
            vehicle_id_stream,
            release_sensor_stream,
            name='top_down_segmented_camera',
            fov=90)
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(TrackVisualizerOperator, op_config, [
        obstacle_tracking_stream, prediction_stream,
        top_down_segmented_camera_stream
    ], FLAGS)


def add_waypoint_visualizer(waypoints_stream,
                            camera_stream,
                            pose_stream,
                            name='waypoint_visualizer_operator'):
    from pylot.debug.waypoint_visualizer_operator import\
            WaypointVisualizerOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(WaypointVisualizerOperator, op_config,
                  [waypoints_stream, camera_stream, pose_stream], FLAGS)


def add_perfect_detector(depth_camera_stream, center_camera_stream,
                         segmented_camera_stream, pose_stream,
                         ground_obstacles_stream,
                         ground_speed_limit_signs_stream,
                         ground_stop_signs_stream):
    from pylot.simulation.perfect_detector_operator import \
        PerfectDetectorOperator
    op_config = erdos.OperatorConfig(name='perfect_detector_operator',
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [obstacles_stream] = erdos.connect(PerfectDetectorOperator, op_config, [
        depth_camera_stream, center_camera_stream, segmented_camera_stream,
        pose_stream, ground_obstacles_stream, ground_speed_limit_signs_stream,
        ground_stop_signs_stream
    ], FLAGS)
    return obstacles_stream


def add_perfect_traffic_light_detector(ground_traffic_lights_stream,
                                       center_camera_stream,
                                       depth_camera_stream,
                                       segmented_camera_stream, pose_stream):
    from pylot.simulation.perfect_traffic_light_detector_operator import \
        PerfectTrafficLightDetectorOperator
    op_config = erdos.OperatorConfig(
        name='perfect_traffic_light_detector_operator',
        log_file_name=FLAGS.log_file_name,
        csv_log_file_name=FLAGS.csv_log_file_name,
        profile_file_name=FLAGS.profile_file_name)
    [traffic_lights_stream
     ] = erdos.connect(PerfectTrafficLightDetectorOperator, op_config, [
         ground_traffic_lights_stream, center_camera_stream,
         depth_camera_stream, segmented_camera_stream, pose_stream
     ], FLAGS)
    return traffic_lights_stream


def add_perfect_lane_detector(pose_stream):
    from pylot.simulation.perfect_lane_detector_operator import \
        PerfectLaneDetectionOperator
    op_config = erdos.OperatorConfig(name='perfect_lane_detection_operator',
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [detected_lanes_stream] = erdos.connect(PerfectLaneDetectionOperator,
                                            op_config, [pose_stream], FLAGS)
    return detected_lanes_stream


def add_perfect_tracking(vehicle_id_stream, ground_obstacles_stream,
                         pose_stream):
    from pylot.simulation.perfect_tracker_operator import \
        PerfectTrackerOperator
    op_config = erdos.OperatorConfig(name='perfect_tracking_operator',
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [ground_tracking_stream] = erdos.connect(
        PerfectTrackerOperator, op_config,
        [vehicle_id_stream, ground_obstacles_stream, pose_stream], FLAGS)
    return ground_tracking_stream


def add_time_to_decision(pose_stream, obstacles_stream):
    from pylot.control.time_to_decision_operator import TimeToDecisionOperator
    op_config = erdos.OperatorConfig(name='time_to_decision_operator',
                                     flow_watermarks=False,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [time_to_decision] = erdos.connect(TimeToDecisionOperator, op_config,
                                       [pose_stream, obstacles_stream], FLAGS)
    return time_to_decision
