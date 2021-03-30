from absl import flags

import erdos

import pylot.utils

# TODO: Hack to avoid a tensorflow import error.
import tensorflow as tf  # noqa: F401

FLAGS = flags.FLAGS


def add_simulator_bridge(control_stream, sensor_ready_stream,
                         pipeline_finish_notify_stream):
    from pylot.simulation.carla_operator import CarlaOperator
    op_config = erdos.OperatorConfig(name='simulator_bridge_operator',
                                     flow_watermarks=False,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    return erdos.connect(
        CarlaOperator, op_config,
        [control_stream, sensor_ready_stream, pipeline_finish_notify_stream],
        FLAGS)


def add_efficientdet_obstacle_detection(camera_stream,
                                        time_to_decision_stream,
                                        csv_file_name=None):
    """Adds an operator that uses EfficientDet for obstacle detection."""
    from pylot.perception.detection.efficientdet_operator import \
        EfficientDetOperator
    if csv_file_name is None:
        csv_file_name = FLAGS.csv_log_file_name
    op_config = erdos.OperatorConfig(name='efficientdet_operator',
                                     flow_watermarks=False,
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
            flow_watermarks=False,
            log_file_name=FLAGS.log_file_name,
            csv_log_file_name=csv_file_name,
            profile_file_name=FLAGS.profile_file_name)
        obstacles_streams += erdos.connect(
            DetectionOperator, op_config,
            [camera_stream, time_to_decision_stream],
            FLAGS.obstacle_detection_model_paths[i], FLAGS)
    return obstacles_streams


def add_obstacle_location_finder(obstacles_stream, depth_stream, pose_stream,
                                 camera_setup):
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
    [obstacles_with_loc_stream
     ] = erdos.connect(ObstacleLocationFinderOperator, op_config,
                       [obstacles_stream, depth_stream, pose_stream], FLAGS,
                       camera_setup)
    return obstacles_with_loc_stream


def add_obstacle_location_history(obstacles_stream, depth_stream, pose_stream,
                                  camera_setup):
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
                                     flow_watermarks=False,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [tracked_obstacles
     ] = erdos.connect(ObstacleLocationHistoryOperator, op_config,
                       [obstacles_stream, depth_stream, pose_stream], FLAGS,
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
                             evaluate_timely=False,
                             matching_policy='ceil',
                             frame_gap=None,
                             name='detection_eval_operator'):
    from pylot.perception.detection.detection_eval_operator import \
        DetectionEvalOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [finished_indicator_stream
     ] = erdos.connect(DetectionEvalOperator, op_config,
                       [obstacles_stream, ground_obstacles_stream],
                       evaluate_timely, matching_policy, frame_gap, FLAGS)


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


def add_traffic_light_detector(traffic_light_camera_stream,
                               time_to_decision_stream):
    from pylot.perception.detection.traffic_light_det_operator import \
        TrafficLightDetOperator
    op_config = erdos.OperatorConfig(name='traffic_light_detector_operator',
                                     flow_watermarks=False,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [traffic_lights_stream
     ] = erdos.connect(TrafficLightDetOperator, op_config,
                       [traffic_light_camera_stream, time_to_decision_stream],
                       FLAGS)
    return traffic_lights_stream


def add_traffic_light_invasion_sensor(ground_vehicle_id_stream, pose_stream):
    from pylot.drivers.carla_traffic_light_invasion_sensor_operator import \
        CarlaTrafficLightInvasionSensorOperator
    op_config = erdos.OperatorConfig(
        name='traffic_light_invasion_sensor_operator',
        log_file_name=FLAGS.log_file_name,
        csv_log_file_name=FLAGS.csv_log_file_name,
        profile_file_name=FLAGS.profile_file_name)
    [traffic_light_invasion_stream
     ] = erdos.connect(CarlaTrafficLightInvasionSensorOperator, op_config,
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


def add_lanenet_detection(bgr_camera_stream, name='lanenet_lane_detection'):
    from pylot.perception.detection.lanenet_detection_operator import \
        LanenetDetectionOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [lane_detection_stream] = erdos.connect(LanenetDetectionOperator,
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


def add_center_track_tracking(bgr_camera_stream,
                              camera_setup,
                              name='center_track'):
    from pylot.perception.tracking.center_track_operator import \
        CenterTrackOperator
    op_config = erdos.OperatorConfig(name='center_track_operator',
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [obstacle_tracking_stream] = erdos.connect(CenterTrackOperator, op_config,
                                               [bgr_camera_stream], FLAGS,
                                               camera_setup)
    return obstacle_tracking_stream


def add_tracking_evaluation(obstacle_tracking_stream,
                            ground_obstacles_stream,
                            evaluate_timely=False,
                            matching_policy='ceil',
                            frame_gap=None,
                            name='tracking_eval_operator'):
    from pylot.perception.tracking.tracking_eval_operator import \
        TrackingEvalOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [finished_indicator_stream
     ] = erdos.connect(TrackingEvalOperator, op_config,
                       [obstacle_tracking_stream, ground_obstacles_stream],
                       evaluate_timely, matching_policy, frame_gap, FLAGS)
    return finished_indicator_stream


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


def add_linear_prediction(tracking_stream, time_to_decision_stream):
    from pylot.prediction.linear_predictor_operator import \
        LinearPredictorOperator
    op_config = erdos.OperatorConfig(name='linear_prediction_operator',
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [prediction_stream
     ] = erdos.connect(LinearPredictorOperator, op_config,
                       [tracking_stream, time_to_decision_stream], FLAGS)
    return prediction_stream


def add_r2p2_prediction(point_cloud_stream, obstacles_tracking_stream,
                        time_to_decision_stream, lidar_setup):
    from pylot.prediction.r2p2_predictor_operator import \
        R2P2PredictorOperator
    op_config = erdos.OperatorConfig(name='r2p2_prediction_operator',
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [prediction_stream] = erdos.connect(R2P2PredictorOperator, op_config, [
        point_cloud_stream, obstacles_tracking_stream, time_to_decision_stream
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


def add_behavior_planning(pose_stream,
                          open_drive_stream,
                          global_trajectory_stream,
                          goal_location,
                          name='behavior_planning'):
    from pylot.planning.behavior_planning_operator import \
        BehaviorPlanningOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [trajectory_stream] = erdos.connect(
        BehaviorPlanningOperator, op_config,
        [pose_stream, open_drive_stream, global_trajectory_stream], FLAGS,
        goal_location)
    return trajectory_stream


def add_planning(pose_stream,
                 prediction_stream,
                 traffic_lights_stream,
                 lanes_stream,
                 global_trajectory_stream,
                 open_drive_stream,
                 time_to_decision_stream,
                 name='planning_operator'):
    from pylot.planning.planning_operator import PlanningOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [waypoints_stream] = erdos.connect(PlanningOperator, op_config, [
        pose_stream, prediction_stream, traffic_lights_stream, lanes_stream,
        global_trajectory_stream, open_drive_stream, time_to_decision_stream
    ], FLAGS)
    return waypoints_stream


def add_left_right_cameras(transform,
                           vehicle_id_stream,
                           release_sensor_stream,
                           fov=90):
    (left_camera_setup, right_camera_setup) = \
        pylot.drivers.sensor_setup.create_left_right_camera_setups(
            'camera',
            transform.location,
            FLAGS.camera_image_width,
            FLAGS.camera_image_height,
            FLAGS.offset_left_right_cameras,
            fov)
    left_camera_stream, notify_left_stream = add_camera_driver(
        left_camera_setup, vehicle_id_stream, release_sensor_stream)
    right_camera_stream, notify_right_stream = add_camera_driver(
        right_camera_setup, vehicle_id_stream, release_sensor_stream)
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
    op_config = erdos.OperatorConfig(
        name='simulator_collision_sensor_operator',
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
        name='simulator_lane_invasion_sensor_operator',
        flow_watermarks=False,
        log_file_name=FLAGS.log_file_name,
        csv_log_file_name=FLAGS.csv_log_file_name,
        profile_file_name=FLAGS.profile_file_name)
    [lane_invasion_stream
     ] = erdos.connect(CarlaLaneInvasionSensorDriverOperator, op_config,
                       [vehicle_id_stream], FLAGS)
    return lane_invasion_stream


def add_camera_driver(camera_setup, vehicle_id_stream, release_sensor_stream):
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
              name='center_lidar',
              legacy=False):
    # Ensure that each lidar reading offers a 360 degree view.
    rotation_frequency = FLAGS.simulator_lidar_frequency
    if rotation_frequency == -1:
        # If no lidar reading frequency is specified, set the
        # rotation frequency to the tick frequency.
        rotation_frequency = FLAGS.simulator_fps
    lidar_setup = pylot.drivers.sensor_setup.create_center_lidar_setup(
        transform.location, rotation_frequency, legacy=legacy)
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


def add_gnss(transform, vehicle_id_stream, name='gnss'):
    from pylot.drivers.carla_gnss_driver_operator import \
        CarlaGNSSDriverOperator
    gnss_setup = pylot.drivers.sensor_setup.GNSSSetup(name, transform)
    op_config = erdos.OperatorConfig(name=gnss_setup.get_name() + '_operator',
                                     flow_watermarks=False,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [gnss_stream] = erdos.connect(CarlaGNSSDriverOperator, op_config,
                                  [vehicle_id_stream], gnss_setup, FLAGS)
    return (gnss_stream, gnss_setup)


def add_localization(imu_stream,
                     gnss_stream,
                     ground_pose_stream,
                     name="localization"):
    from pylot.localization.localization_operator import LocalizationOperator
    op_config = erdos.OperatorConfig(name=name + "_operator",
                                     flow_watermarks=False,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [pose_stream
     ] = erdos.connect(LocalizationOperator, op_config,
                       [imu_stream, gnss_stream, ground_pose_stream], FLAGS)
    return pose_stream


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


def add_mpc(pose_stream, waypoints_stream):
    from pylot.control.mpc.mpc_operator import MPCOperator
    op_config = erdos.OperatorConfig(name='mpc_operator',
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [control_stream] = erdos.connect(MPCOperator, op_config,
                                     [pose_stream, waypoints_stream], FLAGS)
    return control_stream


def add_pid_control(pose_stream, waypoints_stream):
    from pylot.control.pid_control_operator import PIDControlOperator
    op_config = erdos.OperatorConfig(name='pid_control_operator',
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [control_stream] = erdos.connect(PIDControlOperator, op_config,
                                     [pose_stream, waypoints_stream], FLAGS)
    return control_stream


def add_synchronizer(ground_vehicle_id_stream, stream_to_sync_on):
    from pylot.simulation.synchronizer_operator import SynchronizerOperator
    op_config = erdos.OperatorConfig(name='synchronizer_operator',
                                     flow_watermarks=False,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    (control_stream, ) = erdos.connect(
        SynchronizerOperator, op_config,
        [ground_vehicle_id_stream, stream_to_sync_on], FLAGS)
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
    [finished_indicator_stream] = erdos.connect(BoundingBoxLoggerOperator,
                                                op_config, [obstacles_stream],
                                                FLAGS)
    return finished_indicator_stream


def add_camera_logging(stream, name, filename_prefix):
    from pylot.loggers.camera_logger_operator import CameraLoggerOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [finished_indicator_stream] = erdos.connect(CameraLoggerOperator,
                                                op_config, [stream], FLAGS,
                                                filename_prefix)
    return finished_indicator_stream


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
                            pose_stream):
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
    [finished_indicator_stream
     ] = erdos.connect(EvalMetricLoggerOperator, op_config, [
         collision_stream, lane_invasion_stream, traffic_light_invasion_stream,
         imu_stream, pose_stream
     ], FLAGS)
    return finished_indicator_stream


def add_imu_logging(imu_stream, name='imu_logger_operator'):
    from pylot.loggers.imu_logger_operator import IMULoggerOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [finished_indicator_stream] = erdos.connect(IMULoggerOperator, op_config,
                                                [imu_stream], FLAGS)
    return finished_indicator_stream


def add_lidar_logging(point_cloud_stream,
                      name='lidar_logger_operator',
                      filename_prefix='lidar'):
    from pylot.loggers.lidar_logger_operator import LidarLoggerOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [finished_indicator_stream] = erdos.connect(LidarLoggerOperator, op_config,
                                                [point_cloud_stream], FLAGS,
                                                filename_prefix)
    return finished_indicator_stream


def add_multiple_object_tracker_logging(
        obstacles_stream, name='multiple_object_tracker_logger_operator'):
    from pylot.loggers.multiple_object_tracker_logger_operator import \
        MultipleObjectTrackerLoggerOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [finished_indicator_stream
     ] = erdos.connect(MultipleObjectTrackerLoggerOperator, op_config,
                       [obstacles_stream], FLAGS)
    return finished_indicator_stream


def add_trajectory_logging(obstacles_tracking_stream,
                           name='trajectory_logger_operator'):
    from pylot.loggers.trajectory_logger_operator import \
        TrajectoryLoggerOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [finished_indicator_stream
     ] = erdos.connect(TrajectoryLoggerOperator, op_config,
                       [obstacles_tracking_stream], FLAGS)
    return finished_indicator_stream


def add_visualizer(pose_stream=None,
                   camera_stream=None,
                   tl_camera_stream=None,
                   prediction_camera_stream=None,
                   depth_stream=None,
                   point_cloud_stream=None,
                   segmentation_stream=None,
                   imu_stream=None,
                   obstacles_stream=None,
                   traffic_lights_stream=None,
                   tracked_obstacles_stream=None,
                   lane_detection_stream=None,
                   prediction_stream=None,
                   waypoints_stream=None,
                   control_stream=None,
                   name='visualizer_operator'):
    from pylot.debug.visualizer_operator import VisualizerOperator
    import pygame
    pygame.init()
    pygame_display = pygame.display.set_mode(
        (FLAGS.camera_image_width, FLAGS.camera_image_height),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("Pylot")
    streams_to_send_top_on = []
    if pose_stream is None:
        pose_stream = erdos.IngestStream()
        streams_to_send_top_on.append(pose_stream)
    if (camera_stream is None
            or not (FLAGS.visualize_rgb_camera
                    or FLAGS.visualize_detected_obstacles
                    or FLAGS.visualize_detected_traffic_lights
                    or FLAGS.visualize_tracked_obstacles
                    or FLAGS.visualize_waypoints)):
        camera_stream = erdos.IngestStream()
        streams_to_send_top_on.append(camera_stream)
    if depth_stream is None or not FLAGS.visualize_depth_camera:
        depth_stream = erdos.IngestStream()
        streams_to_send_top_on.append(depth_stream)
    if point_cloud_stream is None or not FLAGS.visualize_lidar:
        point_cloud_stream = erdos.IngestStream()
        streams_to_send_top_on.append(point_cloud_stream)
    if segmentation_stream is None or not FLAGS.visualize_segmentation:
        segmentation_stream = erdos.IngestStream()
        streams_to_send_top_on.append(segmentation_stream)
    if imu_stream is None or not FLAGS.visualize_imu:
        imu_stream = erdos.IngestStream()
        streams_to_send_top_on.append(imu_stream)
    if obstacles_stream is None or not FLAGS.visualize_detected_obstacles:
        obstacles_stream = erdos.IngestStream()
        streams_to_send_top_on.append(obstacles_stream)
    if tl_camera_stream is None or not FLAGS.visualize_detected_traffic_lights:
        tl_camera_stream = erdos.IngestStream()
        streams_to_send_top_on.append(tl_camera_stream)
    if (traffic_lights_stream is None
            or not (FLAGS.visualize_detected_traffic_lights
                    or FLAGS.visualize_world)):
        traffic_lights_stream = erdos.IngestStream()
        streams_to_send_top_on.append(traffic_lights_stream)
    if (tracked_obstacles_stream is None
            or not FLAGS.visualize_tracked_obstacles):
        tracked_obstacles_stream = erdos.IngestStream()
        streams_to_send_top_on.append(tracked_obstacles_stream)
    if lane_detection_stream is None or not FLAGS.visualize_detected_lanes:
        lane_detection_stream = erdos.IngestStream()
        streams_to_send_top_on.append(lane_detection_stream)
    if prediction_camera_stream is None or not FLAGS.visualize_prediction:
        prediction_camera_stream = erdos.IngestStream()
        streams_to_send_top_on.append(prediction_camera_stream)
    if (prediction_stream is None
            or not (FLAGS.visualize_prediction or FLAGS.visualize_world)):
        prediction_stream = erdos.IngestStream()
        streams_to_send_top_on.append(prediction_stream)
    if waypoints_stream is None or not (FLAGS.visualize_waypoints
                                        or FLAGS.visualize_world):
        waypoints_stream = erdos.IngestStream()
        streams_to_send_top_on.append(waypoints_stream)
    if control_stream is None:
        control_stream = erdos.IngestStream()
        streams_to_send_top_on.append(control_stream)

    control_display_stream = erdos.IngestStream()
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(VisualizerOperator, op_config, [
        pose_stream, camera_stream, tl_camera_stream, prediction_camera_stream,
        depth_stream, point_cloud_stream, segmentation_stream, imu_stream,
        obstacles_stream, traffic_lights_stream, tracked_obstacles_stream,
        lane_detection_stream, prediction_stream, waypoints_stream,
        control_stream, control_display_stream
    ], pygame_display, FLAGS)
    return control_display_stream, streams_to_send_top_on


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


def add_perfect_lane_detector(pose_stream, open_drive_stream,
                              center_camera_stream):
    from pylot.simulation.perfect_lane_detector_operator import \
        PerfectLaneDetectionOperator
    op_config = erdos.OperatorConfig(name='perfect_lane_detection_operator',
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [detected_lanes_stream
     ] = erdos.connect(PerfectLaneDetectionOperator, op_config,
                       [pose_stream, open_drive_stream, center_camera_stream],
                       FLAGS)
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
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [time_to_decision] = erdos.connect(TimeToDecisionOperator, op_config,
                                       [pose_stream, obstacles_stream], FLAGS)
    return time_to_decision
