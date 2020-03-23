from absl import flags

import copy
import erdos

import pylot.utils
# Control operators.
from pylot.control.mpc.mpc_agent_operator import MPCAgentOperator
from pylot.control.pid_agent_operator import PIDAgentOperator
# Visualizing operators.
from pylot.debug.camera_visualizer_operator import CameraVisualizerOperator
from pylot.debug.can_bus_visualizer_operator import CanBusVisualizerOperator
from pylot.debug.lidar_visualizer_operator import LidarVisualizerOperator
from pylot.debug.track_visualizer_operator import TrackVisualizerOperator
from pylot.debug.waypoint_visualizer_operator import WaypointVisualizerOperator
# Sensor setups.
from pylot.drivers.sensor_setup import DepthCameraSetup, RGBCameraSetup, \
    SegmentedCameraSetup
# Logging operators.
from pylot.loggers.bounding_box_logger_operator import \
    BoundingBoxLoggerOperator
from pylot.loggers.camera_logger_operator import CameraLoggerOperator
from pylot.loggers.imu_logger_operator import IMULoggerOperator
from pylot.loggers.lidar_logger_operator import LidarLoggerOperator
from pylot.loggers.multiple_object_tracker_logger_operator import \
    MultipleObjectTrackerLoggerOperator
from pylot.loggers.trajectory_logger_operator import TrajectoryLoggerOperator
# Perception operators.
from pylot.perception.detection.detection_decay_operator import \
    DetectionDecayOperator
from pylot.perception.detection.detection_eval_operator import \
    DetectionEvalOperator
from pylot.perception.detection.detection_operator import DetectionOperator
from pylot.perception.detection.lane_detection_canny_operator import \
    CannyEdgeLaneDetectionOperator
from pylot.perception.detection.obstacle_location_finder_operator import \
    ObstacleLocationFinderOperator
from pylot.perception.detection.traffic_light_det_operator import \
    TrafficLightDetOperator
from pylot.perception.fusion.fusion_operator import FusionOperator
from pylot.perception.fusion.fusion_verification_operator import \
    FusionVerificationOperator
from pylot.perception.segmentation.segmentation_decay_operator import \
    SegmentationDecayOperator
from pylot.perception.segmentation.segmentation_drn_operator import\
    SegmentationDRNOperator
from pylot.perception.segmentation.segmentation_eval_operator import \
    SegmentationEvalOperator
from pylot.perception.tracking.object_tracker_operator import \
    ObjectTrackerOperator
from pylot.perception.tracking.tracking_eval_operator import \
    TrackingEvalOperator
# Planning operators.
from pylot.planning.frenet_optimal_trajectory.fot_planning_operator \
    import FOTPlanningOperator
from pylot.planning.waypoint_planning_operator import WaypointPlanningOperator
# Prediction operators.
from pylot.prediction.linear_predictor_operator import LinearPredictorOperator
from pylot.prediction.prediction_eval_operator import PredictionEvalOperator
from pylot.simulation.synchronizer_operator import SynchronizerOperator

FLAGS = flags.FLAGS


def add_carla_bridge(control_stream):
    from pylot.simulation.carla_operator import CarlaOperator
    op_config = erdos.OperatorConfig(name='carla_operator',
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    return erdos.connect(CarlaOperator, op_config, [control_stream], FLAGS)


def add_obstacle_detection(camera_stream, csv_file_name=None):
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
            DetectionOperator, op_config, [camera_stream],
            FLAGS.obstacle_detection_model_paths[i], FLAGS)
    return obstacles_streams


def add_obstacle_location_finder(obstacles_stream, point_cloud_stream,
                                 can_bus_stream, camera_stream, camera_setup):
    """Adds an operator that finds the world locations of the obstacles.

    Args:
        obstacles_stream (:py:class:`erdos.ReadStream`): Stream on which
            detected obstacles are received.
        point_cloud_stream (:py:class:`erdos.ReadStream`): Stream on
            which point cloud messages are received.
        can_bus_stream (:py:class:`erdos.ReadStream`, optional): Stream on
            which can bus info is received.
        camera_stream (:py:class:`erdos.ReadStream`): The stream on which
            camera frames are received.
        camera_setup (:py:class:`~pylot.drivers.sensor_setup.CameraSetup`):
            The setup of the center camera.

    Returns:
        :py:class:`erdos.ReadStream`: Stream on which
        :py:class:`~pylot.perception.messages.ObstaclesMessage` messages with
        world locations are published.
    """
    op_config = erdos.OperatorConfig(name=camera_setup.get_name() +
                                     '_location_finder_operator',
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [obstacles_with_loc_stream] = erdos.connect(
        ObstacleLocationFinderOperator, op_config,
        [obstacles_stream, point_cloud_stream, can_bus_stream, camera_stream],
        FLAGS, camera_setup)
    return obstacles_with_loc_stream


def add_detection_decay(ground_obstacles_stream):
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
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(DetectionEvalOperator, op_config,
                  [obstacles_stream, ground_obstacles_stream], FLAGS)


def add_traffic_light_detector(traffic_light_camera_stream):
    op_config = erdos.OperatorConfig(name='traffic_light_detector_operator',
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [traffic_lights_stream] = erdos.connect(TrafficLightDetOperator, op_config,
                                            [traffic_light_camera_stream],
                                            FLAGS)
    return traffic_lights_stream


def add_canny_edge_lane_detection(bgr_camera_stream,
                                  name='canny_edge_lane_detection'):
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
                          name_prefix='tracker_'):
    op_config = erdos.OperatorConfig(name=name_prefix + FLAGS.tracker_type,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)

    [obstacle_tracking_stream
     ] = erdos.connect(ObjectTrackerOperator, op_config,
                       [obstacles_stream, bgr_camera_stream],
                       FLAGS.tracker_type, FLAGS)
    return obstacle_tracking_stream


def add_tracking_evaluation(obstacle_tracking_stream,
                            ground_obstacles_stream,
                            name='tracking_eval_operator'):
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
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(SegmentationEvalOperator, op_config,
                  [ground_segmented_stream, segmented_stream], FLAGS)


def add_segmentation_decay(ground_segmented_stream,
                           name='segmentation_decay_operator'):
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [iou_stream] = erdos.connect(SegmentationDecayOperator, op_config,
                                 [ground_segmented_stream], FLAGS)
    return iou_stream


def add_linear_prediction(tracking_stream):
    op_config = erdos.OperatorConfig(name='linear_prediction_operator',
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [prediction_stream] = erdos.connect(LinearPredictorOperator, op_config,
                                        [tracking_stream], FLAGS)
    return prediction_stream


def add_prediction_evaluation(can_bus_stream,
                              tracking_stream,
                              prediction_stream,
                              name='prediction_eval_operator'):
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(PredictionEvalOperator, op_config,
                  [can_bus_stream, tracking_stream, prediction_stream], FLAGS)


def add_fot_planning(can_bus_stream,
                     prediction_stream,
                     global_trajectory_stream,
                     open_drive_stream,
                     goal_location,
                     name='fot_planning_operator'):
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [waypoints_stream] = erdos.connect(FOTPlanningOperator, op_config, [
        can_bus_stream, prediction_stream, global_trajectory_stream,
        open_drive_stream
    ], FLAGS, goal_location)
    return waypoints_stream


def add_rrt_star_planning(can_bus_stream,
                          prediction_stream,
                          goal_location,
                          name='rrt_star_planning_operator'):
    from pylot.planning.rrt_star.rrt_star_planning_operator import \
        RRTStarPlanningOperator
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [waypoints_stream] = erdos.connect(RRTStarPlanningOperator, op_config,
                                       [can_bus_stream, prediction_stream],
                                       FLAGS, goal_location)
    return waypoints_stream


def add_waypoint_planning(can_bus_stream,
                          open_drive_stream,
                          global_trajectory_stream,
                          obstacles_stream,
                          traffic_lights_stream,
                          goal_location,
                          name='waypoint_planning_operator'):
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [waypoints_stream] = erdos.connect(WaypointPlanningOperator, op_config, [
        can_bus_stream, open_drive_stream, global_trajectory_stream,
        obstacles_stream, traffic_lights_stream
    ], FLAGS, goal_location)
    return waypoints_stream


def add_rgb_camera(transform,
                   vehicle_id_stream,
                   name='center_rgb_camera',
                   fov=90):
    rgb_camera_setup = RGBCameraSetup(name, FLAGS.carla_camera_image_width,
                                      FLAGS.carla_camera_image_height,
                                      transform, fov)
    camera_stream = _add_camera_driver(vehicle_id_stream, rgb_camera_setup)
    return (camera_stream, rgb_camera_setup)


def add_depth_camera(transform,
                     vehicle_id_stream,
                     name='center_depth_camera',
                     fov=90):
    depth_camera_setup = DepthCameraSetup(name, FLAGS.carla_camera_image_width,
                                          FLAGS.carla_camera_image_height,
                                          transform, fov)
    ground_depth_camera_stream = _add_camera_driver(vehicle_id_stream,
                                                    depth_camera_setup)
    return (ground_depth_camera_stream, depth_camera_setup)


def add_segmented_camera(transform,
                         vehicle_id_stream,
                         name='center_segmented_camera',
                         fov=90):
    segmented_camera_setup = SegmentedCameraSetup(
        name, FLAGS.carla_camera_image_width, FLAGS.carla_camera_image_height,
        transform, fov)
    ground_segmented_camera_stream = _add_camera_driver(
        vehicle_id_stream, segmented_camera_setup)
    return (ground_segmented_camera_stream, segmented_camera_setup)


def add_left_right_cameras(transform, vehicle_id_stream, fov=90):
    (left_camera_setup, right_camera_setup) = \
        pylot.drivers.sensor_setup.create_left_right_camera_setups(
            'camera',
            transform.location,
            FLAGS.carla_camera_image_width,
            FLAGS.carla_camera_image_height,
            FLAGS.offset_left_right_cameras,
            fov)
    left_camera_stream = _add_camera_driver(vehicle_id_stream,
                                            left_camera_setup)
    right_camera_stream = _add_camera_driver(vehicle_id_stream,
                                             right_camera_setup)
    return (left_camera_stream, right_camera_stream)


def _add_camera_driver(vehicle_id_stream, camera_setup):
    from pylot.drivers.carla_camera_driver_operator import \
        CarlaCameraDriverOperator
    op_config = erdos.OperatorConfig(name=camera_setup.get_name() +
                                     '_operator',
                                     flow_watermarks=False,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [camera_stream] = erdos.connect(CarlaCameraDriverOperator, op_config,
                                    [vehicle_id_stream], camera_setup, FLAGS)
    return camera_stream


def add_lidar(transform, vehicle_id_stream, name='center_lidar'):
    lidar_setup = pylot.drivers.sensor_setup.create_center_lidar_setup(
        transform.location)
    point_cloud_stream = _add_lidar_driver(vehicle_id_stream, lidar_setup)
    return (point_cloud_stream, lidar_setup)


def _add_lidar_driver(vehicle_id_stream, lidar_setup):
    from pylot.drivers.carla_lidar_driver_operator import \
        CarlaLidarDriverOperator
    op_config = erdos.OperatorConfig(name=lidar_setup.get_name() + '_operator',
                                     flow_watermarks=False,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [point_cloud_stream] = erdos.connect(CarlaLidarDriverOperator, op_config,
                                         [vehicle_id_stream], lidar_setup,
                                         FLAGS)
    return point_cloud_stream


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


def add_fusion(can_bus_stream, obstacles_stream, depth_stream,
               ground_obstacles_stream):
    op_config = erdos.OperatorConfig(name='fusion_operator',
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [obstacle_pos_stream
     ] = erdos.connect(FusionOperator, op_config,
                       [can_bus_stream, obstacles_stream, depth_stream], FLAGS)

    if FLAGS.evaluate_fusion:
        eval_op_config = erdos.OperatorConfig(
            name='fusion_verification_operator',
            log_file_name=FLAGS.log_file_name,
            csv_log_file_name=FLAGS.csv_log_file_name,
            profile_file_name=FLAGS.profile_file_name)
        erdos.connect(FusionVerificationOperator, eval_op_config,
                      [ground_obstacles_stream, obstacle_pos_stream])
    return obstacle_pos_stream


def add_mpc_agent(can_bus_stream, waypoints_stream):
    op_config = erdos.OperatorConfig(name='mpc_agent_operator',
                                     flow_watermarks=False,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [control_stream] = erdos.connect(MPCAgentOperator, op_config,
                                     [can_bus_stream, waypoints_stream], FLAGS)
    return control_stream


def add_pid_agent(can_bus_stream, waypoints_stream):
    op_config = erdos.OperatorConfig(name='pid_agent_operator',
                                     flow_watermarks=False,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [control_stream] = erdos.connect(PIDAgentOperator, op_config,
                                     [can_bus_stream, waypoints_stream], FLAGS)
    return control_stream


def add_synchronizer(stream_to_sync_on):
    op_config = erdos.OperatorConfig(name='synchronizer_operator',
                                     flow_watermarks=False,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    (control_stream, ) = erdos.connect(SynchronizerOperator, op_config,
                                       [stream_to_sync_on], FLAGS)
    return control_stream


def add_bounding_box_logging(obstacles_stream,
                             name='bounding_box_logger_operator'):
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(BoundingBoxLoggerOperator, op_config, [obstacles_stream],
                  FLAGS)


def add_camera_logging(stream, name, filename_prefix):
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(CameraLoggerOperator, op_config, [stream], FLAGS,
                  filename_prefix)


def add_chauffeur_logging(vehicle_id_stream, can_bus_stream,
                          obstacle_tracking_stream, top_down_camera_stream,
                          top_down_segmentation_stream, top_down_camera_setup):
    from pylot.loggers.chauffeur_logger_operator import ChauffeurLoggerOperator
    op_config = erdos.OperatorConfig(name='chauffeur_logger_operator',
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)

    erdos.connect(ChauffeurLoggerOperator, op_config, [
        vehicle_id_stream, can_bus_stream, obstacle_tracking_stream,
        top_down_camera_stream, top_down_segmentation_stream
    ], FLAGS, top_down_camera_setup)


def add_carla_collision_logging(vehicle_id_stream, can_bus_stream):
    from pylot.loggers.carla_collision_logger_operator import \
        CarlaCollisionLoggerOperator
    op_config = erdos.OperatorConfig(name='carla_collision_logger_operator',
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(CarlaCollisionLoggerOperator, op_config,
                  [vehicle_id_stream, can_bus_stream], FLAGS)


def add_imu_logging(imu_stream, name='imu_logger_operator'):
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(IMULoggerOperator, op_config, [imu_stream], FLAGS)


def add_lidar_logging(point_cloud_stream,
                      name='lidar_logger_operator',
                      filename_prefix='carla-lidar-'):
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(LidarLoggerOperator, op_config, [point_cloud_stream], FLAGS,
                  filename_prefix)


def add_multiple_object_tracker_logging(
    obstacles_stream, name='multiple_object_tracker_logger_operator'):
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(MultipleObjectTrackerLoggerOperator, op_config,
                  [obstacles_stream], FLAGS)


def add_trajectory_logging(obstacles_tracking_stream,
                           name='trajectory_logger_operator'):
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(TrajectoryLoggerOperator, op_config,
                  [obstacles_tracking_stream], FLAGS)


def add_sensor_visualizers(camera_stream, depth_camera_stream,
                           point_cloud_stream, segmented_stream, imu_stream,
                           can_bus_stream):
    """ Adds operators for visualizing sensors streams. """
    if FLAGS.visualize_rgb_camera:
        add_camera_visualizer(camera_stream, 'rgb_camera')
    if FLAGS.visualize_depth_camera:
        add_camera_visualizer(depth_camera_stream, 'depth_camera')
    if FLAGS.visualize_imu:
        add_imu_visualizer(imu_stream)
    if FLAGS.visualize_can_bus:
        add_can_bus_visualizer(can_bus_stream)
    if FLAGS.visualize_lidar:
        add_lidar_visualizer(point_cloud_stream)
    if FLAGS.visualize_segmentation:
        add_camera_visualizer(segmented_stream, 'segmented_camera')


def add_lidar_visualizer(point_cloud_stream, name='lidar_visualizer_operator'):
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(LidarVisualizerOperator, op_config, [point_cloud_stream])


def add_camera_visualizer(camera_stream, name):
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


def add_can_bus_visualizer(can_bus_stream, name='can_bus_visualizer_operator'):
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(CanBusVisualizerOperator, op_config, [can_bus_stream], FLAGS)


def add_prediction_visualizer(obstacle_tracking_stream,
                              prediction_stream,
                              vehicle_id_stream,
                              camera_transform,
                              name='top_down_tracking_visualizer_operator'):
    top_down_transform = pylot.utils.get_top_down_transform(
        camera_transform, FLAGS.top_down_lateral_view)
    (top_down_segmented_camera_stream,
     top_down_segmented_camera_setup) = \
        pylot.operator_creator.add_segmented_camera(
            top_down_transform,
            vehicle_id_stream,
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
                            can_bus_stream,
                            name='waypoint_visualizer_operator'):
    op_config = erdos.OperatorConfig(name=name,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    erdos.connect(WaypointVisualizerOperator, op_config,
                  [waypoints_stream, camera_stream, can_bus_stream], FLAGS)


def add_perfect_detector(depth_camera_stream, center_camera_stream,
                         segmented_camera_stream, can_bus_stream,
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
        can_bus_stream, ground_obstacles_stream,
        ground_speed_limit_signs_stream, ground_stop_signs_stream
    ], FLAGS)
    return obstacles_stream


def add_perfect_traffic_light_detector(ground_traffic_lights_stream,
                                       center_camera_stream,
                                       depth_camera_stream,
                                       segmented_camera_stream,
                                       can_bus_stream):
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
         depth_camera_stream, segmented_camera_stream, can_bus_stream
     ], FLAGS)
    return traffic_lights_stream


def add_perfect_lane_detector(can_bus_stream):
    from pylot.simulation.perfect_lane_detector_operator import \
        PerfectLaneDetectionOperator
    op_config = erdos.OperatorConfig(name='perfect_lane_detection_operator',
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [detected_lanes_stream] = erdos.connect(PerfectLaneDetectionOperator,
                                            op_config, [can_bus_stream], FLAGS)
    return detected_lanes_stream


def add_perfect_tracking(vehicle_id_stream, ground_obstacles_stream,
                         can_bus_stream):
    from pylot.simulation.perfect_tracker_operator import \
        PerfectTrackerOperator
    op_config = erdos.OperatorConfig(name='perfect_tracking_operator',
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    [ground_tracking_stream] = erdos.connect(
        PerfectTrackerOperator, op_config,
        [vehicle_id_stream, ground_obstacles_stream, can_bus_stream], FLAGS)
    return ground_tracking_stream
