from absl import flags

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
from pylot.planning.waypoint_planning_operator import WaypointPlanningOperator
# Prediction operators.
from pylot.prediction.linear_predictor_operator import LinearPredictorOperator
from pylot.prediction.prediction_eval_operator import PredictionEvalOperator
from pylot.simulation.synchronizer_operator import SynchronizerOperator

FLAGS = flags.FLAGS


def add_carla_bridge(control_stream):
    from pylot.simulation.carla_operator import CarlaOperator
    return erdos.connect(CarlaOperator, [control_stream],
                         'carla_operator',
                         True,
                         FLAGS,
                         log_file_name=FLAGS.log_file_name,
                         csv_file_name=FLAGS.csv_log_file_name)


def add_obstacle_detection(camera_stream, csv_file_name=None):
    obstacles_streams = []
    if csv_file_name is None:
        csv_file_name = FLAGS.csv_log_file_name
    for i in range(0, len(FLAGS.obstacle_detection_model_paths)):
        obstacles_streams += erdos.connect(
            DetectionOperator, [camera_stream],
            FLAGS.obstacle_detection_model_names[i],
            True,
            FLAGS.obstacle_detection_model_paths[i],
            FLAGS,
            log_file_name=FLAGS.log_file_name,
            csv_file_name=csv_file_name)
    return obstacles_streams


def add_obstacle_location_finder(obstacles_stream, point_cloud_stream,
                                 can_bus_stream, camera_setup):
    """Adds an operator that finds the world locations of the obstacles.

    Args:
        obstacles_stream (:py:class:`erdos.ReadStream`): Stream on which
            detected obstacles are received.
        point_cloud_stream (:py:class:`erdos.ReadStream`): Stream on
            which point cloud messages are received.
        can_bus_stream (:py:class:`erdos.ReadStream`, optional): Stream on
            which can bus info is received.
        camera_setup (:py:class:`~pylot.drivers.sensor_setup.CameraSetup`):
            The setup of the center camera.

    Returns:
        :py:class:`erdos.ReadStream`: Stream on which
        :py:class:`~pylot.perception.messages.ObstaclesMessage` messages with
        world locations are published.
    """
    [obstacles_with_loc_stream
     ] = erdos.connect(ObstacleLocationFinderOperator,
                       [obstacles_stream, point_cloud_stream, can_bus_stream],
                       camera_setup.name + '_location_finder_operator',
                       True,
                       FLAGS,
                       camera_setup,
                       log_file_name=FLAGS.log_file_name,
                       csv_file_name=FLAGS.csv_log_file_name)
    return obstacles_with_loc_stream


def add_detection_decay(ground_obstacles_stream):
    [map_stream] = erdos.connect(DetectionDecayOperator,
                                 [ground_obstacles_stream],
                                 'detection_decay_operator',
                                 True,
                                 FLAGS,
                                 log_file_name=FLAGS.log_file_name,
                                 csv_file_name=FLAGS.csv_log_file_name)
    return map_stream


def add_detection_evaluation(obstacles_stream,
                             ground_obstacles_stream,
                             name='detection_eval_operator'):
    erdos.connect(DetectionEvalOperator,
                  [obstacles_stream, ground_obstacles_stream],
                  name,
                  True,
                  FLAGS,
                  log_file_name=FLAGS.log_file_name,
                  csv_file_name=FLAGS.csv_log_file_name)


def add_traffic_light_detector(traffic_light_camera_stream):
    [traffic_lights_stream
     ] = erdos.connect(TrafficLightDetOperator, [traffic_light_camera_stream],
                       'traffic_light_detector_operator',
                       True,
                       FLAGS,
                       log_file_name=FLAGS.log_file_name,
                       csv_file_name=FLAGS.csv_log_file_name)
    return traffic_lights_stream


def add_canny_edge_lane_detection(bgr_camera_stream,
                                  name='canny_edge_lane_detection'):
    [lane_detection_stream
     ] = erdos.connect(CannyEdgeLaneDetectionOperator, [bgr_camera_stream],
                       name,
                       True,
                       FLAGS,
                       log_file_name=FLAGS.log_file_name,
                       csv_file_name=FLAGS.csv_log_file_name)
    return lane_detection_stream


def add_obstacle_tracking(obstacles_stream,
                          bgr_camera_stream,
                          name_prefix='tracker_'):
    [obstacle_tracking_stream
     ] = erdos.connect(ObjectTrackerOperator,
                       [obstacles_stream, bgr_camera_stream],
                       name_prefix + FLAGS.tracker_type,
                       True,
                       FLAGS.tracker_type,
                       FLAGS,
                       log_file_name=FLAGS.log_file_name,
                       csv_file_name=FLAGS.csv_log_file_name)
    return obstacle_tracking_stream


def add_tracking_evaluation(obstacle_tracking_stream,
                            ground_obstacles_stream,
                            name='tracking_eval_operator'):
    erdos.connect(TrackingEvalOperator,
                  [obstacle_tracking_stream, ground_obstacles_stream],
                  name,
                  True,
                  FLAGS,
                  log_file_name=FLAGS.log_file_name,
                  csv_file_name=FLAGS.csv_log_file_name)


def add_depth_estimation(left_camera_stream,
                         right_camera_stream,
                         center_camera_setup,
                         name='depth_estimation_operator'):
    try:
        from pylot.perception.depth_estimation.depth_estimation_operator\
            import DepthEstimationOperator
    except ImportError:
        raise Exception("Error importing AnyNet depth estimation.")

    [depth_estimation_stream
     ] = erdos.connect(DepthEstimationOperator,
                       [left_camera_stream, right_camera_stream],
                       name,
                       True,
                       center_camera_setup.get_transform(),
                       center_camera_setup.get_fov(),
                       FLAGS,
                       log_file_name=FLAGS.log_file_name,
                       csv_file_name=FLAGS.csv_log_file_name)
    return depth_estimation_stream


def add_segmentation(bgr_camera_stream, name='drn_segmentation_operator'):
    [segmented_stream] = erdos.connect(SegmentationDRNOperator,
                                       [bgr_camera_stream],
                                       name,
                                       True,
                                       FLAGS,
                                       log_file_name=FLAGS.log_file_name,
                                       csv_file_name=FLAGS.csv_log_file_name)
    return segmented_stream


def add_segmentation_evaluation(ground_segmented_stream,
                                segmented_stream,
                                name='segmentation_evaluation_operator'):
    erdos.connect(SegmentationEvalOperator,
                  [ground_segmented_stream, segmented_stream],
                  name,
                  True,
                  FLAGS,
                  log_file_name=FLAGS.log_file_name,
                  csv_file_name=FLAGS.csv_log_file_name)


def add_segmentation_decay(ground_segmented_stream,
                           name='segmentation_decay_operator'):
    [iou_stream] = erdos.connect(SegmentationDecayOperator,
                                 [ground_segmented_stream],
                                 name,
                                 True,
                                 FLAGS,
                                 log_file_name=FLAGS.log_file_name,
                                 csv_file_name=FLAGS.csv_log_file_name)
    return iou_stream


def add_linear_prediction(tracking_stream):
    [prediction_stream] = erdos.connect(LinearPredictorOperator,
                                        [tracking_stream],
                                        'linear_prediction_operator',
                                        True,
                                        FLAGS,
                                        log_file_name=FLAGS.log_file_name)
    return prediction_stream


def add_prediction_evaluation(can_bus_stream,
                              tracking_stream,
                              prediction_stream,
                              name='prediction_eval_operator'):
    erdos.connect(PredictionEvalOperator,
                  [can_bus_stream, tracking_stream, prediction_stream],
                  name,
                  True,
                  FLAGS,
                  log_file_name=FLAGS.log_file_name,
                  csv_file_name=FLAGS.csv_log_file_name)


def add_fot_planning(can_bus_stream,
                     prediction_stream,
                     goal_location,
                     name='fot_planning_operator'):
    from pylot.planning.frenet_optimal_trajectory.fot_planning_operator \
        import FOTPlanningOperator
    [waypoints_stream] = erdos.connect(FOTPlanningOperator,
                                       [can_bus_stream, prediction_stream],
                                       name,
                                       True,
                                       FLAGS,
                                       goal_location,
                                       log_file_name=FLAGS.log_file_name,
                                       csv_file_name=FLAGS.csv_log_file_name)
    return waypoints_stream


def add_rrt_star_planning(can_bus_stream,
                          prediction_stream,
                          goal_location,
                          name='rrt_star_planning_operator'):
    from pylot.planning.rrt_star.rrt_star_planning_operator import \
        RRTStarPlanningOperator
    [waypoints_stream] = erdos.connect(RRTStarPlanningOperator,
                                       [can_bus_stream, prediction_stream],
                                       name,
                                       True,
                                       FLAGS,
                                       goal_location,
                                       log_file_name=FLAGS.log_file_name,
                                       csv_file_name=FLAGS.csv_log_file_name)
    return waypoints_stream


def add_waypoint_planning(can_bus_stream,
                          open_drive_stream,
                          global_trajectory_stream,
                          obstacles_stream,
                          traffic_lights_stream,
                          goal_location,
                          name='waypoint_planning_operator'):
    [waypoints_stream] = erdos.connect(WaypointPlanningOperator, [
        can_bus_stream, open_drive_stream, global_trajectory_stream,
        obstacles_stream, traffic_lights_stream
    ],
                                       name,
                                       True,
                                       FLAGS,
                                       goal_location,
                                       log_file_name=FLAGS.log_file_name,
                                       csv_file_name=FLAGS.csv_log_file_name)
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
    [camera_stream] = erdos.connect(CarlaCameraDriverOperator,
                                    [vehicle_id_stream],
                                    camera_setup.get_name() + "_operator",
                                    False,
                                    camera_setup,
                                    FLAGS,
                                    log_file_name=FLAGS.log_file_name)

    return camera_stream


def add_lidar(transform, vehicle_id_stream, name='center_lidar'):
    lidar_setup = pylot.drivers.sensor_setup.create_center_lidar_setup(
        transform.location)
    point_cloud_stream = _add_lidar_driver(vehicle_id_stream, lidar_setup)
    return (point_cloud_stream, lidar_setup)


def _add_lidar_driver(vehicle_id_stream, lidar_setup):
    from pylot.drivers.carla_lidar_driver_operator import \
        CarlaLidarDriverOperator
    [point_cloud_stream] = erdos.connect(CarlaLidarDriverOperator,
                                         [vehicle_id_stream],
                                         lidar_setup.get_name() + "_operator",
                                         False,
                                         lidar_setup,
                                         FLAGS,
                                         log_file_name=FLAGS.log_file_name)
    return point_cloud_stream


def add_imu(transform, vehicle_id_stream, name='imu'):
    from pylot.drivers.carla_imu_driver_operator import CarlaIMUDriverOperator
    imu_setup = pylot.drivers.sensor_setup.IMUSetup(name, transform)
    [imu_stream] = erdos.connect(CarlaIMUDriverOperator, [vehicle_id_stream],
                                 imu_setup.get_name() + "_operator",
                                 False,
                                 imu_setup,
                                 FLAGS,
                                 log_file_name=FLAGS.log_file_name)
    return (imu_stream, imu_setup)


def add_fusion(can_bus_stream, obstacles_stream, depth_stream,
               ground_obstacles_stream):
    [obstacle_pos_stream
     ] = erdos.connect(FusionOperator,
                       [can_bus_stream, obstacles_stream, depth_stream],
                       'fusion_operator',
                       True,
                       FLAGS,
                       log_file_name=FLAGS.log_file_name,
                       csv_file_name=FLAGS.csv_log_file_name)

    if FLAGS.evaluate_fusion:
        erdos.connect(FusionVerificationOperator,
                      [ground_obstacles_stream, obstacle_pos_stream],
                      'fusion_verification_operator',
                      True,
                      log_file_name=FLAGS.log_file_name)
    return obstacle_pos_stream


def add_mpc_agent(can_bus_stream, waypoints_stream):
    [control_stream] = erdos.connect(MPCAgentOperator,
                                     [can_bus_stream, waypoints_stream],
                                     'mpc_agent_operator',
                                     True,
                                     FLAGS,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_file_name=FLAGS.csv_log_file_name)
    return control_stream


def add_pid_agent(can_bus_stream, waypoints_stream):
    input_streams = [can_bus_stream, waypoints_stream]
    [control_stream] = erdos.connect(PIDAgentOperator,
                                     input_streams,
                                     'pid_agent_operator',
                                     True,
                                     FLAGS,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_file_name=FLAGS.csv_log_file_name)
    return control_stream


def add_synchronizer(stream_to_sync_on):
    (control_stream, ) = erdos.connect(
        SynchronizerOperator,
        [stream_to_sync_on],
        'synchronizer_operator',
        False,  # Does not flow watermarks.
        FLAGS)
    return control_stream


def add_bounding_box_logging(obstacles_stream,
                             name='bounding_box_logger_operator'):
    erdos.connect(BoundingBoxLoggerOperator, [obstacles_stream],
                  name,
                  True,
                  FLAGS,
                  log_file_name=FLAGS.log_file_name)


def add_camera_logging(stream, name, filename_prefix):
    erdos.connect(CameraLoggerOperator, [stream],
                  name,
                  True,
                  FLAGS,
                  filename_prefix,
                  log_file_name=FLAGS.log_file_name)


def add_chauffeur_logging(vehicle_id_stream, can_bus_stream,
                          obstacle_tracking_stream, top_down_camera_stream,
                          top_down_segmentation_stream, top_down_camera_setup):
    from pylot.loggers.chauffeur_logger_operator import ChauffeurLoggerOperator
    erdos.connect(ChauffeurLoggerOperator, [
        vehicle_id_stream, can_bus_stream, obstacle_tracking_stream,
        top_down_camera_stream, top_down_segmentation_stream
    ], 'chauffeur_logger_operator', True, FLAGS, top_down_camera_setup)


def add_imu_logging(imu_stream, name='imu_logger_operator'):
    erdos.connect(IMULoggerOperator, [imu_stream],
                  name,
                  True,
                  FLAGS,
                  log_file_name=FLAGS.log_file_name)


def add_lidar_logging(point_cloud_stream,
                      name='lidar_logger_operator',
                      filename_prefix='carla-lidar-'):
    erdos.connect(LidarLoggerOperator, [point_cloud_stream],
                  name,
                  True,
                  FLAGS,
                  filename_prefix,
                  log_file_name=FLAGS.log_file_name)


def add_multiple_object_tracker_logging(
    obstacles_stream, name='multiple_object_tracker_logger_operator'):
    erdos.connect(MultipleObjectTrackerLoggerOperator, [obstacles_stream],
                  name,
                  True,
                  FLAGS,
                  log_file_name=FLAGS.log_file_name)


def add_trajectory_logging(obstacles_tracking_stream,
                           name='trajectory_logger_operator'):
    erdos.connect(TrajectoryLoggerOperator, [obstacles_tracking_stream],
                  name,
                  True,
                  FLAGS,
                  log_file_name=FLAGS.log_file_name)


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
    erdos.connect(LidarVisualizerOperator, [point_cloud_stream],
                  name,
                  True,
                  log_file_name=FLAGS.log_file_name)


def add_camera_visualizer(camera_stream, name):
    erdos.connect(CameraVisualizerOperator, [camera_stream],
                  name,
                  True,
                  log_file_name=FLAGS.log_file_name)


def add_imu_visualizer(imu_stream, name='imu_visualizer_operator'):
    from pylot.debug.imu_visualizer_operator import IMUVisualizerOperator
    erdos.connect(IMUVisualizerOperator, [imu_stream],
                  name,
                  True,
                  FLAGS,
                  log_file_name=FLAGS.log_file_name)


def add_can_bus_visualizer(can_bus_stream, name='can_bus_visualizer_operator'):
    erdos.connect(CanBusVisualizerOperator, [can_bus_stream],
                  name,
                  True,
                  FLAGS,
                  log_file_name=FLAGS.log_file_name)


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

    erdos.connect(TrackVisualizerOperator, [
        obstacle_tracking_stream, prediction_stream,
        top_down_segmented_camera_stream
    ],
                  name,
                  True,
                  FLAGS,
                  log_file_name=FLAGS.log_file_name)


def add_waypoint_visualizer(waypoints_stream,
                            camera_stream,
                            can_bus_stream,
                            name='waypoint_visualizer_operator'):
    erdos.connect(WaypointVisualizerOperator,
                  [waypoints_stream, camera_stream, can_bus_stream],
                  name,
                  True,
                  FLAGS,
                  log_file_name=FLAGS.log_file_name)


def add_perfect_detector(depth_camera_stream, center_camera_stream,
                         segmented_camera_stream, can_bus_stream,
                         ground_obstacles_stream,
                         ground_speed_limit_signs_stream,
                         ground_stop_signs_stream):
    from pylot.simulation.perfect_detector_operator import \
        PerfectDetectorOperator
    [obstacles_stream] = erdos.connect(PerfectDetectorOperator, [
        depth_camera_stream, center_camera_stream, segmented_camera_stream,
        can_bus_stream, ground_obstacles_stream,
        ground_speed_limit_signs_stream, ground_stop_signs_stream
    ],
                                       'perfect_detector_operator',
                                       True,
                                       FLAGS,
                                       log_file_name=FLAGS.log_file_name)
    return obstacles_stream


def add_perfect_traffic_light_detector(ground_traffic_lights_stream,
                                       center_camera_stream,
                                       depth_camera_stream,
                                       segmented_camera_stream,
                                       can_bus_stream):
    from pylot.simulation.perfect_traffic_light_detector_operator import \
        PerfectTrafficLightDetectorOperator
    [traffic_lights_stream
     ] = erdos.connect(PerfectTrafficLightDetectorOperator, [
         ground_traffic_lights_stream, center_camera_stream,
         depth_camera_stream, segmented_camera_stream, can_bus_stream
     ],
                       'perfect_traffic_light_detector_operator',
                       True,
                       FLAGS,
                       log_file_name=FLAGS.log_file_name)
    return traffic_lights_stream


def add_perfect_lane_detector(can_bus_stream):
    from pylot.simulation.perfect_lane_detector_operator import \
        PerfectLaneDetectionOperator
    [detected_lanes_stream] = erdos.connect(PerfectLaneDetectionOperator,
                                            [can_bus_stream],
                                            'perfect_lane_detection_operator',
                                            True,
                                            FLAGS,
                                            log_file_name=FLAGS.log_file_name)
    return detected_lanes_stream


def add_perfect_tracking(ground_obstacles_stream, can_bus_stream):
    from pylot.simulation.perfect_tracker_operator import \
        PerfectTrackerOperator
    [ground_tracking_stream
     ] = erdos.connect(PerfectTrackerOperator,
                       [ground_obstacles_stream, can_bus_stream],
                       'perfect_tracking_operator',
                       True,
                       FLAGS,
                       log_file_name=FLAGS.log_file_name)
    return ground_tracking_stream
