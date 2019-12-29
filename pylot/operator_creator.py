from absl import flags
import erdos

# Simulation operators.
from pylot.simulation.camera_driver_operator import CameraDriverOperator
from pylot.simulation.carla_operator import CarlaOperator
from pylot.simulation.imu_driver_operator import IMUDriverOperator
from pylot.simulation.lidar_driver_operator import LidarDriverOperator
from pylot.simulation.synchronizer_operator import SynchronizerOperator
# Perception operators.
from pylot.perception.detection.detection_operator import DetectionOperator
from pylot.perception.detection.detection_eval_operator import \
    DetectionEvalOperator
from pylot.perception.detection.detection_decay_operator import \
    DetectionDecayOperator
from pylot.perception.detection.lane_detection_operator import \
    LaneDetectionOperator
from pylot.perception.detection.traffic_light_det_operator import \
    TrafficLightDetOperator
from pylot.perception.tracking.object_tracker_operator import \
    ObjectTrackerOperator
from pylot.perception.fusion.fusion_operator import FusionOperator
from pylot.perception.fusion.fusion_verification_operator import \
    FusionVerificationOperator
from pylot.perception.segmentation.segmentation_eval_operator import \
    SegmentationEvalOperator
from pylot.perception.segmentation.segmentation_decay_operator import \
    SegmentationDecayOperator
# Planning operators.
from pylot.planning.rrt_star.rrt_star_planning_operator import \
    RRTStarPlanningOperator
from pylot.planning.waypoint_planning_operator import WaypointPlanningOperator
# Prediction operators.
from pylot.prediction.linear_predictor_operator import LinearPredictorOperator
from pylot.prediction.prediction_eval_operator import PredictionEvalOperator
# Control operators.
from pylot.control.ground_agent_operator import GroundAgentOperator
from pylot.control.mpc.mpc_agent_operator import MPCAgentOperator
from pylot.control.pid_control_operator import PIDControlOperator
from pylot.control.pylot_agent_operator import PylotAgentOperator
# Logging operators.
from pylot.loggers.bounding_box_logger_operator import \
    BoundingBoxLoggerOperator
from pylot.loggers.camera_logger_operator import CameraLoggerOperator
from pylot.loggers.chauffeur_logger_operator import ChauffeurLoggerOperator
from pylot.loggers.depth_camera_logger_operator import \
    DepthCameraLoggerOperator
from pylot.loggers.imu_logger_operator import IMULoggerOperator
from pylot.loggers.lidar_logger_operator import LidarLoggerOperator
from pylot.loggers.multiple_object_tracker_logger_operator import \
    MultipleObjectTrackerLoggerOperator
from pylot.loggers.trajectory_logger_operator import TrajectoryLoggerOperator
# Visualizing operators.
from pylot.debug.camera_visualizer_operator import CameraVisualizerOperator
from pylot.debug.can_bus_visualizer_operator import CanBusVisualizerOperator
from pylot.debug.imu_visualizer_operator import IMUVisualizerOperator
from pylot.debug.lidar_visualizer_operator import LidarVisualizerOperator
from pylot.debug.track_visualizer_operator import TrackVisualizerOperator
from pylot.debug.waypoint_visualizer_operator import WaypointVisualizerOperator
# Perfect versions of operators.
from pylot.simulation.perfect_detector_operator import PerfectDetectorOperator
from pylot.simulation.perfect_lane_detector_operator import \
    PerfectLaneDetectionOperator
from pylot.simulation.perfect_tracker_operator import PerfectTrackerOperator
from pylot.simulation.perfect_traffic_light_detector_operator import \
    PerfectTrafficLightDetectorOperator
# Sensor setups.
from pylot.simulation.sensor_setup import DepthCameraSetup, RGBCameraSetup, \
    SegmentedCameraSetup
import pylot.simulation.utils

FLAGS = flags.FLAGS


def add_carla_bridge(control_stream):
    return erdos.connect(CarlaOperator, [control_stream],
                         True,
                         'carla_operator',
                         FLAGS,
                         log_file_name=FLAGS.log_file_name,
                         csv_file_name=FLAGS.csv_log_file_name)


def add_obstacle_detection(camera_stream):
    obstacles_streams = []
    for i in range(0, len(FLAGS.obstacle_detection_model_paths)):
        obstacles_streams += erdos.connect(
            DetectionOperator, [camera_stream],
            True,
            FLAGS.obstacle_detection_model_names[i],
            FLAGS.obstacle_detection_model_paths[i],
            FLAGS,
            log_file_name=FLAGS.log_file_name,
            csv_file_name=FLAGS.csv_log_file_name)
    return obstacles_streams


def add_detection_decay(ground_obstacles_stream):
    [map_stream] = erdos.connect(DetectionDecayOperator,
                                 [ground_obstacles_stream],
                                 True,
                                 'detection_decay_operator',
                                 FLAGS,
                                 log_file_name=FLAGS.log_file_name,
                                 csv_file_name=FLAGS.csv_log_file_name)
    return map_stream


def add_detection_evaluation(obstacles_stream,
                             ground_obstacles_stream,
                             name='detection_eval_operator'):
    erdos.connect(DetectionEvalOperator,
                  [obstacles_stream, ground_obstacles_stream],
                  True,
                  name,
                  FLAGS,
                  log_file_name=FLAGS.log_file_name,
                  csv_file_name=FLAGS.csv_log_file_name)


def add_traffic_light_detector(traffic_light_camera_stream):
    [traffic_lights_stream
     ] = erdos.connect(TrafficLightDetOperator, [traffic_light_camera_stream],
                       True,
                       'traffic_light_detector_operator',
                       FLAGS,
                       log_file_name=FLAGS.log_file_name,
                       csv_file_name=FLAGS.csv_log_file_name)
    return traffic_lights_stream


def add_lane_detection(bgr_camera_stream, name='lane_detection'):
    [lane_detection_stream
     ] = erdos.connect(LaneDetectionOperator, [bgr_camera_stream],
                       True,
                       name,
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
                       True,
                       name_prefix + FLAGS.tracker_type,
                       FLAGS.tracker_type,
                       FLAGS,
                       log_file_name=FLAGS.log_file_name,
                       csv_file_name=FLAGS.csv_log_file_name)
    return obstacle_tracking_stream


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
                       True,
                       name,
                       center_camera_setup.get_transform(),
                       center_camera_setup.get_fov(),
                       FLAGS,
                       log_file_name=FLAGS.log_file_name,
                       csv_file_name=FLAGS.csv_log_file_name)
    return depth_estimation_stream


def add_segmentation(bgr_camera_stream, name='drn_segmentation_operator'):
    try:
        from pylot.perception.segmentation.segmentation_drn_operator import\
            SegmentationDRNOperator
    except ImportError:
        raise Exception("Error importing DRN segmentation.")

    [segmented_stream] = erdos.connect(SegmentationDRNOperator,
                                       [bgr_camera_stream],
                                       True,
                                       name,
                                       FLAGS,
                                       log_file_name=FLAGS.log_file_name,
                                       csv_file_name=FLAGS.csv_log_file_name)
    return segmented_stream


def add_segmentation_evaluation(ground_segmented_stream,
                                segmented_stream,
                                name='segmentation_evaluation_operator'):
    erdos.connect(SegmentationEvalOperator,
                  [ground_segmented_stream, segmented_stream],
                  True,
                  name,
                  FLAGS,
                  log_file_name=FLAGS.log_file_name,
                  csv_file_name=FLAGS.csv_log_file_name)


def add_segmentation_decay(ground_segmented_stream,
                           name='segmentation_decay_operator'):
    [iou_stream] = erdos.connect(SegmentationDecayOperator,
                                 [ground_segmented_stream],
                                 True,
                                 name,
                                 FLAGS,
                                 log_file_name=FLAGS.log_file_name,
                                 csv_file_name=FLAGS.csv_log_file_name)
    return iou_stream


def add_linear_prediction(tracking_stream):
    [prediction_stream] = erdos.connect(LinearPredictorOperator,
                                        [tracking_stream],
                                        True,
                                        'linear_prediction_operator',
                                        FLAGS,
                                        log_file_name=FLAGS.log_file_name)
    return prediction_stream


def add_prediction_evaluation(can_bus_stream,
                              tracking_stream,
                              prediction_stream,
                              name='prediction_eval_operator'):
    erdos.connect(PredictionEvalOperator,
                  [can_bus_stream, tracking_stream, prediction_stream],
                  True,
                  name,
                  FLAGS,
                  log_file_name=FLAGS.log_file_name,
                  csv_file_name=FLAGS.csv_log_file_name)


def add_waypoint_planning(can_bus_stream,
                          open_drive_stream,
                          global_trajectory_stream,
                          goal_location,
                          name='waypoint_planning_operator'):
    [waypoints_stream] = erdos.connect(
        WaypointPlanningOperator,
        [can_bus_stream, open_drive_stream, global_trajectory_stream],
        True,
        name,
        FLAGS,
        goal_location,
        log_file_name=FLAGS.log_file_name,
        csv_file_name=FLAGS.csv_log_file_name)
    return waypoints_stream


def add_rrt_start_planning(can_bus_stream,
                           prediction_stream,
                           goal_location,
                           name='rrt_star_planning_operator'):
    [waypoints_stream] = erdos.connect(RRTStarPlanningOperator,
                                       [can_bus_stream, prediction_stream],
                                       True,
                                       name,
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
        pylot.simulation.sensor_setup.create_left_right_camera_setups(
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
    [camera_stream] = erdos.connect(CameraDriverOperator, [vehicle_id_stream],
                                    False,
                                    camera_setup.get_name() + "_operator",
                                    camera_setup,
                                    FLAGS,
                                    log_file_name=FLAGS.log_file_name)

    return camera_stream


def add_lidar(transform, vehicle_id_stream, name='center_lidar'):
    lidar_setup = pylot.simulation.sensor_setup.create_center_lidar_setup(
        transform.location)
    point_cloud_stream = _add_lidar_driver(vehicle_id_stream, lidar_setup)
    return (point_cloud_stream, lidar_setup)


def _add_lidar_driver(vehicle_id_stream, lidar_setup):
    [point_cloud_stream] = erdos.connect(LidarDriverOperator,
                                         [vehicle_id_stream],
                                         False,
                                         lidar_setup.get_name() + "_operator",
                                         lidar_setup,
                                         FLAGS,
                                         log_file_name=FLAGS.log_file_name)
    return point_cloud_stream


def add_imu(transform, vehicle_id_stream, name='imu'):
    imu_setup = pylot.simulation.sensor_setup.IMUSetup(name, transform)
    [imu_stream] = erdos.connect(IMUDriverOperator, [vehicle_id_stream],
                                 False,
                                 imu_setup.get_name() + "_operator",
                                 imu_setup,
                                 FLAGS,
                                 log_file_name=FLAGS.log_file_name)
    return (imu_stream, imu_setup)


def add_fusion(can_bus_stream, obstacles_stream, depth_stream,
               ground_obstacles_stream):
    [obstacle_pos_stream
     ] = erdos.connect(FusionOperator,
                       [can_bus_stream, obstacles_stream, depth_stream],
                       True,
                       'fusion_operator',
                       FLAGS,
                       log_file_name=FLAGS.log_file_name,
                       csv_file_name=FLAGS.csv_log_file_name)

    if FLAGS.evaluate_fusion:
        erdos.connect(FusionVerificationOperator,
                      [ground_obstacles_stream, obstacle_pos_stream],
                      True,
                      'fusion_verification_operator',
                      log_file_name=FLAGS.log_file_name)
    return obstacle_pos_stream


def add_pid_control(waypoints_stream, can_bus_stream):
    [control_stream] = erdos.connect(PIDControlOperator,
                                     [waypoints_stream, can_bus_stream],
                                     True,
                                     'pid_control_operator',
                                     FLAGS,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_file_name=FLAGS.csv_log_file_name)
    return control_stream


def add_ground_agent(can_bus_stream, ground_obstacles_stream,
                     ground_traffic_lights_stream, waypoints_stream):
    [control_stream] = erdos.connect(GroundAgentOperator, [
        can_bus_stream, ground_obstacles_stream, ground_traffic_lights_stream,
        waypoints_stream
    ],
                                     True,
                                     'ground_agent_operator',
                                     FLAGS,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_file_name=FLAGS.csv_log_file_name)
    return control_stream


def add_mpc_agent(can_bus_stream, ground_obstacles_stream,
                  ground_traffic_lights_stream, waypoints_stream):
    [control_stream] = erdos.connect(MPCAgentOperator, [
        can_bus_stream, ground_obstacles_stream, ground_traffic_lights_stream,
        waypoints_stream
    ],
                                     True,
                                     'mpc_agent_operator',
                                     FLAGS,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_file_name=FLAGS.csv_log_file_name)
    return control_stream


def add_pylot_agent(can_bus_stream, waypoints_stream, traffic_lights_stream,
                    obstacles_stream, lidar_stream, open_drive_stream,
                    depth_camera_stream, camera_setup):
    input_streams = [
        can_bus_stream, waypoints_stream, traffic_lights_stream,
        obstacles_stream, lidar_stream, open_drive_stream, depth_camera_stream
    ]
    [control_stream] = erdos.connect(PylotAgentOperator,
                                     input_streams,
                                     True,
                                     'pylot_agent_operator',
                                     FLAGS,
                                     camera_setup,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_file_name=FLAGS.csv_log_file_name)
    return control_stream


def add_synchronizer(stream_to_sync_on):
    (control_stream, ) = erdos.connect(
        SynchronizerOperator,
        [stream_to_sync_on],
        False,  # Does not flow watermarks.
        FLAGS)
    return control_stream


def add_bounding_box_logging(obstacles_stream,
                             name='bounding_box_logger_operator'):
    erdos.connect(BoundingBoxLoggerOperator, [obstacles_stream],
                  True,
                  name,
                  FLAGS,
                  log_file_name=FLAGS.log_file_name)


def add_camera_logging(stream, name, filename_prefix):
    erdos.connect(CameraLoggerOperator, [stream],
                  True,
                  name,
                  FLAGS,
                  filename_prefix,
                  log_file_name=FLAGS.log_file_name)


def add_depth_camera_logging(depth_camera_stream,
                             name='depth_camera_logger_operator',
                             filename_prefix='carla-depth-'):
    erdos.connect(DepthCameraLoggerOperator, [depth_camera_stream],
                  True,
                  name,
                  FLAGS,
                  filename_prefix,
                  log_file_name=FLAGS.log_file_name)


def add_chauffeur_logging(vehicle_id_stream, can_bus_stream,
                          obstacle_tracking_stream, top_down_camera_stream,
                          top_down_segmentation_stream, top_down_camera_setup):
    erdos.connect(ChauffeurLoggerOperator, [
        vehicle_id_stream, can_bus_stream, obstacle_tracking_stream,
        top_down_camera_stream, top_down_segmentation_stream
    ], True, 'chauffeur_logger_operator', FLAGS, top_down_camera_setup)


def add_imu_logging(imu_stream, name='imu_logger_operator'):
    erdos.connect(IMULoggerOperator, [imu_stream],
                  True,
                  name,
                  FLAGS,
                  log_file_name=FLAGS.log_file_name)


def add_lidar_logging(point_cloud_stream, name='lidar_logger_operator'):
    erdos.connect(LidarLoggerOperator, [point_cloud_stream],
                  True,
                  name,
                  FLAGS,
                  log_file_name=FLAGS.log_file_name)


def add_multiple_object_tracker_logging(
    obstacles_stream, name='multiple_object_tracker_logger_operator'):
    erdos.connect(MultipleObjectTrackerLoggerOperator, [obstacles_stream],
                  True,
                  name,
                  FLAGS,
                  log_file_name=FLAGS.log_file_name)


def add_trajectory_logging(obstacles_tracking_stream,
                           name='trajectory_logger_operator'):
    erdos.connect(TrajectoryLoggerOperator, [obstacles_tracking_stream],
                  True,
                  name,
                  FLAGS,
                  log_file_name=FLAGS.log_file_name)


def add_visualizers(camera_stream, depth_camera_stream, point_cloud_stream,
                    segmented_stream, imu_stream, can_bus_stream,
                    top_down_segmented_stream, obstacle_tracking_stream,
                    prediction_stream, top_down_camera_setup):
    if FLAGS.visualize_rgb_camera:
        add_camera_visualizer(camera_stream, 'rgb_camera')
    if FLAGS.visualize_depth_camera:
        add_camera_visualizer(depth_camera_stream, 'depth_camera')
    if FLAGS.visualize_imu:
        add_imu_visualizer(imu_stream)
    if FLAGS.visualize_can_bus:
        add_can_bus_visualize(can_bus_stream)
    if FLAGS.visualize_lidar:
        add_lidar_visualizer(point_cloud_stream)
    if FLAGS.visualize_segmentation:
        add_camera_visualizer(segmented_stream, 'segmented_camera')
    if FLAGS.visualize_top_down_segmentation:
        add_camera_visualizer(top_down_segmented_stream,
                              'top_down_segmented_camera')
    if FLAGS.visualize_top_down_tracker_output:
        add_top_down_tracking_visualizer(obstacle_tracking_stream,
                                         prediction_stream,
                                         top_down_segmented_stream,
                                         top_down_camera_setup)


def add_lidar_visualizer(point_cloud_stream, name='lidar_visualizer_operator'):
    erdos.connect(LidarVisualizerOperator, [point_cloud_stream],
                  True,
                  name,
                  log_file_name=FLAGS.log_file_name)


def add_camera_visualizer(camera_stream, name):
    erdos.connect(CameraVisualizerOperator, [camera_stream],
                  True,
                  name,
                  log_file_name=FLAGS.log_file_name)


def add_imu_visualizer(imu_stream, name='imu_visualizer_operator'):
    erdos.connect(IMUVisualizerOperator, [imu_stream],
                  True,
                  name,
                  FLAGS,
                  log_file_name=FLAGS.log_file_name)


def add_can_bus_visualize(can_bus_stream, name='can_bus_visualizer_operator'):
    erdos.connect(CanBusVisualizerOperator, [can_bus_stream],
                  True,
                  name,
                  FLAGS,
                  log_file_name=FLAGS.log_file_name)


def add_top_down_tracking_visualizer(
    obstacle_tracking_stream,
    prediction_stream,
    top_down_segmented_camera_stream,
    top_down_camera_setup,
    name='top_down_tracking_visualizer_operator'):
    erdos.connect(TrackVisualizerOperator, [
        obstacle_tracking_stream, prediction_stream,
        top_down_segmented_camera_stream
    ],
                  True,
                  name,
                  FLAGS,
                  top_down_camera_setup,
                  log_file_name=FLAGS.log_file_name)


def add_waypoint_visualizer(waypoints_stream,
                            name='waypoint_visualizer_operator'):
    erdos.connect(WaypointVisualizerOperator, [waypoints_stream],
                  True,
                  name,
                  FLAGS,
                  log_file_name=FLAGS.log_file_name)


def add_perfect_detector(depth_camera_stream, center_camera_stream,
                         segmented_camera_stream, can_bus_stream,
                         ground_obstacles_stream,
                         ground_speed_limit_signs_stream,
                         ground_stop_signs_stream, camera_setup):
    [obstacles_stream] = erdos.connect(PerfectDetectorOperator, [
        depth_camera_stream, center_camera_stream, segmented_camera_stream,
        can_bus_stream, ground_obstacles_stream,
        ground_speed_limit_signs_stream, ground_stop_signs_stream
    ],
                                       True,
                                       'perfect_detector_operator',
                                       camera_setup,
                                       FLAGS,
                                       log_file_name=FLAGS.log_file_name)
    return obstacles_stream


def add_perfect_traffic_light_detector(ground_traffic_lights_stream,
                                       center_camera_stream,
                                       depth_camera_stream,
                                       segmented_camera_stream,
                                       can_bus_stream):
    [traffic_lights_stream
     ] = erdos.connect(PerfectTrafficLightDetectorOperator, [
         ground_traffic_lights_stream, center_camera_stream,
         depth_camera_stream, segmented_camera_stream, can_bus_stream
     ],
                       True,
                       'perfect_traffic_light_detector_operator',
                       FLAGS,
                       log_file_name=FLAGS.log_file_name)
    return traffic_lights_stream


def add_perfect_lane_detector(can_bus_stream):
    [detected_lanes_stream] = erdos.connect(PerfectLaneDetectionOperator,
                                            [can_bus_stream],
                                            True,
                                            'perfect_lane_detection_operator',
                                            FLAGS,
                                            log_file_name=FLAGS.log_file_name)
    return detected_lanes_stream


def add_perfect_tracking(ground_obstacles_stream, can_bus_stream):
    [ground_tracking_stream
     ] = erdos.connect(PerfectTrackerOperator,
                       [ground_obstacles_stream, can_bus_stream],
                       True,
                       'perfect_tracking_operator',
                       FLAGS,
                       log_file_name=FLAGS.log_file_name)
    return ground_tracking_stream
