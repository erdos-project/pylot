from absl import flags


# import Control operators.
from pylot.control.pylot_agent_operator import PylotAgentOperator
from pylot.control.ground_agent_operator import GroundAgentOperator
# Import debug operators.
from pylot.debug.camera_replay_operator import CameraReplayOperator
from pylot.debug.depth_camera_visualizer import DepthCameraVisualizer
from pylot.debug.lidar_visualizer_operator import LidarVisualizerOperator
from pylot.debug.segmented_video_operator import SegmentedVideoOperator
from pylot.debug.video_operator import VideoOperator
# Import logging operators.
from pylot.loggers.bounding_box_logger_operator import BoundingBoxLoggerOp
from pylot.loggers.camera_logger_operator import CameraLoggerOp
from pylot.loggers.lidar_logger_operator import LidarLoggerOp
from pylot.loggers.trajectory_logger_operator import TrajectoryLoggerOp
# Import perception operators.
from pylot.perception.detection.detection_operator import DetectionOperator
from pylot.perception.detection.detection_eval_ground_operator import DetectionEvalGroundOperator
try:
    from pylot.perception.detection.detection_center_net_operator import DetectionCenterNetOperator
except ImportError:
    print("Error importing CenterNet detector.")
from pylot.perception.detection.lane_detection_operator import LaneDetectionOperator
from pylot.perception.detection.obstacle_accuracy_operator import ObstacleAccuracyOperator
from pylot.perception.detection.traffic_light_det_operator import TrafficLightDetOperator
try:
    from pylot.perception.depth_estimation.depth_est_operator import DepthEstOperator
except ImportError:
    print("Error importing AnyNet depth estimation.")
from pylot.perception.fusion.fusion_operator import FusionOperator
from pylot.perception.fusion.fusion_verification_operator import FusionVerificationOperator
from pylot.perception.segmentation.segmentation_drn_operator import SegmentationDRNOperator
try:
    from pylot.perception.segmentation.segmentation_dla_operator import SegmentationDLAOperator
except ImportError:
    print("Error importing DLA segmentation.")
from pylot.perception.segmentation.segmentation_eval_operator import SegmentationEvalOperator
from pylot.perception.segmentation.segmentation_eval_ground_operator import SegmentationEvalGroundOperator
from pylot.perception.tracking.object_tracker_operator import ObjectTrackerOp
# Import planning operators.
from pylot.planning.legacy_planning_operator import LegacyPlanningOperator

FLAGS = flags.FLAGS


def create_carla_legacy_op(graph, camera_setups, lidar_setups, auto_pilot):
    # Import operator that works with Carla 0.8.4
    from pylot.simulation.carla_legacy_operator import CarlaLegacyOperator
    carla_op = graph.add(
        CarlaLegacyOperator,
        name='carla',
        init_args={
            'flags': FLAGS,
            'auto_pilot': auto_pilot,
            'camera_setups': camera_setups,
            'lidar_setups': lidar_setups,
            'log_file_name': FLAGS.log_file_name,
            'csv_file_name': FLAGS.csv_log_file_name
        },
        setup_args={
            'camera_setups': camera_setups,
            'lidar_setups': lidar_setups
        })
    return carla_op


def create_carla_op(graph, auto_pilot):
    from pylot.simulation.carla_operator import CarlaOperator
    carla_op = graph.add(
        CarlaOperator,
        name='carla',
        init_args={
            'auto_pilot': auto_pilot,
            'flags': FLAGS,
            'log_file_name': FLAGS.log_file_name,
            'csv_file_name': FLAGS.csv_log_file_name
        })
    return carla_op


def create_carla_replay_op(graph):
    from pylot.simulation.carla_replay_operator import CarlaReplayOperator
    carla_replay_op = graph.add(
        CarlaReplayOperator,
        name='carla_replay',
        init_args={
            'flags': FLAGS,
            'log_file_name': FLAGS.log_file_name,
            'csv_file_name': FLAGS.csv_log_file_name

        })
    return carla_replay_op


def create_camera_driver_op(graph, camera_setup):
    from pylot.simulation.camera_driver_operator import CameraDriverOperator
    camera_op = graph.add(
        CameraDriverOperator,
        name=camera_setup.name,
        init_args={
            'camera_setup': camera_setup,
            'flags': FLAGS,
            'log_file_name': FLAGS.log_file_name
        },
        setup_args={'camera_setup': camera_setup})
    return camera_op


def create_driver_ops(graph, camera_setups, lidar_setups, auto_pilot=False):
    camera_ops = []
    lidar_ops = []
    if '0.8' in FLAGS.carla_version:
        carla_op = create_carla_legacy_op(
            graph, camera_setups, lidar_setups, auto_pilot)
        # The legacy carla op implements the camera drivers.
        camera_ops = [carla_op]
        lidar_ops = [carla_op]
    elif '0.9' in FLAGS.carla_version:
        if FLAGS.carla_replay_file == '':
            carla_op = create_carla_op(graph, auto_pilot)
        else:
            carla_op = create_carla_replay_op(graph)
        camera_ops = [create_camera_driver_op(graph, cs)
                      for cs in camera_setups]
        lidar_ops = [create_lidar_driver_op(graph, ls)
                     for ls in lidar_setups]
        graph.connect([carla_op], camera_ops + lidar_ops)
    else:
        raise ValueError(
            'Unexpected Carla version {}'.format(FLAGS.carla_version))
    return (carla_op, camera_ops, lidar_ops)


def create_camera_logger_op(graph):
    camera_logger_op = graph.add(
        CameraLoggerOp,
        name='camera_logger_op',
        init_args={'flags': FLAGS,
                   'log_file_name': FLAGS.log_file_name,
                   'csv_file_name': FLAGS.csv_log_file_name})
    return camera_logger_op


def create_bounding_box_logger_op(graph):
    bbox_logger_op = graph.add(
        BoundingBoxLoggerOp,
        name='bounding_box_logger',
        init_args={'flags': FLAGS})
    return bbox_logger_op


def create_lidar_driver_op(graph, lidar_setup):
    from pylot.simulation.lidar_driver_operator import LidarDriverOperator
    lidar_op = graph.add(
        LidarDriverOperator,
        name=lidar_setup.name,
        init_args={
            'lidar_setup': lidar_setup,
            'flags': FLAGS,
            'log_file_name': FLAGS.log_file_name
        },
        setup_args={'lidar_setup': lidar_setup})
    return lidar_op


def create_lidar_logger_op(graph):
    lidar_logger_op = graph.add(
        LidarLoggerOp,
        name='lidar_logger_op',
        init_args={'flags': FLAGS,
                   'log_file_name': FLAGS.log_file_name,
                   'csv_file_name': FLAGS.csv_log_file_name})
    return lidar_logger_op

def create_trajectory_logger_op(graph):
    trajectory_logger_op = graph.add(
        TrajectoryLoggerOp,
        name='trajectory_logger_op',
        init_args={'flags': FLAGS})
    return trajectory_logger_op

def create_planning_op(graph, goal_location):
    from pylot.planning.planning_operator import PlanningOperator
    planning_op = graph.add(
        PlanningOperator,
        name='planning',
        init_args={
            'goal_location': goal_location,
            'flags': FLAGS,
            'log_file_name': FLAGS.log_file_name,
            'csv_file_name': FLAGS.csv_log_file_name
        })
    return planning_op


def create_control_op(graph):
    from pylot.control.pid_control_operator import PIDControlOperator
    control_op = graph.add(
        PIDControlOperator,
        name='controller',
        init_args={
            'longitudinal_control_args': {
                'K_P': FLAGS.pid_p,
                'K_I': FLAGS.pid_i,
                'K_D': FLAGS.pid_d,
            },
            'flags': FLAGS,
            'log_file_name': FLAGS.log_file_name,
            'csv_file_name': FLAGS.csv_log_file_name
        })
    return control_op


def create_waypoint_visualizer_op(graph):
    from pylot.debug.waypoint_visualize_operator import WaypointVisualizerOperator
    waypoint_viz_op = graph.add(
        WaypointVisualizerOperator,
        name='waypoint_viz',
        init_args={
            'flags': FLAGS,
            'log_file_name': FLAGS.log_file_name
        })
    return waypoint_viz_op


def create_camera_replay_ops(graph):
    camera_ops = []
    for i in range(0, FLAGS.num_cameras, 1):
        op_name = 'camera{}'.format(i)
        camera_op = graph.add(
            CameraReplayOperator,
            name=op_name,
            init_args={'log_file_name': FLAGS.log_file_name},
            setup_args={'op_name': op_name})
        camera_ops.append(camera_op)
    # replay_rgb_op = ReplayOp('pylot_rgb_camera_data.erdos',
    #                          frequency=10,
    #                          name='replay_rgb_camera')
    # camera_streams = replay_rgb_op([])
    return camera_ops


def create_pylot_agent_op(graph, bgr_camera_setup):
    agent_op = graph.add(
        PylotAgentOperator,
        name='pylot_agent',
        init_args={
            'flags': FLAGS,
            'bgr_camera_setup': bgr_camera_setup,
            'log_file_name': FLAGS.log_file_name,
            'csv_file_name': FLAGS.csv_log_file_name
        })
    return agent_op


def create_ground_agent_op(graph):
    agent_op = graph.add(
        GroundAgentOperator,
        name='ground_agent',
        # TODO(ionel): Do not hardcode city name!
        init_args={
            'city_name': 'Town01',
            'flags': FLAGS,
            'log_file_name': FLAGS.log_file_name,
            'csv_file_name': FLAGS.csv_log_file_name
        })
    return agent_op


def create_legacy_planning_op(
        graph, city_name, goal_location, goal_orientation):
    planning_op = graph.add(
        LegacyPlanningOperator,
        name='legacy_planning',
        init_args={
            'city_name': city_name,
            'goal_location': goal_location,
            'goal_orientation': goal_orientation,
            'flags': FLAGS,
            'log_file_name': FLAGS.log_file_name,
            'csv_file_name': FLAGS.csv_log_file_name
        })
    return planning_op


def create_lidar_visualizer_op(graph):
    lidar_visualizer_op = graph.add(
        LidarVisualizerOperator,
        name='lidar_visualizer',
        init_args={'flags': FLAGS,
                   'log_file_name': FLAGS.log_file_name})
    return lidar_visualizer_op


def create_lidar_record_op(graph):
    record_lidar_op = graph.add(
        RecordOp,
        name='record_lidar',
        init_args={'filename': 'pylot_lidar_data.erdos'},
        setup_args={'filter': 'lidar'})
    return record_lidar_op


def create_camera_video_op(graph, name, filter_name):
    video_op = graph.add(
        VideoOperator,
        name=name,
        init_args={'flags': FLAGS,
                   'log_file_name': FLAGS.log_file_name},
        setup_args={'filter_name': filter_name})
    return video_op


def create_depth_camera_visualizer_op(graph, name, filter_name):
    depth_visualizer_op = graph.add(
        DepthCameraVisualizer,
        name=name,
        init_args={'flags': FLAGS,
                   'log_file_name': FLAGS.log_file_name},
        setup_args={'filter_name': filter_name})
    return depth_visualizer_op


def create_segmented_video_op(graph):
    segmented_video_op = graph.add(
        SegmentedVideoOperator,
        name='segmented_video',
        init_args={'flags': FLAGS,
                   'log_file_name': FLAGS.log_file_name})
    return segmented_video_op


def create_record_op(graph, name, filename, filter_name):
    record_op = graph.add(
        RecordOp,
        name=name,
        init_args={'filename': filename},
        setup_args={'filter_name': filter_name})
    return record_op


def create_record_carla_op(graph):
    input_names = [
        'can_bus', 'traffic_lights', 'pedestrians', 'vehicles', 'traffic_signs'
    ]
    record_carla_op = graph.add(
        RecordOp,
        name='record_carla',
        init_args={'filename': 'pylot_carla_data.erdos'},
        setup_args={'filter': input_names})
    return record_carla_op


def create_perfect_detector_op(graph, bgr_camera_setup, output_stream_name):
    from pylot.simulation.perfect_detector_operator import PerfectDetectorOp
    perfect_det_op = graph.add(
        PerfectDetectorOp,
        name='perfect_detector',
        init_args={'bgr_camera_setup': bgr_camera_setup,
                   'output_stream_name': output_stream_name,
                   'flags': FLAGS,
                   'log_file_name': FLAGS.log_file_name,
                   'csv_file_name': FLAGS.csv_log_file_name},
        setup_args={'output_stream_name': output_stream_name})
    return perfect_det_op


def create_perfect_lane_detector_op(graph, output_stream_name):
    from pylot.simulation import perfect_lane_detector_operator
    perfect_lane_detector_op = graph.add(
        perfect_lane_detector_operator.PerfectLaneDetectionOperator,
        name='perfect_lane_detection_op',
        init_args={
            'output_stream_name': output_stream_name,
            'flags': FLAGS,
            'log_file_name': FLAGS.log_file_name,
            'csv_file_name': FLAGS.csv_log_file_name
        },
        setup_args={'output_stream_name': output_stream_name})
    return perfect_lane_detector_op


def create_detector_op_helper(graph, name, model_path, gpu_memory_fraction):
    obj_detector_op = graph.add(
        DetectionOperator,
        name=name,
        setup_args={'output_stream_name': 'obj_stream'},
        init_args={'output_stream_name': 'obj_stream',
                   'model_path': model_path,
                   'flags': FLAGS,
                   'log_file_name': FLAGS.log_file_name,
                   'csv_file_name': FLAGS.csv_log_file_name},
        _resources = {"GPU": gpu_memory_fraction})
    return obj_detector_op

def create_perfect_tracking_op(graph, output_stream_name):
    from pylot.simulation.perfect_tracker_operator import PerfectTrackerOp
    perfect_tracker_op = graph.add(
        PerfectTrackerOp,
        name='perfect_tracker',
        init_args={'output_stream_name': output_stream_name,
            'flags': FLAGS,
            'log_file_name': FLAGS.log_file_name,
            'csv_file_name': FLAGS.csv_log_file_name
        },
        setup_args={'output_stream_name': output_stream_name})
    return perfect_tracker_op

def create_depth_estimation_op(graph, center_transform,
                               left_camera_name, right_camera_name):
    depth_estimation_op = graph.add(
        DepthEstOperator,
        name='depth_estimation',
        init_args={
            'output_stream_name': 'depth_estimation',
            'transform': center_transform,
            'flags': FLAGS,
            'log_file_name': FLAGS.log_file_name
        },
        setup_args={
            'output_stream_name': 'depth_estimation',
            'left_camera_name': left_camera_name,
            'right_camera_name': right_camera_name})
    return depth_estimation_op


def create_detector_ops(graph):
    detector_ops = []
    if FLAGS.detector_ssd_mobilenet_v1:
        detector_ops.append(create_detector_op_helper(
            graph,
            'detector_ssd_mobilenet_v1',
            FLAGS.detector_ssd_mobilenet_v1_model_path,
            FLAGS.obj_detection_gpu_memory_fraction))
    if FLAGS.detector_frcnn_resnet101:
        detector_ops.append(create_detector_op_helper(
            graph,
            'detector_faster_rcnn_resnet101',
            FLAGS.detector_frcnn_resnet101_model_path,
            FLAGS.obj_detection_gpu_memory_fraction))
    if FLAGS.detector_ssd_resnet50_v1:
        detector_ops.append(create_detector_op_helper(
            graph,
            'detector_ssd_resnet50_v1',
            FLAGS.detector_ssd_resnet50_v1_model_path,
            FLAGS.obj_detection_gpu_memory_fraction))
    if FLAGS.detector_center_net:
        obj_det_op = graph.add(
            DetectionCenterNetOperator,
            name='detector_center_net',
            setup_args={'output_stream_name': 'obj_stream'},
            init_args={'output_stream_name': 'obj_stream',
                       'flags': FLAGS,
                       'log_file_name': FLAGS.log_file_name,
                       'csv_file_name': FLAGS.csv_log_file_name})
        detector_ops.append(obj_det_op)
    return detector_ops


def create_eval_ground_truth_detector_op(graph):
    ground_truth_op = graph.add(
        DetectionEvalGroundOperator,
        name='eval_ground_detection',
        init_args={
            'flags': FLAGS,
            'log_file_name': FLAGS.log_file_name,
            'csv_file_name': FLAGS.csv_log_file_name
        },
    )
    return ground_truth_op


def create_obstacle_accuracy_op(graph, ground_obstacles_stream_name):
    obstacle_accuracy_op = graph.add(
        ObstacleAccuracyOperator,
        name='obstacle_accuracy',
        setup_args={'ground_obstacles_stream': ground_obstacles_stream_name},
        init_args={'flags': FLAGS,
                   'log_file_name': FLAGS.log_file_name,
                   'csv_file_name': FLAGS.csv_log_file_name})
    return obstacle_accuracy_op


def create_traffic_light_op(graph):
    traffic_light_det_op = graph.add(
        TrafficLightDetOperator,
        name='traffic_light_detector',
        setup_args={'output_stream_name': 'detected_traffic_lights'},
        init_args={'output_stream_name': 'detected_traffic_lights',
                   'flags': FLAGS,
                   'log_file_name': FLAGS.log_file_name,
                   'csv_file_name': FLAGS.csv_log_file_name},
        _resources = {"GPU": FLAGS.traffic_light_det_gpu_memory_fraction})
    return traffic_light_det_op


def create_object_tracking_op(graph):
    tracker_op = None
    name = 'tracker_' + FLAGS.tracker_type
    setup_args = {'output_stream_name': 'tracker_stream'}
    init_args = {'output_stream_name': 'tracker_stream',
                 'tracker_type': FLAGS.tracker_type,
                 'flags': FLAGS,
                 'log_file_name': FLAGS.log_file_name,
                 'csv_file_name': FLAGS.csv_log_file_name}
    if FLAGS.tracker_type == 'cv2':
        # Doesn't require a GPU.
        tracker_op = graph.add(
            ObjectTrackerOp,
            name=name,
            setup_args=setup_args,
            init_args=init_args)
    else:
        # Other trackers require a GPU.
        tracker_op = graph.add(
            ObjectTrackerOp,
            name=name,
            setup_args=setup_args,
            init_args=init_args,
            _resources = {"GPU": FLAGS.obj_tracking_gpu_memory_fraction})
    return tracker_op


def create_segmentation_drn_op(graph):
    segmentation_op = graph.add(
        SegmentationDRNOperator,
        name='segmentation_drn',
        setup_args={'output_stream_name': 'segmented_stream'},
        init_args={'output_stream_name': 'segmented_stream',
                   'flags': FLAGS,
                   'log_file_name': FLAGS.log_file_name,
                   'csv_file_name': FLAGS.csv_log_file_name},
        _resources = {"GPU": FLAGS.segmentation_drn_gpu_memory_fraction})
    return segmentation_op


def create_segmentation_dla_op(graph):
    segmentation_op = graph.add(
        SegmentationDLAOperator,
        name='segmentation_dla',
        setup_args={'output_stream_name': 'segmented_stream'},
        init_args={'output_stream_name': 'segmented_stream',
                   'flags': FLAGS,
                   'log_file_name': FLAGS.log_file_name,
                   'csv_file_name': FLAGS.csv_log_file_name},
        _resources = {"GPU": FLAGS.segmentation_dla_gpu_memory_fraction})
    return segmentation_op


def create_segmentation_eval_op(graph, ground_stream_name, segmented_stream_name):
    segmentation_eval_op = graph.add(
        SegmentationEvalOperator,
        name='segmentation_eval',
        init_args={'flags': FLAGS,
                   'log_file_name': FLAGS.log_file_name,
                   'csv_file_name': FLAGS.csv_log_file_name},
        setup_args={'ground_stream_name': ground_stream_name,
                    'segmented_stream_name': segmented_stream_name})
    return segmentation_eval_op


def create_lane_detection_op(graph):
    lane_det_op = graph.add(
        LaneDetectionOperator,
        name='lane_detection',
        init_args={'output_stream_name': 'lane_det',
                   'flags': FLAGS,
                   'log_file_name': FLAGS.log_file_name,
                   'csv_file_name': FLAGS.csv_log_file_name},
        setup_args={'output_stream_name': 'lane_det'})
    return lane_det_op


def create_segmentation_ground_eval_op(graph, ground_stream_name):
    seg_eval_op = graph.add(
        SegmentationEvalGroundOperator,
        name='segmentation_ground_eval',
        init_args={'flags': FLAGS,
                   'log_file_name': FLAGS.log_file_name,
                   'csv_file_name': FLAGS.csv_log_file_name},
        setup_args={'ground_stream_name': ground_stream_name})
    return seg_eval_op


def create_fusion_ops(graph):
    fusion_op = graph.add(
        FusionOperator,
        name='fusion',
        setup_args={'output_stream_name': 'fusion_vehicles'},
        init_args={'flags': FLAGS,
                   'output_stream_name': 'fusion_vehicles',
                   'log_file_name': FLAGS.log_file_name,
                   'csv_file_name': FLAGS.csv_log_file_name})
    fusion_verification_op = graph.add(
        FusionVerificationOperator,
        name='fusion_verifier',
        init_args={'log_file_name': FLAGS.log_file_name})
    return (fusion_op, fusion_verification_op)


def add_visualization_operators(graph,
                                camera_ops,
                                lidar_ops,
                                rgb_camera_name,
                                depth_camera_name):
    if FLAGS.visualize_rgb_camera:
        camera_video_op = create_camera_video_op(graph,
                                                 'rgb_camera',
                                                 rgb_camera_name)
        graph.connect(camera_ops, [camera_video_op])

    if FLAGS.visualize_depth_camera:
        depth_video_op = create_depth_camera_visualizer_op(
            graph,
            'depth_camera_video',
            depth_camera_name)
        graph.connect(camera_ops, [depth_video_op])

    if FLAGS.visualize_lidar:
        lidar_visualizer_op = create_lidar_visualizer_op(graph)
        graph.connect(lidar_ops, [lidar_visualizer_op])

    if FLAGS.visualize_segmentation:
        # Segmented camera. The stream comes from CARLA.
        segmented_video_op = create_segmented_video_op(graph)
        graph.connect(camera_ops, [segmented_video_op])


def add_recording_operators(graph,
                            camera_ops,
                            carla_op,
                            lidar_ops=None,
                            rgb_camera_name=None,
                            depth_camera_name=None):
    if FLAGS.record_rgb_camera:
        record_rgb_op = create_record_op(graph,
                                         'record_rgb_camera',
                                         'pylot_rgb_camera_data.erdos',
                                         rgb_camera_name)
        graph.connect(camera_ops, [record_rgb_op])

    if FLAGS.record_depth_camera:
        record_depth_camera_op = create_record_op(
            graph,
            'record_depth_camera',
            'pylot_depth_camera_data.erdos',
            depth_camera_name)
        graph.connect(camera_ops, [record_depth_camera_op])

    if FLAGS.record_lidar:
        record_lidar_op = create_lidar_record_op(graph)
        graph.connect(lidar_ops, [record_lidar_op])

    if FLAGS.record_ground_truth:
        record_carla_op = create_record_carla_op(graph)
        graph.connect([carla_op], [record_carla_op])
