from absl import flags

# import Control operators.
from control.erdos_agent_operator import ERDOSAgentOperator
from control.ground_agent_operator import GroundAgentOperator
# Import debug operators.
from debug.camera_replay_operator import CameraReplayOperator
from debug.depth_camera_visualizer import DepthCameraVisualizer
from debug.depth_estimation_operator import DepthEstimationOp
from debug.lidar_visualizer_operator import LidarVisualizerOperator
from debug.segmented_video_operator import SegmentedVideoOperator
from debug.video_operator import VideoOperator
# Import perception operators.
from perception.detection.detection_operator import DetectionOperator
from perception.detection.detection_eval_ground_operator import DetectionEvalGroundOperator
try:
    from perception.detection.detection_center_net_operator import DetectionCenterNetOperator
except ImportError:
    print("Error importing CenterNet detector.")
from perception.detection.lane_detection_operator import LaneDetectionOperator
from perception.detection.obstacle_accuracy_operator import ObstacleAccuracyOperator
from perception.detection.traffic_light_det_operator import TrafficLightDetOperator
from perception.fusion.fusion_operator import FusionOperator
from perception.fusion.fusion_verification_operator import FusionVerificationOperator
from perception.segmentation.segmentation_drn_operator import SegmentationDRNOperator
try:
    from perception.segmentation.segmentation_dla_operator import SegmentationDLAOperator
except ImportError:
    print("Error importing DLA segmentation.")
from perception.segmentation.segmentation_eval_operator import SegmentationEvalOperator
from perception.segmentation.segmentation_eval_ground_operator import SegmentationEvalGroundOperator
from perception.tracking.object_tracker_operator import ObjectTrackerOp
# Import planning operators.
from planning.waypointer_operator import WaypointerOperator

FLAGS = flags.FLAGS


def create_carla_legacy_op(graph, camera_setups, lidar_setups):
    # Import operator that works with Carla 0.8.4
    from simulation.carla_legacy_operator import CarlaLegacyOperator
    carla_op = graph.add(
        CarlaLegacyOperator,
        name='carla',
        init_args={
            'flags': FLAGS,
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


def create_carla_op(graph):
    from simulation.carla_operator import CarlaOperator
    carla_op = graph.add(
        CarlaOperator,
        name='carla',
        init_args={
            'flags': FLAGS,
            'log_file_name': FLAGS.log_file_name,
            'csv_file_name': FLAGS.csv_log_file_name
        })
    return carla_op


def create_camera_driver_op(graph, camera_setup):
    from simulation.camera_driver_operator import CameraDriverOperator
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


def create_lidar_driver_op(graph, lidar_setup):
    from simulation.lidar_driver_operator import LidarDriverOperator
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


def create_planning_op(graph, goal_location):
    from planning.planning_operator import PlanningOperator
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
    from control.pid_control_operator import PIDControlOperator
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
    from debug.waypoint_visualize_operator import WaypointVisualizerOperator
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


def create_erdos_agent_op(graph, depth_camera_name):
    agent_op = graph.add(
        ERDOSAgentOperator,
        name='erdos_agent',
        init_args={
            'flags': FLAGS,
            'log_file_name': FLAGS.log_file_name,
            'csv_file_name': FLAGS.csv_log_file_name
        },
        setup_args={'depth_camera_name': depth_camera_name})
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


def create_waypointer_op(graph, goal_location, goal_orientation):
    waypointer_op = graph.add(
        WaypointerOperator,
        name='waypointer',
        # TODO(ionel): Do not hardcode Town name!
        init_args={
            'city_name': 'Town01',
            'goal_location': goal_location,
            'goal_orientation': goal_orientation,
            'flags': FLAGS,
            'log_file_name': FLAGS.log_file_name,
            'csv_file_name': FLAGS.csv_log_file_name
        })
    return waypointer_op


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


def create_depth_estimation_op(graph, left_camera_name, right_camera_name):
    depth_estimation_op = graph.add(
        DepthEstimationOp,
        name='depth_estimation',
        init_args={
            'flags': FLAGS,
            'log_file_name': FLAGS.log_file_name
        },
        setup_args={
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


def create_eval_ground_truth_detector_op(graph,
                                         rgb_camera_setup,
                                         depth_camera_name):
    ground_truth_op = graph.add(
        DetectionEvalGroundOperator,
        name='eval_ground_detection',
        setup_args={'depth_camera_name': depth_camera_name},
        init_args={
            'rgb_camera_setup': rgb_camera_setup,
            'flags': FLAGS,
            'log_file_name': FLAGS.log_file_name,
            'csv_file_name': FLAGS.csv_log_file_name
        },
    )
    return ground_truth_op


def create_obstacle_accuracy_op(graph,
                             rgb_camera_setup,
                             depth_camera_name):
    obstacle_accuracy_op = graph.add(
        ObstacleAccuracyOperator,
        name='obstacle_accuracy',
        setup_args={'depth_camera_name': depth_camera_name},
        init_args={'rgb_camera_setup': rgb_camera_setup,
                   'flags': FLAGS,
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
