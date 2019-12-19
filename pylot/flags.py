from absl import flags

import pylot.control.flags

FLAGS = flags.FLAGS

# Files where to log information.
flags.DEFINE_string('log_file_name', None, 'Name of the log file')
flags.DEFINE_string('csv_log_file_name', None,
                    'csv file into which to log runtime stats')

######################################################################
# Perception
######################################################################
flags.DEFINE_bool('obj_detection', False,
                  'True to enable object detection operator')
flags.DEFINE_bool('detector_ssd_mobilenet_v1', False,
                  'True to enable SSD mobilenet v1 detector')
flags.DEFINE_string(
    'detector_ssd_mobilenet_v1_model_path',
    'dependencies/models/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb',
    'Path to the model')
flags.DEFINE_bool('detector_frcnn_resnet101', False,
                  'True to enable faster RCNN resnet101 detector')
flags.DEFINE_string(
    'detector_frcnn_resnet101_model_path',
    'dependencies/models/faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.pb',
    'Path to the model')
flags.DEFINE_bool('detector_ssd_resnet50_v1', False,
                  'True to enable SSD resnet50 v1 detector')
flags.DEFINE_string(
    'detector_ssd_resnet50_v1_model_path',
    'dependencies/models/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb',
    'Path to the model')
flags.DEFINE_bool('obj_tracking', False,
                  'True to enable object tracking operator')
flags.DEFINE_enum('tracker_type',
                  'cv2',
                  ['cv2', 'da_siam_rpn', 'deep_sort', 'sort'],
                  'Tracker type')
flags.DEFINE_bool('lane_detection', False, 'True to enable lane detection')
flags.DEFINE_bool('fusion', False, 'True to enable fusion operator')
flags.DEFINE_bool('evaluate_fusion', False, 'True to enable fusion evaluation')
flags.DEFINE_bool('traffic_light_det', False,
                  'True to enable traffic light detection operator')
flags.DEFINE_bool('segmentation', False,
                  'True to enable segmantation operator')
flags.DEFINE_bool('depth_estimation', False,
                  'True to depth estimation using cameras')
flags.DEFINE_float(
    'offset_left_right_cameras',
    0.4,
    'How much we offset the left and right cameras from the center.')

######################################################################
# Prediction
######################################################################
flags.DEFINE_bool('prediction', False, 'True to enable prediction.')
flags.DEFINE_enum('prediction_type',
                  'linear',
                  ['linear'],
                  'Type of prediction module to use')

######################################################################
# Planning
######################################################################
flags.DEFINE_enum('planning_type',
                  'single_waypoint',
                  ['single_waypoint', 'multiple_waypoints', 'rrt_star'],
                  'Type of planning module to use')
flags.DEFINE_bool('imu', False, 'True to enable the IMU sensor')

######################################################################
# Control
######################################################################
flags.DEFINE_enum('control_agent_operator',
                  'pylot',
                  ['pylot', 'ground', 'mpc'],
                  'Control agent operator to use to drive')

######################################################################
# Perfect components
######################################################################
flags.DEFINE_bool('use_perfect_perception', False,
                  'True to enable the agent to use perfect ground detection')
flags.DEFINE_bool('perfect_tracking', False,
                  'True to enable perfect object tracker.')

######################################################################
# Carla flags
######################################################################
flags.DEFINE_integer('carla_camera_image_width', 1920,
                     'Carla camera image width')
flags.DEFINE_integer('carla_camera_image_height', 1080,
                     'Carla camera image height')

# Visualizing operators
flags.DEFINE_bool('visualize_depth_camera', False,
                  'True to enable depth camera video operator')
flags.DEFINE_bool('visualize_lidar', False,
                  'True to enable CARLA Lidar visualizer operator')
flags.DEFINE_bool('visualize_depth_estimation', False,
                  'True to enable depth estimation visualization')
flags.DEFINE_bool('visualize_rgb_camera', False,
                  'True to enable RGB camera video operator')
flags.DEFINE_bool('visualize_imu', False,
                  'True to enable CARLA IMU visualizer operator')
flags.DEFINE_bool('visualize_segmentation', False,
                  'True to enable CARLA segmented video operator')
flags.DEFINE_bool('visualize_ground_obstacles', False,
                  'True to enable visualization of ground obstacles')
flags.DEFINE_bool('visualize_tracker_output', False,
                  'True to enable visualization of tracker output')
flags.DEFINE_bool('visualize_segmentation_output', False,
                  'True to enable visualization of segmentation output')
flags.DEFINE_bool('visualize_detector_output', False,
                  'True to enable visualization of detector output')
flags.DEFINE_bool('visualize_traffic_light_output', False,
                  'True to enable visualization of traffic light output')
flags.DEFINE_bool('visualize_lane_detection', False,
                  'True to visualize lane detection')
flags.DEFINE_bool('visualize_waypoints', False, 'True to visualize waypoints')
flags.DEFINE_bool('visualize_can_bus', False, 'True to visualize can bus.')
flags.DEFINE_bool('visualize_top_down_segmentation', False,
                  'True to visualize top-down segmentation')
flags.DEFINE_bool('visualize_top_down_tracker_output', False,
                  'True to enable visualization of top-down tracker output')

# Accuracy evaluation flags.
flags.DEFINE_bool('evaluate_obj_detection', False,
                  'True to enable object detection accuracy evaluation')
flags.DEFINE_bool('evaluate_segmentation', False,
                  'True to enable segmentation evaluation')

########################################
# Recording operators.
########################################
flags.DEFINE_string('data_path', 'data/', 'Path where to logged data')
flags.DEFINE_bool('log_detector_output', False,
                  'Enable recording of bbox annotated detector images')
flags.DEFINE_bool('log_traffic_light_detector_output', False,
                  'Enable recording of bbox annotated tl detector images')

# Other flags
flags.DEFINE_integer('top_down_lateral_view', 20,
                     'Distance in meters to the left and right of the '
                     'ego-vehicle that the top-down camera shows.')

# Flag validators.
flags.register_multi_flags_validator(
    ['obj_detection', 'detector_ssd_mobilenet_v1',
     'detector_frcnn_resnet101', 'detector_ssd_resnet50_v1'],
    lambda flags_dict: (not flags_dict['obj_detection'] or
                        (flags_dict['obj_detection'] and
                         (flags_dict['detector_ssd_mobilenet_v1'] or
                          flags_dict['detector_frcnn_resnet101'] or
                          flags_dict['detector_ssd_resnet50_v1']))),
    message='a detector must be active when --obj_detection is set')


def tracker_flag_validator(flags_dict):
    if flags_dict['obj_tracking']:
        return flags_dict['obj_detection']
    return True


flags.register_multi_flags_validator(
    ['obj_detection', 'obj_tracking'],
    tracker_flag_validator,
    message='--obj_detection must be set if --obj_tracking is set')


def detector_accuracy_validator(flags_dict):
    if flags_dict['evaluate_obj_detection']:
        return flags_dict['obj_detection']
    return True


flags.register_multi_flags_validator(
    ['obj_detection', 'evaluate_obj_detection'],
    detector_accuracy_validator,
    message='--obj_detection must be set if --evaluate_obj_detection is set')
