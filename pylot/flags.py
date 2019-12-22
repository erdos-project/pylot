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
flags.DEFINE_bool('obstacle_detection', False,
                  'True to enable obstacle detection operator')
flags.DEFINE_bool('perfect_obstacle_detection', False,
                  'True to enable perfect obstacle detection')
flags.DEFINE_list('obstacle_detection_model_paths',
                  'dependencies/models/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb',
                  'Comma-separated list of model paths')
flags.DEFINE_list('obstacle_detection_model_names',
                  'ssd_resnet50_v1_fpn',
                  'Comma-separated list of model names')
flags.DEFINE_bool('obstacle_tracking', False,
                  'True to enable obstacle tracking operator')
flags.DEFINE_bool('perfect_obstacle_tracking', False,
                  'True to enable perfect obstacle tracking')
flags.DEFINE_enum('tracker_type',
                  'cv2',
                  ['cv2', 'da_siam_rpn', 'deep_sort', 'sort'],
                  'Tracker type')
flags.DEFINE_bool('lane_detection', False, 'True to enable lane detection')
flags.DEFINE_bool('perfect_lane_detection', False,
                  'True to enable perfect lane detection')
flags.DEFINE_bool('fusion', False, 'True to enable fusion operator')
flags.DEFINE_bool('traffic_light_detection', False,
                  'True to enable traffic light detection operator')
flags.DEFINE_bool('perfect_traffic_light_detection', False,
                  'True to enable perfect traffic light detection')
flags.DEFINE_bool('segmentation', False,
                  'True to enable segmentation operator')
flags.DEFINE_bool('perfect_segmentation', False,
                  'True to enable perfect segmentation')
flags.DEFINE_bool('depth_estimation', False,
                  'True to estimate depth using cameras')
flags.DEFINE_bool('perfect_depth_estimation', False,
                  'True to use perfect depth')
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
flags.DEFINE_bool('evaluate_obstacle_detection', False,
                  'True to enable object detection accuracy evaluation')
flags.DEFINE_bool('evaluate_fusion', False, 'True to enable fusion evaluation')
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
    ['obstacle_detection', 'obstacle_detection_model_paths',
     'obstacle_detection_model_names'],
    lambda flags_dict: (not flags_dict['obstacle_detection'] or
                        (flags_dict['obstacle_detection'] and
                         (len(flags_dict['obstacle_detection_model_paths']) ==
                          len(flags_dict['obstacle_detection_model_names'])))),
    message='--obstacle_detection_model_paths and '
    '--obstacle_detection_model_names must have the same length')


def ground_agent_validator(flags_dict):
    if flags_dict['control_agent_operator'] == 'ground':
        return flags_dict['planning_type'] == 'single_waypoint'
    return True


flags.register_multi_flags_validator(
    ['planning_type',
     'control_agent_operator'],
    ground_agent_validator,
    message='ground agent requires single_waypoint planning')


def mpc_agent_validator(flags_dict):
    if flags_dict['control_agent_operator'] == 'mpc':
        return (flags_dict['planning_type'] == 'multiple_waypoints' or
                flags_dict['planning_type'] == 'rrt_star')
    return True


flags.register_multi_flags_validator(
    ['planning_type',
     'control_agent_operator'],
    mpc_agent_validator,
    message='mpc agent requires multiple_waypoints or rrt_star planning')


def pylot_agent_validator(flags_dict):
    if flags_dict['control_agent_operator'] == 'pylot':
        has_obstacle_detector = (
            flags_dict['obstacle_detection'] or
            flags_dict['perfect_obstacle_detection'])
        has_traffic_light_detector = (
            flags_dict['traffic_light_detection'] or
            flags_dict['perfect_traffic_light_detection'])
        # TODO: Add lane detection, obstacle tracking and prediction once
        # the agent depends on these components.
        has_depth = (flags_dict['depth_estimation'] or
                     flags_dict['perfect_depth_estimation'])
        has_planner = flags_dict['planning_type'] == 'single_waypoint'
        return (has_obstacle_detector and
                has_traffic_light_detector and
                has_planner and
                has_depth)
    return True


flags.register_multi_flags_validator(
    ['obstacle_detection',
     'perfect_obstacle_detection',
     'traffic_light_detection',
     'perfect_traffic_light_detection',
     'depth_estimation',
     'perfect_depth_estimation',
     'planning_type',
     'control_agent_operator'],
    pylot_agent_validator,
    message='pylot agent requires obstacle detection, traffic light detection,'
    ' depth, and single waypoint planning')


def obstacle_tracking_validator(flags_dict):
    if flags_dict['obstacle_tracking']:
        return (flags_dict['obstacle_detection'] or
                flags_dict['perfect_obstacle_detection'])
    return True


flags.register_multi_flags_validator(
    ['obstacle_detection',
     'perfect_obstacle_detection',
     'obstacle_tracking'],
    obstacle_tracking_validator,
    message='--obstacle_detection or --perfect_obstacle_detection must be set'
    ' when --obstacle_tracking is enabled')


def detector_evaluation_validator(flags_dict):
    if flags_dict['evaluate_obstacle_detection']:
        return flags_dict['obstacle_detection']
    return True


flags.register_multi_flags_validator(
    ['obstacle_detection', 'evaluate_obstacle_detection'],
    detector_evaluation_validator,
    message='--obstacle_detection must be set when '
    '--evaluate_obstacle_detection is enabled')


def fusion_evaluation_validator(flags_dict):
    if flags_dict['evaluate_fusion']:
        return flags_dict['fusion']
    return True


flags.register_multi_flags_validator(
    ['fusion', 'evaluate_fusion'],
    fusion_evaluation_validator,
    message='--fusion must be set when --evaluate_fusion is enabled')
