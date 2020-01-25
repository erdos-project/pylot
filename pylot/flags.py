from absl import flags

import pylot.control.flags
import pylot.prediction.flags
import pylot.simulation.flags

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
flags.DEFINE_list(
    'obstacle_detection_model_paths',
    'dependencies/models/obstacle_detection/faster-rcnn/frozen_inference_graph.pb',
    'Comma-separated list of model paths')
flags.DEFINE_list('obstacle_detection_model_names', 'faster-rcnn',
                  'Comma-separated list of model names')
flags.DEFINE_bool('obstacle_tracking', False,
                  'True to enable obstacle tracking operator')
flags.DEFINE_bool('perfect_obstacle_tracking', False,
                  'True to enable perfect obstacle tracking')
flags.DEFINE_enum('tracker_type', 'cv2',
                  ['cv2', 'da_siam_rpn', 'deep_sort', 'sort'], 'Tracker type')
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
    'offset_left_right_cameras', 0.4,
    'How much we offset the left and right cameras from the center.')

######################################################################
# Prediction
######################################################################
flags.DEFINE_bool('prediction', False, 'True to enable prediction.')
flags.DEFINE_enum('prediction_type', 'linear', ['linear'],
                  'Type of prediction module to use')

######################################################################
# Planning
######################################################################
flags.DEFINE_enum('planning_type', 'waypoint', ['waypoint', 'rrt_star'],
                  'Type of planning module to use')
flags.DEFINE_bool('imu', False, 'True to enable the IMU sensor')

######################################################################
# Control
######################################################################
flags.DEFINE_enum('control_agent', 'carla_auto_pilot',
                  ['pylot', 'mpc', 'carla_auto_pilot'],
                  'Control agent operator to use to drive')

######################################################################
# Sensor visualizing flags
######################################################################
flags.DEFINE_bool('visualize_rgb_camera', False,
                  'True to enable RGB camera sensor visualizer')
flags.DEFINE_bool('visualize_depth_camera', False,
                  'True to enable depth camera sensor visualizer')
flags.DEFINE_bool('visualize_segmentation', False,
                  'True to enable segmented carmera sensor visualizer')
flags.DEFINE_bool('visualize_lidar', False,
                  'True to enable CARLA Lidar visualizer operator')
flags.DEFINE_bool('visualize_imu', False,
                  'True to enable CARLA IMU visualizer operator')
flags.DEFINE_bool('visualize_can_bus', False, 'True to visualize can bus.')

######################################################################
# Visualizing flags for components.
######################################################################
flags.DEFINE_bool('visualize_detected_obstacles', False,
                  'True to enable visualization of detected obstacles')
flags.DEFINE_bool('visualize_detected_traffic_lights', False,
                  'True to enable visualization of detected traffic lights')
flags.DEFINE_bool('visualize_waypoints', False,
                  'True to enable visualization of waypoing planning')
flags.DEFINE_bool('visualize_prediction', False,
                  'True to enable visualization of prediction output')

# Accuracy evaluation flags.
flags.DEFINE_bool('evaluate_obstacle_detection', False,
                  'True to enable object detection accuracy evaluation')
flags.DEFINE_bool('evaluate_prediction', False,
                  'True to enable prediction evaluation')
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

# Flag validators.
flags.register_multi_flags_validator(
    [
        'obstacle_detection', 'obstacle_detection_model_paths',
        'obstacle_detection_model_names'
    ],
    lambda flags_dict: (not flags_dict['obstacle_detection'] or
                        (flags_dict['obstacle_detection'] and
                         (len(flags_dict['obstacle_detection_model_paths']) ==
                          len(flags_dict['obstacle_detection_model_names'])))),
    message='--obstacle_detection_model_paths and '
    '--obstacle_detection_model_names must have the same length')


def prediction_validator(flags_dict):
    if flags_dict['prediction']:
        return (flags_dict['obstacle_tracking']
                or flags_dict['perfect_obstacle_tracking'])
    return True


flags.register_multi_flags_validator(
    ['prediction', 'obstacle_tracking', 'perfect_obstacle_tracking'],
    prediction_validator,
    message=
    'prediction requires --obstacle_tracking or --perfect_obstacle_tracking')

flags.register_multi_flags_validator(
    [
        'evaluate_prediction', 'prediction_num_future_steps',
        'perfect_tracking_num_steps'
    ],
    lambda flags_dict: (not flags_dict['evaluate_prediction'] or
                        (flags_dict['evaluate_prediction'] and flags_dict[
                            'prediction_num_future_steps'] <= flags_dict[
                                'perfect_tracking_num_steps'])),
    message='must track at least as many steps as we predict when'
    ' --evaluate_prediction is enabled')


def rrt_star_validator(flags_dict):
    if flags_dict['planning_type'] == 'rrt_star':
        return flags_dict['prediction']
    return True


flags.register_multi_flags_validator(['planning_type', 'prediction'],
                                     rrt_star_validator,
                                     message='rrt star requires --prediction')


def pylot_agent_validator(flags_dict):
    if flags_dict['control_agent'] == 'pylot':
        has_obstacle_detector = (flags_dict['obstacle_detection']
                                 or flags_dict['perfect_obstacle_detection'])
        has_traffic_light_detector = (
            flags_dict['traffic_light_detection']
            or flags_dict['perfect_traffic_light_detection'])
        # TODO: Add lane detection, obstacle tracking and prediction once
        # the agent depends on these components.
        return (has_obstacle_detector and has_traffic_light_detector)
    return True


flags.register_multi_flags_validator(
    [
        'obstacle_detection', 'perfect_obstacle_detection',
        'traffic_light_detection', 'perfect_traffic_light_detection',
        'control_agent'
    ],
    pylot_agent_validator,
    message='pylot agent requires obstacle detection, traffic light detection,'
)


def obstacle_detection_validator(flags_dict):
    return (flags_dict['obstacle_detection'] is False
            or flags_dict['perfect_obstacle_detection'] is False)


flags.register_multi_flags_validator(
    [
        'obstacle_detection',
        'perfect_obstacle_detection',
    ],
    obstacle_detection_validator,
    message=
    '--obstacle_detection and --perfect_obstacle_detection cannot be both True'
)


def obstacle_detection_eval_validator(flags_dict):
    if flags_dict['evaluate_obstacle_detection']:
        return flags_dict['obstacle_detection']
    return True


flags.register_multi_flags_validator(
    ['obstacle_detection', 'evaluate_obstacle_detection'],
    obstacle_detection_eval_validator,
    message=
    '--obstacle_detection must be True when --evaluate_obstacle_detection is True'
)


def obstacle_tracking_validator(flags_dict):
    if flags_dict['obstacle_tracking']:
        return (flags_dict['obstacle_detection']
                or flags_dict['perfect_obstacle_detection'])
    return True


flags.register_multi_flags_validator(
    ['obstacle_detection', 'perfect_obstacle_detection', 'obstacle_tracking'],
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
