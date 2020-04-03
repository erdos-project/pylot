from absl import flags

import pylot.perception.flags
import pylot.prediction.flags
import pylot.planning.flags
import pylot.control.flags
import pylot.simulation.flags
import pylot.debug.flags

FLAGS = flags.FLAGS

# Files where to log information.
flags.DEFINE_string('log_file_name', None, 'Name of the log file')
flags.DEFINE_string('csv_log_file_name', None,
                    'csv file into which to log runtime stats')
flags.DEFINE_string(
    'profile_file_name', 'pylot_profile.json',
    'file into which to log runtime profile stats in Chrome trace format')

######################################################################
# Perception
######################################################################
flags.DEFINE_bool('obstacle_detection', False,
                  'True to enable obstacle detection operator')
flags.DEFINE_bool('perfect_obstacle_detection', False,
                  'True to enable perfect obstacle 2D detection')
flags.DEFINE_bool(
    'carla_obstacle_detection', True,
    'True to enable usage of obstacles received directly from CARLA')
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
                  ['cv2', 'da_siam_rpn', 'deep_sort', 'sort'],
                  'Sets which obstacle tracker to use')
flags.DEFINE_integer(
    'tracking_num_steps', 10,
    'Limit on number of past steps returned by the object tracker.')
flags.DEFINE_bool('lane_detection', False, 'True to enable lane detection')
flags.DEFINE_bool('perfect_lane_detection', False,
                  'True to enable perfect lane detection')
flags.DEFINE_bool('fusion', False, 'True to enable fusion operator')
flags.DEFINE_bool('traffic_light_detection', False,
                  'True to enable traffic light detection operator')
flags.DEFINE_bool('perfect_traffic_light_detection', False,
                  'True to enable perfect traffic light 2D detection')
flags.DEFINE_bool(
    'carla_traffic_light_detection', True,
    'True to enable usage of traffic lights received directly from CARLA')
flags.DEFINE_bool('segmentation', False,
                  'True to enable segmentation operator')
flags.DEFINE_bool('perfect_segmentation', False,
                  'True to enable perfect segmentation')
flags.DEFINE_enum('obstacle_location_finder_sensor', 'lidar',
                  ['lidar', 'depth_camera'],
                  'Sets which sensor to use to compute obstacle locations')
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
flags.DEFINE_enum('planning_type', 'waypoint',
                  ['waypoint', 'rrt_star', 'frenet_optimal_trajectory'],
                  'Type of planning module to use')
flags.DEFINE_bool('imu', False, 'True to enable the IMU sensor')

######################################################################
# Control
######################################################################
flags.DEFINE_enum('control_agent', 'carla_auto_pilot',
                  ['pid', 'mpc', 'carla_auto_pilot', 'manual'],
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
flags.DEFINE_bool('visualize_pose', False,
                  'True to visualize ego-vehicle pose')

######################################################################
# Visualizing flags for components.
######################################################################
flags.DEFINE_bool('visualize_detected_obstacles', False,
                  'True to enable visualization of detected obstacles')
flags.DEFINE_bool('visualize_detected_traffic_lights', False,
                  'True to enable visualization of detected traffic lights')
flags.DEFINE_bool(
    'visualize_obstacles_with_distance', False,
    'True to include distance into the visualization of obstacles')
flags.DEFINE_bool('visualize_waypoints', False,
                  'True to enable visualization of waypoing planning')
flags.DEFINE_bool('visualize_prediction', False,
                  'True to enable visualization of prediction output')

# Accuracy evaluation flags.
flags.DEFINE_bool('evaluate_obstacle_detection', False,
                  'True to enable object detection accuracy evaluation')
flags.DEFINE_bool('evaluate_obstacle_tracking', False,
                  'True to enable object tracking evaluation')
flags.DEFINE_bool('evaluate_prediction', False,
                  'True to enable prediction evaluation')
flags.DEFINE_bool('evaluate_control', False,
                  'True to enable control evaluation')
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

########################################
# Evaluation operators.
########################################
flags.DEFINE_bool('evaluation', False,
                  'Enable end-to-end evaluation of the pipeline.')


def must_add_depth_camera_sensor():
    """Returns true if the depth camera sensor must be added.

    We don't add all sensors by default because they slow donwn the simulation
    """
    return (FLAGS.perfect_obstacle_detection
            or FLAGS.evaluate_obstacle_detection
            or FLAGS.perfect_depth_estimation or FLAGS.visualize_depth_camera
            or FLAGS.obstacle_location_finder_sensor == 'depth_camera')


def must_add_segmented_camera_sensor():
    """Returns true if the segmented camera sensor must be added.

    We don't add all sensors by default because they slow donwn the simulation
    """
    return (FLAGS.visualize_segmentation or FLAGS.evaluate_segmentation
            or FLAGS.perfect_segmentation or FLAGS.perfect_obstacle_detection
            or FLAGS.evaluate_obstacle_detection)


def must_add_lidar_sensor():
    """Returns true if the lidar sensor must be added.

    We don't add all sensors by default because they slow donwn the simulation
    """
    return (FLAGS.visualize_lidar
            or FLAGS.obstacle_location_finder_sensor == 'lidar')


def must_add_imu_sensor():
    """ Returns true if the IMU sensor must be added.

    We don't add all sensors by default because they slow down the simulation
    """
    return (FLAGS.imu or FLAGS.evaluation)


# Flag validators.
flags.register_multi_flags_validator(
    [
        'obstacle_detection', 'obstacle_detection_model_paths',
        'obstacle_detection_model_names'
    ],
    lambda flags_dict: (not flags_dict['obstacle_detection'] or (flags_dict[
        'obstacle_detection'] and (len(flags_dict[
            'obstacle_detection_model_paths']) == len(flags_dict[
                'obstacle_detection_model_names'])))),
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
        'tracking_num_steps'
    ],
    lambda flags_dict: (not flags_dict['evaluate_prediction'] or (flags_dict[
        'evaluate_prediction'] and flags_dict[
            'prediction_num_future_steps'] <= flags_dict['tracking_num_steps'])
                        ),
    message='must track at least as many steps as we predict when'
    ' --evaluate_prediction is enabled')


def rrt_star_validator(flags_dict):
    if flags_dict['planning_type'] == 'rrt_star':
        return flags_dict['prediction']
    return True


flags.register_multi_flags_validator(['planning_type', 'prediction'],
                                     rrt_star_validator,
                                     message='rrt star requires --prediction')


def waypoint_planning_validator(flags_dict):
    if flags_dict['planning_type'] == 'waypoint':
        has_obstacle_detector = (flags_dict['obstacle_detection']
                                 or flags_dict['perfect_obstacle_detection']
                                 or flags_dict['carla_obstacle_detection'])
        has_traffic_light_detector = (
            flags_dict['traffic_light_detection']
            or flags_dict['perfect_traffic_light_detection']
            or flags_dict['carla_traffic_light_detection'])
        # TODO: Add lane detection, obstacle tracking and prediction once
        # the agent depends on these components.
        return (has_obstacle_detector and has_traffic_light_detector)
    return True


flags.register_multi_flags_validator(
    [
        'obstacle_detection', 'perfect_obstacle_detection',
        'carla_obstacle_detection', 'traffic_light_detection',
        'perfect_traffic_light_detection', 'carla_traffic_light_detection',
        'planning_type'
    ],
    waypoint_planning_validator,
    message='waypoint planner requires obstacle detection and '
    'traffic light detection')


def obstacle_detection_validator(flags_dict):
    if flags_dict['obstacle_detection']:
        return not (flags_dict['perfect_obstacle_detection']
                    or flags_dict['carla_obstacle_detection'])
    if flags_dict['perfect_obstacle_detection']:
        return not (flags_dict['obstacle_detection']
                    or flags_dict['carla_obstacle_detection'])
    if flags_dict['carla_obstacle_detection']:
        return not (flags_dict['obstacle_detection']
                    or flags_dict['perfect_obstacle_detection'])
    return False


flags.register_multi_flags_validator(
    [
        'obstacle_detection',
        'perfect_obstacle_detection',
        'carla_obstacle_detection',
    ],
    obstacle_detection_validator,
    message='Only one of --obstacle_detection, --perfect_obstacle_detection, '
    'or --carla_obstacle_detection can be True')


def obstacle_detection_eval_validator(flags_dict):
    if flags_dict['evaluate_obstacle_detection']:
        return flags_dict['obstacle_detection']
    return True


flags.register_multi_flags_validator(
    ['obstacle_detection', 'evaluate_obstacle_detection'],
    obstacle_detection_eval_validator,
    message='--obstacle_detection must be enabled when '
    '--evaluate_obstacle_detection is enabled')


def traffic_light_detection_validator(flags_dict):
    if flags_dict['traffic_light_detection']:
        return not (flags_dict['perfect_traffic_light_detection']
                    or flags_dict['carla_traffic_light_detection'])
    if flags_dict['perfect_traffic_light_detection']:
        return not (flags_dict['traffic_light_detection']
                    or flags_dict['carla_traffic_light_detection'])
    if flags_dict['carla_traffic_light_detection']:
        return not (flags_dict['traffic_light_detection']
                    or flags_dict['perfect_traffic_light_detection'])


flags.register_multi_flags_validator(
    [
        'traffic_light_detection', 'perfect_traffic_light_detection',
        'carla_traffic_light_detection'
    ],
    traffic_light_detection_validator,
    message='Only one of --traffic_light_detection, '
    '--perfect_traffic_light_detection, or --carla_traffic_light_detection'
    ' can be True')


def obstacle_tracking_validator(flags_dict):
    if flags_dict['obstacle_tracking']:
        return (flags_dict['obstacle_detection']
                or flags_dict['perfect_obstacle_detection']
                or flags_dict['carla_obstacle_detection'])
    return True


flags.register_multi_flags_validator(
    [
        'obstacle_detection', 'perfect_obstacle_detection',
        'carla_obstacle_detection', 'obstacle_tracking'
    ],
    obstacle_tracking_validator,
    message='--obstacle_detection, --perfect_obstacle_detection, or '
    '--carla_obstacle_detection must be set when --obstacle_tracking is'
    ' enabled')


def obstacle_tracking_evaluation_validator(flags_dict):
    if flags_dict['evaluate_obstacle_tracking']:
        return (flags_dict['obstacle_tracking']
                and (flags_dict['perfect_obstacle_detection']
                     or flags_dict['obstacle_detection']))
    return True


flags.register_multi_flags_validator(
    [
        'obstacle_tracking', 'evaluate_obstacle_tracking',
        'perfect_obstacle_detection', 'obstacle_detection'
    ],
    obstacle_tracking_evaluation_validator,
    message='--obstacle_tracking and either --obstacle_detection or '
    '--perfect_obstacle_detection must be set when --evaluate_obstacle_tracking'
    ' is enabled')


def fusion_evaluation_validator(flags_dict):
    if flags_dict['evaluate_fusion']:
        return flags_dict['fusion']
    return True


flags.register_multi_flags_validator(
    ['fusion', 'evaluate_fusion'],
    fusion_evaluation_validator,
    message='--fusion must be set when --evaluate_fusion is enabled')
