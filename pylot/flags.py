from absl import flags

import pylot.control.flags
import pylot.debug.flags
import pylot.perception.flags
import pylot.planning.flags
import pylot.prediction.flags
import pylot.simulation.flags  # noqa: F401

FLAGS = flags.FLAGS

# Files where to log information.
flags.DEFINE_string('log_file_name', None, 'Name of the log file')
flags.DEFINE_string('csv_log_file_name', None,
                    'csv file into which to log runtime stats')
flags.DEFINE_string(
    'profile_file_name', 'pylot_profile.json',
    'file into which to log runtime profile stats in Chrome trace format')

flags.DEFINE_enum(
    'execution_mode', 'simulation',
    ['simulation', 'real-world', 'challenge-map', 'challenge-sensors'],
    'Sets the mode in which the pipeline executes')

######################################################################
# Perception
######################################################################
flags.DEFINE_integer('camera_image_width', 1920, 'Camera image width')
flags.DEFINE_integer('camera_image_height', 1080, 'Camera image height')
flags.DEFINE_float('camera_fov', 90.0, 'Camera field of view')
flags.DEFINE_bool('obstacle_detection', False,
                  'True to enable obstacle detection operator')
flags.DEFINE_bool('perfect_obstacle_detection', False,
                  'True to enable perfect obstacle 2D detection')
flags.DEFINE_bool(
    'simulator_obstacle_detection', True,
    'True to enable usage of obstacles received from the simulator')
flags.DEFINE_bool('obstacle_tracking', False,
                  'True to enable obstacle tracking operator')
flags.DEFINE_bool('perfect_obstacle_tracking', False,
                  'True to enable perfect obstacle tracking')
flags.DEFINE_enum('tracker_type', 'sort',
                  ['da_siam_rpn', 'deep_sort', 'sort', 'center_track'],
                  'Sets which obstacle tracker to use')
flags.DEFINE_bool('lane_detection', False, 'True to enable lane detection')
flags.DEFINE_bool('perfect_lane_detection', False,
                  'True to enable perfect lane detection')
flags.DEFINE_enum('lane_detection_type', 'lanenet', ['lanenet', 'canny'],
                  'Sets which lane detection operator to use')
flags.DEFINE_bool('fusion', False, 'True to enable fusion operator')
flags.DEFINE_bool('traffic_light_detection', False,
                  'True to enable traffic light detection operator')
flags.DEFINE_bool('perfect_traffic_light_detection', False,
                  'True to enable perfect traffic light 2D detection')
flags.DEFINE_bool(
    'simulator_traffic_light_detection', True,
    'True to enable usage of traffic lights received from the simulator')
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
flags.DEFINE_bool(
    'localization', False,
    'True to run localization. False for ground truth localization')
flags.DEFINE_float(
    'offset_left_right_cameras', 0.4,
    'How much we offset the left and right cameras from the center.')

######################################################################
# Prediction
######################################################################
flags.DEFINE_bool('prediction', False, 'True to enable prediction.')
flags.DEFINE_enum('prediction_type', 'linear', ['linear', 'r2p2'],
                  'Type of prediction module to use')

######################################################################
# Planning
######################################################################
flags.DEFINE_enum(
    'planning_type', 'waypoint',
    ['waypoint', 'rrt_star', 'frenet_optimal_trajectory', 'hybrid_astar'],
    'Type of planning module to use')
flags.DEFINE_bool('imu', False, 'True to enable the IMU sensor')

######################################################################
# Control
######################################################################
flags.DEFINE_enum('control', 'simulator_auto_pilot',
                  ['pid', 'mpc', 'simulator_auto_pilot', 'manual'],
                  'Control operator to use to drive')

######################################################################
# Deadline enforcement
######################################################################
flags.DEFINE_enum(
    'deadline_enforcement', 'none', ['none', 'static', 'dynamic'],
    'Controls how end-to-end deadlines are enforced. None means no enforcement'
)
flags.DEFINE_float(
    'detection_deadline', None,
    'Detection deadline (in ms) when using static deadline enforcement')
flags.DEFINE_float(
    'tracking_deadline', None,
    'Tracking deadline (in ms) when using static deadline enforcement')
flags.DEFINE_float(
    'planning_deadline', None,
    'Planning deadline (on ms) when using static deadline enforcement')

######################################################################
# Visualizing flags
######################################################################
flags.DEFINE_bool('visualize_rgb_camera', False,
                  'True to enable RGB camera sensor visualization')
flags.DEFINE_bool('visualize_depth_camera', False,
                  'True to enable depth camera sensor visualization')
flags.DEFINE_bool('visualize_lidar', False,
                  'True to enable Lidar visualization')
flags.DEFINE_bool('visualize_imu', False,
                  'True to enable simulator IMU visualization')
flags.DEFINE_bool('visualize_pose', False,
                  'True to enable simulator ego-vehicle pose visualization')
flags.DEFINE_bool('visualize_detected_obstacles', False,
                  'True to enable visualization of detected obstacles')
flags.DEFINE_bool('visualize_detected_traffic_lights', False,
                  'True to enable visualization of detected traffic lights')
flags.DEFINE_bool('visualize_detected_lanes', False,
                  'True to enable visualization of detected lanes')
flags.DEFINE_bool('visualize_tracked_obstacles', False,
                  'True to enable visualization of tracked obstacles')
flags.DEFINE_bool('visualize_segmentation', False,
                  'True to enable visualization of segmentaation')
flags.DEFINE_bool('visualize_waypoints', False,
                  'True to enable visualization of planning waypoints')
flags.DEFINE_bool('visualize_prediction', False,
                  'True to enable visualization of obstacle predictions')
flags.DEFINE_bool(
    'visualize_world', False,
    'True to enable visualization of the planning world representation')
flags.DEFINE_integer('top_down_camera_altitude', 40,
                     'Altitude of the top-down world camera (in meters).')

########################################
# Recording operators.
########################################
flags.DEFINE_string('simulation_recording_file', None,
                    'Path to where the simulation is recorded')
flags.DEFINE_string('data_path', 'data/', 'Path where to logged data')
flags.DEFINE_bool('log_detector_output', False,
                  'Enable recording of bbox annotated detector images')
flags.DEFINE_bool('log_traffic_light_detector_output', False,
                  'Enable recording of bbox annotated tl detector images')
flags.DEFINE_bool('log_lane_detection_camera', False,
                  'True to enable lane detection camera logging')
flags.DEFINE_integer('log_every_nth_message', 1,
                     'Control how often the script logs messages')
########################################
# Evaluation operators.
########################################
flags.DEFINE_bool('evaluation', False,
                  'Enable end-to-end evaluation of the pipeline.')
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


def must_visualize():
    """Returns True if any visualization flag is True."""
    return (FLAGS.visualize_rgb_camera or FLAGS.visualize_depth_camera
            or FLAGS.visualize_lidar or FLAGS.visualize_imu
            or FLAGS.visualize_pose or FLAGS.visualize_detected_obstacles
            or FLAGS.visualize_detected_traffic_lights
            or FLAGS.visualize_detected_lanes
            or FLAGS.visualize_tracked_obstacles
            or FLAGS.visualize_segmentation or FLAGS.visualize_waypoints
            or FLAGS.visualize_prediction or FLAGS.visualize_world)


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
    return (FLAGS.evaluate_segmentation or FLAGS.perfect_segmentation
            or FLAGS.perfect_obstacle_detection
            or FLAGS.evaluate_obstacle_detection)


def must_add_lidar_sensor():
    """Returns true if the lidar sensor must be added.

    We don't add all sensors by default because they slow donwn the simulation
    """
    return (FLAGS.visualize_lidar
            or (FLAGS.prediction and FLAGS.prediction_type == 'r2p2')
            or FLAGS.obstacle_location_finder_sensor == 'lidar')


def must_add_imu_sensor():
    """ Returns true if the IMU sensor must be added.

    We don't add all sensors by default because they slow down the simulation
    """
    return (FLAGS.imu or FLAGS.evaluation or FLAGS.localization)


def must_add_gnss_sensor():
    """ Returns true if the GNSS sensor must be added.

    We don't add all sensors by default because they slow down the simulation
    """
    return FLAGS.localization


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
    message='prediction requires --obstacle_tracking or'
    ' --perfect_obstacle_tracking')

flags.register_multi_flags_validator(
    ['visualize_prediction', 'execution_mode'],
    lambda flags_dict: ((not flags_dict['visualize_prediction']) or flags_dict[
        'execution_mode'] == 'simulator'),
    message='--visualize_prediction can only be enabled when running atop the'
    ' simulator')


def prediction_ego_agent_validator(flags_dict):
    if flags_dict['prediction_ego_agent']:
        return flags_dict['perfect_obstacle_tracking']
    return True


flags.register_multi_flags_validator(
    ['prediction_ego_agent', 'perfect_obstacle_tracking'],
    prediction_ego_agent_validator,
    message='ego-agent prediction requires --perfect_obstacle_tracking')


def prediction_eval_validator(flags_dict):
    if flags_dict['evaluate_prediction']:
        return flags_dict['prediction']
    return True


flags.register_multi_flags_validator(
    ['prediction', 'evaluate_prediction'],
    prediction_eval_validator,
    message='--prediction must be enabled when --evaluate_prediction is '
    'enabled')

flags.register_multi_flags_validator(
    [
        'evaluate_prediction', 'prediction_num_future_steps',
        'tracking_num_steps'
    ],
    lambda flags_dict:
    (not flags_dict['evaluate_prediction'] or
     (flags_dict['evaluate_prediction'] and flags_dict[
         'prediction_num_future_steps'] <= flags_dict['tracking_num_steps'])),
    message='must track at least as many steps as we predict when'
    ' --evaluate_prediction is enabled')


def planning_validator(flags_dict):
    if (flags_dict['planning_type'] == 'rrt_star'
            or flags_dict['planning_type'] == 'frenet_optimal_trajectory'
            or flags_dict['planning_type'] == 'hybrid_astar'):
        return flags_dict['prediction']
    return True


flags.register_multi_flags_validator(
    ['planning_type', 'prediction'],
    planning_validator,
    message='This type of planner requires --prediction')


def waypoint_planning_validator(flags_dict):
    if flags_dict['planning_type'] == 'waypoint':
        has_obstacle_detector = (flags_dict['obstacle_detection']
                                 or flags_dict['perfect_obstacle_detection']
                                 or flags_dict['simulator_obstacle_detection'])
        has_traffic_light_detector = (
            flags_dict['traffic_light_detection']
            or flags_dict['perfect_traffic_light_detection']
            or flags_dict['simulator_traffic_light_detection'])
        # TODO: Add lane detection, obstacle tracking and prediction once
        # the agent depends on these components.
        return (has_obstacle_detector and has_traffic_light_detector)
    return True


flags.register_multi_flags_validator(
    [
        'obstacle_detection', 'perfect_obstacle_detection',
        'simulator_obstacle_detection', 'traffic_light_detection',
        'perfect_traffic_light_detection', 'simulator_traffic_light_detection',
        'planning_type'
    ],
    waypoint_planning_validator,
    message='waypoint planner requires obstacle detection and '
    'traffic light detection')


def obstacle_detection_validator(flags_dict):
    if flags_dict['obstacle_detection']:
        return not (flags_dict['perfect_obstacle_detection']
                    or flags_dict['simulator_obstacle_detection'])
    if flags_dict['perfect_obstacle_detection']:
        return not (flags_dict['obstacle_detection']
                    or flags_dict['simulator_obstacle_detection'])
    if flags_dict['simulator_obstacle_detection']:
        return not (flags_dict['obstacle_detection']
                    or flags_dict['perfect_obstacle_detection'])
    # return False if neither flag is set.
    return False


flags.register_multi_flags_validator(
    [
        'obstacle_detection',
        'perfect_obstacle_detection',
        'simulator_obstacle_detection',
    ],
    obstacle_detection_validator,
    message='Only one of --obstacle_detection, --perfect_obstacle_detection, '
    'or --simulator_obstacle_detection can be True')


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
                    or flags_dict['simulator_traffic_light_detection'])
    if flags_dict['perfect_traffic_light_detection']:
        return not (flags_dict['traffic_light_detection']
                    or flags_dict['simulator_traffic_light_detection'])
    if flags_dict['simulator_traffic_light_detection']:
        return not (flags_dict['traffic_light_detection']
                    or flags_dict['perfect_traffic_light_detection'])
    # return False if neither flag is set.
    return False


flags.register_multi_flags_validator(
    [
        'traffic_light_detection', 'perfect_traffic_light_detection',
        'simulator_traffic_light_detection'
    ],
    traffic_light_detection_validator,
    message='Only one of --traffic_light_detection, '
    '--perfect_traffic_light_detection, or --simulator_traffic_light_detection'
    ' can be True')


def obstacle_tracking_validator(flags_dict):
    if flags_dict['obstacle_tracking']:
        return (flags_dict['obstacle_detection']
                or flags_dict['perfect_obstacle_detection']
                ) and not (flags_dict['perfect_obstacle_tracking'])
    if flags_dict['perfect_obstacle_tracking']:
        return not flags_dict['obstacle_tracking']
    return True


flags.register_multi_flags_validator(
    [
        'obstacle_detection', 'perfect_obstacle_detection',
        'obstacle_tracking', 'perfect_obstacle_tracking'
    ],
    obstacle_tracking_validator,
    message='--obstacle_detection or --perfect_obstacle_detection must be set '
    'when --obstacle_tracking is enabled. Only one of --obstacle_tracking, or '
    '--perfect_obstacle_tracking can be enabled')


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
    '--perfect_obstacle_detection must be set when '
    '--evaluate_obstacle_tracking is enabled')


def fusion_evaluation_validator(flags_dict):
    if flags_dict['evaluate_fusion']:
        return flags_dict['fusion']
    return True


flags.register_multi_flags_validator(
    ['fusion', 'evaluate_fusion'],
    fusion_evaluation_validator,
    message='--fusion must be set when --evaluate_fusion is enabled')


def segmentation_validator(flags_dict):
    if flags_dict['segmentation']:
        return not flags_dict['perfect_segmentation']
    if flags_dict['perfect_segmentation']:
        return not flags_dict['segmentation']
    return True


flags.register_multi_flags_validator(
    ['perfect_segmentation', 'segmentation'],
    segmentation_validator,
    message='--segmentation and --perfect_segmentation cannot be both set')


def segmentation_evaluation_validator(flags_dict):
    if flags_dict['evaluate_segmentation']:
        return flags_dict['segmentation']
    return True


flags.register_multi_flags_validator(
    ['evaluate_segmentation', 'segmentation'],
    segmentation_evaluation_validator,
    message='--segmentation must be set when --evaluate_segmentation is '
    'enabled')


def depth_validator(flags_dict):
    if flags_dict['depth_estimation']:
        return not flags_dict['perfect_depth_estimation']
    if flags_dict['perfect_depth_estimation']:
        return not flags_dict['depth_estimation']
    return True


flags.register_multi_flags_validator(
    ['depth_estimation', 'perfect_depth_estimation'],
    depth_validator,
    message='--depth_estimation and --perfect_depth_estimation cannot be'
    ' both set')


def lane_detection_validator(flags_dict):
    if flags_dict['lane_detection']:
        return not flags_dict['perfect_lane_detection']
    if flags_dict['perfect_lane_detection']:
        return not flags_dict['lane_detection']
    return True


flags.register_multi_flags_validator(
    ['lane_detection', 'perfect_lane_detection'],
    lane_detection_validator,
    message='--lane_detection and --perfect_lane_detection cannot be both set')
