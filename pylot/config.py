from absl import flags

FLAGS = flags.FLAGS

# Files where to log information.
flags.DEFINE_string('log_file_name', None, 'Name of the log file')
flags.DEFINE_string('csv_log_file_name', None,
                    'csv file into which to log runtime stats')

######################################################################
# Perception
######################################################################
flags.DEFINE_bool('use_perfect_perception', False,
                  'True to enable the agent to use perfect ground detection')
###################################
# Obstacle detection flags.
###################################
flags.DEFINE_bool('obj_detection', False,
                  'True to enable object detection operator')
flags.DEFINE_float(
    'obj_detection_gpu_memory_fraction',
    0.3,
    'GPU memory fraction allocated to each obj detector operator')
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
flags.DEFINE_float('detector_min_score_threshold', 0.5,
                   'Min score threshold for bounding box')
flags.DEFINE_string('path_coco_labels', 'dependencies/models/coco.names',
                    'Path to the COCO labels')
###################################
# Obstacle tracking flags.
###################################
flags.DEFINE_bool('obj_tracking', False,
                  'True to enable object tracking operator')
flags.DEFINE_enum('tracker_type',
                  'cv2',
                  ['cv2', 'da_siam_rpn', 'deep_sort', 'sort'],
                  'Tracker type')
flags.DEFINE_float(
    'obj_tracking_gpu_memory_fraction',
    0.3,
    'GPU memory fraction allocated to each obj tracker operator')
flags.DEFINE_string('da_siam_rpn_model_path',
                    'dependencies/models/SiamRPNVOT.model',
                    'Path to the model')
flags.DEFINE_string(
    'deep_sort_tracker_pedestrian_weights_path',
    'dependencies/nanonets_object_tracking/ped_feature_extractor',
    'Path to weights for pedestrian feature extractor model')
flags.DEFINE_bool('perfect_tracking', False,
                  'True to enable perfect object tracker.')
flags.DEFINE_integer(
    'perfect_tracking_num_steps',
    None,
    'Limit on number of past steps returned by the perfect object tracker.')
###################################
# Lane detection flags.
###################################
flags.DEFINE_bool('lane_detection', False, 'True to enable lane detection')
###################################
# Fusion flags.
###################################
flags.DEFINE_bool('fusion', False, 'True to enable fusion operator')
flags.DEFINE_bool('evaluate_fusion', False, 'True to enable fusion evaluation')
###################################
# Traffic light detection flags.
###################################
flags.DEFINE_bool('traffic_light_det', False,
                  'True to enable traffic light detection operator')
flags.DEFINE_string(
    'traffic_light_det_model_path',
    'dependencies/models/traffic_light_det_inference_graph.pb',
    'Path to the traffic light model protobuf')
flags.DEFINE_float('traffic_light_det_min_score_threshold', 0.3,
                   'Min score threshold for bounding box')
flags.DEFINE_float(
    'traffic_light_det_gpu_memory_fraction',
    0.3,
    'GPU memory fraction allocated to each traffic light detector')
###################################
# Semantic segmentation flags.
###################################
flags.DEFINE_bool('segmentation', False,
                  'True to enable segmantation operator')
flags.DEFINE_string('segmentation_model_path',
                    'dependencies/models/drn_d_22_cityscapes.pth',
                    'Path to the model')
flags.DEFINE_float(
    'segmentation_drn_gpu_memory_fraction',
    0.2,
    'GPU memory fraction allocated to DRN segmentation')
###################################
# Depth estimation flags.
###################################
flags.DEFINE_bool('depth_estimation', False,
                  'True to depth estimation using cameras')
flags.DEFINE_string('depth_estimation_model_path', 'dependencies/anynet/',
                    'Path to AnyNet depth estimation model')

######################################################################
# Prediction
######################################################################
flags.DEFINE_bool('prediction', False, 'True to enable prediction.')
flags.DEFINE_enum('prediction_type',
                  'linear',
                  ['linear'],
                  'Type of prediction module to use')
flags.DEFINE_integer(
    'prediction_num_past_steps',
    None,
    'Number of past steps of each agent given to the prediction module.')
flags.DEFINE_integer(
    'prediction_num_future_steps',
    None,
    'Number of future steps outputted by the prediction module.')

######################################################################
# Planning
######################################################################
flags.DEFINE_bool('waypoint_planning_operator', False,
                  'True to use the waypoint planning operator.')

######################################################################
# Control
######################################################################
# By default we use the Pylot agent.
flags.DEFINE_enum('control_agent_operator',
                  'pylot',
                  ['pylot', 'ground', 'mpc'],
                  'Control agent operator to use to drive')
# Agent flags.
flags.DEFINE_bool('stop_for_traffic_lights', True,
                  'True to enable traffic light stopping')
flags.DEFINE_bool('stop_for_pedestrians', True,
                  'True to enable pedestrian stopping')
flags.DEFINE_bool('stop_for_vehicles', True,
                  'True to enable vehicle stopping')
# Agent stopping parameters.
flags.DEFINE_integer('traffic_light_min_dist_thres', 5,
                     'Min distance threshold traffic light')
flags.DEFINE_integer('traffic_light_max_dist_thres', 20,
                     'Max distance threshold traffic light')
flags.DEFINE_float('traffic_light_angle_thres', 0.5,
                   'Traffic light angle threshold')
flags.DEFINE_integer('vehicle_distance_thres', 15,
                     'Vehicle distance threshold')
flags.DEFINE_float('vehicle_angle_thres', 0.4,
                   'Vehicle angle threshold')
flags.DEFINE_integer('pedestrian_distance_hit_thres', 35,
                     'Pedestrian hit zone distance threshold')
flags.DEFINE_float('pedestrian_angle_hit_thres', 0.15,
                   'Pedestrian hit zone angle threshold')
flags.DEFINE_integer('pedestrian_distance_emergency_thres', 12,
                     'Pedestrian emergency zone distance threshold')
flags.DEFINE_float('pedestrian_angle_emergency_thres', 0.5,
                   'Pedestrian emergency zone angle threshold')
# PID controller parameters
flags.DEFINE_float('pid_p', 0.25, 'PID p parameter')
flags.DEFINE_float('pid_i', 0.20, 'PID i parameter')
flags.DEFINE_float('pid_d', 0.0, 'PID d parameter')
# Steering control parameters
flags.DEFINE_float('default_throttle', 0.0, 'Default throttle')
flags.DEFINE_float('throttle_max', 0.75, 'Max throttle')
flags.DEFINE_integer(
    'target_speed', 10,
    'Target speed in m/s , could be controlled by the speed limit')
flags.DEFINE_float('steer_gain', 0.7, 'Gain on computed steering angle')

flags.DEFINE_float('brake_strength', 1,
                   'Strength for applying brake; between 0 and 1')
flags.DEFINE_integer('coast_factor', 2, 'Factor to control coasting')

######################################################################
# Carla flags.
######################################################################
flags.DEFINE_enum('carla_version',
                  '0.9.6',
                  ['0.9.5', '0.9.6'],
                  'Carla simulator version')
flags.DEFINE_string('carla_host', 'localhost', 'Carla host.')
flags.DEFINE_integer('carla_port', 2000, 'Carla port.')
flags.DEFINE_integer('carla_timeout', 10,
                     'Timeout for connecting to the Carla simulator.')
flags.DEFINE_bool('carla_synchronous_mode', True,
                  'Run Carla in synchronous mode.')
flags.DEFINE_integer('carla_town', 1, 'Sets which Carla town to use.')
flags.DEFINE_integer('carla_fps', 10,
                     'Carla simulator FPS; do not set bellow 10.')
flags.DEFINE_float('carla_step_frequency', -1,
                   'Target frequency of sending control commands. -1 if '
                   'commands should be applied as fast as possible.')
flags.DEFINE_integer('carla_num_vehicles', 20, 'Carla num vehicles.')
flags.DEFINE_integer('carla_num_pedestrians', 40, 'Carla num pedestrians.')
flags.DEFINE_string('carla_weather', 'ClearNoon', 'Carla Weather Presets')
flags.DEFINE_integer(
    'carla_spawn_point_index', -1,
    'Index of spawn point where to place ego vehicle. -1 to randomly assign.')
flags.DEFINE_integer('carla_camera_image_width', 1920,
                     'Carla camera image width')
flags.DEFINE_integer('carla_camera_image_height', 1080,
                     'Carla camera image height')
flags.DEFINE_string('carla_replay_file', '', 'Path to the Carla log file')
flags.DEFINE_float('carla_replay_start_time', 0.0,
                   'The time at which to start replaying')
flags.DEFINE_float('carla_replay_duration', 0.0,
                   'The duration of the replay run')
flags.DEFINE_integer('carla_replay_id', 0,
                     'The actor id to follow during the replay')
flags.DEFINE_bool('carla_auto_pilot', False,
                  'Use auto pilot to drive the ego vehicle')

# Visualizing operators
flags.DEFINE_bool('visualize_depth_camera', False,
                  'True to enable depth camera video operator')
flags.DEFINE_bool('visualize_lidar', False,
                  'True to enable CARLA Lidar visualizer operator')
flags.DEFINE_bool('visualize_depth_estimation', False,
                  'True to enable depth estimation visualization')
flags.DEFINE_bool('visualize_rgb_camera', False,
                  'True to enable RGB camera video operator')
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
flags.DEFINE_bool('visualize_top_down_segmentation', False,
                  'True to visualize top-down segmentation')
flags.DEFINE_bool('visualize_top_down_tracker_output', False,
                  'True to enable visualization of top-down tracker output')

# Accuracy evaluation flags.
flags.DEFINE_bool('evaluate_obj_detection', False,
                  'True to enable object detection accuracy evaluation')
flags.DEFINE_bool('compute_detection_decay', False,
                  'True to enable ground truth object detection evaluation.')
flags.DEFINE_enum('detection_metric', 'mAP', ['mAP', 'timely-mAP'],
                  'Detection evaluation metric')
flags.DEFINE_bool(
    'detection_eval_use_accuracy_model',
    False,
    'Enable to use a model for detection accuracy decay over time')
flags.DEFINE_integer('decay_max_latency', 400,
                     'Max latency to evaluate in ground truth experiments')
flags.DEFINE_bool('evaluate_segmentation', False,
                  'True to enable segmentation evaluation')
flags.DEFINE_enum('segmentation_metric', 'mIoU', ['mIoU', 'timely-mIoU'],
                  'Segmentation evaluation metric')
flags.DEFINE_bool('compute_segmentation_decay', False,
                  'True to enable ground truth segmentation evaluation')
flags.DEFINE_bool(
    'segmentation_eval_use_accuracy_model',
    False,
    'Enable to use a model for segmentation accuracy decay over time')

########################################
# Recording operators.
########################################
flags.DEFINE_string('data_path', 'data/', 'Path where to logged data')
flags.DEFINE_bool('log_detector_output', False,
                  'Enable recording of bbox annotated detector images')
flags.DEFINE_bool('log_traffic_light_detector_output', False,
                  'Enable recording of bbox annotated tl detector images')
flags.DEFINE_bool('record_depth_camera', False, 'True to record depth camera')
flags.DEFINE_bool('record_lidar', False, 'True to record lidar')
flags.DEFINE_bool('record_rgb_camera', False, 'True to record RGB camera')
flags.DEFINE_bool(
    'record_ground_truth', False,
    'True to carla data (e.g., vehicle position, traffic lights)')

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

flags.register_multi_flags_validator(
    ['perfect_tracking', 'perfect_tracking_num_steps'],
    lambda flags_dict: (not flags_dict['perfect_tracking'] or
                        (flags_dict['perfect_tracking'] and
                         flags_dict['perfect_tracking_num_steps'])),
    message='--perfect_tracking_num_steps must be set if --perfect_tracking is set')

flags.register_multi_flags_validator(
    ['prediction', 'prediction_num_past_steps', 'prediction_num_future_steps'],
    lambda flags_dict: (not flags_dict['prediction'] or
                        (flags_dict['prediction'] and
                         flags_dict['prediction_num_past_steps'] and
                         flags_dict['prediction_num_future_steps'])),
    message='--prediction_num_past_steps and --prediction_num_future_steps must be set if --prediction is set')

def detector_accuracy_validator(flags_dict):
    if flags_dict['evaluate_obj_detection']:
        return flags_dict['obj_detection']
    return True


flags.register_multi_flags_validator(
    ['obj_detection', 'evaluate_obj_detection'],
    detector_accuracy_validator,
    message='--obj_detection must be set if --evaluate_obj_detection is set')
