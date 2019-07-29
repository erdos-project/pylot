from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('framework', 'ros',
                    'Execution framework to use: ros | ray.')
flags.DEFINE_bool('replay', False,
                  ('True if run in replay mode, otherwise run '
                   'Carla in server mode using `./CarlaUE4.sh -carla-server`'))
flags.DEFINE_string('log_file_name', None, 'Name of the log file')
flags.DEFINE_string('csv_log_file_name', None,
                    'csv file into which to log runtime stats')
flags.DEFINE_bool('fail_on_message_loss', True,
                  'True to enable operator failure when messages are lost')
flags.DEFINE_bool('ground_agent_operator', True,
                  'True to use the ground truth controller')

# Sensors to enable.
flags.DEFINE_bool('lidar', False, 'True to enable the lidar sensor')

# Modules to enable.
flags.DEFINE_bool('segmentation_drn', False,
                  'True to enable DRN segmantation operator')
flags.DEFINE_string('segmentation_drn_model_path',
                    'dependencies/models/drn_d_22_cityscapes.pth',
                    'Path to the model')
flags.DEFINE_bool('segmentation_dla', False,
                  'True to enable DLA segmantation operator')
flags.DEFINE_string('segmentation_dla_model_path',
                    'dependencies/dla/DLASeg.pth',
                    'Path to the model')
flags.DEFINE_bool('segmentation_gpu', True,
                  'True, if segmentation should use a GPU')
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
flags.DEFINE_bool('detector_center_net', False,
                  'True to enable CenterNet detector')
flags.DEFINE_string('detector_center_net_model_path',
                    'dependencies/CenterNet/models/ctdet_coco_dla_2x.pth',
                    'Path to the model')
flags.DEFINE_float('detector_min_score_threshold', 0.5,
                   'Min score threshold for bounding box')
flags.DEFINE_string('path_coco_labels', 'dependencies/models/coco.names',
                    'Path to the COCO labels')
flags.DEFINE_bool('obj_tracking', False,
                  'True to enable object tracking operator')
flags.DEFINE_string('tracker_type', 'cv2',
                    'Tracker type: cv2 | crv | da_siam_rpn')
flags.DEFINE_string('da_siam_rpn_model_path',
                    'dependencies/models/SiamRPNVOT.model',
                    'Path to the model')
flags.DEFINE_bool('lane_detection', False, 'True to enable lane detection')
flags.DEFINE_bool('fusion', False, 'True to enable fusion operator')
flags.DEFINE_bool('traffic_light_det', False,
                  'True to enable traffic light detection operator')
flags.DEFINE_string(
    'traffic_light_det_model_path',
    'dependencies/models/traffic_light_det_inference_graph.pb',
    'Path to the traffic light model protobuf')
flags.DEFINE_float('traffic_light_det_min_score_threshold', 0.3,
                   'Min score threshold for bounding box')
flags.DEFINE_string('depth_est_model_path', 'dependencies/anynet/',
                    'Path to AnyNet depth estimation model')

# Agent flags.
flags.DEFINE_bool('stop_for_traffic_lights', True,
                  'True to enable traffic light stopping')
flags.DEFINE_bool('stop_for_pedestrians', True,
                  'True to enable pedestrian stopping')
flags.DEFINE_bool('stop_for_vehicles', True,
                  'True to enable vehicle stopping')
flags.DEFINE_bool('use_perfect_perception', False,
                  'True to enable the agent to use perfect ground detection')
# Traffic light stopping parameters.
flags.DEFINE_integer('traffic_light_min_dist_thres', 9,
                     'Min distance threshold traffic light')
flags.DEFINE_integer('traffic_light_max_dist_thres', 20,
                     'Max distance threshold traffic light')
flags.DEFINE_float('traffic_light_angle_thres', 0.5,
                   'Traffic light angle threshold')
# Vehicle stopping parameters.
flags.DEFINE_integer('vehicle_distance_thres', 15,
                     'Vehicle distance threshold')
flags.DEFINE_float('vehicle_angle_thres', 0.4,
                   'Vehicle angle threshold')
# Pedestrian stopping parameters.
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
flags.DEFINE_integer('target_speed', 36,
                     'Target speed, could be controlled by the speed limit')
flags.DEFINE_float('steer_gain', 0.7, 'Gain on computed steering angle')
flags.DEFINE_float('brake_strength', 1,
                   'Strength for applying brake; between 0 and 1')
flags.DEFINE_integer('coast_factor', 2, 'Factor to control coasting')

# Carla flags.
flags.DEFINE_string('carla_version', '0.8.4', 'Carla simulator version.')
flags.DEFINE_string('carla_host', 'localhost', 'Carla host.')
flags.DEFINE_integer('carla_port', 2000, 'Carla port.')
flags.DEFINE_integer('carla_timeout', 10,
                     'Timeout limit for Carla operator')
flags.DEFINE_bool('carla_synchronous_mode', True,
                  'Run Carla in synchronous mode.')
flags.DEFINE_integer('carla_town', 1, 'Sets which Carla town to use.')
flags.DEFINE_integer('carla_fps', 10, 'Carla FPS; do not set bellow 10.')
flags.DEFINE_float('carla_step_frequency', -1,
                   'Target frequency of sending control commands. -1 if '
                   'commands should be applied as soon ASAP.')
flags.DEFINE_integer('carla_num_vehicles', 20, 'Carla num vehicles.')
flags.DEFINE_integer('carla_num_pedestrians', 40, 'Carla num pedestrians.')
flags.DEFINE_bool('carla_high_quality', False,
                  'True to enable high quality Carla simulations.')
flags.DEFINE_integer('carla_weather', 2,
                     'Carla weather preset; between 1 and 14')
flags.DEFINE_bool('carla_random_player_start', True,
                  'True to randomly assign a car to the player')
flags.DEFINE_integer('carla_start_player_num', 0,
                     'Number of the assigned start player')
flags.DEFINE_integer('carla_camera_image_width', 800,
                     'Carla camera image width')
flags.DEFINE_integer('carla_camera_image_height', 600,
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

# Estimate depth using two cameras.
flags.DEFINE_bool('depth_estimation', False,
                  'True to depth estimation using cameras')

# Visualizing operators
flags.DEFINE_bool('visualize_depth_camera', False,
                  'True to enable depth camera video operator')
flags.DEFINE_bool('visualize_lidar', False,
                  'True to enable CARLA Lidar visualizer operator')
flags.DEFINE_bool('visualize_depth_est', False,
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

# Accuracy evaluation flags.
flags.DEFINE_bool('evaluate_obj_detection', False,
                  'True to enable object detection accuracy evaluation')
flags.DEFINE_bool('evaluate_segmentation', False,
                  'True to enable segmentation evaluation')
flags.DEFINE_bool('eval_ground_truth_segmentation', False,
                  'True to enable ground truth segmentation evaluation')
flags.DEFINE_bool('eval_ground_truth_object_detection', False,
                  'True to enable ground truth object detection evaluation.')
flags.DEFINE_string('eval_detection_metric', 'mAP',
                    'Metric to evaluate detection on: mAP | timely-mAP')
flags.DEFINE_bool('detection_eval_use_accuracy_model', False,
                  'Enable to use a model for detection accuracy decay over time')
flags.DEFINE_integer('eval_ground_truth_max_latency', 2000,
                     'Max latency to evaluate in ground truth experiments')
flags.DEFINE_string('eval_segmentation_metric', 'mIoU',
                    'Metric to evaluate segmentation on: mIoU | timely-mIoU')
flags.DEFINE_bool('segmentation_eval_use_accuracy_model', False,
                  'Enable to use a model for segmentation accuracy decay over time')

# GPU memory fractions.
flags.DEFINE_float('obj_detection_gpu_memory_fraction', 0.3,
                   'GPU memory fraction allocated to each obj detector operator')
flags.DEFINE_float('obj_tracking_gpu_memory_fraction', 0.3,
                   'GPU memory fraction allocated to each obj tracker operator')
flags.DEFINE_float('traffic_light_det_gpu_memory_fraction', 0.3,
                   'GPU memory fraction allocated to each traffic light detector')
flags.DEFINE_float('segmentation_dla_gpu_memory_fraction', 0.2,
                   'GPU memory fraction allocated to DLA segmentation')
flags.DEFINE_float('segmentation_drn_gpu_memory_fraction', 0.2,
                   'GPU memory fraction allocated to DRN segmentation')

# Recording operators
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
flags.DEFINE_integer('num_cameras', 5, 'Number of cameras.')

# Flag validators.
flags.register_validator('framework',
                         lambda value: value == 'ros' or value == 'ray',
                         message='--framework must be: ros | ray')
flags.register_multi_flags_validator(
    ['replay', 'evaluate_obj_detection'],
    lambda flags_dict: not (flags_dict['replay'] and flags_dict['evaluate_obj_detection']),
    message='--evaluate_obj_detection cannot be set when --replay is set')
flags.register_multi_flags_validator(
    ['replay', 'fusion'],
    lambda flags_dict: not (flags_dict['replay'] and flags_dict['fusion']),
    message='--fusion cannot be set when --replay is set')
# flags.register_multi_flags_validator(
#     ['ground_agent_operator', 'obj_detection', 'traffic_light_det', 'segmentation_drn', 'segmentation_dla'],
#     lambda flags_dict: (flags_dict['ground_agent_operator'] or
#                         (flags_dict['obj_detection'] and
#                          flags_dict['traffic_light_det'] and
#                          (flags_dict['segmentation_drn'] or flags_dict['segmentation_dla']))),
#     message='ERDOS agent requires obj detection, segmentation and traffic light detection')
flags.register_multi_flags_validator(
    ['obj_detection', 'detector_center_net', 'detector_ssd_mobilenet_v1',
     'detector_frcnn_resnet101', 'detector_ssd_resnet50_v1'],
    lambda flags_dict: (not flags_dict['obj_detection'] or
                        (flags_dict['obj_detection'] and
                         (flags_dict['detector_center_net'] or
                          flags_dict['detector_ssd_mobilenet_v1'] or
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
    message='--obj_detection mustg be set if --evaluate_obj_detection is set')
