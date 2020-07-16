from absl import flags

# Detection flags.
flags.DEFINE_list(
    'obstacle_detection_model_paths',
    'dependencies/models/obstacle_detection/faster-rcnn/frozen_inference_graph.pb',  # noqa: E501
    'Comma-separated list of model paths')
flags.DEFINE_list('obstacle_detection_model_names', 'faster-rcnn',
                  'Comma-separated list of model names')
flags.DEFINE_float(
    'obstacle_detection_gpu_memory_fraction', 0.3,
    'GPU memory fraction allocated to each obstacle detector operator')
flags.DEFINE_integer('obstacle_detection_gpu_index', 0,
                     'The index of the GPU to deploy the model on')
flags.DEFINE_float('obstacle_detection_min_score_threshold', 0.5,
                   'Min score threshold for bounding box')
flags.DEFINE_string('path_coco_labels', 'dependencies/models/pylot.names',
                    'Path to the COCO labels')

# Traffic light detector flags.
flags.DEFINE_string(
    'traffic_light_det_model_path',
    'dependencies/models/traffic_light_detection/faster-rcnn/frozen_inference_graph.pb',  # noqa: E501
    'Path to the traffic light model protobuf')
flags.DEFINE_float('traffic_light_det_min_score_threshold', 0.3,
                   'Min score threshold for bounding box')
flags.DEFINE_float(
    'traffic_light_det_gpu_memory_fraction', 0.3,
    'GPU memory fraction allocated to each traffic light detector')
flags.DEFINE_integer('traffic_light_det_gpu_index', 0,
                     'The index of the GPU to deploy the model on')

# DRN Segmentation flags.
flags.DEFINE_string(
    'segmentation_model_path',
    'dependencies/models/segmentation/drn/drn_d_22_cityscapes.pth',
    'Path to the model')

# Depth estimation flags.
flags.DEFINE_string('depth_estimation_model_path',
                    'dependencies/models/depth_estimation/AnyNet/',
                    'Path to AnyNet depth estimation model')

# Tracking flags.
flags.DEFINE_integer(
    'tracking_num_steps', 10,
    'Limit on number of past steps returned by the object tracker.')
flags.DEFINE_float(
    'min_matching_iou', 0.5,
    'IoU required between detection and track for matching to be considered')
flags.DEFINE_integer('obstacle_track_max_age', 3,
                     'Number of frames to track without a detection update')
flags.DEFINE_integer(
    'ignore_obstacles_with_short_history', 1,
    'Ignore obstacles with history smaller than specified value.')
flags.DEFINE_integer(
    'track_every_nth_detection', 1,
    'Determines how often detections are used by tracker (1=every detection).')

# DeepSORT tracking flags.
flags.DEFINE_string(
    'deep_sort_tracker_weights_path',
    'dependencies/models/tracking/deep-sort-carla/feature_extractor',
    'Path to weights for feature extractor model')

# DaSiamRPN tracking flags.
flags.DEFINE_string('da_siam_rpn_model_path',
                    'dependencies/models/tracking/DASiamRPN/SiamRPNVOT.model',
                    'Path to the model')

# Lane detection flags.
flags.DEFINE_float('lane_detection_gpu_memory_fraction', 0.3,
                   'GPU memory fraction allocated to Lanenet')
flags.DEFINE_integer('lane_detection_gpu_index', 0,
                     'The index of the GPU to deploy the model on')
flags.DEFINE_string(
    'lanenet_detection_model_path',
    'dependencies/models/lane_detection/lanenet/tusimple_lanenet.ckpt',
    'Path to the saved lanenet model')

# Evaluation metrics.

# Segmentation eval flags.
flags.DEFINE_enum('segmentation_metric', 'mIoU', ['mIoU', 'timely-mIoU'],
                  'Segmentation evaluation metric')

# Detection eval flags.
flags.DEFINE_enum('detection_metric', 'mAP', ['mAP', 'timely-mAP'],
                  'Detection evaluation metric')

# Tracking eval flags.
flags.DEFINE_list('tracking_metrics', [
    'num_misses', 'num_switches', 'num_false_positives', 'mota', 'motp',
    'mostly_tracked', 'mostly_lost', 'partially_tracked', 'idf1'
], 'Tracking evaluation metrics')
