""" If this operator needs to be run with the faster NMS implementation,
follow these steps:
    1. Clone google/automl and tensorflow/models from Github.
    2. Run the following commands in automl/efficientdet:
        a. grep -lIR "from object_detection" ./* | xargs sed -i "s/from object_detection/from object_det/g"
        b. mv object_detection object_det
        c. grep -lIR "inference\." ./* | xargs sed -i 's/inference\./infer\./g'
        d. grep -lIR "import inference" ./* | xargs sed -i 's/import inference/import infer/g'
        e. mv inference.py infer.py
        f. grep -lIR " utils" ./* | xargs sed -i 's/ utils/ util/g'
	g. grep -lIR "utils.TpuBatchNormalization" ./* | xargs sed -i 's/utils\.TpuBatchNormalization/util\.TpuBatchNormalization/g'
	h. mv utils.py util.py
    3. Set the MODIFIED_AUTOML flag to on.
"""
import erdos
import time
import copy

import pylot.utils
from pylot.perception.detection.utils import BoundingBox2D, DetectedObstacle,\
    load_coco_bbox_colors, load_coco_labels
from pylot.perception.messages import ObstaclesMessage

# Detection related imports.
import numpy as np
import tensorflow as tf
import hparams_config
try:
    import infer
except ImportError:
    import inference as infer
import anchors

NUM_CLASSES = 90
MODIFIED_AUTOML = False


class EfficientDetOperator(erdos.Operator):
    """ Detects obstacles using the EfficientDet set of models.

    The operator receives frames on camera stream, and runs a model for each
    frame.

    Args:
        camera_stream (:py:class:`erdos.ReadStream`): The stream on which
            camera frames are received.
        obstacles_stream (:py:class:`erdos.WriteStream`): Stream on which the
            operator sends
            :py:class:`~pylot.perception.messages.ObstaclesMessage` messages.
        model_path(:obj:`str`): Path to the model pb file.
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, camera_stream, time_to_decision_stream,
                 obstacles_stream, model_names, model_paths, flags):
        camera_stream.add_callback(self.on_msg_camera_stream,
                                   [obstacles_stream])
        time_to_decision_stream.add_callback(self.on_time_to_decision_update)
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            self.config.name + '-csv', self.config.csv_log_file_name)

        # Load the COCO labels.
        self._coco_labels = load_coco_labels(self._flags.path_coco_labels)
        self._bbox_colors = load_coco_bbox_colors(self._coco_labels)
        self._important_labels = {
            'car', 'bicycle', 'motorcycle', 'bus', 'truck', 'vehicle',
            'person', 'stop sign', 'parking meter', 'cat', 'dog',
            'speed limit 30', 'speed limit 60', 'speed limit 90'
        }

        # Build inputs and preprocessing.
        tf.compat.v1.disable_eager_execution()

        assert len(model_names) == len(
            model_paths), 'Model names and paths do not have same length'
        self._models = {}
        for index, model_path in enumerate(model_paths):
            model_name = model_names[index]
            self._models[model_name] = self.load_model(model_name, model_path,
                                                       flags)
            if index == 0:
                # Use the first model by default.
                (self._tf_session, self._image_placeholder,
                 self._detections_batch) = self._models[model_name]
        self._unique_id = 0

    def load_model(self, model_name, model_path, flags):
        graph = tf.Graph()
        # Initialize the Config and Session.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=flags.
                                    obstacle_detection_gpu_memory_fraction,
                                    allow_growth=True)

        tf_session = tf.Session(graph=graph,
                                config=tf.ConfigProto(gpu_options=gpu_options))

        with graph.as_default():
            # Get the required arguments to initialize Inference.
            image_size = hparams_config.get_detection_config(
                model_name).image_size
            driver = infer.InferenceDriver(model_name, model_path, image_size,
                                           NUM_CLASSES)

            image_placeholder = tf.placeholder("uint8", [None, None, 3],
                                               name="image_tensor")
            raw_images, images, scales = \
                EfficientDetOperator.build_inputs_with_placeholder(
                    image_placeholder, image_size)

            # Build model.
            class_outputs, box_outputs = infer.build_model(model_name, images)
            infer.restore_ckpt(tf_session,
                               model_path,
                               enable_ema=True,
                               export_ckpt=None)

            # Build postprocessing.
            params = copy.deepcopy(driver.params)
            params.update(dict(batch_size=1))
            detections_batch = None
            if MODIFIED_AUTOML:
                detections_batch = EfficientDetOperator.det_post_process(
                    params, class_outputs, box_outputs, scales)
            else:
                detections_batch = infer.det_post_process(
                    params, class_outputs, box_outputs, scales)

        return tf_session, image_placeholder, detections_batch

    @staticmethod
    def connect(camera_stream, time_to_decision_stream):
        """Connects the operator to other streams.

        Args:
            camera_stream (:py:class:`erdos.ReadStream`): The stream on which
                camera frames are received.

        Returns:
            :py:class:`erdos.WriteStream`: Stream on which the operator sends
            :py:class:`~pylot.perception.messages.ObstaclesMessage` messages.
        """
        obstacles_stream = erdos.WriteStream()
        return [obstacles_stream]

    def pick_model(self, ttd):
        """Decides which model to use based on time to decision."""
        runtimes = [('efficientdet-d7', 262), ('efficientdet-d6', 190),
                    ('efficientdet-d5', 141), ('efficientdet-d4', 74),
                    ('efficientdet-d3', 42), ('efficientdet-d2', 24),
                    ('efficientdet-d1', 20), ('efficientdet-d0', 16)]
        fastest_loaded_model_name = None
        for index, (model_name, runtime) in enumerate(runtimes):
            # Pick the model if it is preloaded and if we have enough time to
            # run it.
            if ttd >= runtime and model_name in self._models:
                self._logger.debug(
                    'Using detection model {}'.format(model_name))
                return self._models[model_name]
            if model_name in self._models:
                fastest_loaded_model_name = model_name
        # Not enough time to run detection.
        self._logger.error(
            'Insufficient time to run detection. Using detection model {}'.
            format(fastest_loaded_model_name))
        return self._models[fastest_loaded_model_name]

    def on_time_to_decision_update(self, msg):
        self._logger.debug('@{}: {} received ttd update {}'.format(
            msg.timestamp, self.config.name, msg))
        if self._flags.deadline_enforcement == 'dynamic':
            (self._tf_session, self._image_placeholder,
             self._detections_batch) = self.pick_model(msg.data)
        elif self._flags.deadline_enforcement == 'static':
            (self._tf_session, self._image_placeholder,
             self._detections_batch) = self.pick_model(
                 self._flags.detection_deadline)
        else:
            return

    @erdos.profile_method()
    def on_msg_camera_stream(self, msg, obstacles_stream):
        """Invoked whenever a frame message is received on the stream.

        Args:
            msg (:py:class:`~pylot.perception.messages.FrameMessage`): Message
                received.
            obstacles_stream (:py:class:`erdos.WriteStream`): Stream on which
                the operator sends
                :py:class:`~pylot.perception.messages.ObstaclesMessage`
                messages.
        """
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self.config.name))
        inputs = msg.frame.as_rgb_numpy_array()

        results = []
        start_time = time.time()
        if MODIFIED_AUTOML:
            (boxes_np, scores_np, classes_np,
             num_detections_np) = self._tf_session.run(
                 self._detections_batch,
                 feed_dict={self._image_placeholder: inputs})
            num_detections = num_detections_np[0]
            boxes = boxes_np[0][:num_detections]
            scores = scores_np[0][:num_detections]
            classes = classes_np[0][:num_detections]
            results = zip(boxes, scores, classes)
        else:
            outputs_np = self._tf_session.run(
                self._detections_batch,
                feed_dict={self._image_placeholder: inputs})[0]
            for _, x, y, width, height, score, _class in outputs_np:
                results.append(((y, x, y + height, x + width), score, _class))
        end_time = time.time()
        obstacles = []
        for (ymin, xmin, ymax, xmax), score, _class in results:
            if np.isclose(ymin, ymax) or np.isclose(xmin, xmax):
                continue
            if MODIFIED_AUTOML:
                # The alternate NMS implementation screws up the class labels.
                _class = int(_class) + 1
            if _class in self._coco_labels:
                if (score >= self._flags.obstacle_detection_min_score_threshold
                        and
                        self._coco_labels[_class] in self._important_labels):
                    camera_setup = msg.frame.camera_setup
                    width, height = camera_setup.width, camera_setup.height
                    xmin, xmax = max(0, int(xmin)), min(int(xmax), width)
                    ymin, ymax = max(0, int(ymin)), min(int(ymax), height)
                    obstacles.append(
                        DetectedObstacle(BoundingBox2D(xmin, xmax, ymin, ymax),
                                         score,
                                         self._coco_labels[_class],
                                         id=self._unique_id))
                    self._unique_id += 1
                    self._csv_logger.info(
                        "{}, detection, {}, {:4f}, {}".format(
                            msg.timestamp, self._coco_labels[_class], score,
                            (end_time - start_time) * 1000))
            else:
                self._logger.debug(
                    'Filtering unknown class: {}'.format(_class))

        if (self._flags.visualize_detected_obstacles
                or self._flags.log_detector_output):
            msg.frame.annotate_with_bounding_boxes(msg.timestamp, obstacles,
                                                   None, self._bbox_colors)
            if self._flags.visualize_detected_obstacles:
                msg.frame.visualize(self.config.name,
                                    pygame_display=pylot.utils.PYGAME_DISPLAY)
            if self._flags.log_detector_output:
                msg.frame.save(msg.timestamp.coordinates[0],
                               self._flags.data_path,
                               'detector-{}'.format(self.config.name))
        obstacles_stream.send(ObstaclesMessage(msg.timestamp, obstacles, 0))

    @staticmethod
    def build_inputs_with_placeholder(image, image_size):
        """ Builds the input image using a placeholder.

        Args:
            image: A placeholder for the image.
            image_size: a single integer for image width and height.

        Returns
            (raw_images, processed_image_placeholder, scales)
        """
        raw_images, images, scales = [], [], []
        raw_images.append(image)
        image, scale = infer.image_preprocess(image, image_size)
        images.append(image)
        scales.append(scale)
        return raw_images, tf.stack(images), tf.stack(scales)

    @staticmethod
    def det_post_process(params, class_outputs, box_outputs, scales):
        from object_detection.core.post_processing import \
            batch_multiclass_non_max_suppression
        cls_outputs_all, box_outputs_all = [], []
        for level in range(params['min_level'], params['max_level'] + 1):
            cls_outputs_all.append(
                tf.reshape(class_outputs[level],
                           [params['batch_size'], -1, params['num_classes']]))
            box_outputs_all.append(
                tf.reshape(box_outputs[level], [params['batch_size'], -1, 4]))
        cls_outputs_all = tf.concat(cls_outputs_all, 1)
        box_outputs_all = tf.concat(box_outputs_all, 1)
        probs = tf.math.sigmoid(cls_outputs_all)

        # Generate location of anchors.
        eval_anchors = tf.transpose(
            anchors.Anchors(params['min_level'], params['max_level'],
                            params['num_scales'], params['aspect_ratios'],
                            params['anchor_scale'],
                            params['image_size']).boxes)
        ycenter_a = (eval_anchors[0] + eval_anchors[2]) / 2
        xcenter_a = (eval_anchors[1] + eval_anchors[3]) / 2
        ha = eval_anchors[2] - eval_anchors[0]
        wa = eval_anchors[3] - eval_anchors[1]

        # Generate absolute bboxes in the units of pixels of the image.
        box_outputs_per_sample = tf.transpose(box_outputs_all[0])
        ty, tx, th, tw = (box_outputs_per_sample[0], box_outputs_per_sample[1],
                          box_outputs_per_sample[2], box_outputs_per_sample[3])
        w, h = tf.math.exp(tw) * wa, tf.math.exp(th) * ha
        ycenter, xcenter = ty * ha + ycenter_a, tx * wa + xcenter_a
        ymin, ymax = ycenter - h / 2.0, ycenter + h / 2.0
        xmin, xmax = xcenter - w / 2.0, xcenter + w / 2.0
        boxes = tf.transpose(tf.stack([ymin, xmin, ymax, xmax]))

        # Generate the outputs
        boxes_all = tf.reshape(boxes, [params['batch_size'], -1, 1, 4])
        probs_all = tf.reshape(
            probs, [params['batch_size'], -1, params['num_classes']])
        (boxes_tf, scores_tf, classes_tf, _, _, num_detections_tf) = \
            batch_multiclass_non_max_suppression(
                boxes=boxes_all, scores=probs_all, score_thresh=0.5,
                iou_thresh=0.5,
                max_size_per_class=anchors.MAX_DETECTIONS_PER_IMAGE,
                max_total_size=anchors.MAX_DETECTIONS_PER_IMAGE,
                use_combined_nms=False, use_class_agnostic_nms=True)
        boxes_tf *= scales
        return [boxes_tf, scores_tf, classes_tf, num_detections_tf]
