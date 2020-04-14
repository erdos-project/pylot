import erdos
import copy

import pylot.utils
from pylot.perception.detection.utils import BoundingBox2D, DetectedObstacle,\
    load_coco_bbox_colors, load_coco_labels
from pylot.perception.messages import ObstaclesMessage

# Detection related imports.
import numpy as np
import tensorflow as tf
import hparams_config
import inference

NUM_CLASSES = 90


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
    def __init__(self, camera_stream, obstacles_stream, model_path, flags):
        camera_stream.add_callback(self.on_msg_camera_stream,
                                   [obstacles_stream])
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)

        # Get the required arguments to initialize Inference.
        model_name = self.config.name
        image_size = hparams_config.get_detection_config(model_name).image_size
        driver = inference.InferenceDriver(model_name, model_path, image_size,
                                           NUM_CLASSES)

        # Initialize the Config and Session.
        self._gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=flags.
            obstacle_detection_gpu_memory_fraction,
            allow_growth=True)
        self._tf_session = tf.Session(config=tf.ConfigProto(
            gpu_options=self._gpu_options))

        # Build inputs and preprocessing.
        self._image_placeholder = tf.placeholder("uint8", [None, None, 3],
                                                 name="image_tensor")
        results = EfficientDetOperator.build_inputs_with_placeholder(
            self._image_placeholder, image_size)
        self._raw_images, self._images, self._scales = results

        # Build model.
        self._class_outputs, self._box_outputs = inference.build_model(
            model_name, self._images)
        inference.restore_ckpt(self._tf_session,
                               model_path,
                               enable_ema=True,
                               export_ckpt=None)

        # Build postprocessing.
        params = copy.deepcopy(driver.params)
        params.update(dict(batch_size=1))
        self._detections_batch = inference.det_post_process(
            params, self._class_outputs, self._box_outputs, self._scales)

        # Load the COCO labels.
        self._coco_labels = load_coco_labels(self._flags.path_coco_labels)
        self._bbox_colors = load_coco_bbox_colors(self._coco_labels)
        self._important_labels = {
            'car', 'bicycle', 'motorcycle', 'bus', 'truck', 'vehicle',
            'person', 'stop sign', 'parking meter', 'cat', 'dog',
            'speed limit 30', 'speed limit 60', 'speed limit 90'
        }
        self._unique_id = 0

    @staticmethod
    def connect(camera_stream):
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

        outputs_np = self._tf_session.run(
            self._detections_batch,
            feed_dict={self._image_placeholder: inputs})[0]
        obstacles = []
        for image_id, x, y, width, height, score, _class in outputs_np:
            if np.isclose(width, 0) or np.isclose(height, 0):
                continue
            _class = int(_class)
            if _class in self._coco_labels:
                if (score >= self._flags.obstacle_detection_min_score_threshold
                        and
                        self._coco_labels[_class] in self._important_labels):
                    min_x = int(x)
                    min_y = int(y)
                    max_x = min(int(x + width), msg.frame.camera_setup.width)
                    max_y = min(int(y + height), msg.frame.camera_setup.height)
                    if min_x < max_x and min_y < max_y:
                        obstacles.append(
                            DetectedObstacle(BoundingBox2D(
                                min_x, max_x, min_y, max_y),
                                             score,
                                             self._coco_labels[_class],
                                             id=self._unique_id))
                    self._unique_id += 1
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
        image, scale = inference.image_preprocess(image, image_size)
        images.append(image)
        scales.append(scale)
        return raw_images, tf.stack(images), tf.stack(scales)
