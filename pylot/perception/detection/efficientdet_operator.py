import time
from collections import deque

import erdos

import numpy as np

import tensorflow as tf

import pylot.utils
from pylot.perception.detection.utils import BoundingBox2D, DetectedObstacle,\
    load_coco_bbox_colors, load_coco_labels
from pylot.perception.messages import ObstaclesMessage

# Detection related imports.
import automl.efficientdet.hparams_config as hparams_config
try:
    import automl.efficientdet.infer as infer
except ImportError:
    import automl.efficientdet.inference as infer


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
        camera_stream.add_callback(self.on_msg_camera_stream)
        time_to_decision_stream.add_callback(self.on_time_to_decision_update)
        erdos.add_watermark_callback([camera_stream], [obstacles_stream],
                                     self.on_watermark)
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

        assert len(model_names) == len(
            model_paths), 'Model names and paths do not have same length'
        self._models = {}
        self._driver = None
        for index, model_path in enumerate(model_paths):
            model_name = model_names[index]
            self._models[model_name] = self.load_serving_model(
                model_name, model_path)
            if index == 0:
                # Use the first model by default.
                self._model_name, self._driver = self._models[model_name]
                # Serve some junk image to load up the model.
                inputs = np.zeros((108, 192, 3))
                _ = self._driver.serve_images([inputs])[0]
        self._unique_id = 0

        self._frame_msgs = deque()
        self._ttd_msgs = deque()

    def load_serving_model(self, model_name, model_path):
        graph = tf.Graph()
        driver = infer.ServingDriver(
            graph,
            model_name,
            model_path,
            batch_size=1,
            use_xla=True,
            model_params=hparams_config.get_detection_config(
                model_name).as_dict(),
            gpu_memory_fraction=self._flags.
            obstacle_detection_gpu_memory_fraction)
        driver.load(model_path)
        return model_name, driver

    @staticmethod
    def connect(camera_stream, time_to_decision_stream):
        """Connects the operator to other streams.

        Args:
            camera_stream (:py:class:`erdos.ReadStream`): The stream on which
                camera frames are received.

        Returns:
            [:py:class:`erdos.WriteStream`, :py:class:`erdos.WriteStream`]:
            Streams on which the operator sends
            :py:class:`~pylot.perception.messages.ObstaclesMessage` messages
            for detection and :py:class:`erdos.Message` for runtimes.
        """
        obstacles_stream = erdos.WriteStream()
        return [obstacles_stream]

    def _pick_model(self, ttd):
        """Decides which model to use based on time to decision."""
        runtimes = [('efficientdet-d6', 190), ('efficientdet-d5', 141),
                    ('efficientdet-d4', 74), ('efficientdet-d3', 42),
                    ('efficientdet-d2', 24)]
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

    def update_model_choice(self, detection_deadline):
        if self._flags.deadline_enforcement == 'dynamic':
            self._model_name, self._driver = self._pick_model(
                detection_deadline)
        elif self._flags.deadline_enforcement == 'static':
            self._model_name, self._driver = self._pick_model(
                self._flags.detection_deadline)

    def on_time_to_decision_update(self, msg):
        self._logger.debug('@{}: {} received ttd update {}'.format(
            msg.timestamp, self.config.name, msg))
        self._ttd_msgs.append(msg)

    @erdos.profile_method()
    def on_msg_camera_stream(self, msg):
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self.config.name))
        self._frame_msgs.append(msg)

    @erdos.profile_method()
    def on_watermark(self, timestamp, obstacles_stream):
        """Invoked whenever a frame message is received on the stream.

        Args:
            msg (:py:class:`~pylot.perception.messages.FrameMessage`): Message
                received.
            obstacles_stream (:py:class:`erdos.WriteStream`): Stream on which
                the operator sends
                :py:class:`~pylot.perception.messages.ObstaclesMessage`
                messages.
        """
        start_time = time.time()
        #ttd_msg = self._ttd_msgs.popleft()
        frame_msg = self._frame_msgs.popleft()
        #ttd, detection_deadline = ttd_msg.data
        #self.update_model_choice(detection_deadline)
        frame = frame_msg.frame
        inputs = frame.as_rgb_numpy_array()
        detector_start_time = time.time()
        outputs_np = self._driver.serve_images([inputs])[0]
        detector_end_time = time.time()
        self._logger.debug("@{}: detector runtime {}".format(
            timestamp, (detector_end_time - detector_start_time) * 1000))
        obstacles = []
        camera_setup = frame.camera_setup
        for _, y, x, height, width, score, _class in outputs_np:
            xmin = int(x)
            ymin = int(y)
            xmax = int(x + width)
            ymax = int(y + height)
            if _class in self._coco_labels:
                if (score >= self._flags.obstacle_detection_min_score_threshold
                        and
                        self._coco_labels[_class] in self._important_labels):
                    xmin, xmax = max(0, xmin), min(xmax, camera_setup.width)
                    ymin, ymax = max(0, ymin), min(ymax, camera_setup.height)
                    if xmin < xmax and ymin < ymax:
                        obstacles.append(
                            DetectedObstacle(BoundingBox2D(
                                xmin, xmax, ymin, ymax),
                                             score,
                                             self._coco_labels[_class],
                                             id=self._unique_id))
                        self._unique_id += 1
                        self._csv_logger.info(
                            "{},{},detection,{},{:4f}".format(
                                pylot.utils.time_epoch_ms(),
                                timestamp.coordinates[0],
                                self._coco_labels[_class], score))
            else:
                self._logger.debug(
                    'Filtering unknown class: {}'.format(_class))

        if (self._flags.visualize_detected_obstacles
                or self._flags.log_detector_output):
            frame.annotate_with_bounding_boxes(timestamp, obstacles, None,
                                               self._bbox_colors)
            if self._flags.visualize_detected_obstacles:
                frame.visualize(self.config.name,
                                pygame_display=pylot.utils.PYGAME_DISPLAY)
            if self._flags.log_detector_output:
                frame.save(timestamp.coordinates[0], self._flags.data_path,
                           'detector-{}'.format(self.config.name))
        end_time = time.time()
        obstacles_stream.send(ObstaclesMessage(timestamp, obstacles, 0))
        obstacles_stream.send(erdos.WatermarkMessage(timestamp))
        operator_time_total_end = time.time()
        self._logger.debug("@{}: total time spent: {}".format(
            timestamp, (operator_time_total_end - start_time) * 1000))
