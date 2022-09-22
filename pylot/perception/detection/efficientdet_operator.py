import time
from collections import deque
from typing import List

import erdos
from erdos.context import TwoInOneOutContext
from erdos.operator import TwoInOneOut

import numpy as np
from pylot.perception.messages import ObstaclesMessageTuple

import pylot.utils
from pylot.perception.camera_frame import CameraFrame
from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.detection.utils import BoundingBox2D, \
    OBSTACLE_LABELS, load_coco_bbox_colors, load_coco_labels

import tensorflow as tf


# TODO: Remove once transition to TF2 is complete
class EfficientDetOperator(TwoInOneOut[CameraFrame, float,
                                       ObstaclesMessageTuple]):
    """ Detects obstacles using the EfficientDet set of models.

    The operator receives frames on camera stream, and runs a model for each
    frame.

    Args:
        model_names (list): List of model names
        model_path (list): List of paths to the model pb file.
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, model_names: List[str], model_paths: List[str], flags):
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            self.config.name + '-csv', self.config.csv_log_file_name)

        # Load the COCO labels.
        self._coco_labels = load_coco_labels(self._flags.path_coco_labels)
        self._bbox_colors = load_coco_bbox_colors(self._coco_labels)

        assert len(model_names) == len(
            model_paths), 'Model names and paths do not have same length'
        self._models = {}
        self._tf_session = None
        self._signatures = {
            'image_files': 'image_files:0',
            'image_arrays': 'image_arrays:0',
            'prediction': 'detections:0',
        }
        for index, model_path in enumerate(model_paths):
            model_name = model_names[index]
            self._models[model_name] = self.load_serving_model(
                model_name, model_path,
                flags.obstacle_detection_gpu_memory_fraction)
            if index == 0:
                # Use the first model by default.
                self._model_name, self._tf_session = self._models[model_name]
                # Serve some junk image to load up the model.
                inputs = np.zeros((108, 192, 3))
                self._tf_session.run(
                    self._signatures['prediction'],
                    feed_dict={self._signatures['image_arrays']: [inputs]})[0]
        self._unique_id = 0
        self._frames = deque()
        self._ttd_data = deque()
        self._last_ttd = 400

    def load_serving_model(self, model_name: str, model_path: str,
                           gpu_memory_fraction: float):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            # Load a frozen graph.
            graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(model_path, 'rb') as f:
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
        gpu_options = tf.compat.v1.GPUOptions(
            allow_growth=True,
            visible_device_list=str(self._flags.obstacle_detection_gpu_index),
            per_process_gpu_memory_fraction=gpu_memory_fraction)
        return model_name, tf.compat.v1.Session(
            graph=detection_graph,
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

    def _pick_model(self, ttd: float):
        """Decides which model to use based on time to decision."""
        # 99th percetile runtime on RTX 2080.
        runtimes = [('efficientdet-d7', 269), ('efficientdet-d6', 200),
                    ('efficientdet-d5', 158), ('efficientdet-d4', 105),
                    ('efficientdet-d3', 77), ('efficientdet-d2', 59),
                    ('efficientdet-d1', 55)]
        # # 99th percentile runitme on V100
        # runtimes = [('efficientdet-d7', 238), ('efficientdet-d6', None),
        #             ('efficientdet-d5', None), ('efficientdet-d4', 98),
        #             ('efficientdet-d3', None), ('efficientdet-d2', None),
        #             ('efficientdet-d1', 56)]
        # Runtimes from EfficientDet paper.
        # runtimes = [('efficientdet-d6', 190), ('efficientdet-d5', 141),
        #             ('efficientdet-d4', 74), ('efficientdet-d3', 42),
        #             ('efficientdet-d2', 24)]
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

    def update_model_choice(self, detection_deadline: float):
        if self._flags.deadline_enforcement == 'dynamic':
            self._model_name, self._tf_session = self._pick_model(
                detection_deadline)
        elif self._flags.deadline_enforcement == 'static':
            self._model_name, self._tf_session = self._pick_model(
                self._flags.detection_deadline)

    def on_left_data(self, context: TwoInOneOutContext[List[Obstacle]],
                     data: CameraFrame):
        """Invoked whenever a camera message is received on the stream."""
        self._logger.debug('@{}: {} received message'.format(
            context.timestamp, self.config.name))
        self._frames.append(data)
        if context.timestamp.is_top:
            return
        start_time = time.time()
        if len(self._ttd_data) > 0:
            ttd = self._ttd_data.popleft()
            self._last_ttd = ttd
        frame = self._frames.popleft()
        self.update_model_choice(self._last_ttd)
        inputs = frame.as_rgb_numpy_array()
        detector_start_time = time.time()
        outputs_np = self._tf_session.run(
            self._signatures['prediction'],
            feed_dict={self._signatures['image_arrays']: [inputs]})[0]
        detector_end_time = time.time()
        runtime = (detector_end_time - detector_start_time) * 1000
        self._logger.debug("@{}: detector runtime {}".format(
            context.timestamp, runtime))
        obstacles = []
        camera_setup = frame.camera_setup
        for _, ymin, xmin, ymax, xmax, score, _class in outputs_np:
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            if _class in self._coco_labels:
                if (score >= self._flags.obstacle_detection_min_score_threshold
                        and self._coco_labels[_class] in OBSTACLE_LABELS):
                    xmin, xmax = max(0, xmin), min(xmax, camera_setup.width)
                    ymin, ymax = max(0, ymin), min(ymax, camera_setup.height)
                    if xmin < xmax and ymin < ymax:
                        obstacles.append(
                            Obstacle(BoundingBox2D(xmin, xmax, ymin, ymax),
                                     score,
                                     self._coco_labels[_class],
                                     id=self._unique_id))
                        self._unique_id += 1
                        self._csv_logger.info(
                            "{},{},detection,{},{:4f}".format(
                                pylot.utils.time_epoch_ms(),
                                context.timestamp.coordinates[0],
                                self._coco_labels[_class], score))
            else:
                self._logger.debug(
                    'Filtering unknown class: {}'.format(_class))

        if self._flags.log_detector_output:
            frame.annotate_with_bounding_boxes(context.timestamp, obstacles,
                                               None, self._bbox_colors)
            frame.save(context.timestamp.coordinates[0], self._flags.data_path,
                       'detector-{}'.format(self.config.name))
        context.write_stream.send(
            erdos.Message(context.timestamp,
                          ObstaclesMessageTuple(obstacles, runtime)))
        context.write_stream.send(erdos.WatermarkMessage(context.timestamp))

        operator_time_total_end = time.time()
        self._logger.debug("@{}: total time spent: {}".format(
            context.timestamp, (operator_time_total_end - start_time) * 1000))

    def on_right_data(self, context: TwoInOneOutContext[List[Obstacle]],
                      data: float):
        self._logger.debug('@{}: {} received ttd update {}'.format(
            context.timestamp, self.config.name, data))
        self._ttd_data.append(data)

    def destroy(self):
        self._logger.warn('destroying {}'.format(self.config.name))
