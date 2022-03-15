"""Implements an operator that detects traffic lights."""
import logging
from typing import Any

import erdos
from erdos.operator import TwoInOneOut
from erdos.context import TwoInOneOutContext

import numpy as np
from pylot.perception.camera_frame import CameraFrame

import pylot.utils
from pylot.perception.detection.traffic_light import TrafficLight, \
    TrafficLightColor
from pylot.perception.detection.utils import BoundingBox2D

import tensorflow as tf


class TrafficLightDetOperator(TwoInOneOut):
    """Detects traffic lights using a TensorFlow model.

    The operator receives frames on a camera stream, and runs a model for each
    frame.

    Args:
        flags (absl.flags): Object to be used to access absl flags.
    """

    def __init__(self, flags):
        # Register a callback on the camera input stream.
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags

        # Load the model from the model file.
        pylot.utils.set_tf_loglevel(logging.ERROR)

        # Only sets memory growth for flagged GPU
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(
            [physical_devices[self._flags.traffic_light_det_gpu_index]], 'GPU')
        tf.config.experimental.set_memory_growth(
            physical_devices[self._flags.traffic_light_det_gpu_index], True)

        # Load the model from the saved_model format file.
        self._model = tf.saved_model.load(
            self._flags.traffic_light_det_model_path)

        self._labels = {
            1: TrafficLightColor.GREEN,
            2: TrafficLightColor.YELLOW,
            3: TrafficLightColor.RED,
            4: TrafficLightColor.OFF
        }
        # Unique bounding box id. Incremented for each bounding box.
        self._unique_id = 0
        # Serve some junk image to load up the model.
        self.__run_model(np.zeros((108, 192, 3), dtype='uint8'))

    def on_left_data(self, context: TwoInOneOutContext, data: CameraFrame):
        """Invoked whenever a frame message is received on the stream."""
        self._logger.debug('@{}: {} received message'.format(
            context.timestamp, self.config.name))
        assert data.encoding == 'BGR', 'Expects BGR frames'
        boxes, scores, labels = self.__run_model(data.as_rgb_numpy_array())

        traffic_lights = self.__convert_to_detected_tl(
            boxes, scores, labels, data.camera_setup.height,
            data.camera_setup.width)

        self._logger.debug('@{}: {} detected traffic lights {}'.format(
            context.timestamp, self.config.name, traffic_lights))

        context.write_stream.send(
            erdos.Message(context.timestamp, traffic_lights))
        context.write_stream.send(erdos.WatermarkMessage(context.timestamp))

        if self._flags.log_traffic_light_detector_output:
            data.annotate_with_bounding_boxes(context.timestamp,
                                              traffic_lights)
            data.save(context.timestamp.coordinates[0], self._flags.data_path,
                      'tl-detector-{}'.format(self.config.name))

    def on_right_data(self, context: TwoInOneOutContext, data: Any):
        self._logger.debug('@{}: {} received ttd update {}'.format(
            context.timestamp, self.config.name, data))

    def __run_model(self, image_np):
        # Expand dimensions since the model expects images to have
        # shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        infer = self._model.signatures['serving_default']
        result = infer(tf.convert_to_tensor(value=image_np_expanded))

        boxes = result['boxes']
        scores = result['scores']
        classes = result['classes']
        num_detections = result['detections']

        num_detections = int(num_detections[0])
        res_labels = [
            self._labels[int(label)] for label in classes[0][:num_detections]
        ]

        res_boxes = boxes[0][:num_detections]
        res_scores = scores[0][:num_detections]
        return res_boxes, res_scores, res_labels

    def __convert_to_detected_tl(self, boxes, scores, labels, height, width):
        traffic_lights = []
        for index in range(len(scores)):
            if scores[
                    index] > self._flags.traffic_light_det_min_score_threshold:
                bbox = BoundingBox2D(
                    int(boxes[index][1] * width),  # x_min
                    int(boxes[index][3] * width),  # x_max
                    int(boxes[index][0] * height),  # y_min
                    int(boxes[index][2] * height)  # y_max
                )
                traffic_lights.append(
                    TrafficLight(scores[index],
                                 labels[index],
                                 id=self._unique_id,
                                 bounding_box=bbox))
                self._unique_id += 1
        return traffic_lights

    def destroy(self):
        self._logger.warn('destroying {}'.format(self.config.name))
