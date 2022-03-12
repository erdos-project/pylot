"""Implements an operator that detects obstacles."""
import logging
import time
from typing import Any

import erdos
from erdos.operator import TwoInOneOut
from erdos.context import TwoInOneOutContext

import numpy as np

import pylot.utils
from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.detection.utils import BoundingBox2D, \
    OBSTACLE_LABELS, load_coco_bbox_colors, load_coco_labels
from pylot.perception.messages import ObstaclesMessage

import tensorflow as tf

class DetectionOperator(TwoInOneOut):
    """Detects obstacles using a TensorFlow model.

    The operator receives frames on a camera stream, and runs a model for each
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
    def __init__(self, model_path: str, flags):
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)

        pylot.utils.set_tf_loglevel(logging.ERROR)

        # Only sets memory growth for flagged GPU
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(
            [physical_devices[0]],
            'GPU')
        tf.config.experimental.set_memory_growth(
            physical_devices[0], True)

        # Load the model from the saved_model format file.
        self._model = tf.saved_model.load(model_path)

        self._coco_labels = load_coco_labels(self._flags.path_coco_labels)
        self._bbox_colors = load_coco_bbox_colors(self._coco_labels)
        # Unique bounding box id. Incremented for each bounding box.
        self._unique_id = 0

        # Serve some junk image to load up the model.
        self.__run_model(np.zeros((108, 192, 3), dtype='uint8'))

    def on_left_data(self, context: TwoInOneOutContext, data: Any):
        self.on_msg_camera_stream(data, context.write_stream, context.timestamp)

    def on_right_data(self, context: TwoInOneOutContext, data: Any):
        self.on_time_to_decision_update(data)

    def on_msg_camera_stream(self, msg: erdos.Message,
                             obstacles_stream: erdos.WriteStream,
                             timestamp: erdos.Timestamp):
        """Invoked whenever a frame message is received on the stream.

        Args:
            msg (:py:class:`~pylot.perception.messages.FrameMessage`): Message
                received.
            obstacles_stream (:py:class:`erdos.WriteStream`): Stream on which
                the operator sends
                :py:class:`~pylot.perception.messages.ObstaclesMessage`
                messages.
            timestamp (:py:class:`erdos.Timestamp`): Timestamp associated with message recieved
        """
        self._logger.debug('@{} received message'.format(self.config.name))
        start_time = time.time()

        # The models expect BGR images.
        assert msg.encoding == 'BGR', 'Expects BGR frames'
        num_detections, res_boxes, res_scores, res_classes = self.__run_model(msg.frame)
        obstacles = []
        for i in range(0, num_detections):
            if res_classes[i] in self._coco_labels:
                if (res_scores[i] >= self._flags.obstacle_detection_min_score_threshold):
                    if (self._coco_labels[res_classes[i]] in OBSTACLE_LABELS):
                        obstacles.append(
                            Obstacle(BoundingBox2D(
                                int(res_boxes[i][1] *
                                    msg.camera_setup.width),
                                int(res_boxes[i][3] *
                                    msg.camera_setup.width),
                                int(res_boxes[i][0] *
                                    msg.camera_setup.height),
                                int(res_boxes[i][2] *
                                    msg.camera_setup.height)),
                                     res_scores[i],
                                     self._coco_labels[res_classes[i]],
                                     id=self._unique_id))
                        self._unique_id += 1
                    else:
                        self._logger.warning('@{} Ignoring non essential detection {}'.format(timestamp, self._coco_labels[res_classes[i]]))
            else:
                self._logger.warning('@{} Filtering unknown class: {}'.format(timestamp, res_classes[i]))

        print('@{}: {}'.format(timestamp, obstacles))

        # Get runtime in ms.
        runtime = (time.time() - start_time) * 1000
        # Send out obstacles.
        obstacles_stream.send(ObstaclesMessage(timestamp, obstacles, runtime))   # AttributeError: 'CameraFrame' object has no attribute 'timestamp'
        obstacles_stream.send(erdos.WatermarkMessage(timestamp))

        if self._flags.log_detector_output:
            msg.annotate_with_bounding_boxes(timestamp, obstacles,
                                                    None, self._bbox_colors)
            msg.save(timestamp.coordinates[0], self._flags.data_path,
                    'detector-{}'.format(self.config.name))
        # msg.save(timestamp=timestamp, data_path='./', file_base='detection')

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
        res_classes = [int(cls) for cls in classes[0][:num_detections]]
        res_boxes = boxes[0][:num_detections]
        res_scores = scores[0][:num_detections]
        return num_detections, res_boxes, res_scores, res_classes

    def on_time_to_decision_update(self, msg: erdos.Message):
        self._logger.debug('@{}: {} received ttd update {}'.format(
            msg.timestamp, self.config.name, msg))

    def destroy(self):
        self._logger.warn('destroying {}'.format(self.config.name))
        # Sending top watermark because the operator is not flowing
        # watermarks.
        # self._obstacles_stream.send(
        #     erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))

