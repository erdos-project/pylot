"""Implements an operator that detects obstacles."""

from absl import flags
import erdos
import logging
import numpy as np
import tensorflow as tf
import time

from pylot.perception.detection.utils import BoundingBox2D, DetectedObstacle,\
    load_coco_labels, load_coco_bbox_colors
from pylot.perception.messages import ObstaclesMessage
from pylot.utils import set_tf_loglevel, time_epoch_ms

flags.DEFINE_float(
    'obstacle_detection_gpu_memory_fraction', 0.3,
    'GPU memory fraction allocated to each obstacle detector operator')
flags.DEFINE_float('obstacle_detection_min_score_threshold', 0.5,
                   'Min score threshold for bounding box')
flags.DEFINE_string('path_coco_labels', 'dependencies/models/pylot.names',
                    'Path to the COCO labels')


class DetectionOperator(erdos.Operator):
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
    def __init__(self, camera_stream, obstacles_stream, model_path, flags):
        camera_stream.add_callback(self.on_msg_camera_stream,
                                   [obstacles_stream])
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._detection_graph = tf.Graph()
        # Load the model from the model file.
        set_tf_loglevel(logging.ERROR)
        with self._detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self._gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=flags.
            obstacle_detection_gpu_memory_fraction)
        # Create a TensorFlow session.
        self._tf_session = tf.Session(
            graph=self._detection_graph,
            config=tf.ConfigProto(gpu_options=self._gpu_options))
        # Get the tensors we're interested in.
        self._image_tensor = self._detection_graph.get_tensor_by_name(
            'image_tensor:0')
        self._detection_boxes = self._detection_graph.get_tensor_by_name(
            'detection_boxes:0')
        self._detection_scores = self._detection_graph.get_tensor_by_name(
            'detection_scores:0')
        self._detection_classes = self._detection_graph.get_tensor_by_name(
            'detection_classes:0')
        self._num_detections = self._detection_graph.get_tensor_by_name(
            'num_detections:0')
        self._coco_labels = load_coco_labels(self._flags.path_coco_labels)
        self._bbox_colors = load_coco_bbox_colors(self._coco_labels)
        self._important_labels = {
            'car', 'bicycle', 'motorcycle', 'bus', 'truck', 'vehicle',
            'person', 'stop sign', 'parking meter', 'cat', 'dog',
            'speed limit 30', 'speed limit 60', 'speed limit 90'
        }

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
        start_time = time.time()
        # The models expect BGR images.
        assert msg.frame.encoding == 'BGR', 'Expects BGR frames'
        # Expand dimensions since the model expects images to have
        # shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(msg.frame.frame, axis=0)
        (boxes, scores, classes, num_detections) = self._tf_session.run(
            [
                self._detection_boxes, self._detection_scores,
                self._detection_classes, self._num_detections
            ],
            feed_dict={self._image_tensor: image_np_expanded})

        num_detections = int(num_detections[0])
        res_classes = [int(cls) for cls in classes[0][:num_detections]]
        res_boxes = boxes[0][:num_detections]
        res_scores = scores[0][:num_detections]

        obstacles = []
        for i in range(0, num_detections):
            if res_classes[i] in self._coco_labels:
                if (res_scores[i] >=
                        self._flags.obstacle_detection_min_score_threshold
                        and self._coco_labels[
                            res_classes[i]] in self._important_labels):
                    obstacles.append(
                        DetectedObstacle(
                            BoundingBox2D(
                                int(res_boxes[i][1] *
                                    msg.frame.camera_setup.width),
                                int(res_boxes[i][3] *
                                    msg.frame.camera_setup.width),
                                int(res_boxes[i][0] *
                                    msg.frame.camera_setup.height),
                                int(res_boxes[i][2] *
                                    msg.frame.camera_setup.height)),
                            res_scores[i], self._coco_labels[res_classes[i]]))
            else:
                self._logger.warning('Filtering unknown class: {}'.format(
                    res_classes[i]))

        self._logger.debug('@{}: {} obstacles: {}'.format(
            msg.timestamp, self.config.name, obstacles))

        if (self._flags.visualize_detected_obstacles
                or self._flags.log_detector_output):
            msg.frame.annotate_with_bounding_boxes(msg.timestamp, obstacles,
                                                   self._bbox_colors)
            if self._flags.visualize_detected_obstacles:
                msg.frame.visualize(self.config.name)
            if self._flags.log_detector_output:
                msg.frame.save(msg.timestamp.coordinates[0],
                               self._flags.data_path,
                               'detector-{}'.format(self.config.name))

        # Get runtime in ms.
        runtime = (time.time() - start_time) * 1000
        # Send out obstacles.
        obstacles_stream.send(
            ObstaclesMessage(msg.timestamp, obstacles, runtime))
