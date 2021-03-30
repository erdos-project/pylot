"""Implements an operator that detects traffic lights."""
import logging

import erdos

import numpy as np

import pylot.utils
from pylot.perception.detection.traffic_light import TrafficLight, \
    TrafficLightColor
from pylot.perception.detection.utils import BoundingBox2D
from pylot.perception.messages import TrafficLightsMessage

import tensorflow as tf


class TrafficLightDetOperator(erdos.Operator):
    """Detects traffic lights using a TensorFlow model.

    The operator receives frames on a camera stream, and runs a model for each
    frame.

    Args:
        camera_stream (:py:class:`erdos.ReadStream`): The stream on which
            camera frames are received.
        traffic_lights_stream (:py:class:`erdos.WriteStream`): Stream on which
            the operator sends
            :py:class:`~pylot.perception.messages.TrafficLightsMessage`
            messages.
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, camera_stream: erdos.ReadStream,
                 time_to_decision_stream: erdos.ReadStream,
                 traffic_lights_stream: erdos.WriteStream, flags):
        # Register a callback on the camera input stream.
        camera_stream.add_callback(self.on_frame, [traffic_lights_stream])
        time_to_decision_stream.add_callback(self.on_time_to_decision_update)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags
        self._traffic_lights_stream = traffic_lights_stream
        self._detection_graph = tf.Graph()
        # Load the model from the model file.
        pylot.utils.set_tf_loglevel(logging.ERROR)
        with self._detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self._flags.traffic_light_det_model_path,
                                'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self._gpu_options = tf.GPUOptions(
            allow_growth=True,
            visible_device_list=str(self._flags.traffic_light_det_gpu_index),
            per_process_gpu_memory_fraction=flags.
            traffic_light_det_gpu_memory_fraction)
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
        self._labels = {
            1: TrafficLightColor.GREEN,
            2: TrafficLightColor.YELLOW,
            3: TrafficLightColor.RED,
            4: TrafficLightColor.OFF
        }
        # Serve some junk image to load up the model.
        self.__run_model(np.zeros((108, 192, 3)))

    @staticmethod
    def connect(camera_stream: erdos.ReadStream,
                time_to_decision_stream: erdos.ReadStream):
        """Connects the operator to other streams.

        Args:
            camera_stream (:py:class:`erdos.ReadStream`): The stream on which
                camera frames are received.

        Returns:
            :py:class:`erdos.WriteStream`: Stream on which the operator sends
            :py:class:`~pylot.perception.messages.TrafficLightsMessage`
            messages for traffic lights.
        """
        traffic_lights_stream = erdos.WriteStream()
        return [traffic_lights_stream]

    def destroy(self):
        self._logger.warn('destroying {}'.format(self.config.name))
        self._traffic_lights_stream.send(
            erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))

    def on_time_to_decision_update(self, msg: erdos.Message):
        self._logger.debug('@{}: {} received ttd update {}'.format(
            msg.timestamp, self.config.name, msg))

    @erdos.profile_method()
    def on_frame(self, msg: erdos.Message,
                 traffic_lights_stream: erdos.WriteStream):
        """Invoked whenever a frame message is received on the stream.

        Args:
            msg: A :py:class:`~pylot.perception.messages.FrameMessage`.
            obstacles_stream (:py:class:`erdos.WriteStream`): Stream on which
                the operator sends
                :py:class:`~pylot.perception.messages.TrafficLightsMessage`
                messages for traffic lights.
        """
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self.config.name))
        assert msg.frame.encoding == 'BGR', 'Expects BGR frames'
        boxes, scores, labels = self.__run_model(
            msg.frame.as_rgb_numpy_array())

        traffic_lights = self.__convert_to_detected_tl(
            boxes, scores, labels, msg.frame.camera_setup.height,
            msg.frame.camera_setup.width)

        self._logger.debug('@{}: {} detected traffic lights {}'.format(
            msg.timestamp, self.config.name, traffic_lights))

        traffic_lights_stream.send(
            TrafficLightsMessage(msg.timestamp, traffic_lights))
        traffic_lights_stream.send(erdos.WatermarkMessage(msg.timestamp))

        if self._flags.log_traffic_light_detector_output:
            msg.frame.annotate_with_bounding_boxes(msg.timestamp,
                                                   traffic_lights)
            msg.frame.save(msg.timestamp.coordinates[0], self._flags.data_path,
                           'tl-detector-{}'.format(self.config.name))

    def __run_model(self, image_np):
        # Expand dimensions since the model expects images to have
        # shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        (boxes, scores, classes, num) = self._tf_session.run(
            [
                self._detection_boxes, self._detection_scores,
                self._detection_classes, self._num_detections
            ],
            feed_dict={self._image_tensor: image_np_expanded})

        num_detections = int(num[0])
        labels = [self._labels[label] for label in classes[0][:num_detections]]
        boxes = boxes[0][:num_detections]
        scores = scores[0][:num_detections]
        return boxes, scores, labels

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
                                 bounding_box=bbox))
        return traffic_lights
