from absl import flags
import erdust
import numpy as np
import tensorflow as tf
import time

from pylot.perception.detection.utils import DetectedObject,\
    TrafficLightColor, annotate_image_with_bboxes, save_image, visualize_image
from pylot.perception.messages import DetectorMessage
from pylot.utils import bgr_to_rgb, rgb_to_bgr, time_epoch_ms

flags.DEFINE_string(
    'traffic_light_det_model_path',
    'dependencies/models/traffic_light_det_inference_graph.pb',
    'Path to the traffic light model protobuf')
flags.DEFINE_float('traffic_light_det_min_score_threshold', 0.3,
                   'Min score threshold for bounding box')
flags.DEFINE_float(
    'traffic_light_det_gpu_memory_fraction',
    0.3,
    'GPU memory fraction allocated to each traffic light detector')


class TrafficLightDetOperator(erdust.Operator):
    """ Subscribes to a camera stream, and runs a model for each frame."""
    def __init__(self,
                 camera_stream,
                 traffic_lights_stream,
                 name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        # Register a callback on the camera input stream.
        camera_stream.add_callback(self.on_frame, [traffic_lights_stream])
        self._name = name
        self._logger = erdust.utils.setup_logging(name, log_file_name)
        self._csv_logger = erdust.utils.setup_csv_logging(
            name + '-csv', csv_file_name)
        self._flags = flags
        self._detection_graph = tf.Graph()
        # Load the model from the model file.
        with self._detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(
                    self._flags.traffic_light_det_model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self._gpu_options = tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=flags.traffic_light_det_gpu_memory_fraction)
        # Create a TensorFlow session.
        self._tf_session = tf.compat.v1.Session(
            graph=self._detection_graph,
            config=tf.compat.v1.ConfigProto(gpu_options=self._gpu_options))
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
            2: TrafficLightColor.RED,
            3: TrafficLightColor.YELLOW,
            4: TrafficLightColor.OFF
        }
        # The bounding box colors to use in the visualizer.
        self._bbox_colors = {TrafficLightColor.GREEN: [0, 128, 0],
                             TrafficLightColor.RED: [255, 0, 0],
                             TrafficLightColor.YELLOW: [255, 255, 0],
                             TrafficLightColor.OFF: [0, 0, 0]}

    @staticmethod
    def connect(camera_stream):
        traffic_lights_stream = erdust.WriteStream()
        return [traffic_lights_stream]

    def on_frame(self, msg, traffic_lights_stream):
        """ Invoked when the operator receives a message on the data stream."""
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self._name))
        start_time = time.time()
        assert msg.encoding == 'BGR', 'Expects BGR frames'
        image_np = bgr_to_rgb(msg.frame)
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
        labels = [self._labels[label]
                  for label in classes[0][:num_detections]]
        boxes = boxes[0][:num_detections]
        scores = scores[0][:num_detections]

        traffic_lights = self.__convert_to_detected_tl(
            boxes, scores, labels, msg.height, msg.width)

        self._logger.debug('@{}: {} detected traffic lights {}'.format(
            msg.timestamp, self._name, traffic_lights))

        if (self._flags.visualize_traffic_light_output or
            self._flags.log_traffic_light_detector_output):
            annotate_image_with_bboxes(msg.timestamp,
                                       rgb_to_bgr(image_np),
                                       traffic_lights,
                                       self._bbox_colors)
            if self._flags.visualize_traffic_light_output:
                visualize_image(self._name, rgb_to_bgr(image_np))
            if self._flags.log_traffic_light_detector_output:
                save_image(image_np,
                           msg.timestamp,
                           self._flags.data_path,
                           'tl-detector-{}'.format(self._name))

        # Get runtime in ms.
        runtime = (time.time() - start_time) * 1000
        self._csv_logger.info('{},{},"{}",{}'.format(
            time_epoch_ms(), self._name, msg.timestamp, runtime))

        traffic_lights_stream.send(
            DetectorMessage(traffic_lights, runtime, msg.timestamp))

    def __convert_to_detected_tl(self, boxes, scores, labels, height, width):
        traffic_lights = []
        index = 0
        while index < len(boxes) and index < len(scores):
            if scores[index] > self._flags.traffic_light_det_min_score_threshold:
                ymin = int(boxes[index][0] * height)
                xmin = int(boxes[index][1] * width)
                ymax = int(boxes[index][2] * height)
                xmax = int(boxes[index][3] * width)
                corners = (xmin, xmax, ymin, ymax)
                traffic_lights.append(
                    DetectedObject(corners, scores[index], labels[index]))
            index += 1
        return traffic_lights
