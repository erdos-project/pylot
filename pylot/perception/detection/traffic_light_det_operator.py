import numpy as np
import tensorflow as tf
import time

from erdos.op import Op
from erdos.utils import setup_csv_logging, setup_logging, time_epoch_ms

from pylot.perception.detection.utils import DetectedObject, load_coco_labels, load_coco_bbox_colors, visualize_bboxes
from pylot.perception.messages import DetectorMessage
from pylot.utils import bgr_to_rgb, rgb_to_bgr, create_traffic_lights_stream, is_camera_stream


class TrafficLightDetOperator(Op):
    """ Subscribes to a camera stream, and runs a model for each frame."""
    def __init__(self,
                 name,
                 output_stream_name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        super(TrafficLightDetOperator, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._output_stream_name = output_stream_name
        self._flags = flags
        self._detection_graph = tf.Graph()
        # Load the model from the model file.
        with self._detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self._flags.traffic_light_det_model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self._gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=flags.traffic_light_det_gpu_memory_fraction)
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
            1: 'Green',
            2: 'Red',
            3: 'Yellow',
            4: 'Off'
        }
        # The bounding box colors to use in the visualizer.
        self._bbox_colors = {'Green': [0, 128, 0],
                             'Red': [255, 0, 0],
                             'Yellow': [255, 255, 0],
                             'Off': [0, 0, 0]}

    @staticmethod
    def setup_streams(input_streams,
                      output_stream_name,
                      camera_stream_name=None):
        # Select camera input streams.
        camera_streams = input_streams.filter(is_camera_stream)
        if camera_stream_name:
            # Select only the camera the operator is interested in.
            camera_streams = camera_streams.filter_name(camera_stream_name)
        # Register a callback on the camera input stream.
        camera_streams.add_callback(TrafficLightDetOperator.on_frame)
        return [create_traffic_lights_stream(output_stream_name)]

    def on_frame(self, msg):
        """ Invoked when the operator receives a message on the data stream."""
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

        self._logger.info('Traffic light boxes {}'.format(boxes))
        self._logger.info('Traffic light scores {}'.format(scores))
        self._logger.info('Traffic light labels {}'.format(labels))

        traffic_lights = self.__convert_to_detected_tl(
            boxes, scores, labels, msg.height, msg.width)

        if self._flags.visualize_traffic_light_output:
            visualize_bboxes(self.name, msg.timestamp, rgb_to_bgr(image_np),
                             traffic_lights, self._bbox_colors)

        # Get runtime in ms.
        runtime = (time.time() - start_time) * 1000
        self._csv_logger.info('{},{},"{}",{}'.format(
            time_epoch_ms(), self.name, msg.timestamp, runtime))

        output_msg = DetectorMessage(traffic_lights, runtime, msg.timestamp)
        self.get_output_stream(self._output_stream_name).send(output_msg)

    def execute(self):
        self.spin()

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
