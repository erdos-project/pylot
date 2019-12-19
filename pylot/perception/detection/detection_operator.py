from absl import flags
import erdust
import numpy as np
import tensorflow as tf
import time

from pylot.perception.detection.utils import DetectedObject,\
    load_coco_labels, load_coco_bbox_colors, annotate_image_with_bboxes,\
    save_image, visualize_image
from pylot.perception.messages import DetectorMessage
from pylot.utils import bgr_to_rgb, time_epoch_ms

flags.DEFINE_float(
    'obj_detection_gpu_memory_fraction',
    0.3,
    'GPU memory fraction allocated to each obj detector operator')
flags.DEFINE_float('detector_min_score_threshold', 0.5,
                   'Min score threshold for bounding box')
flags.DEFINE_string('path_coco_labels', 'dependencies/models/coco.names',
                    'Path to the COCO labels')


class DetectionOperator(erdust.Operator):
    """ Subscribes to a camera stream, and runs a model for each frame."""
    def __init__(self,
                 camera_stream,
                 obstacles_stream,
                 name,
                 model_path,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        camera_stream.add_callback(self.on_msg_camera_stream,
                                   [obstacles_stream])
        self._name = name
        self._flags = flags
        self._logger = erdust.utils.setup_logging(name, log_file_name)
        self._csv_logger = erdust.utils.setup_csv_logging(
            name + '-csv', csv_file_name)
        self._detection_graph = tf.Graph()
        # Load the model from the model file.
        with self._detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self._gpu_options = tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=flags.obj_detection_gpu_memory_fraction)
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
        self._coco_labels = load_coco_labels(self._flags.path_coco_labels)
        self._bbox_colors = load_coco_bbox_colors(self._coco_labels)

    @staticmethod
    def connect(camera_stream):
        obstacles_stream = erdust.WriteStream()
        return [obstacles_stream]

    def on_msg_camera_stream(self, msg, obstacles_stream):
        """ Invoked when the operator receives a message on the data stream."""
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self._name))
        start_time = time.time()
        # The models expect BGR images.
        assert msg.encoding == 'BGR', 'Expects BGR frames'
        image_np = msg.frame
        # Expand dimensions since the model expects images to have
        # shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        (boxes, scores, classes, num_detections) = self._tf_session.run(
            [
                self._detection_boxes, self._detection_scores,
                self._detection_classes, self._num_detections
            ],
            feed_dict={self._image_tensor: image_np_expanded})

        num_detections = int(num_detections[0])
        res_classes = classes[0][:num_detections]
        res_boxes = boxes[0][:num_detections]
        res_scores = scores[0][:num_detections]

        # TODO(ionel): BIG HACK TO FILTER OUT UNKNOWN CLASSES!
        boxes = []
        scores = []
        labels = []
        for i in range(0, num_detections):
            if res_classes[i] in self._coco_labels:
                labels.append(self._coco_labels[res_classes[i]])
                boxes.append(res_boxes[i])
                scores.append(res_scores[i])

        detected_objects = self.__convert_to_detected_objs(
            boxes, scores, labels, msg.height, msg.width)
        self._logger.debug('@{}: {} detected objects: {}'.format(
            msg.timestamp, self._name, detected_objects))

        if (self._flags.visualize_detector_output or
            self._flags.log_detector_output):
            annotate_image_with_bboxes(
                msg.timestamp, image_np, detected_objects, self._bbox_colors)
            if self._flags.visualize_detector_output:
                visualize_image(self._name, image_np)
            if self._flags.log_detector_output:
                save_image(bgr_to_rgb(image_np),
                           msg.timestamp,
                           self._flags.data_path,
                           'detector-{}'.format(self._name))

        # Get runtime in ms.
        runtime = (time.time() - start_time) * 1000
        self._csv_logger.info('{},{},"{}",{}'.format(
            time_epoch_ms(), self._name, msg.timestamp, runtime))
        # Send out obstacles.
        obstacles_stream.send(
            DetectorMessage(detected_objects, runtime, msg.timestamp))

    def __convert_to_detected_objs(self, boxes, scores, labels, height, width):
        index = 0
        detected_objects = []
        while index < len(boxes) and index < len(scores):
            if scores[index] >= self._flags.detector_min_score_threshold:
                ymin = int(boxes[index][0] * height)
                xmin = int(boxes[index][1] * width)
                ymax = int(boxes[index][2] * height)
                xmax = int(boxes[index][3] * width)
                corners = (xmin, xmax, ymin, ymax)
                detected_objects.append(
                    DetectedObject(corners, scores[index], labels[index]))
            index += 1
        return detected_objects
