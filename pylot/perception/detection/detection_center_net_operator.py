import cv2
import time

from erdos.op import Op
from erdos.utils import setup_csv_logging, setup_logging, time_epoch_ms

from pylot.perception.detection.utils import DetectedObject, load_coco_labels, load_coco_bbox_colors, visualize_bboxes
from pylot.perception.messages import DetectorMessage
from pylot.utils import create_obstacles_stream, is_camera_stream

from detectors.detector_factory import detector_factory
from opts import opts


class DetectionCenterNetOperator(Op):
    def __init__(self,
                 name,
                 output_stream_name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        super(DetectionCenterNetOperator, self).__init__(name)
        self._flags = flags
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._output_stream_name = output_stream_name
        self._opt = opts().init()
        self._opt.load_model = self._flags.detector_center_net_model_path
        Detector = detector_factory[self._opt.task]
        self._detector = Detector(self._opt)
        self._coco_labels = load_coco_labels(self._flags.path_coco_labels)
        self._bbox_colors = load_coco_bbox_colors(self._coco_labels)

    @staticmethod
    def setup_streams(input_streams, output_stream_name):
        input_streams.filter(is_camera_stream).add_callback(
            DetectionCenterNetOperator.on_msg_camera_stream)
        return [create_obstacles_stream(output_stream_name)]

    def on_msg_camera_stream(self, msg):
        self._logger.info('{} received frame {}'.format(self.name, msg.timestamp))
        start_time = time.time()
        assert msg.encoding == 'BGR', 'Expects BGR frames'
        image_np = msg.frame
        results = self._detector.run(image_np)
        detected_objs = self.__get_output_bboxes(results['results'])
        if self._flags.visualize_detector_output:
            visualize_bboxes(self.name, msg.timestamp, image_np, detected_objs,
                             self._bbox_colors)

        # Get runtime in ms.
        runtime = (time.time() - start_time) * 1000
        self._csv_logger.info('{},{},"{}",{}'.format(
            time_epoch_ms(), self.name, msg.timestamp, runtime))
        output_msg = DetectorMessage(detected_objs, runtime, msg.timestamp)
        self.get_output_stream(self._output_stream_name).send(output_msg)

    def execute(self):
        self.spin()

    def __get_output_bboxes(self, results):
        """ Transform from CenterNet output format to our format."""
        output = []
        for category in range(1, self._opt.num_classes + 1):
            for bbox in results[category]:
                confidence = bbox[4]
                if confidence >= self._flags.detector_min_score_threshold:
                    corners = (int(bbox[0]), int(bbox[2]), int(bbox[1]), int(bbox[3]))
                    output.append(DetectedObject(corners, confidence, self._coco_labels[category]))
        return output
