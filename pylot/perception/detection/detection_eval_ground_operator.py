import cv2
import time

from collections import deque

from erdos.op import Op
from erdos.utils import setup_csv_logging, setup_logging, time_epoch_ms

from pylot.perception.detection.utils import get_precision_recall_at_iou
import pylot.utils
from pylot.simulation.utils import get_2d_bbox_from_3d_box, get_camera_intrinsic_and_transform


class DetectionEvalGroundOperator(Op):
    def __init__(self,
                 name,
                 rgb_camera_setup,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        super(DetectionEvalGroundOperator, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._flags = flags
        # Queue of incoming data.
        self._bgr_imgs = deque()
        self._vehicle_transforms = deque()
        self._depth_imgs = deque()
        self._pedestrians = deque()
        self._ground_bboxes = deque()
        (camera_name, _, img_size, pos) = rgb_camera_setup
        (self._rgb_intrinsic, self._rgb_transform,
         self._rgb_img_size) = get_camera_intrinsic_and_transform(
             image_size=img_size,
             position=pos)
        self._iou_thresholds = [0.1 * i for i in range(1, 10)]

    @staticmethod
    def setup_streams(input_streams, depth_camera_name):
        input_streams.filter(pylot.utils.is_depth_camera_stream).filter_name(
            depth_camera_name).add_callback(
                DetectionEvalGroundOperator.on_depth_camera_update)
        input_streams.filter(pylot.utils.is_camera_stream).add_callback(
            DetectionEvalGroundOperator.on_bgr_camera_update)
        input_streams.filter(pylot.utils.is_vehicle_transform_stream).add_callback(
            DetectionEvalGroundOperator.on_vehicle_transform_update)
        input_streams.filter(pylot.utils.is_ground_pedestrians_stream).add_callback(
            DetectionEvalGroundOperator.on_pedestrians_update)

        input_streams.add_completion_callback(
            DetectionEvalGroundOperator.on_notification)
        return []

    def on_notification(self, msg):
        # Ignore the first several seconds of the simulation because the car is
        # not moving at the beginning.
        if msg.timestamp.coordinates[
                0] <= self._flags.eval_ground_truth_ignore_first:
            return

        # Get the data from the start of all the queues and make sure that
        # we did not miss any data.
        depth_msg = self._depth_imgs.popleft()
        bgr_msg = self._bgr_imgs.popleft()
        vehicle_trans_msg = self._vehicle_transforms.popleft()
        pedestrians_msg = self._pedestrians.popleft()

        self._logger.info('Timestamps {} {} {} {}'.format(
            depth_msg.timestamp, bgr_msg.timestamp, vehicle_trans_msg.timestamp,
            pedestrians_msg.timestamp))

        assert (depth_msg.timestamp == bgr_msg.timestamp ==
                vehicle_trans_msg.timestamp == pedestrians_msg.timestamp)

        vehicle_transform = vehicle_trans_msg.data

        bboxes = []
        self._logger.info('Number of pedestrians {}'.format(
            len(pedestrians_msg.pedestrians)))
        for pedestrian in pedestrians_msg.pedestrians:
            bbox = get_2d_bbox_from_3d_box(depth_msg.frame, vehicle_transform,
                                           pedestrian.transform, pedestrian.bounding_box,
                                           self._rgb_transform,
                                           self._rgb_intrinsic,
                                           self._rgb_img_size, 1.5, 3.0)
            if bbox is not None:
                bboxes.append(bbox)

        # Remove the buffered bboxes that are too old.
        while (len(self._ground_bboxes) > 0 and msg.timestamp.coordinates[0] -
               self._ground_bboxes[0][0].coordinates[0] >
               self._flags.eval_ground_truth_max_latency):
            self._ground_bboxes.popleft()

        for (old_timestamp, old_bboxes) in self._ground_bboxes:
            # Ideally, you would like to take multiple precision values at different
            # recalls and average them, but we can't vary model confidence, so we just
            # return the actual precision.
            if (len(bboxes) > 0 or len(old_bboxes) > 0):
                latency = msg.timestamp.coordinates[0] - old_timestamp.coordinates[0]
                precisions = []
                for iou in self._iou_thresholds:
                    (precision, _) = get_precision_recall_at_iou(
                        bboxes, old_bboxes, iou)
                    precisions.append(precision)
                self._logger.info("Precision {}".format(precisions))
                avg_precision = float(sum(precisions)) / len(precisions)
                self._logger.info(
                    "The latency is {} and the average precision is {}".format(
                        latency, avg_precision))
                self._csv_logger.info('{},{},{},{}'.format(
                    time_epoch_ms(), self.name, latency, avg_precision))

        # Buffer the new bounding boxes.
        self._ground_bboxes.append((msg.timestamp, bboxes))

    def on_depth_camera_update(self, msg):
        if msg.timestamp.coordinates[
                0] > self._flags.eval_ground_truth_ignore_first:
            self._depth_imgs.append(msg)

    def on_bgr_camera_update(self, bgr_msg):
        if bgr_msg.timestamp.coordinates[
                0] > self._flags.eval_ground_truth_ignore_first:
            self._bgr_imgs.append(bgr_msg)

    def on_vehicle_transform_update(self, transform_msg):
        if transform_msg.timestamp.coordinates[
                0] > self._flags.eval_ground_truth_ignore_first:
            self._vehicle_transforms.append(transform_msg)

    def on_pedestrians_update(self, msg):
        if msg.timestamp.coordinates[
                0] > self._flags.eval_ground_truth_ignore_first:
            self._pedestrians.append(msg)
