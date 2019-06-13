from collections import deque
import time

from erdos.op import Op
from erdos.utils import setup_csv_logging, setup_logging, time_epoch_ms

from pylot.perception.segmentation.utils import tf_compute_semantic_iou, generate_masks, compute_semantic_iou_from_masks
from pylot.utils import is_segmented_camera_stream


class SegmentationEvalGroundOperator(Op):
    """ Computes how much segmentation accuracy decreases over time.

    The operator subscribes to the Carla perfect segmented frames stream.
    """
    def __init__(self, name, flags, log_file_name=None, csv_file_name=None):
        super(SegmentationEvalGroundOperator, self).__init__(name)
        self._flags = flags
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._time_delta = None
        self._ground_masks = deque()

    @staticmethod
    def setup_streams(input_streams, ground_stream_name):
        input_streams.filter(is_segmented_camera_stream) \
                     .filter_name(ground_stream_name).add_callback(
            SegmentationEvalGroundOperator.on_ground_segmented_frame)
        return []

    def on_ground_segmented_frame(self, msg):
        # We ignore the first several seconds of the simulation because the car
        # is not moving at beginning.
        if msg.timestamp.coordinates[0] > self._flags.eval_ground_truth_ignore_first:
            start_time = time.time()
            # We don't fully transform it to cityscapes palette to avoid
            # introducing extra latency.
            frame_masks = generate_masks(msg.frame)

            if len(self._ground_masks) > 0:
                if self._time_delta is None:
                    self._time_delta = (msg.timestamp.coordinates[0] -
                                        self._ground_masks[0][0])
                else:
                    # Check that no frames got dropped.
                    recv_time_delta = (msg.timestamp.coordinates[0] -
                                       self._ground_masks[-1][0])
                    assert self._time_delta == recv_time_delta

                # Pop the oldest frame if it's older than the max latency
                # we're interested in.
                if (msg.timestamp.coordinates[0] - self._ground_masks[0][0] >
                    self._flags.eval_ground_truth_max_latency):
                    self._ground_masks.popleft()

                cur_time = time_epoch_ms()
                for timestamp, ground_masks in self._ground_masks:
                    (mean_iou, class_iou) = compute_semantic_iou_from_masks(
                        ground_masks, frame_masks)
                    time_diff = msg.timestamp.coordinates[0] - timestamp
                    self._logger.info(
                        'Segmentation ground latency {} ; mean IoU {}'.format(
                            time_diff, mean_iou))
                    self._csv_logger.info('{},{},mIoU,{},{}'.format(
                        cur_time, self.name, time_diff, mean_iou))
                    pedestrian_key = 4
                    if pedestrian_key in class_iou:
                        self._logger.info(
                            'Segmentation ground latency {} ; pedestrian IoU {}'.format(
                                time_diff, class_iou[pedestrian_key]))
                        self._csv_logger.info('{},{},pedestrianIoU,{},{}'.format(
                            cur_time,
                            self.name,
                            time_diff,
                            class_iou[pedestrian_key]))

                    vehicle_key = 10
                    if vehicle_key in class_iou:
                        self._logger.info(
                            'Segmentation ground latency {} ; vehicle IoU {}'.format(
                                time_diff, class_iou[vehicle_key]))
                        self._csv_logger.info('{},{},vehicleIoU,{},{}'.format(
                            cur_time,
                            self.name,
                            time_diff,
                            class_iou[vehicle_key]))

            # Append the processed image to the buffer.
            self._ground_masks.append(
                (msg.timestamp.coordinates[0], frame_masks))

            runtime = (time.time() - start_time) * 1000
            self._logger.info('Segmentation eval ground runtime {}'.format(
                runtime))

    def execute(self):
        self.spin()
