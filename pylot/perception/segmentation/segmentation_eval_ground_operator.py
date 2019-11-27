from collections import deque
import erdust
import time

from pylot.perception.segmentation.utils import generate_masks,\
    compute_semantic_iou_from_masks
from pylot.utils import time_epoch_ms


class SegmentationEvalGroundOperator(erdust.Operator):
    """ Computes how much segmentation accuracy decreases over time.

    The operator subscribes to the Carla perfect segmented frames stream.
    """
    def __init__(self,
                 ground_segmented_stream,
                 name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        ground_segmented_stream.add_callback(self.on_ground_segmented_frame)
        self._name = name
        self._flags = flags
        self._logger = erdust.setup_logging(name, log_file_name)
        self._csv_logger = erdust.setup_csv_logging(
            name + '-csv', csv_file_name)
        self._time_delta = None
        self._ground_masks = deque()

    @staticmethod
    def connect(ground_segmented_stream):
        return []

    def on_ground_segmented_frame(self, msg):
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
                    frame_masks, ground_masks)
                time_diff = msg.timestamp.coordinates[0] - timestamp
                self._logger.info(
                    'Segmentation ground latency {} ; mean IoU {}'.format(
                        time_diff, mean_iou))
                self._csv_logger.info('{},{},mIoU,{},{}'.format(
                    cur_time, self._name, time_diff, mean_iou))
                pedestrian_key = 4
                if pedestrian_key in class_iou:
                    self._logger.info(
                        'Segmentation ground latency {} ; pedestrian IoU {}'.format(
                            time_diff, class_iou[pedestrian_key]))
                    self._csv_logger.info('{},{},pedestrianIoU,{},{}'.format(
                        cur_time,
                        self._name,
                        time_diff,
                        class_iou[pedestrian_key]))

                vehicle_key = 10
                if vehicle_key in class_iou:
                    self._logger.info(
                        'Segmentation ground latency {} ; vehicle IoU {}'.format(
                            time_diff, class_iou[vehicle_key]))
                    self._csv_logger.info('{},{},vehicleIoU,{},{}'.format(
                        cur_time,
                        self._name,
                        time_diff,
                        class_iou[vehicle_key]))

        # Append the processed image to the buffer.
        self._ground_masks.append(
            (msg.timestamp.coordinates[0], frame_masks))

        runtime = (time.time() - start_time) * 1000
        self._logger.info('Segmentation eval ground runtime {}'.format(
                runtime))
