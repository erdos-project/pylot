from collections import deque
import erdust
import time

from pylot.utils import time_epoch_ms


class SegmentationDecayOperator(erdust.Operator):
    """ Computes how much segmentation accuracy decreases over time.

    The operator subscribes to the Carla perfect segmented frames stream.
    """
    def __init__(self,
                 ground_segmented_stream,
                 iou_stream,
                 name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        ground_segmented_stream.add_callback(
            self.on_ground_segmented_frame, [iou_stream])
        self._name = name
        self._flags = flags
        self._logger = erdust.utils.setup_logging(name, log_file_name)
        self._csv_logger = erdust.utils.setup_csv_logging(
            name + '-csv', csv_file_name)
        self._ground_frames = deque()

    @staticmethod
    def connect(ground_segmented_stream):
        iou_stream = erdust.WriteStream()
        return [iou_stream]

    def on_ground_segmented_frame(self, msg, iou_stream):
        start_time = time.time()
        # We don't fully transform it to cityscapes palette to avoid
        # introducing extra latency.
        frame = msg.frame

        if len(self._ground_frames) > 0:
            # Pop the oldest frame if it's older than the max latency
            # we're interested in.
            if (msg.timestamp.coordinates[0] - self._ground_frames[0][0] >
                self._flags.decay_max_latency):
                self._ground_frames.popleft()

            cur_time = time_epoch_ms()
            for timestamp, ground_frame in self._ground_frames:
                (mean_iou, class_iou) = \
                    ground_frame.compute_semantic_iou_using_masks(frame)
                time_diff = msg.timestamp.coordinates[0] - timestamp
                self._logger.info(
                    'Segmentation ground latency {} ; mean IoU {}'.format(
                        time_diff, mean_iou))
                self._csv_logger.info('{},{},mIoU,{},{}'.format(
                    cur_time, self._name, time_diff, mean_iou))
                iou_stream.send(erdust.Message(msg.timestamp,
                                               (time_diff, mean_iou)))
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
        self._ground_frames.append((msg.timestamp.coordinates[0], frame))

        runtime = (time.time() - start_time) * 1000
        self._logger.info('Segmentation eval ground runtime {}'.format(
                runtime))
