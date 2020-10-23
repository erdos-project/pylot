import time
from collections import deque

import erdos

from pylot.utils import time_epoch_ms


class SegmentationDecayOperator(erdos.Operator):
    """Computes how much segmentation accuracy decreases over time.

    The operator subscribes to the perfect segmented frames stream.

    Args:
        ground_segmented_stream (:py:class:`erdos.ReadStream`): Stream on
            which perfectly segmented
            :py:class:`~pylot.perception.messages.SegmentedFrameMessage` are
            received.
        iou_stream (:py:class:`erdos.WriteStream`): Stream on which the
            operator publishes IoU accuracy results.
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, ground_segmented_stream, iou_stream, flags):
        ground_segmented_stream.add_callback(self.on_ground_segmented_frame,
                                             [iou_stream])
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            self.config.name + '-csv', self.config.csv_log_file_name)
        self._ground_frames = deque()

    @staticmethod
    def connect(ground_segmented_stream):
        """Connects the operator to other streams.

        Args:
            ground_segmented_stream (:py:class:`erdos.ReadStream`): Stream on
                which perfectly segmented
                :py:class:`~pylot.perception.messages.SegmentedFrameMessage`
                are received.

        Returns:
            :py:class:`erdos.WriteStream`: Stream on which the operator
            publishes IoU accuracy results.
        """
        iou_stream = erdos.WriteStream()
        return [iou_stream]

    def on_ground_segmented_frame(self, msg, iou_stream):
        assert len(msg.timestamp.coordinates) == 1
        start_time = time.time()
        # We don't fully transform it to cityscapes palette to avoid
        # introducing extra latency.
        frame = msg.frame

        sim_time = msg.timestamp[0]
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
                self._csv_logger.info('{},{},{},mIoU,{},{:.4f}'.format(
                    cur_time, sim_time, self.config.name, time_diff, mean_iou))
                iou_stream.send(
                    erdos.Message(msg.timestamp, (time_diff, mean_iou)))
                person_key = 4
                if person_key in class_iou:
                    self._logger.info(
                        'Segmentation ground latency {} ; person IoU {}'.
                        format(time_diff, class_iou[person_key]))
                    self._csv_logger.info(
                        '{},{},{},personIoU,{},{:.4f}'.format(
                            cur_time, sim_time, self.config.name, time_diff,
                            class_iou[person_key]))

                vehicle_key = 10
                if vehicle_key in class_iou:
                    self._logger.info(
                        'Segmentation ground latency {} ; vehicle IoU {}'.
                        format(time_diff, class_iou[vehicle_key]))
                    self._csv_logger.info(
                        '{},{},{},vehicleIoU,{},{:.4f}'.format(
                            cur_time, sim_time, self.config.name, time_diff,
                            class_iou[vehicle_key]))

        # Append the processed image to the buffer.
        self._ground_frames.append((msg.timestamp.coordinates[0], frame))

        runtime = (time.time() - start_time) * 1000
        self._logger.info(
            'Segmentation eval ground runtime {}'.format(runtime))
