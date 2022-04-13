"""Implements an operator that eveluates segmentation output."""

import heapq

import erdos
from erdos.operator import TwoInOneOut
from erdos.context import TwoInOneOutContext

from pylot.perception.messages import SegmentedMessageTuple
from pylot.perception.segmentation.segmented_frame import SegmentedFrame
from pylot.utils import time_epoch_ms


class SegmentationEvalOperator(TwoInOneOut):
    """Operator that computes accuracy metrics using segmented frames.

    Args:
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, flags):
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            self.config.name + '-csv', self.config.csv_log_file_name)
        # Buffer of ground truth segmented frames.
        self._ground_frames = []
        # Buffer of segmentation output frames.
        self._segmented_frames = []
        # Heap storing pairs of (ground/output time, game time).
        self._segmented_start_end_times = []
        self._sim_interval = None
        self._last_notification = None

    def on_watermark(self, context: TwoInOneOutContext):
        """Invoked when all input streams have received a watermark."""
        if context.timestamp.is_top:
            return
        assert len(context.timestamp.coordinates) == 1
        if not self._last_notification:
            self._last_notification = context.timestamp.coordinates[0]
            return
        else:
            self._sim_interval = context.timestamp.coordinates[
                0] - self._last_notification
            self._last_notification = context.timestamp.coordinates[0]

        game_time = context.timestamp.coordinates[0]
        while len(self._segmented_start_end_times) > 0:
            (end_time, start_time) = self._segmented_start_end_times[0]
            # We can compute mIoU if the end time is not greater than the
            # ground time.
            if end_time <= game_time:
                # This is the closest ground segmentation to the end time.
                heapq.heappop(self._segmented_start_end_times)
                end_frame = self.__get_ground_segmentation_at(end_time)
                self._logger.debug('Computing for times {} {}'.format(
                    start_time, end_time))
                start_frame = self.__get_segmented_at(start_time)
                self.__compute_mean_iou(context.timestamp, end_frame,
                                        start_frame)
            else:
                # The remaining entries are newer ground segmentated frames.
                break

        self.__garbage_collect_segmentation()

    def on_left_data(self, context: TwoInOneOutContext, data: SegmentedFrame):
        """Invoked on each ground truth frame."""
        # Buffer the ground truth frames.
        game_time = context.timestamp.coordinates[0]
        self._ground_frames.append((game_time, data))

    def on_right_data(self, context: TwoInOneOutContext,
                      data: SegmentedMessageTuple):
        """Invoked on each segmented frame."""
        game_time = context.timestamp.coordinates[0]
        self._segmented_frames.append((game_time, data.frame))
        # Two metrics: 1) mIoU, and 2) timely-mIoU
        if self._flags.segmentation_metric == 'mIoU':
            # We will compare with segmented ground frame with the same game
            # time.
            heapq.heappush(self._segmented_start_end_times,
                           (game_time, game_time))
        elif self._flags.segmentation_metric == 'timely-mIoU':
            # Ground segmented frame time should be as close as possible to
            # the time game time + segmentation runtime.
            segmented_time = self.__compute_closest_frame_time(game_time +
                                                               data.runtime)
            # Round time to nearest frame.
            heapq.heappush(self._segmented_start_end_times,
                           (segmented_time, game_time))
        else:
            self._logger.fatal('Unexpected segmentation metric {}'.format(
                self._flags.segmentation_metric))

    def destroy(self):
        self._logger.warn('destroying {}'.format(self.config.name))

    def __compute_closest_frame_time(self, time):
        base = int(time) / self._sim_interval * self._sim_interval
        if time - base < self._sim_interval / 2:
            return base
        else:
            return base + self._sim_interval

    def __compute_mean_iou(self, timestamp, ground_frame, segmented_frame):
        ground_frame.transform_to_cityscapes()
        (mean_iou,
         class_iou) = ground_frame.compute_semantic_iou(segmented_frame)
        self._logger.info('IoU class scores: {}'.format(class_iou))
        self._logger.info('mean IoU score: {}'.format(mean_iou))
        self._csv_logger.info('{},{},{},{},{:.4f}'.format(
            time_epoch_ms(), timestamp.coordinates[0], self.config.name,
            self._flags.segmentation_metric, mean_iou))

    def __get_ground_segmentation_at(self, timestamp):
        for (time, frame) in self._ground_frames:
            if time == timestamp:
                return frame
            elif time > timestamp:
                break
        self._logger.fatal(
            'Could not find ground segmentation for {}'.format(timestamp))

    def __get_segmented_at(self, timestamp):
        for (time, frame) in self._segmented_frames:
            if time == timestamp:
                return frame
            elif time > timestamp:
                break
        self._logger.fatal(
            'Could not find segmentaed frame for {}'.format(timestamp))

    def __garbage_collect_segmentation(self):
        # Get the minimum watermark.
        watermark = None
        for (_, start_time) in self._segmented_start_end_times:
            if watermark is None or start_time < watermark:
                watermark = start_time
        if watermark is None:
            return
        # Remove all segmentations that are below the watermark.
        index = 0
        while (index < len(self._segmented_frames)
               and self._segmented_frames[index][0] < watermark):
            index += 1
        if index > 0:
            self._segmented_frames = self._segmented_frames[index:]
        # Remove all the ground segmentations that are below the watermark.
        index = 0
        while (index < len(self._ground_frames)
               and self._ground_frames[index][0] < watermark):
            index += 1
        if index > 0:
            self._ground_frames = self._ground_frames[index:]
