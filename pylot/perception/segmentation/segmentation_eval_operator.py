"""Implements an operator that eveluates segmentation output."""

import heapq

import erdos

from pylot.utils import time_epoch_ms


class SegmentationEvalOperator(erdos.Operator):
    """Operator that computes accuracy metrics using segmented frames.

    Args:
        ground_segmented_stream (:py:class:`erdos.ReadStream`): Stream on
            which perfectly segmented
            :py:class:`~pylot.perception.messages.SegmentedFrameMessage` are
            received.
        segmented_stream (:py:class:`erdos.ReadStream`): Stream on which
            segmented
            :py:class:`~pylot.perception.messages.SegmentedFrameMessage` are
            received.
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, ground_segmented_stream, segmented_stream, flags):
        ground_segmented_stream.add_callback(self.on_ground_segmented_frame)
        segmented_stream.add_callback(self.on_segmented_frame)
        # Register a watermark callback.
        erdos.add_watermark_callback(
            [ground_segmented_stream, segmented_stream], [], self.on_watermark)
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

    @staticmethod
    def connect(ground_segmented_stream, segmented_stream):
        """Connects the operator to other streams.

        Args:
            ground_segmented_stream (:py:class:`erdos.ReadStream`): Stream on
                 which perfectly segmented
                 :py:class:`~pylot.perception.messages.SegmentedFrameMessage`
                 are received.
            segmented_stream (:py:class:`erdos.ReadStream`): Stream on which
                segmented
                :py:class:`~pylot.perception.messages.SegmentedFrameMessage`
                are received.
        """
        return []

    def destroy(self):
        self._logger.warn('destroying {}'.format(self.config.name))

    def on_watermark(self, timestamp):
        """Invoked when all input streams have received a watermark.

        Args:
            timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of
                the watermark.
        """
        if timestamp.is_top:
            return
        assert len(timestamp.coordinates) == 1
        if not self._last_notification:
            self._last_notification = timestamp.coordinates[0]
            return
        else:
            self._sim_interval = timestamp.coordinates[
                0] - self._last_notification
            self._last_notification = timestamp.coordinates[0]

        game_time = timestamp.coordinates[0]
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
                self.__compute_mean_iou(timestamp, end_frame, start_frame)
            else:
                # The remaining entries are newer ground segmentated frames.
                break

        self.__garbage_collect_segmentation()

    def on_ground_segmented_frame(self, msg):
        # Buffer the ground truth frames.
        game_time = msg.timestamp.coordinates[0]
        self._ground_frames.append((game_time, msg.frame))

    def on_segmented_frame(self, msg):
        game_time = msg.timestamp.coordinates[0]
        self._segmented_frames.append((game_time, msg.frame))
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
                                                               msg.runtime)
            # Round time to nearest frame.
            heapq.heappush(self._segmented_start_end_times,
                           (segmented_time, game_time))
        else:
            self._logger.fatal('Unexpected segmentation metric {}'.format(
                self._flags.segmentation_metric))

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
