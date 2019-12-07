import erdust
import heapq

from pylot.utils import time_epoch_ms


class SegmentationEvalOperator(erdust.Operator):

    def __init__(self,
                 ground_segmented_stream,
                 segmented_stream,
                 name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        ground_segmented_stream.add_callback(self.on_ground_segmented_frame)
        segmented_stream.add_callback(self.on_segmented_frame)
        # Register a watermark callback.
        erdust.add_watermark_callback(self.on_notification)
        self._name = name
        self._flags = flags
        self._logger = erdust.utils.setup_logging(name, log_file_name)
        self._csv_logger = erdust.utils.setup_csv_logging(
            name + '-csv', csv_file_name)
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
        return []

    def on_notification(self, timestamp):
        if not self._last_notification:
            self._last_notification = timestamp.coordinates[0]
            return
        else:
            self._sim_interval = timestamp.coordinates[0] - self._last_notification
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
                self._logger.info('Computing for times {} {}'.format(
                    start_time, end_time))
                if self._flags.segmentation_eval_use_accuracy_model:
                    # Not using the segmentation output => get ground
                    # segmentation.
                    start_frame = self.__get_ground_segmentation_at(start_time)
                    self.__compute_mean_iou(end_frame, start_frame)
                else:
                    start_frame = self.__get_segmented_at(start_time)
                    self.__compute_mean_iou(end_frame, start_frame)
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
            segmented_time = game_time + msg.runtime
            if self._flags.segmentation_eval_use_accuracy_model:
                # Include the decay of segmentation with time if we do not
                # want to use the accuracy of our models.
                # TODO(ionel): We must pass model mIoU to this method.
                segmented_time += self.__mean_iou_to_latency(1)
            segmented_time = self.__compute_closest_frame_time(segmented_time)
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

    def __compute_mean_iou(self, ground_frame, segmented_frame):
        (mean_iou, class_iou) = ground_frame.compute_semantic_iou(
            segmented_frame)
        self._logger.info('IoU class scores: {}'.format(class_iou))
        self._logger.info('mean IoU score: {}'.format(mean_iou))
        self._csv_logger.info('{},{},{},{}'.format(
            time_epoch_ms(), self._name, self._flags.segmentation_metric,
            mean_iou))

    def __mean_iou_to_latency(self, mean_iou):
        """ Function that gives a latency estimate of how much
        simulation time must pass such that a 1.0 IoU decays to mean_iou.
        """
        # TODO(ionel): Implement!
        return 0

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
        while (index < len(self._segmented_frames) and
               self._segmented_frames[index][0] < watermark):
            index += 1
        if index > 0:
            self._segmented_frames = self._segmented_frames[index:]
        # Remove all the ground segmentations that are below the watermark.
        index = 0
        while (index < len(self._ground_frames) and
               self._ground_frames[index][0] < watermark):
            index += 1
        if index > 0:
            self._ground_frames = self._ground_frames[index:]
