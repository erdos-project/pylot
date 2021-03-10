"""Implements an operator that eveluates tracking output."""
import math

import erdos

import motmetrics as mm

import numpy as np

from pylot.utils import time_epoch_ms


class TrackingEvalOperator(erdos.Operator):
    """Operator that computes accuracy metrics using tracked obstacles.

    Args:
        obstacle_tracking_stream (:py:class:`erdos.ReadStream`): The stream on
            which tracked obstacles are received.
        ground_obstacles_stream: The stream on which
            :py:class:`~pylot.perception.messages.ObstaclesMessage` are
            received from the simulator.
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, obstacle_tracking_stream, ground_obstacles_stream,
                 finished_indicator_stream, evaluate_timely, frame_gap, flags):
        obstacle_tracking_stream.add_callback(self.on_tracker_obstacles)
        ground_obstacles_stream.add_callback(self.on_ground_obstacles)
        erdos.add_watermark_callback(
            [obstacle_tracking_stream, ground_obstacles_stream],
            [finished_indicator_stream], self.on_watermark)
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            self.config.name + '-csv', self.config.csv_log_file_name)
        self._last_notification = None
        # Buffer of tracked obstacles.
        self._tracked_obstacles = []
        # Buffer of ground obstacles.
        self._ground_obstacles = []
        # Storing pairs of (game time, ground/output time).
        self._tracker_start_end_times = []
        self._frame_gap = frame_gap
        self._start_anchored_accumulator = mm.MOTAccumulator(auto_id=True)
        self._end_anchored_accumulator = mm.MOTAccumulator(auto_id=True)
        self._metrics_host = mm.metrics.create()
        self._evaluate_timely = evaluate_timely
        # The start time of the most recent inference that completed before
        # the timestamp of the watermark.
        self._start_time_best_inference = None
        # Index in tracker_start_end_times to the inference with the next
        # unprocessed start time. We need to maintain this index because
        # the start_tracker_end_times list might contain entries with
        # start time beyond current watermark.
        self._start_time_frontier = 0
        # Buffer storing start times and ground times of predictions we have
        # to compute accuracy for. The buffer is used to ensure that these
        # accuracies are computed only once the ground data is available.
        self._accuracy_compute_buffer = []

    @staticmethod
    def connect(obstacle_tracking_stream, ground_obstacles_stream):
        """Connects the operator to other streams.

        Args:
            obstacle_tracking_stream (:py:class:`erdos.ReadStream`): The stream
                on which obstacle tracks are received from object trackers.
            ground_obstacles_stream: The stream on which
                :py:class:`~pylot.perception.messages.ObstaclesMessage` are
                received from the simulator.
        """
        finished_indicator_stream = erdos.WriteStream()
        return [finished_indicator_stream]

    def on_watermark(self, timestamp, finished_indicator_stream):
        """Invoked when all input streams have received a watermark.

        Args:
            timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of
                the watermark.
        """
        if timestamp.is_top:
            return
        assert len(timestamp.coordinates) == 1
        game_time = timestamp.coordinates[0]
        (st, et) = self._tracker_start_end_times[self._start_time_frontier]
        assert st == game_time, 'Incorrect frontier'
        self._accuracy_compute_buffer.append((st, et, False))
        index = self._start_time_frontier
        while index >= 0:
            (p_start_time, p_end_time) = self._tracker_start_end_times[index]
            if p_end_time == st:
                # This is the result that arrived before start_time, and
                # uses the most up-to-date sensor data (tracker_start_end_times
                # is sorted by start_times).
                self._start_time_best_inference = p_start_time
                self.__gc_obstacles_earlier_than(p_start_time)
                self._tracker_start_end_times = \
                    self._tracker_start_end_times[index:]
                self._start_time_frontier -= index
                break
            index -= 1
        if self._start_time_best_inference:
            # Compute accuracy if we have a prediction with end_time less than
            # current frontier.
            self._accuracy_compute_buffer.append(
                (self._start_time_best_inference, st, True))
        self.__drain_accuracy_compute_buffer(st)
        # Increase the frontier to process the next start time.
        self._start_time_frontier += 1

    def __compute_frame_gap(self, game_time):
        """Infer frame gap if not explicitly provided in constructor."""
        if not self._frame_gap:
            if not self._last_notification:
                self._last_notification = game_time
                return
            else:
                self._frame_gap = (game_time - self._last_notification)
                self._last_notification = game_time

    def __drain_accuracy_compute_buffer(self, up_to_time):
        for (st, et, end_anchored) in self._accuracy_compute_buffer:
            if et <= up_to_time:
                self.compute_accuracy(st, et, end_anchored)
        self._accuracy_compute_buffer = list(
            filter(lambda args: args[1] > up_to_time,
                   self._accuracy_compute_buffer))

    def compute_accuracy(self, frame_time, ground_time, end_anchored):
        tracker_obstacles = self.__get_tracked_obstacles_at(frame_time)
        ground_obstacles = self.__get_ground_obstacles_at(ground_time)
        if end_anchored:
            metrics_summary_df = self.__get_tracker_metrics(
                tracker_obstacles, ground_obstacles,
                self._end_anchored_accumulator)
            self.__write_metrics_to_csv(metrics_summary_df, "end", ground_time)
        else:
            metrics_summary_df = self.__get_tracker_metrics(
                tracker_obstacles, ground_obstacles,
                self._start_anchored_accumulator)
            self.__write_metrics_to_csv(metrics_summary_df, "start",
                                        frame_time)

    def __write_metrics_to_csv(self, metrics_summary_df, anchor_type,
                               anchor_time):
        for metric_name in self._flags.tracking_metrics:
            if metric_name in metrics_summary_df.columns:
                if (metric_name == 'mostly_tracked'
                        or metric_name == 'mostly_lost'
                        or metric_name == 'partially_tracked'):
                    ratio = metrics_summary_df[metric_name].values[
                        0] / metrics_summary_df['num_unique_objects'].values[0]
                    self._csv_logger.info("{},{},{},{},{},{:.4f}".format(
                        time_epoch_ms(), anchor_type, anchor_time,
                        self.config.name, 'ratio_' + metric_name, ratio))
                elif metric_name == 'motp':
                    # See https://github.com/cheind/py-motmetrics/issues/92
                    motp = (1 -
                            metrics_summary_df[metric_name].values[0]) * 100
                    self._csv_logger.info('{},{},{},{},{},{:.4f}'.format(
                        time_epoch_ms(), anchor_type, anchor_time,
                        self.config.name, metric_name, motp))
                elif (metric_name == 'idf1' or metric_name == 'mota'):
                    metric_val = \
                        metrics_summary_df[metric_name].values[0] * 100
                    self._csv_logger.info('{},{},{},{},{},{:.4f}'.format(
                        time_epoch_ms(), anchor_type, anchor_time,
                        self.config.name, metric_name, metric_val))
                else:
                    self._csv_logger.info('{},{},{},{},{},{:.4f}'.format(
                        time_epoch_ms(), anchor_type, anchor_time,
                        self.config.name, metric_name,
                        metrics_summary_df[metric_name].values[0]))
            else:
                raise ValueError(
                    'Unexpected tracking metric: {}'.format(metric_name))

    def __get_ground_obstacles_at(self, timestamp):
        for (ground_time, obstacles) in self._ground_obstacles:
            if ground_time == timestamp:
                return obstacles
            elif ground_time > timestamp:
                break
        self._logger.fatal(
            'Could not find ground obstacles for {}'.format(timestamp))

    def __get_tracked_obstacles_at(self, timestamp):
        for (start_time, obstacles) in self._tracked_obstacles:
            if start_time == timestamp:
                return obstacles
            elif start_time > timestamp:
                break
        self._logger.fatal(
            'Could not find tracked obstacles for {}'.format(timestamp))

    def __gc_obstacles_earlier_than(self, game_time):
        index = 0
        while (index < len(self._tracked_obstacles)
               and self._tracked_obstacles[index][0] < game_time):
            index += 1
        if index > 0:
            self._tracked_obstacles = self._tracked_obstacles[index:]
        index = 0
        while (index < len(self._ground_obstacles)
               and self._ground_obstacles[index][0] < game_time):
            index += 1
        if index > 0:
            self._ground_obstacles = self._ground_obstacles[index:]

    def on_tracker_obstacles(self, msg):
        game_time = msg.timestamp.coordinates[0]
        self._tracked_obstacles.append((game_time, msg.obstacles))
        if len(self._tracked_obstacles) > 1:
            assert game_time >= self._tracked_obstacles[-2][0], \
                'Obstacle messages did not arrive in order'
        # Two metrics: 1) mAP, and 2) timely-mAP
        if not self._evaluate_timely:
            # We will compare the obstacles with the ground truth at the same
            # game time.
            self._tracker_start_end_times.append((game_time, game_time))
        else:
            # Ground obstacles time should be as close as possible to the time
            # of the obstacles + detector + tracker runtime.
            ground_obstacles_time = self.__compute_closest_frame_time(
                game_time + msg.runtime)
            self._tracker_start_end_times.append(
                (game_time, ground_obstacles_time))

    def on_ground_obstacles(self, msg):
        game_time = msg.timestamp.coordinates[0]
        self._ground_obstacles.append((game_time, msg.obstacles))

    def __get_tracker_metrics(self, tracked_obstacles, ground_obstacles,
                              accumulator):
        """Computes several tracker accuracy metrics using motmetrics library.

        Args:
            tracked_obstacles: list of Obstacles from trackers
            ground_obstacles: list of Obstacles from perfect detector

        Returns:
            tracker_metrics_df: one-row pandas.DataFrame with columns from
                motmetrics.metrics.motchallenge_metrics for the current frame
        """
        ground_ids = [ob.id for ob in ground_obstacles]
        track_ids = [ob.id for ob in tracked_obstacles]
        ground_bboxes = np.array([
            ob.bounding_box.as_width_height_bbox() for ob in ground_obstacles
        ])
        tracked_bboxes = np.array([
            ob.bounding_box.as_width_height_bbox() for ob in tracked_obstacles
        ])
        cost_matrix = mm.distances.iou_matrix(ground_bboxes,
                                              tracked_bboxes,
                                              max_iou=1 -
                                              self._flags.min_matching_iou)
        accumulator.update(ground_ids, track_ids, cost_matrix)
        # Calculate all motchallenge metrics by default. Logged metrics
        # determined by list passed to --tracking_metrics
        tracker_metrics_df = self._metrics_host.compute(
            accumulator,
            metrics=list(
                set(self._flags.tracking_metrics).union(
                    set(mm.metrics.motchallenge_metrics))))
        return tracker_metrics_df

    def __compute_closest_frame_time(self, time):
        base = math.ceil(int(time) / self._frame_gap) * self._frame_gap
        return base

    # def __compute_closest_frame_time(self, time):
    #     base = int(time) / self._frame_gap * self._frame_gap
    #     if time - base < self._frame_gap / 2:
    #         return base
    #     else:
    #         return base + self._frame_gap
