"""Implements an operator that eveluates tracking output."""
import erdos

import motmetrics as mm

import numpy as np

from pylot.perception.base_perception_eval import BasePerceptionEvalOperator
from pylot.utils import time_epoch_ms


class TrackingEvalOperator(BasePerceptionEvalOperator):
    def __init__(self, prediction_stream, ground_truth_stream,
                 finished_indicator_stream, evaluate_timely, matching_policy,
                 frame_gap, flags):
        super().__init__(prediction_stream, ground_truth_stream,
                         finished_indicator_stream, evaluate_timely,
                         matching_policy, frame_gap, flags)
        self._flags = flags
        self._start_anchored_accumulator = mm.MOTAccumulator(auto_id=True)
        self._end_anchored_accumulator = mm.MOTAccumulator(auto_id=True)
        self._metrics_host = mm.metrics.create()
        self._csv_logger = erdos.utils.setup_csv_logging(
            self.config.name + '-csv', self.config.csv_log_file_name)

    def compute_accuracy(self, frame_time, ground_time, end_anchored):
        tracker_obstacles = self.get_prediction_at(frame_time)
        ground_obstacles = self.get_ground_truth_at(ground_time)
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

    def __write_metrics_to_csv(self, metrics_summary_df, anchor_type,
                               anchor_time):
        for metric_name in self._flags.tracking_metrics:
            if metric_name in metrics_summary_df.columns:
                if (metric_name == 'mostly_tracked'
                        or metric_name == 'mostly_lost'
                        or metric_name == 'partially_tracked'):
                    ratio = metrics_summary_df[metric_name].values[
                        0] / metrics_summary_df['num_unique_objects'].values[0]
                    self._csv_logger.info("{},{},{},{},{},{},{:.4f}".format(
                        time_epoch_ms(), anchor_time, self.config.name,
                        anchor_type, self._matching_policy,
                        'ratio_' + metric_name, ratio))
                elif metric_name == 'motp':
                    # See https://github.com/cheind/py-motmetrics/issues/92
                    motp = (1 -
                            metrics_summary_df[metric_name].values[0]) * 100
                    self._csv_logger.info('{},{},{},{},{},{},{:.4f}'.format(
                        time_epoch_ms(), anchor_time, self.config.name,
                        anchor_type, self._matching_policy, metric_name, motp))
                elif (metric_name == 'idf1' or metric_name == 'mota'):
                    metric_val = \
                        metrics_summary_df[metric_name].values[0] * 100
                    self._csv_logger.info('{},{},{},{},{},{},{:.4f}'.format(
                        time_epoch_ms(), anchor_time, self.config.name,
                        anchor_type, self._matching_policy, metric_name,
                        metric_val))
                else:
                    self._csv_logger.info('{},{},{},{},{},{},{:.4f}'.format(
                        time_epoch_ms(), anchor_time, self.config.name,
                        anchor_type, self._matching_policy, metric_name,
                        metrics_summary_df[metric_name].values[0]))
            else:
                raise ValueError(
                    'Unexpected tracking metric: {}'.format(metric_name))
