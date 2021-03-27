"""Implements an operator that eveluates tracking output."""
import erdos

import motmetrics as mm

import numpy as np

from pylot.perception.base_perception_eval_operator import (
    BasePerceptionEvalOperator, ScoringModule)


class TrackingEvalOperator(BasePerceptionEvalOperator):
    def __init__(self, prediction_stream: erdos.ReadStream,
                 ground_truth_stream: erdos.ReadStream,
                 finished_indicator_stream: erdos.WriteStream,
                 evaluate_timely: bool, matching_policy: str, frame_gap: int,
                 flags):
        super().__init__(prediction_stream, ground_truth_stream,
                         finished_indicator_stream, evaluate_timely,
                         matching_policy, frame_gap, TrackingScoringModule,
                         flags)


class TrackingScoringModule(ScoringModule):
    def __init__(self, flags):
        self._flags = flags
        self.accumulator = mm.MOTAccumulator(auto_id=True)
        self._metrics_host = mm.metrics.create()

    def get_scores(self):
        # Calculate all motchallenge metrics by default. Logged metrics
        # determined by list passed to --tracking_metrics
        metrics_summary_df = self._metrics_host.compute(
            self.accumulator,
            metrics=list(
                set(self._flags.tracking_metrics).union(
                    set(mm.metrics.motchallenge_metrics))))
        final_scores = {}
        for metric_name in self._flags.tracking_metrics:
            if metric_name in metrics_summary_df.columns:
                if (metric_name == 'mostly_tracked'
                        or metric_name == 'mostly_lost'
                        or metric_name == 'partially_tracked'):
                    ratio = metrics_summary_df[metric_name].values[
                        0] / metrics_summary_df['num_unique_objects'].values[0]
                    final_scores['ratio_' + metric_name] = ratio
                elif metric_name == 'motp':
                    # See https://github.com/cheind/py-motmetrics/issues/92
                    motp = (1 -
                            metrics_summary_df[metric_name].values[0]) * 100
                    final_scores[metric_name] = motp
                elif (metric_name == 'idf1' or metric_name == 'mota'):
                    metric_val = \
                        metrics_summary_df[metric_name].values[0] * 100
                    final_scores[metric_name] = metric_val
                else:
                    final_scores[metric_name] = metrics_summary_df[
                        metric_name].values[0]
            else:
                raise ValueError(
                    'Unexpected tracking metric: {}'.format(metric_name))
        return final_scores

    def add_datapoint(self, tracked_obstacles, ground_obstacles):
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
            ob.bounding_box_2D.as_width_height_bbox()
            for ob in ground_obstacles
        ])
        tracked_bboxes = np.array([
            ob.bounding_box_2D.as_width_height_bbox()
            for ob in tracked_obstacles
        ])
        cost_matrix = mm.distances.iou_matrix(
            ground_bboxes,
            tracked_bboxes,
            max_iou=1 - self._flags.eval_min_matching_iou)
        self.accumulator.update(ground_ids, track_ids, cost_matrix)
