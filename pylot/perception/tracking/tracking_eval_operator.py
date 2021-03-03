"""Implements an operator that eveluates tracking output."""
import heapq
import math
import time

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
                 finished_indicator_stream, evaluate_timely, flags):
        obstacle_tracking_stream.add_callback(self.on_tracker_obstacles)
        ground_obstacles_stream.add_callback(self.on_ground_obstacles)
        erdos.add_watermark_callback(
            [obstacle_tracking_stream, ground_obstacles_stream],
            [finished_indicator_stream],
            self.on_watermark)
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
        # Heap storing pairs of (ground/output time, game time).
        self._tracker_start_end_times = []
        self._sim_interval = None
        self._accumulator = mm.MOTAccumulator(auto_id=True)
        self._metrics_host = mm.metrics.create()
        self._evaluate_timely = evaluate_timely

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
        assert len(timestamp.coordinates) == 1
        op_start_time = time.time()
        game_time = timestamp.coordinates[0]
        if not self._last_notification:
            self._last_notification = game_time
            return
        else:
            self._sim_interval = (game_time - self._last_notification)
            self._last_notification = game_time

        sim_time = timestamp.coordinates[0]
        while len(self._tracker_start_end_times) > 0:
            (end_time, start_time) = self._tracker_start_end_times[0]
            # We can compute tracker metrics if the endtime is not greater than
            # the ground time.
            if end_time <= game_time:
                # This is the closest ground bounding box to the end time.
                heapq.heappop(self._tracker_start_end_times)
                ground_obstacles = self.__get_ground_obstacles_at(end_time)
                # Get tracker output obstacles.
                tracker_obstacles = self.__get_tracked_obstacles_at(start_time)
                metrics_summary_df = self.__get_tracker_metrics(
                    tracker_obstacles, ground_obstacles)
                # Get runtime in ms
                runtime = (time.time() - op_start_time) * 1000
                self._csv_logger.info('{},{},{},{},{}'.format(
                    time_epoch_ms(), sim_time, self.config.name, 'runtime',
                    runtime))
                # Write metrics to csv log file
                self.__write_metrics_to_csv(metrics_summary_df, sim_time)
                self._logger.debug('Computing accuracy for {} {}'.format(
                    end_time, start_time))
            else:
                # The remaining entries require newer ground obstacles.
                break
        self.__garbage_collect_obstacles()

    def __write_metrics_to_csv(self, metrics_summary_df, sim_time):
        for metric_name in self._flags.tracking_metrics:
            if metric_name in metrics_summary_df.columns:
                if (metric_name == 'mostly_tracked'
                        or metric_name == 'mostly_lost'
                        or metric_name == 'partially_tracked'):
                    ratio = metrics_summary_df[metric_name].values[
                        0] / metrics_summary_df[
                            'num_unique_objects'].values[0]
                    self._csv_logger.info(
                        "{},{},{},{},{:.4f}".format(
                            time_epoch_ms(), sim_time,
                            self.config.name,
                            'ratio_' + metric_name, ratio))
                elif metric_name == 'motp':
                    # See https://github.com/cheind/py-motmetrics/issues/92
                    motp = (1 - metrics_summary_df[metric_name].
                            values[0]) * 100
                    self._csv_logger.info(
                        '{},{},{},{},{:.4f}'.format(
                            time_epoch_ms(), sim_time,
                            self.config.name, metric_name, motp))
                elif (metric_name == 'idf1'
                        or metric_name == 'mota'):
                    metric_val = metrics_summary_df[
                        metric_name].values[0] * 100
                    self._csv_logger.info(
                        '{},{},{},{},{:.4f}'.format(
                            time_epoch_ms(), sim_time,
                            self.config.name, metric_name,
                            metric_val))
                else:
                    self._csv_logger.info(
                        '{},{},{},{},{:.4f}'.format(
                            time_epoch_ms(), sim_time,
                            self.config.name, metric_name,
                            metrics_summary_df[metric_name].
                            values[0]))
            else:
                raise ValueError(
                    'Unexpected tracking metric: {}'.format(
                        metric_name))

    def __get_ground_obstacles_at(self, timestamp):
        for (ground_time, obstacles) in self._ground_obstacles:
            if ground_time == timestamp:
                return obstacles
            elif ground_time > timestamp:
                break
        self._logger.fatal(
            'Could not find ground obstacles for {}'.format(timestamp))

    def __get_tracked_obstacles_at(self, timestamp):
        for (ground_time, obstacles) in self._tracked_obstacles:
            if ground_time == timestamp:
                return obstacles
            elif ground_time > timestamp:
                break
        self._logger.fatal(
            'Could not find tracked obstacles for {}'.format(timestamp))

    def __garbage_collect_obstacles(self):
        # Get the minimum watermark.
        watermark = None
        for (_, start_time) in self._tracker_start_end_times:
            if watermark is None or start_time < watermark:
                watermark = start_time
        if watermark is None:
            return
        # Remove all tracked obstacles that are below the watermark.
        index = 0
        while (index < len(self._tracked_obstacles)
               and self._tracked_obstacles[index][0] < watermark):
            index += 1
        if index > 0:
            self._tracked_obstacles = self._tracked_obstacles[index:]
        # Remove all the ground obstacles that are below the watermark.
        index = 0
        while (index < len(self._ground_obstacles)
               and self._ground_obstacles[index][0] < watermark):
            index += 1
        if index > 0:
            self._ground_obstacles = self._ground_obstacles[index:]

    def on_tracker_obstacles(self, msg):
        game_time = msg.timestamp.coordinates[0]
        self._tracked_obstacles.append((game_time, msg.obstacles))
        # Two metrics: 1) mAP, and 2) timely-mAP
        if not self._evaluate_timely:
            # We will compare the obstacles with the ground truth at the same
            # game time.
            heapq.heappush(self._tracker_start_end_times,
                           (game_time, game_time))
        else:
            # Ground obstacles time should be as close as possible to the time
            # of the obstacles + detector + tracker runtime.
            ground_obstacles_time = self.__compute_closest_frame_time(
                game_time + msg.runtime)
            heapq.heappush(self._tracker_start_end_times,
                           (ground_obstacles_time, game_time))

    def on_ground_obstacles(self, msg):
        game_time = msg.timestamp.coordinates[0]
        self._ground_obstacles.append((game_time, msg.obstacles))

    def __get_tracker_metrics(self, tracked_obstacles, ground_obstacles):
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
        self._accumulator.update(ground_ids, track_ids, cost_matrix)
        # Calculate all motchallenge metrics by default. Logged metrics
        # determined by list passed to --tracking_metrics
        tracker_metrics_df = self._metrics_host.compute(
            self._accumulator, metrics=list(
                set(self._flags.tracking_metrics).union(
                    set(mm.metrics.motchallenge_metrics))
            )
        )
        return tracker_metrics_df

    def __compute_closest_frame_time(self, time):
        base = math.ceil(int(time) / self._sim_interval) * self._sim_interval
        return base

    # def __compute_closest_frame_time(self, time):
    #     base = int(time) / self._sim_interval * self._sim_interval
    #     if time - base < self._sim_interval / 2:
    #         return base
    #     else:
    #         return base + self._sim_interval
