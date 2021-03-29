import math

import erdos

from pylot.utils import time_epoch_ms


class BasePerceptionEvalOperator(erdos.Operator):
    """Operator that computes accuracy metrics using tracked obstacles.

    Args:
        prediction_stream (:py:class:`erdos.ReadStream`): The stream on
            which tracked obstacles are received.
        ground_truth_stream: The stream on which
            :py:class:`~pylot.perception.messages.ObstaclesMessage` are
            received from the simulator.
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, prediction_stream: erdos.ReadStream,
                 ground_truth_stream: erdos.ReadStream,
                 finished_indicator_stream: erdos.WriteStream,
                 evaluate_timely: bool, matching_policy: str, frame_gap: int,
                 scoring_module, flags):
        prediction_stream.add_callback(self.on_prediction)
        ground_truth_stream.add_callback(self.on_ground_truth)
        erdos.add_watermark_callback([prediction_stream, ground_truth_stream],
                                     [finished_indicator_stream],
                                     self.on_watermark)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._last_notification = None
        # Buffer of predictions.
        self._predictions = []
        # Buffer of ground data.
        self._ground_truths = []
        # Storing pairs of (game time, ground/output time).
        self._prediction_start_end_times = []
        self._frame_gap = frame_gap
        self._evaluate_timely = evaluate_timely
        self._matching_policy = matching_policy
        # The start time of the most recent inference that completed before
        # the timestamp of the watermark.
        self._start_time_best_inference = None
        # Index in prediction_start_end_times to the inference with the next
        # unprocessed start time. We need to maintain this index because
        # the start_tracker_end_times list might contain entries with
        # start time beyond current watermark.
        self._start_time_frontier = 0
        # Buffer storing start times and ground times of predictions we have
        # to compute accuracy for. The buffer is used to ensure that these
        # accuracies are computed only once the ground data is available.
        self._accuracy_compute_buffer = []
        # See interface below this function
        self._scoring_module = scoring_module
        self.scoring_modules = {
            "start": scoring_module(flags),
            "end": scoring_module(flags)
        }
        self._csv_logger = erdos.utils.setup_csv_logging(
            self.config.name + '-csv', self.config.csv_log_file_name)

    @staticmethod
    def connect(prediction_stream: erdos.ReadStream,
                ground_truth_stream: erdos.ReadStream):
        """Connects the operator to other streams.

        Args:
            prediction_stream (:py:class:`erdos.ReadStream`): The stream
                on which obstacle tracks are received from object trackers.
            ground_truth_stream: The stream on which
                :py:class:`~pylot.perception.messages.ObstaclesMessage` are
                received from the simulator.
        """
        finished_indicator_stream = erdos.WriteStream()
        return [finished_indicator_stream]

    @erdos.profile_method()
    def on_watermark(self, timestamp: erdos.Timestamp,
                     finished_indicator_stream: erdos.WriteStream):
        """Invoked when all input streams have received a watermark.

        Args:
            timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of
                the watermark.
        """
        if timestamp.is_top:
            return
        assert len(timestamp.coordinates) == 1
        game_time = timestamp.coordinates[0]
        self.__compute_frame_gap(game_time)
        on_new_prediction = False
        if self._start_time_frontier < len(self._prediction_start_end_times):
            (st,
             et) = self._prediction_start_end_times[self._start_time_frontier]
            if st == game_time:
                on_new_prediction = True
                self._accuracy_compute_buffer.append((st, et, False))
        index = self._start_time_frontier
        if not on_new_prediction:
            index = self._start_time_frontier - 1
        while index >= 0:
            (p_start_time,
             p_end_time) = self._prediction_start_end_times[index]
            if p_end_time == game_time:
                # This is the result that arrived before start_time, and
                # uses the most up-to-date sensor data (tracker_start_end_times
                # is sorted by start_times).
                self._start_time_best_inference = p_start_time
                self._prediction_start_end_times = \
                    self._prediction_start_end_times[index:]
                self._start_time_frontier -= index
                break
            index -= 1
        if self._start_time_best_inference:
            # Compute accuracy if we have a prediction with end_time less than
            # current frontier.
            self._accuracy_compute_buffer.append(
                (self._start_time_best_inference, game_time, True))
        self.__drain_accuracy_compute_buffer(game_time)
        # Increase the frontier to process the next start time.
        if on_new_prediction:
            self._start_time_frontier += 1

    def __drain_accuracy_compute_buffer(self, up_to_time: int):
        for (st, et, end_anchored) in self._accuracy_compute_buffer:
            if et <= up_to_time:
                self.compute_accuracy(st, et, end_anchored)
        self._accuracy_compute_buffer = list(
            filter(lambda args: args[1] > up_to_time,
                   self._accuracy_compute_buffer))
        # Can safely GC until the minimum between the min start time in the
        # accuracy compute buffer and the best inference start time.
        gc_threshold = min(self._accuracy_compute_buffer, default=[None])[0]
        if (self._start_time_best_inference is not None
                and (gc_threshold is None
                     or gc_threshold > self._start_time_best_inference)):
            gc_threshold = self._start_time_best_inference
        if gc_threshold is not None:
            self.__gc_data_earlier_than(gc_threshold)

    def compute_accuracy(self, frame_time: int, ground_time: int,
                         end_anchored: bool):
        anchor_type = "end" if end_anchored else "start"
        anchor_time = ground_time if end_anchored else frame_time
        predictions = self.get_prediction_at(frame_time)
        ground_truths = self.get_ground_truth_at(ground_time)
        self.scoring_modules[anchor_type].add_datapoint(
            predictions, ground_truths)
        new_scores = self.scoring_modules[anchor_type].get_scores()
        for k, v in new_scores.items():
            self._csv_logger.info("{},{},{},{},{},{},{:.4f}".format(
                time_epoch_ms(), anchor_time, self.config.name, anchor_type,
                self._matching_policy, k, v))

    def __compute_frame_gap(self, game_time: int):
        """Infer frame gap if not explicitly provided in constructor."""
        if not self._frame_gap:
            if not self._last_notification:
                self._last_notification = game_time
                return
            else:
                self._frame_gap = (game_time - self._last_notification)
                self._last_notification = game_time

    def get_ground_truth_at(self, timestamp: int):
        for (ground_time, obstacles) in self._ground_truths:
            if ground_time == timestamp:
                return obstacles
            elif ground_time > timestamp:
                break
        self._logger.fatal(
            'Could not find ground obstacles for {}'.format(timestamp))

    def get_prediction_at(self, timestamp: int):
        for (start_time, obstacles) in self._predictions:
            if start_time == timestamp:
                return obstacles
            elif start_time > timestamp:
                break
        self._logger.fatal(
            'Could not find tracked obstacles for {}'.format(timestamp))

    def __gc_data_earlier_than(self, game_time: int):
        index = 0
        while (index < len(self._predictions)
               and self._predictions[index][0] < game_time):
            index += 1
        if index > 0:
            self._predictions = self._predictions[index:]
        index = 0
        while (index < len(self._ground_truths)
               and self._ground_truths[index][0] < game_time):
            index += 1
        if index > 0:
            self._ground_truths = self._ground_truths[index:]

    def on_prediction(self, msg: erdos.Message):
        game_time = msg.timestamp.coordinates[0]
        self._predictions.append((game_time, msg.obstacles))
        if len(self._predictions) > 1:
            assert game_time >= self._predictions[-2][0], \
                'Obstacle messages did not arrive in order'
        # Two metrics: 1) mAP, and 2) timely-mAP
        if not self._evaluate_timely:
            # We will compare the obstacles with the ground truth at the same
            # game time.
            self._prediction_start_end_times.append((game_time, game_time))
        else:
            # Ground obstacles time should be as close as possible to the time
            # of the obstacles + detector + tracker runtime.
            ground_truth_time = self.__compute_closest_frame_time(game_time +
                                                                  msg.runtime)
            self._prediction_start_end_times.append(
                (game_time, ground_truth_time))

    def on_ground_truth(self, msg: erdos.Message):
        game_time = msg.timestamp.coordinates[0]
        self._ground_truths.append((game_time, msg.obstacles))

    def __compute_closest_frame_time(self, time: float) -> int:
        if self._frame_gap is None:
            self._logger.info(
                'Skipping frame {} because frame gap is not set yet'.format(
                    time))
            return
        if self._matching_policy == 'ceil':
            base = math.ceil(time / self._frame_gap) * self._frame_gap
        elif self._matching_policy == 'round':
            base = round(time / self._frame_gap) * self._frame_gap
        else:
            raise ValueError('Matching policy {} not supported'.format(
                self._matching_policy))
        return base


class ScoringModule:
    def __init__(self, flags):
        raise NotImplementedError("To be implemented by child class")

    def add_datapoint(self, predictions, ground_truth):
        raise NotImplementedError("To be implemented by child class")

    def get_scores(self):
        """
        Returns a dictionary from score name to score value
        """
        raise NotImplementedError("To be implemented by child class")
