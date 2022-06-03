import time
from collections import deque
from typing import Union

import erdos
from erdos.operator import OneInOneOut
from erdos.context import OneInOneOutContext
from pylot.perception.camera_frame import CameraFrame

from pylot.perception.messages import ObstaclesMessageTuple


class ObjectTrackerOperator(OneInOneOut):
    def __init__(self, tracker_type: str, flags):
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            self.config.name + '-csv', self.config.csv_log_file_name)
        self._tracker_type = tracker_type
        # Absolute time when the last tracker run completed.
        self._last_tracker_run_completion_time = 0
        try:
            if tracker_type == 'da_siam_rpn':
                from pylot.perception.tracking.da_siam_rpn_tracker import\
                    MultiObjectDaSiamRPNTracker
                self._tracker = MultiObjectDaSiamRPNTracker(
                    self._flags, self._logger)
            elif tracker_type == 'deep_sort':
                from pylot.perception.tracking.deep_sort_tracker import\
                    MultiObjectDeepSORTTracker
                self._tracker = MultiObjectDeepSORTTracker(
                    self._flags, self._logger)
            elif tracker_type == 'sort':
                from pylot.perception.tracking.sort_tracker import\
                    MultiObjectSORTTracker
                self._tracker = MultiObjectSORTTracker(self._flags,
                                                       self._logger)
            else:
                raise ValueError(
                    'Unexpected tracker type {}'.format(tracker_type))
        except ImportError as error:
            self._logger.fatal('Error importing {}'.format(tracker_type))
            raise error

        self._obstacles = deque()
        self._frames = deque()
        self._detection_update_count = -1

    def destroy(self):
        self._logger.warn('destroying {}'.format(self.config.name))

    def on_frame(self, context: OneInOneOutContext, frame: CameraFrame):
        """Invoked when a CameraFrame is received."""
        self._logger.debug('@{}: {} received frame'.format(
            context.timestamp, self.config.name))
        assert frame.encoding == 'BGR', 'Expects BGR frames'
        self._frames.append(frame)

    def on_obstacles(self, context: OneInOneOutContext,
                     obstacles: ObstaclesMessageTuple):
        """Invoked when obstacles are received on the stream."""
        self._logger.debug('@{}: {} received {} obstacles'.format(
            context.timestamp, self.config.name, len(obstacles.obstacles)))
        self._obstacles.append((context.timestamp, obstacles))

    def on_time_to_decision_update(self, context: OneInOneOutContext,
                                   ttd: float):
        self._logger.debug('@{}: {} received ttd update {}'.format(
            context.timestamp, self.config.name, ttd))

    def _reinit_tracker(self, camera_frame, detected_obstacles):
        start = time.time()
        result = self._tracker.reinitialize(camera_frame, detected_obstacles)
        return (time.time() - start) * 1000, result

    def _run_tracker(self, camera_frame):
        start = time.time()
        result = self._tracker.track(camera_frame)
        return (time.time() - start) * 1000, result

    def on_data(self, context: OneInOneOutContext,
                data: Union[CameraFrame, ObstaclesMessageTuple, float]):
        if isinstance(data, CameraFrame):
            self.on_frame(context, data)
        elif isinstance(data, ObstaclesMessageTuple):
            self.on_obstacles(context, data)
        elif isinstance(data, float):
            self.on_time_to_decision_update(context, data)
        else:
            raise ValueError('Unexpected data type')

    @erdos.profile_method()
    def on_watermark(self, context: OneInOneOutContext):
        self._logger.debug('@{}: received watermark'.format(context.timestamp))
        if context.timestamp.is_top:
            return
        camera_frame = self._frames.popleft()
        tracked_obstacles = []
        detector_runtime = 0
        reinit_runtime = 0
        # Check if the most recent obstacle message has this timestamp.
        # If it doesn't, then the detector might have skipped sending
        # an obstacle message.
        if (len(self._obstacles) > 0
                and self._obstacles[0][0] == context.timestamp):
            _, obstacles = self._obstacles.popleft()
            self._detection_update_count += 1
            if (self._detection_update_count %
                    self._flags.track_every_nth_detection == 0):
                # Reinitialize the tracker with new detections.
                self._logger.debug('Restarting trackers at frame {}'.format(
                    context.timestamp))
                detected_obstacles = []
                for obstacle in obstacles.obstacles:
                    if obstacle.is_vehicle() or obstacle.is_person():
                        detected_obstacles.append(obstacle)
                reinit_runtime, _ = self._reinit_tracker(
                    camera_frame, detected_obstacles)
                detector_runtime = obstacles.runtime
        tracker_runtime, (ok, tracked_obstacles) = \
            self._run_tracker(camera_frame)
        assert ok, 'Tracker failed at timestamp {}'.format(context.timestamp)
        tracker_runtime = tracker_runtime + reinit_runtime
        tracker_delay = self.__compute_tracker_delay(
            context.timestamp.coordinates[0], detector_runtime,
            tracker_runtime)
        context.write_stream.send(
            erdos.Message(
                context.timestamp,
                ObstaclesMessageTuple(tracked_obstacles, tracker_delay)))

    def __compute_tracker_delay(self, world_time, detector_runtime,
                                tracker_runtime):
        # If the tracker runtime does not fit within the frame gap, then
        # the tracker will fall behind. We need a scheduler to better
        # handle such situations.
        if (world_time + detector_runtime >
                self._last_tracker_run_completion_time):
            # The detector finished after the previous tracker invocation
            # completed. Therefore, the tracker is already sequenced.
            tracker_runtime = detector_runtime + tracker_runtime
            self._last_tracker_run_completion_time = \
                world_time + tracker_runtime
        else:
            # The detector finished before the previous tracker invocation
            # completed. The tracker can only run after the previous
            # invocation completes.
            self._last_tracker_run_completion_time += tracker_runtime
            tracker_runtime = \
                self._last_tracker_run_completion_time - world_time
        return tracker_runtime
