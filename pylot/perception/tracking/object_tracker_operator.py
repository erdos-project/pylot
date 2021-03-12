import time
from collections import deque

import erdos

from pylot.perception.messages import ObstaclesMessage


class ObjectTrackerOperator(erdos.Operator):
    def __init__(self, obstacles_stream, camera_stream,
                 time_to_decision_stream, obstacle_tracking_stream,
                 tracker_type, flags):
        obstacles_stream.add_callback(self.on_obstacles_msg)
        camera_stream.add_callback(self.on_frame_msg)
        time_to_decision_stream.add_callback(self.on_time_to_decision_update)
        erdos.add_watermark_callback([obstacles_stream, camera_stream],
                                     [obstacle_tracking_stream],
                                     self.on_watermark)
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

        self._obstacles_msgs = deque()
        self._frame_msgs = deque()
        self._detection_update_count = -1

    @staticmethod
    def connect(obstacles_stream, camera_stream, time_to_decision_stream):
        obstacle_tracking_stream = erdos.WriteStream()
        return [obstacle_tracking_stream]

    def destroy(self):
        self._logger.warn('destroying {}'.format(self.config.name))

    def on_frame_msg(self, msg):
        """Invoked when a FrameMessage is received on the camera stream."""
        self._logger.debug('@{}: {} received frame'.format(
            msg.timestamp, self.config.name))
        assert msg.frame.encoding == 'BGR', 'Expects BGR frames'
        self._frame_msgs.append(msg)

    def on_obstacles_msg(self, msg):
        """Invoked when obstacles are received on the stream."""
        self._logger.debug('@{}: {} received {} obstacles'.format(
            msg.timestamp, self.config.name, len(msg.obstacles)))
        self._obstacles_msgs.append(msg)

    def on_time_to_decision_update(self, msg):
        self._logger.debug('@{}: {} received ttd update {}'.format(
            msg.timestamp, self.config.name, msg))

    def _reinit_tracker(self, camera_frame, detected_obstacles):
        start = time.time()
        result = self._tracker.reinitialize(camera_frame, detected_obstacles)
        return (time.time() - start) * 1000, result

    def _run_tracker(self, camera_frame):
        start = time.time()
        result = self._tracker.track(camera_frame)
        return (time.time() - start) * 1000, result

    @erdos.profile_method()
    def on_watermark(self, timestamp, obstacle_tracking_stream):
        self._logger.debug('@{}: received watermark'.format(timestamp))
        if timestamp.is_top:
            return
        frame_msg = self._frame_msgs.popleft()
        camera_frame = frame_msg.frame
        tracked_obstacles = []
        detector_runtime = 0
        reinit_runtime = 0
        # Check if the most recent obstacle message has this timestamp.
        # If it doesn't, then the detector might have skipped sending
        # an obstacle message.
        if (len(self._obstacles_msgs) > 0
                and self._obstacles_msgs[0].timestamp == timestamp):
            obstacles_msg = self._obstacles_msgs.popleft()
            self._detection_update_count += 1
            if (self._detection_update_count %
                    self._flags.track_every_nth_detection == 0):
                # Reinitialize the tracker with new detections.
                self._logger.debug(
                    'Restarting trackers at frame {}'.format(timestamp))
                detected_obstacles = []
                for obstacle in obstacles_msg.obstacles:
                    if obstacle.is_vehicle() or obstacle.is_person():
                        detected_obstacles.append(obstacle)
                reinit_runtime, _ = self._reinit_tracker(
                    camera_frame, detected_obstacles)
                detector_runtime = obstacles_msg.runtime
        tracker_runtime, (ok, tracked_obstacles) = \
            self._run_tracker(camera_frame)
        assert ok, 'Tracker failed at timestamp {}'.format(timestamp)
        tracker_runtime = tracker_runtime + reinit_runtime
        tracker_delay = self.__compute_tracker_delay(timestamp.coordinates[0],
                                                     detector_runtime,
                                                     tracker_runtime)
        obstacle_tracking_stream.send(
            ObstaclesMessage(timestamp, tracked_obstacles, tracker_delay))

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
