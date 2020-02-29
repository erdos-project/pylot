from absl import flags
from collections import deque
import erdos
import time

from pylot.perception.messages import ObstaclesMessage
from pylot.utils import time_epoch_ms

flags.DEFINE_bool('visualize_tracker_output', False,
                  'True to enable visualization of tracker output')


class ObjectTrackerOperator(erdos.Operator):
    def __init__(self, obstacles_stream, camera_stream,
                 obstacle_tracking_stream, tracker_type, flags):
        obstacles_stream.add_callback(self.on_obstacles_msg,
                                      [obstacle_tracking_stream])
        camera_stream.add_callback(self.on_frame_msg,
                                   [obstacle_tracking_stream])
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            self.config.name + '-csv', self.config.csv_log_file_name)
        self._tracker_type = tracker_type
        try:
            if tracker_type == 'cv2':
                from pylot.perception.tracking.cv2_tracker import\
                    MultiObjectCV2Tracker
                self._tracker = MultiObjectCV2Tracker(self._flags)
            elif tracker_type == 'da_siam_rpn':
                from pylot.perception.tracking.da_siam_rpn_tracker import\
                    MultiObjectDaSiamRPNTracker
                self._tracker = MultiObjectDaSiamRPNTracker(self._flags)
            elif tracker_type == 'deep_sort':
                from pylot.perception.tracking.deep_sort_tracker import\
                    MultiObjectDeepSORTTracker
                self._tracker = MultiObjectDeepSORTTracker(
                    self._flags, self._logger)
            elif tracker_type == 'sort':
                from pylot.perception.tracking.sort_tracker import\
                    MultiObjectSORTTracker
                self._tracker = MultiObjectSORTTracker(self._flags)
            else:
                raise ValueError(
                    'Unexpected tracker type {}'.format(tracker_type))
        except ImportError:
            self._logger.fatal('Error importing {}'.format(tracker_type))
        # Labels the obstacle trackers should track.
        self._tracked_labels = {
            'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck'
        }

        # True when the tracker is ready to update bboxes.
        self._ready_to_update = False
        self._ready_to_update_timestamp = None
        self._to_process = deque()

    @staticmethod
    def connect(obstacles_stream, camera_stream):
        obstacle_tracking_stream = erdos.WriteStream()
        return [obstacle_tracking_stream]

    def on_frame_msg(self, msg, obstacle_tracking_stream):
        """ Invoked when a FrameMessage is received on the camera stream."""
        self._logger.debug('@{}: {} received frame'.format(
            msg.timestamp, self.config.name))
        start_time = time.time()
        assert msg.frame.encoding == 'BGR', 'Expects BGR frames'
        camera_frame = msg.frame
        # Store frames so that they can be re-processed once we receive the
        # next update from the detector.
        self._to_process.append((msg.timestamp, camera_frame))
        # Track if we have a tracker ready to accept new frames.
        if self._ready_to_update:
            self.__track_bboxes_on_frame(camera_frame, msg.timestamp, False,
                                         obstacle_tracking_stream)
        # Get runtime in ms.
        runtime = (time.time() - start_time) * 1000
        self._csv_logger.info('{},{},"{}",{}'.format(time_epoch_ms(),
                                                     self.config.name,
                                                     msg.timestamp, runtime))

    def on_obstacles_msg(self, msg, obstacle_tracking_stream):
        """ Invoked when obstacles are received on the stream."""
        self._logger.debug('@{}: {} received obstacles'.format(
            msg.timestamp, self.config.name))
        self._ready_to_update = False
        self._logger.debug("@{}: received {} bounding boxes".format(
            msg.timestamp, len(msg.obstacles)))
        # Remove frames that are older than the detector update.
        while len(self._to_process
                  ) > 0 and self._to_process[0][0] < msg.timestamp:
            self._logger.debug("@{}: removing stale {} {}".format(
                msg.timestamp, self._to_process[0][0], msg.timestamp))
            self._to_process.popleft()

        detected_obstacles = []
        for obstacle in msg.obstacles:
            if obstacle.label in self._tracked_labels:
                detected_obstacles.append(obstacle)

        if len(detected_obstacles) > 0:
            if len(self._to_process) > 0:
                # Found the frame corresponding to the bounding boxes.
                (timestamp, camera_frame) = self._to_process.popleft()
                assert timestamp == msg.timestamp
                # Re-initialize trackers.
                self._ready_to_update = True
                self._ready_to_update_timestamp = timestamp
                self._logger.debug(
                    'Restarting trackers at frame {}'.format(timestamp))
                self._tracker.reinitialize(camera_frame, detected_obstacles)
                self._logger.debug(
                    'Trackers have {} frames to catch-up'.format(
                        len(self._to_process)))
                for (timestamp, camera_frame) in self._to_process:
                    if self._ready_to_update:
                        self.__track_bboxes_on_frame(camera_frame,
                                                     msg.timestamp, True,
                                                     obstacle_tracking_stream)
            else:
                self._logger.debug(
                    '@{}: received bboxes update, but no frame to process'.
                    format(msg.timestamp))

    def __track_bboxes_on_frame(self, camera_frame, timestamp, catch_up,
                                obstacle_tracking_stream):
        self._logger.debug('Processing frame {}'.format(timestamp))
        # Sequentually update state for each bounding box.
        start_time = time.time()
        ok, tracked_obstacles = self._tracker.track(camera_frame)
        if not ok:
            self._logger.error(
                'Tracker failed at timestamp {} last ready_to_update at {}'.
                format(timestamp, self._ready_to_update_timestamp))
            # The tracker must be reinitialized.
            self._ready_to_update = False
        else:
            # Get runtime in ms.
            runtime = (time.time() - start_time) * 1000
            obstacle_tracking_stream.send(
                ObstaclesMessage(timestamp, tracked_obstacles, runtime))
            if self._flags.visualize_tracker_output and not catch_up:
                # tracked obstacles have no label, draw white bbox.
                camera_frame.annotate_with_bounding_boxes(
                    timestamp, tracked_obstacles)
                camera_frame.visualize(self.config.name)
