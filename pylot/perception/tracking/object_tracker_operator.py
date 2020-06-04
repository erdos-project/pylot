from collections import deque
import erdos

import pylot.utils
from pylot.perception.detection.utils import VEHICLE_LABELS
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

        self._obstacles_msgs = deque()
        self._frame_msgs = deque()
        self._watermark_msg_count = 0

    @staticmethod
    def connect(obstacles_stream, camera_stream, time_to_decision_stream):
        obstacle_tracking_stream = erdos.WriteStream()
        return [obstacle_tracking_stream]

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

    @erdos.profile_method()
    def on_watermark(self, timestamp, obstacle_tracking_stream):
        self._logger.debug('@{}: received watermark'.format(timestamp))
        frame_msg = self._frame_msgs.popleft()
        camera_frame = frame_msg.frame
        tracked_obstacles = []
        self._watermark_msg_count += 1
        if len(self._obstacles_msgs) > 0:
            obstacles_msg = self._obstacles_msgs.popleft()
            assert frame_msg.timestamp == obstacles_msg.timestamp
            self._logger.debug(
                'Restarting trackers at frame {}'.format(timestamp))
            detected_obstacles = []
            for obstacle in obstacles_msg.obstacles:
                if (obstacle.label in VEHICLE_LABELS
                        or obstacle.label == 'person'):
                    detected_obstacles.append(obstacle)
            if self._watermark_msg_count % self._flags.track_every_nth_detection == 0:
                self._tracker.reinitialize(camera_frame, detected_obstacles)

        self._logger.debug('Processing frame {}'.format(timestamp))
        ok, tracked_obstacles = self._tracker.track(camera_frame)
        if not ok:
            self._logger.error(
                'Tracker failed at timestamp {}'.format(timestamp))
        sim_time = timestamp.coordinates[0]
        num_targets = len(tracked_obstacles)
        self._csv_logger.info('{},{},{},{},{}'.format(
            pylot.utils.time_epoch_ms(), sim_time, self.config.name, 'num_targets',
            num_targets))
        obstacle_tracking_stream.send(
            ObstaclesMessage(timestamp, tracked_obstacles, 0))

        if self._flags.visualize_tracker_output:
            # Tracked obstacles have no label, draw white bbox.
            camera_frame.annotate_with_bounding_boxes(timestamp,
                                                      tracked_obstacles)
            camera_frame.visualize(self.config.name,
                                   pygame_display=pylot.utils.PYGAME_DISPLAY)
