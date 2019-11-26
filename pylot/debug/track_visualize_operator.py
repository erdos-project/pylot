from collections import deque
import cv2
import numpy as np
import threading

# ERDOS specific imports.
from erdos.op import Op
from erdos.utils import setup_logging

# Pylot specific imports.
from pylot.perception.segmentation.utils import transform_to_cityscapes_palette
import pylot.utils
import pylot.simulation.carla_utils


class TrackVisualizerOperator(Op):
    """ TrackVisualizerOperator visualizes the past and predicted future
        locations of agents on the top-down segmented image.
    """

    def __init__(self, name, flags, top_down_camera_setup, log_file_name=None):
        """ Initializes the TrackVisualizerOperator with the given
        parameters.

        Args:
            name: The name of the operator.
            flags: A handle to the global flags instance to retrieve the
                configuration.
            log_file_name: The file to log the required information to.
        """
        super(TrackVisualizerOperator, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self._flags = flags
        self._past_colors = {'pedestrian': [255, 0, 0],
                             'vehicle': [128, 128, 0]}
        self._future_colors = {'pedestrian': [0, 0, 255],
                               'vehicle': [0, 255, 0]}

        # Queues of incoming data.
        self._tracking_msgs = deque()
        self._top_down_segmentation_msgs = deque()
        self._prediction_msgs = deque()
        self._lock = threading.Lock()
        self._frame_cnt = 0

        # Get top-down camera.
        self._top_down_camera_setup = top_down_camera_setup

    @staticmethod
    def setup_streams(input_streams, top_down_stream_name):
        input_streams.filter(pylot.utils.is_tracking_stream).add_callback(
            TrackVisualizerOperator.on_tracking_update)
        input_streams.filter(pylot.utils.is_prediction_stream).add_callback(
            TrackVisualizerOperator.on_prediction_update)
        input_streams.filter(
            pylot.utils.is_segmented_camera_stream).filter_name(
                top_down_stream_name).add_callback(
                    TrackVisualizerOperator.on_top_down_segmentation_update)
        # Register a completion watermark callback. The callback is invoked
        # after all the messages with a given timestamp have been received.
        input_streams.add_completion_callback(
           TrackVisualizerOperator.on_notification)
        return []

    def synchronize_msg_buffers(self, timestamp, buffers):
        for buffer in buffers:
            while (len(buffer) > 0 and buffer[0].timestamp < timestamp):
                buffer.popleft()
            if len(buffer) == 0:
                return False
            assert buffer[0].timestamp == timestamp
        return True

    def on_tracking_update(self, msg):
        with self._lock:
            self._tracking_msgs.append(msg)

    def on_top_down_segmentation_update(self, msg):
        with self._lock:
            self._top_down_segmentation_msgs.append(msg)

    def on_prediction_update(self, msg):
        with self._lock:
            self._prediction_msgs.append(msg)

    def on_notification(self, msg):
        # Pop the oldest message from each buffer.
        msg_buffers = [self._tracking_msgs, self._top_down_segmentation_msgs]
        if self._flags.prediction:
            msg_buffers.append(self._prediction_msgs)
        with self._lock:
            if not self.synchronize_msg_buffers(
                    msg.timestamp,
                    msg_buffers):
                return
            tracking_msg = self._tracking_msgs.popleft()
            segmentation_msg = self._top_down_segmentation_msgs.popleft()
            if self._flags.prediction:
                prediction_msg = self._prediction_msgs.popleft()

        if self._flags.prediction:
            self._logger.info('Timestamps {} {} {}'.format(
                tracking_msg.timestamp, segmentation_msg.timestamp, prediction_msg.timestamp))

            assert (tracking_msg.timestamp == segmentation_msg.timestamp == prediction_msg.timestamp)
        else:
            self._logger.info('Timestamps {} {}'.format(
                tracking_msg.timestamp, segmentation_msg.timestamp))

            assert (tracking_msg.timestamp == segmentation_msg.timestamp)

        self._frame_cnt += 1

        display_img = np.uint8(transform_to_cityscapes_palette(
            segmentation_msg.frame))
        for obj in tracking_msg.obj_trajectories:
            self._draw_trajectory_on_img(obj, display_img, False)
        if self._flags.prediction:
            for obj in prediction_msg.predictions:
                display_img = self._draw_trajectory_on_img(obj, display_img, True)
        pylot.utils.add_timestamp(msg.timestamp, display_img)
        cv2.imshow('img', display_img)
        cv2.waitKey(1)

    def _draw_trajectory_on_img(self, obj, img, predict):
        # Intrinsic matrix of the top down segmentation camera.
        intrinsic_matrix = pylot.simulation.utils.create_intrinsic_matrix(
                               self._top_down_camera_setup.width,
                               self._top_down_camera_setup.height,
                               fov=self._top_down_camera_setup.fov)
        # Convert to screen points.
        screen_points = [
            loc.to_camera_view(
                pylot.simulation.utils.camera_to_unreal_transform(
                    self._top_down_camera_setup.transform).matrix,
                intrinsic_matrix) for loc in obj.trajectory
        ]
        if predict:
            point_color = self._future_colors[obj.obj_class]
        else:
            point_color = self._past_colors[obj.obj_class]

        # Draw trajectory points on segmented image.
        for point in screen_points:
            if (0 <= point.x <= self._flags.carla_camera_image_width) and \
               (0 <= point.y <= self._flags.carla_camera_image_height):
                cv2.circle(img,
                           (int(point.x), int(point.y)),
                           3, point_color, -1)
        return img
