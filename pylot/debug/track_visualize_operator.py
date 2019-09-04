from absl import flags
from collections import deque
import cv2
import numpy as np
import threading

import carla

# ERDOS specific imports.
from erdos.op import Op
from erdos.utils import setup_logging

# Pylot specific imports.
from pylot.perception.segmentation.utils import transform_to_cityscapes_palette
import pylot.utils
import pylot.simulation.carla_utils

FLAGS = flags.FLAGS

class TrackVisualizerOperator(Op):
    """ 
        TrackVisualizerOperator visualizes the past locations of agents
        on the top-down segmented image.
    """

    def __init__(self, name, flags, log_file_name=None):
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
        self._colors = {'pedestrian': [255, 0, 0],
                        'vehicle': [0, 255, 0]}

        # Queues of incoming data.
        self._tracking_msgs = deque()
        self._top_down_segmentation_msgs = deque()
        self._lock = threading.Lock()
        self._frame_cnt = 0

    @staticmethod
    def setup_streams(input_streams, top_down_stream_name):
        input_streams.filter(pylot.utils.is_tracking_stream).add_callback(
            TrackVisualizerOperator.on_tracking_update)
        input_streams.filter(pylot.utils.is_segmented_camera_stream).filter_name(
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

    def on_notification(self, msg):
        # Pop the oldest message from each buffer.
        with self._lock:
            if not self.synchronize_msg_buffers(
                    msg.timestamp,
                    [self._tracking_msgs, self._top_down_segmentation_msgs]):
                return
            tracking_msg = self._tracking_msgs.popleft()
            segmentation_msg = self._top_down_segmentation_msgs.popleft()

        self._logger.info('Timestamps {} {}'.format(
            tracking_msg.timestamp, segmentation_msg.timestamp))

        assert (tracking_msg.timestamp == segmentation_msg.timestamp)

        self._frame_cnt += 1

        # Top down camera transform, used to perform the coordinate change
        # for the top-down camera.
        top_down_location = pylot.simulation.utils.Location(1.5, 0.0, 1.4 + FLAGS.top_down_lateral_view)
        top_down_rotation = pylot.simulation.utils.Rotation(-90, 0, 0)
        top_down_transform = pylot.simulation.utils.Transform(
            top_down_location, top_down_rotation)

        display_img = np.uint8(transform_to_cityscapes_palette(segmentation_msg.frame))
        for obj in tracking_msg.obj_trajectories:
            # Intrinsic matrix of the top down segmentation camera.
            intrinsic_matrix = pylot.simulation.utils.create_intrinsic_matrix(
                                   FLAGS.carla_camera_image_width,
                                   FLAGS.carla_camera_image_height,
                                   fov=90)
            # Convert to screen points.
            screen_points = pylot.simulation.utils.locations_3d_to_view(
                                obj.trajectory,
                                top_down_transform.matrix,
                                intrinsic_matrix)
                                                                            
            # Draw trajectory points on segmented image.
            for point in screen_points:
                if (0 <= point.x <= FLAGS.carla_camera_image_width) and \
                   (0 <= point.y <= FLAGS.carla_camera_image_height):
                    cv2.circle(display_img,
                               (int(point.x), int(point.y)),
                               3, self._colors[obj.obj_class], -1)
        pylot.utils.add_timestamp(msg.timestamp, display_img)
        cv2.imshow('img', display_img)
        cv2.waitKey(1)
