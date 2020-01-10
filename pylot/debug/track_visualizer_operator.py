import cv2
import erdos

import pylot.utils


class TrackVisualizerOperator(erdos.Operator):
    """ TrackVisualizerOperator visualizes the past and predicted future
        locations of agents on the top-down segmented image.
    """
    def __init__(self,
                 obstacle_tracking_stream,
                 prediction_stream,
                 segmented_camera_stream,
                 name,
                 flags,
                 top_down_camera_setup,
                 log_file_name=None):
        """ Initializes the TrackVisualizerOperator with the given
        parameters.

        Args:
            name: The name of the operator.
            flags: A handle to the global flags instance to retrieve the
                configuration.
            log_file_name: The file to log the required information to.
        """
        obstacle_tracking_stream.add_callback(self.on_tracking_update)
        prediction_stream.add_callback(self.on_prediction_update)
        segmented_camera_stream.add_callback(
            self.on_top_down_segmentation_update)
        erdos.add_watermark_callback([
            obstacle_tracking_stream, prediction_stream,
            segmented_camera_stream
        ], [], self.on_watermark)
        self._name = name
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._flags = flags
        self._past_colors = {
            'pedestrian': [255, 0, 0],
            'vehicle': [128, 128, 0]
        }
        self._future_colors = {
            'pedestrian': [0, 0, 255],
            'vehicle': [0, 255, 0]
        }
        # Dictionaries to store incoming data.
        self._tracking_msgs = {}
        self._top_down_segmentation_msgs = {}
        self._prediction_msgs = {}
        self._top_down_camera_setup = top_down_camera_setup

    @staticmethod
    def connect(obstacle_tracking_stream, prediction_stream,
                segmented_camera_stream):
        return []

    def on_tracking_update(self, msg):
        self._tracking_msgs[msg.timestamp] = msg

    def on_prediction_update(self, msg):
        self._prediction_msgs[msg.timestamp] = msg

    def on_top_down_segmentation_update(self, msg):
        self._top_down_segmentation_msgs[msg.timestamp] = msg

    def on_watermark(self, timestamp):
        self._logger.debug('@{}: {} received watermark'.format(
            timestamp, self._name))
        tracking_msg = self._tracking_msgs.pop()
        segmentation_msg = self._top_down_segmentation_msgs.pop()
        prediction_msg = self._prediction_msgs.pop()

        display_img = segmentation_msg.frame.as_cityscapes_palette()
        for obstacle in tracking_msg.obstacle_trajectories:
            self._draw_trajectory_on_img(obstacle, display_img, False)
        for obstacle in prediction_msg.predictions:
            display_img = self._draw_trajectory_on_img(obstacle, display_img,
                                                       True)
        pylot.utils.add_timestamp(display_img, timestamp)
        cv2.imshow('img', display_img)
        cv2.waitKey(1)

    def _draw_trajectory_on_img(self, obstacle, img, predict):
        # Intrinsic and extrinsic matrix of the top down segmentation camera.
        extrinsic_matrix = self._top_down_camera_setup.get_extrinsic_matrix()
        intrinsic_matrix = self._top_down_camera_setup.get_intrinsic_matrix()

        # Convert to screen points.
        screen_points = [
            transform.location.to_camera_view(extrinsic_matrix,
                                              intrinsic_matrix)
            for transform in obstacle.trajectory
        ]
        if predict:
            point_color = self._future_colors[obstacle.label]
        else:
            point_color = self._past_colors[obstacle.label]

        # Draw trajectory points on segmented image.
        for point in screen_points:
            if (0 <= point.x <= self._flags.carla_camera_image_width) and \
               (0 <= point.y <= self._flags.carla_camera_image_height):
                cv2.circle(img, (int(point.x), int(point.y)), 3, point_color,
                           -1)
        return img
