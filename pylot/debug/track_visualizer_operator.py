import cv2
import erdust
import numpy as np

# Pylot specific imports.
import pylot.utils
import pylot.simulation.carla_utils


class TrackVisualizerOperator(erdust.Operator):
    """ TrackVisualizerOperator visualizes the past and predicted future
        locations of agents on the top-down segmented image.
    """

    def __init__(self,
                 obstacle_tracking_stream,
                 prediction_stream,
                 segmented_camera_stream,
                 name,
                 flags,
                 top_down_camera_setup):
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
        erdust.add_watermark_callback(
            [obstacle_tracking_stream,
             prediction_stream,
             segmented_camera_stream],
            [],
            self.on_watermark)
        self._flags = flags
        self._past_colors = {'pedestrian': [255, 0, 0],
                             'vehicle': [128, 128, 0]}
        self._future_colors = {'pedestrian': [0, 0, 255],
                               'vehicle': [0, 255, 0]}
        # Dictionaries to store incoming data.
        self._tracking_msgs = {}
        self._top_down_segmentation_msgs = {}
        self._prediction_msgs = {}
        # Get top-down camera.
        self._top_down_camera_setup = top_down_camera_setup

    @staticmethod
    def connect(obstacle_tracking_stream,
                prediction_stream,
                segmented_camera_stream):
        return []

    def on_tracking_update(self, msg):
        self._tracking_msgs[msg.timestamp] = msg

    def on_prediction_update(self, msg):
        self._prediction_msgs[msg.timestamp] = msg

    def on_top_down_segmentation_update(self, msg):
        self._top_down_segmentation_msgs[msg.timestamp] = msg

    def on_watermark(self, timestamp):
        tracking_msg = self._tracking_msgs.pop()
        segmentation_msg = self._top_down_segmentation_msgs.pop()
        prediction_msg = self._prediction_msgs.pop()

        display_img = np.uint8(segmentation_msg.frame)
        for obj in tracking_msg.obj_trajectories:
            self._draw_trajectory_on_img(obj, display_img, False)
        for obj in prediction_msg.predictions:
            display_img = self._draw_trajectory_on_img(obj, display_img, True)
        pylot.utils.add_timestamp(timestamp, display_img)
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
