"""This module implements an operator that visualizes agent predictions."""

from collections import deque
from pylot.perception.detection.obstacle import BoundingBox3D
import erdos


class TrackVisualizerOperator(erdos.Operator):
    """Visualizes the past and predicted future locations of agents.

    The visualization is shown on top-down segmented images.

    Args:
        obstacles_tracking_stream (:py:class:`erdos.ReadStream`): Stream on
            which :py:class:`~pylot.perception.messages.ObstacleTrajectoriesMessage`
            are received.
        prediction_stream: The stream on which
            :py:class:`~pylot.prediction.messages.PredictionMessage` are
            received.
        segmented_camera_stream: The stream on which top-down
            :py:class:`~pylot.perception.messages.SegmentedFrameMessage` are
            received.
        flags (absl.flags): Object to be used to access absl flags.

    Attributes:
        _logger (:obj:`logging.Logger`): Instance to be used to log messages.
        _flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, obstacle_tracking_stream, prediction_stream,
                 segmented_camera_stream, flags):
        obstacle_tracking_stream.add_callback(self.on_tracking_update)
        prediction_stream.add_callback(self.on_prediction_update)
        segmented_camera_stream.add_callback(
            self.on_top_down_segmentation_update)
        erdos.add_watermark_callback([
            obstacle_tracking_stream, prediction_stream,
            segmented_camera_stream
        ], [], self.on_watermark)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags
        self._past_colors = {'person': [255, 0, 0], 'vehicle': [128, 128, 0]}
        self._future_colors = {'person': [0, 0, 255], 'vehicle': [0, 255, 0]}
        # Dictionaries to store incoming data.
        self._tracking_msgs = deque()
        self._top_down_segmentation_msgs = deque()
        self._prediction_msgs = deque()

    @staticmethod
    def connect(obstacle_tracking_stream, prediction_stream,
                segmented_camera_stream):
        return []

    def on_tracking_update(self, msg):
        """Invoked when a msg on the obstacles trajectories stream is received.

        Args:
            msg (:py:class:`~pylot.perception.messages.ObstacleTrajectoriesMessage`):
                Received message.
        """
        self._tracking_msgs.append(msg)

    def on_prediction_update(self, msg):
        """Invoked when a msg on the prediction stream is received.

        Args:
            msg (:py:class:`~pylot.prediction.messages.PredictionMessage`):
                Received message.
        """
        self._prediction_msgs.append(msg)

    def on_top_down_segmentation_update(self, msg):
        """Invoked when a msg on the segmented camera stream is received.

        Args:
            msg (:py:class:`~pylot.prediction.messages.SegmentedFrameMessage`):
                Received message.
        """
        self._top_down_segmentation_msgs.append(msg)

    def on_watermark(self, timestamp):
        """Invoked when all input streams have received a watermark.

        Args:
            timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of
                the watermark.
        """
        self._logger.debug('@{}: {} received watermark'.format(
            timestamp, self.config.name))
        tracking_msg = self._tracking_msgs.popleft()
        segmentation_msg = self._top_down_segmentation_msgs.popleft()
        prediction_msg = self._prediction_msgs.popleft()

        # Transform segmented frame to cityscapes so that the drawn points
        # maintain their color.
        segmentation_msg.frame.transform_to_cityscapes()

        for obstacle in tracking_msg.obstacle_trajectories:
            self._draw_trajectory_on_img(obstacle, segmentation_msg.frame,
                                         False)
        for obstacle in prediction_msg.predictions:
            self._draw_trajectory_on_img(obstacle, segmentation_msg.frame,
                                         True)
        segmentation_msg.frame.visualize('track_visualizer', timestamp)

    def _draw_trajectory_on_img(self, obstacle, segmented_frame, predict):
        # Intrinsic and extrinsic matrix of the top down segmentation camera.
        extrinsic_matrix = segmented_frame.camera_setup.get_extrinsic_matrix()
        intrinsic_matrix = segmented_frame.camera_setup.get_intrinsic_matrix()

        # Set the color of drawing.
        if predict:
            point_color = self._future_colors[obstacle.label]
        else:
            point_color = self._past_colors[obstacle.label]

        # Obstacle trajectory points.
        screen_points = []
        for transform in obstacle.trajectory:
            screen_point = transform.location.to_camera_view(
                extrinsic_matrix, intrinsic_matrix)
            screen_points.append(screen_point)

        # Draw trajectory on segmented image.
        for point in screen_points:
            segmented_frame.draw_point(point, point_color)

        # Obstacle bounding box.
        if isinstance(obstacle.bounding_box, BoundingBox3D):
            start_location = obstacle.bounding_box.transform.location - \
                obstacle.bounding_box.extent
            end_location = obstacle.bounding_box.transform.location + \
                obstacle.bounding_box.extent
            start_points = []
            end_points = []
            for transform in obstacle.trajectory:
                start_transform = transform.transform_points([start_location])
                end_transform = transform.transform_points([end_location])
                start_point = start_transform[0]\
                    .to_camera_view(extrinsic_matrix, intrinsic_matrix)
                end_point = end_transform[0]\
                    .to_camera_view(extrinsic_matrix, intrinsic_matrix)
                start_points.append(start_point)
                end_points.append(end_point)

            # Draw bounding box on segmented image.
            for start_point, end_point in \
                    zip(start_points, end_points):
                if self._in_frame(start_point, segmented_frame) or \
                        self._in_frame(end_point, segmented_frame):
                    segmented_frame.draw_box(start_point, end_point,
                                             point_color)

    @staticmethod
    def _in_frame(point, segmented_frame):
        """ Return if the point is in the segmented frame."""
        return (0 <= point.x <= segmented_frame.camera_setup.width) and \
               (0 <= point.y <= segmented_frame.camera_setup.height)
