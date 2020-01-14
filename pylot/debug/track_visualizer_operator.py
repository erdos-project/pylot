from collections import deque
import erdos


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
        self._past_colors = {'person': [255, 0, 0], 'vehicle': [128, 128, 0]}
        self._future_colors = {'person': [0, 0, 255], 'vehicle': [0, 255, 0]}
        # Dictionaries to store incoming data.
        self._tracking_msgs = deque()
        self._top_down_segmentation_msgs = deque()
        self._prediction_msgs = deque()
        self._top_down_camera_setup = top_down_camera_setup

    @staticmethod
    def connect(obstacle_tracking_stream, prediction_stream,
                segmented_camera_stream):
        return []

    def on_tracking_update(self, msg):
        self._tracking_msgs.append(msg)

    def on_prediction_update(self, msg):
        self._prediction_msgs.append(msg)

    def on_top_down_segmentation_update(self, msg):
        self._top_down_segmentation_msgs.append(msg)

    def on_watermark(self, timestamp):
        self._logger.debug('@{}: {} received watermark'.format(
            timestamp, self._name))
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
            if (0 <= point.x <= segmented_frame.width) and \
               (0 <= point.y <= segmented_frame.height):
                segmented_frame.draw_point(point, point_color)
