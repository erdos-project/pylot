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
import PIL.Image as Image

FLAGS = flags.FLAGS


class ChauffeurLoggerOp(Op):
    """
        TrackVisualizerOperator visualizes the past locations of agents
        on the top-down segmented image.
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
        super(ChauffeurLoggerOp, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self._flags = flags

        # Queues of incoming data.
        self._track_count = 0
        self._frame_count = 0

        # Get top-down camera.
        self._top_down_camera_setup = top_down_camera_setup

    @staticmethod
    def setup_streams(input_streams, top_down_stream_name):
        input_streams.filter(pylot.utils.is_tracking_stream).add_callback(
            ChauffeurLoggerOp.on_tracking_update)
        input_streams.filter(pylot.utils.is_segmented_camera_stream).filter_name(
            top_down_stream_name).add_callback(
            ChauffeurLoggerOp.on_top_down_segmentation_update)

        return []

    def on_tracking_update(self, msg):
        past_poses = np.zeros((self._top_down_camera_setup.height, self._top_down_camera_setup.width, 3), dtype=np.uint8)

        for obj in msg.obj_trajectories:
            # Intrinsic matrix of the top down segmentation camera.
            intrinsic_matrix = pylot.simulation.utils.create_intrinsic_matrix(
                self._top_down_camera_setup.width,
                self._top_down_camera_setup.height,
                fov=self._top_down_camera_setup.fov)
            # Convert to screen points.
            screen_points = pylot.simulation.utils.locations_3d_to_view(
                obj.trajectory,
                self._top_down_camera_setup.transform.matrix,
                intrinsic_matrix)

            # Draw trajectory points on segmented image.
            for point in screen_points:
                if (0 <= point.x <= FLAGS.carla_camera_image_width) and \
                        (0 <= point.y <= FLAGS.carla_camera_image_height):
                    cv2.circle(past_poses,
                               (int(point.x), int(point.y)),
                               3, (100, 100, 100), -1)

        # Save the past poses
        past_poses_img = Image.fromarray(past_poses)
        past_poses_img = past_poses_img.convert('RGB')
        past_poses_img.save('{}{}-{}.png'.format(self._flags.data_path, "past_poses", msg.timestamp.coordinates[0]))

    def on_top_down_segmentation_update(self, msg):
        top_down = np.uint8(transform_to_cityscapes_palette(msg.frame))

        # Save the segmented channels
        for k, v in pylot.utils.LABEL_2_PIXEL.items():
            mask = np.all(top_down == v, axis=-1)
            tmp = np.zeros(top_down.shape[:2])
            tmp[mask] = 1
            name = '{}{}-{}.png'.format(self._flags.data_path, k, msg.timestamp.coordinates[0])
            img = Image.fromarray(tmp)
            img = img.convert('RGB')
            img.save(name)

        top_down_img = Image.fromarray(top_down)
        top_down_img.save('{}{}-{}.png'.format(self._flags.data_path, "top_down_segmentation", msg.timestamp.coordinates[0]))