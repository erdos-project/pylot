from absl import flags
from collections import deque

import cv2
import numpy as np
import threading
import carla
import json

# ERDOS specific imports.
from erdos.op import Op
from erdos.utils import setup_logging

# Pylot specific imports.
from pylot.perception.segmentation.utils import transform_to_cityscapes_palette, LABEL_2_PIXEL
import pylot.utils
import pylot.simulation.carla_utils
import PIL.Image as Image


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
        self._buffer_length = 10

        self._ground_vehicle_id = None
        self._waypoints = None
        self._global_transforms = deque(maxlen=self._buffer_length)  # holds history of global transforms at each timestep

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
        input_streams.filter(pylot.utils.is_ground_vehicle_id_stream).add_callback(
            ChauffeurLoggerOp.on_ground_vehicle_id_update)
        return []

    def on_tracking_update(self, msg):
        past_poses = np.zeros((self._top_down_camera_setup.height, self._top_down_camera_setup.width, 3), dtype=np.uint8)
        future_poses = np.zeros((self._top_down_camera_setup.height, self._top_down_camera_setup.width, 3),
                                dtype=np.uint8)

        # Intrinsic matrix of the top down segmentation camera.
        intrinsic_matrix = pylot.simulation.utils.create_intrinsic_matrix(
            self._top_down_camera_setup.width,
            self._top_down_camera_setup.height,
            fov=self._top_down_camera_setup.fov)

        rotation = pylot.simulation.utils.Rotation(0, 0, 0)
        current_transform = None
        for obj in msg.obj_trajectories:
            if obj.obj_id == -1:
                current_transform = obj.trajectory
                continue

            # Convert to screen points.
            screen_points = pylot.simulation.utils.locations_3d_to_view(
                obj.trajectory,
                self._top_down_camera_setup.transform.matrix,
                intrinsic_matrix)

            # Keep track of ground vehicle waypoints
            if obj.obj_id == self._ground_vehicle_id:
                self._waypoints = obj.trajectory

            # Draw trajectory points on segmented image.
            for point in screen_points:
                if (0 <= point.x <= self._flags.carla_camera_image_width) and \
                        (0 <= point.y <= self._flags.carla_camera_image_height):
                    r = 3
                    if obj.obj_id == self._ground_vehicle_id:
                        r = 10
                    cv2.circle(past_poses,
                               (int(point.x), int(point.y)),
                               r, (100, 100, 100), -1)

        # Make sure transforms deque is full
        while len(self._global_transforms) != self._buffer_length:
            self._global_transforms.append(current_transform)
        previous_transform = self._global_transforms.popleft()
        self._global_transforms.append(current_transform)

        # Transform to previous and back to current frame
        new_transform = current_transform * previous_transform.inverse_transform()
        self._waypoints = [
            (new_transform * pylot.simulation.utils.Transform(pos=wp, rotation=rotation)).location
            for wp in self._waypoints
        ]

        # Center first point at 0, 0
        center_transform = pylot.simulation.utils.Transform(
            pos=self._waypoints[0],
            rotation=rotation
        ).inverse_transform()
        self._waypoints = [
            (center_transform * pylot.simulation.utils.Transform(pos=wp, rotation=rotation)).location
            for wp in self._waypoints
        ]

        # Convert to screen points
        screen_waypoints = pylot.simulation.utils.locations_3d_to_view(
                self._waypoints,
                self._top_down_camera_setup.transform.matrix,
                intrinsic_matrix)

        # Draw screen points
        for point in screen_waypoints:
            cv2.circle(future_poses,
                       (int(point.x), int(point.y)),
                       10, (100, 100, 100), -1)

        # Save screen points
        future_poses_img = Image.fromarray(future_poses)
        future_poses_img = future_poses_img.convert('RGB')
        future_poses_img.save('{}{}-{}.png'.format(self._flags.data_path, "future_poses",
                                                   msg.timestamp.coordinates[
                                                       0] - len(self._waypoints) * 100))

        # Log future poses
        waypoints = [str(wp) for wp in self._waypoints]
        file_name = '{}waypoints-{}.json'.format(self._flags.data_path, msg.timestamp.coordinates[0] - len(self._waypoints) * 100)
        with open(file_name, 'w') as outfile:
            json.dump(waypoints, outfile)

        # Save the past poses
        past_poses_img = Image.fromarray(past_poses)
        past_poses_img = past_poses_img.convert('RGB')
        past_poses_img.save('{}{}-{}.png'.format(self._flags.data_path, "past_poses", msg.timestamp.coordinates[0]))

    def on_top_down_segmentation_update(self, msg):
        top_down = np.uint8(transform_to_cityscapes_palette(msg.frame))

        # Save the segmented channels
        for k, v in LABEL_2_PIXEL.items():
            mask = np.all(top_down == v, axis=-1)
            tmp = np.zeros(top_down.shape[:2])
            tmp[mask] = 1
            name = '{}{}-{}.png'.format(self._flags.data_path, k, msg.timestamp.coordinates[0])
            img = Image.fromarray(tmp)
            img = img.convert('RGB')
            img.save(name)

        top_down_img = Image.fromarray(top_down)
        top_down_img.save('{}{}-{}.png'.format(self._flags.data_path, "top_down_segmentation", msg.timestamp.coordinates[0]))

    def on_ground_vehicle_id_update(self, msg):
        self._ground_vehicle_id = msg.data