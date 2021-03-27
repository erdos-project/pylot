import copy
import os
import pickle

import numpy as np
from numpy.linalg import inv
from numpy.matlib import repmat

import pylot.utils


class DepthFrame(object):
    """Class that stores depth frames.

    Args:
        frame: A numpy array storing the depth frame.
        camera_setup (:py:class:`~pylot.drivers.sensor_setup.DepthCameraSetup`):  # noqa: E501
            The camera setup used by the sensor that generated this frame.
        original_frame: A numpy array storing the RGB encoded depth image.

    Attributes:
        frame: A numpy array storing the depth frame.
        camera_setup (:py:class:`~pylot.drivers.sensor_setup.DepthCameraSetup`):
            The camera setup used by the sensor that generated this frame.
        original_frame: A numpy array storing the RGB encoded depth image.
    """
    def __init__(self, frame, camera_setup, original_frame=None):
        self.frame = frame
        self.camera_setup = camera_setup
        self.original_frame = original_frame
        # Attribute used to cache the depth frame as a point cloud. We're doing
        # this because it is computationally expensive to transform a depth
        # frame to a point cloud.
        self._cached_point_cloud = None

    @classmethod
    def from_simulator_frame(cls,
                             frame,
                             camera_setup,
                             save_original_frame=False):
        """Creates a pylot depth frame from a simulator depth frame.

        Args:
            frame: An image instance containing the depth image.
            camera_setup: The setup of the depth camera.
            save_original_frame: True if the original RGB image needs to be
                saved.

        Returns:
            :py:class:`.DepthFrame`: A depth frame.
        """
        original_frame = None
        # Convert an image containing simulator encoded depth-map to a 2D
        # array containing the depth value of each pixel normalized
        # between [0.0, 1.0]
        _frame = np.frombuffer(frame.raw_data, dtype=np.dtype("uint8"))
        _frame = np.reshape(_frame, (frame.height, frame.width, 4))
        frame = _frame.astype(np.float32)
        if save_original_frame:
            original_frame = copy.deepcopy(frame[:, :, :3])
        # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
        frame = np.dot(frame[:, :, :3], [65536.0, 256.0, 1.0])
        frame /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
        return cls(frame, camera_setup, original_frame)

    def as_numpy_array(self):
        """Returns the depth frame as a numpy array."""
        return self.frame

    def as_point_cloud(self):
        """Converts the depth frame to a 1D array containing the 3D
        position of each pixel in world coordinates.

        See :py:class:`~pylot.drivers.sensor_setup.CameraSetup` for
        coordinate axis orientations.
        """
        far = 1000.0  # max depth in meters.
        intrinsic_mat = self.camera_setup.get_intrinsic_matrix()
        width, height = self.camera_setup.width, self.camera_setup.height
        # 2d pixel coordinates
        pixel_length = width * height
        u_coord = repmat(np.r_[0:width:1], height, 1).reshape(pixel_length)
        v_coord = repmat(np.c_[0:height:1], 1, width).reshape(pixel_length)
        normalized_depth = np.reshape(self.frame, pixel_length)

        # p2d = [u,v,1]
        p2d = np.array([u_coord, v_coord, np.ones_like(u_coord)])

        # P = [X,Y,Z]
        p3d = np.dot(inv(intrinsic_mat), p2d)
        p3d *= normalized_depth * far

        # [[X1,Y1,Z1],[X2,Y2,Z2], ... [Xn,Yn,Zn]]
        locations = np.asarray(np.transpose(p3d))
        # Transform the points in 3D world coordinates.
        to_world_transform = self.camera_setup.get_unreal_transform()
        point_cloud = to_world_transform.transform_points(locations)
        return point_cloud

    def get_pixel_locations(self, pixels):
        """ Gets the 3D world locations from pixel coordinates.

        Args:
            pixels: List of pylot.utils.Vector2D pixel coordinates.
        Returns:
            List of pylot.utils.Locations
        """
        if self._cached_point_cloud is None:
            self._cached_point_cloud = self.as_point_cloud()
        pixel_locations = [
            self._cached_point_cloud[pixel.y * self.camera_setup.width +
                                     pixel.x] for pixel in pixels
        ]
        return [
            pylot.utils.Location(loc[0], loc[1], loc[2])
            for loc in pixel_locations
        ]

    def pixel_has_same_depth(self, x, y, z: float, threshold: float) -> bool:
        """Checks if the depth of pixel (y,x) is within threshold of z."""
        return abs(self.frame[int(y)][int(x)] * 1000 - z) < threshold

    def resize(self, width: int, height: int):
        """Resizes the frame."""
        import cv2
        self.camera_setup.set_resolution(width, height)
        self.frame = cv2.resize(self.frame,
                                dsize=(width, height),
                                interpolation=cv2.INTER_NEAREST)

    def visualize(self, pygame_display, timestamp=None):
        """Visualizes the frame on a pygame display."""
        if self.original_frame is not None:
            import pygame
            image_np = self.original_frame
            image_np = image_np[:, :, ::-1]
            image_np = np.transpose(image_np, (1, 0, 2))
            pygame.surfarray.blit_array(pygame_display, image_np)
            pygame.display.flip()

    def save(self, timestamp: int, data_path: str, file_base: str):
        """Saves the depth frame to a file.

        Args:
            timestamp (:obj:`int`): Timestamp associated with the depth frame.
            data_path (:obj:`str`): Path where to save the depth frame.
            file_base (:obj:`str`): Base name of the file.
        """
        file_name = os.path.join(data_path,
                                 '{}-{}.pkl'.format(file_base, timestamp))
        pickle.dump(self.as_numpy_array(),
                    open(file_name, 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)

    def __repr__(self):
        return 'DepthFrame(camera_setup: {}, frame: {})'.format(
            self.camera_setup, self.frame)

    def __str__(self):
        return 'DepthFrame(camera_setup: {})'.format(self.camera_setup)
