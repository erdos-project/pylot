import numpy as np
from numpy.linalg import inv
from numpy.matlib import repmat
import os
import pickle

import pylot.utils


class DepthFrame(object):
    def __init__(self, frame, camera_setup):
        """ Initializes a depth frame.

        Args:
            frame: A numpy array storring the depth frame.
            camera_setup: The camera setup used to generate this frame.
        """
        self.frame = frame
        self.camera_setup = camera_setup
        # Attribute used to cache the depth frame as a point cloud. We're doing
        # this because it is computationally expensive to transform a depth
        # frame to a point cloud.
        self._cached_point_cloud = None

    @classmethod
    def from_carla_frame(cls, frame, camera_setup):
        """ Creates a DepthFrame from a carla depth frame."""
        # Convert an image containing CARLA encoded depth-map to a 2D
        # array containing the depth value of each pixel normalized
        # between [0.0, 1.0]
        _frame = np.frombuffer(frame.raw_data, dtype=np.dtype("uint8"))
        _frame = np.reshape(_frame, (frame.height, frame.width, 4))
        frame = _frame.astype(np.float32)
        # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
        frame = np.dot(frame[:, :, :3], [65536.0, 256.0, 1.0])
        frame /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
        return cls(frame, camera_setup)

    def as_numpy_array(self):
        return self.frame

    def pixel_has_same_depth(self, x, y, z, threshold):
        return abs(self.frame[int(y)][int(x)] * 1000 - z) < threshold

    def as_point_cloud(self):
        """
        Converts a CARLA-encoded depth-map to a 1D array containing
        the 3D position of each pixel in world coordinates.

        See CameraSetup in pylot/simulation/sensor_setup.py for coordinate
        axis orientations.
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
        locations = [
            pylot.utils.Location(x, y, z)
            for x, y, z in np.asarray(np.transpose(p3d))
        ]
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
        return [
            self._cached_point_cloud[pixel.y * self.camera_setup.width +
                                     pixel.x] for pixel in pixels
        ]

    def save(self, timestamp, data_path, file_base):
        file_name = os.path.join(data_path,
                                 '{}-{}.pkl'.format(file_base, timestamp))
        pickle.dump(self.as_numpy_array(),
                    open(file_name, 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'DepthFrame(camera_setup: {})'.format(self.camera_setup)
