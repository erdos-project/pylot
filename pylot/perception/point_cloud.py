import copy
import numpy as np
from numpy.linalg import inv
import os

from pylot.utils import Location, Transform, Vector2D


class PointCloud(object):
    """Class that stores points clouds.

    Args:
        points: A (number of points) by 3 numpy array, where each row is
            the (x, y, z) coordinates of a point.
        transform (:py:class:`~pylot.utils.Transform`): Transform of the
            point cloud, relative to the ego-vehicle.

    Attributes:
        points: A (number of points) by 3 numpy array, where each row is
            the (x, y, z) coordinates of a point.
        transform (:py:class:`~pylot.utils.Transform`): Transform of the
            point cloud, relative to the ego-vehicle.
    """
    def __init__(self, points, transform):
        # Transform point cloud from lidar to camera coordinates.
        self.points = self._to_camera_coordinates(points)
        self.transform = transform

    @classmethod
    def from_carla_point_cloud(cls, carla_pc, transform):
        """Creates a pylot point cloud from a carla point cloud.

        Returns:
          :py:class:`.PointCloud`: A point cloud.
        """
        points = np.frombuffer(carla_pc.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 3), 3))
        return cls(points, transform)

    def _to_camera_coordinates(self, points):
        # Converts points in lidar coordinates to points in camera coordinates.
        # See CameraSetup in pylot/drivers/sensor_setup.py for coordinate
        # axis orientations.
        to_camera_transform = Transform(matrix=np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]))
        transformed_points = to_camera_transform.transform_points(points)
        return transformed_points

    def get_pixel_location(self, pixel, camera_setup):
        """ Gets the 3D world location from pixel coordinates.

        Args:
            pixel (:py:class:`~pylot.utils.Vector2D`): Pixel coordinates.
            camera_setup (:py:class:`~pylot.drivers.sensor_setup.CameraSetup`):
                The setup of the camera.
        Returns:
            :py:class:`~pylot.utils.Location`: The 3D world location, or None
            if all the point cloud points are behind.
        """
        if len(self.points) == 0:
            return None
        intrinsic_mat = camera_setup.get_intrinsic_matrix()
        # Project our 2D pixel location into 3D space, onto the z=1 plane.
        p3d = np.dot(inv(intrinsic_mat), np.array([[pixel.x], [pixel.y],
                                                   [1.0]]))
        location = self._get_closest_point_in_point_cloud(
            Vector2D(p3d[0], p3d[1]))
        # Normalize our point to have the same depth as our closest point.
        p3d *= np.array([location.z])
        p3d = p3d.transpose()
        # Convert from camera to unreal coordinates.
        to_world_transform = camera_setup.get_unreal_transform()
        camera_point_cloud = to_world_transform.transform_points(p3d)[0]
        pixel_location = Location(camera_point_cloud[0],
                                  camera_point_cloud[1],
                                  camera_point_cloud[2])
        return pixel_location

    def _get_closest_point_in_point_cloud(self, pixel):
        """Finds the closest depth normalized point cloud point.

        Args:
            pixel (:py:class:`~pylot.utils.Vector2D`): Camera coordinates.

        Returns:
            :py:class:`~pylot.utils.Location`: Closest point cloud point.
        """
        # Select only points that are in front.
        fwd_points = self.points[np.where(self.points[:, 2] > 0.0)]
        # Select x and y.
        pc_xy = fwd_points[:, 0:2]
        # Select z
        pc_z = fwd_points[:, 2]
        # Divize x, y by z
        normalized_pc = pc_xy / pc_z[:, None]
        xy = np.array([pixel.x, pixel.y]).transpose()
        # Compute distance
        dist = np.sum((normalized_pc - xy)**2, axis=1)
        # Select index of the closest point.
        closest_index = np.argmin(dist)
        # Return the closest point.
        return Location(fwd_points[closest_index][0],
                        fwd_points[closest_index][1],
                        fwd_points[closest_index][2])

    def save(self, timestamp, data_path, file_base):
        """Saves the point cloud to a file.

        Args:
            timestamp (:obj:`int`): Timestamp associated with the point cloud.
            data_path (:obj:`str`): Path where to save the point cloud.
            file_base (:obj:`str`): Base name of the file.
        """
        import open3d as o3d
        file_name = os.path.join(data_path,
                                 '{}-{}.ply'.format(file_base, timestamp))
        pcd = o3d.PointCloud()
        pcd.points = o3d.Vector3dVector(self.points)
        o3d.write_point_cloud(file_name, pcd)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'PointCloud(transform: {}, number of points: {})'.format(
            self.transform, len(self.points))
