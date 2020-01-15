import copy
import numpy as np
from numpy.linalg import inv
import open3d as o3d
import os

from pylot.utils import Location, Transform, Vector2D


class PointCloud(object):
    def __init__(self, points, transform):
        """ Initializes the point cloud.

        Args:
            points: A list of pylot.utils.Location.
            transform: utils.Transform of the point cloud, relative to the
                ego-vehicle.
        """
        # Transform point cloud from lidar to camera coordinates.
        self.points = self._to_camera_coordinates(points)
        self.transform = transform

    @classmethod
    def from_carla_point_cloud(cls, carla_pc, transform):
        # Transform the raw_data into a point cloud.
        points = np.frombuffer(carla_pc.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 3), 3))
        point_cloud = [Location(x, y, z) for x, y, z in np.asarray(points)]
        return cls(point_cloud, transform)

    def _to_camera_coordinates(self, points):
        # Converts points in lidar coordinates to points in camera coordinates.
        # See CameraSetup in pylot/simulation/sensor_setup.py for coordinate
        # axis orientations.
        to_camera_transform = Transform(matrix=np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]))
        transformed_points = to_camera_transform.transform_points(points)
        return np.asarray([[loc.x, loc.y, loc.z]
                           for loc in transformed_points])

    def get_pixel_location(self, pixel, camera_setup):
        """ Gets the 3D world location from pixel coordinates.

        Args:
            pixel: A pylot.utils.Vector2D denoting pixel coordinates.
            camera_setup: The setup of the camera.
        Returns:
            A pylot.utils.Location of the 3D world location, or None if all the
            point cloud points are behind.
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
        p3d_locations = [
            Location(px, py, pz) for px, py, pz in np.asarray(p3d.transpose())
        ]
        # Convert from camera to unreal coordinates.
        to_world_transform = camera_setup.get_unreal_transform()
        camera_point_cloud = to_world_transform.transform_points(p3d_locations)
        return camera_point_cloud[0]

    def _get_closest_point_in_point_cloud(self, pixel):
        """ Finds the closest depth normalized point cloud point."""
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
