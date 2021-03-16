import copy
import os

import numpy as np
from numpy.linalg import inv

from pylot.drivers.sensor_setup import CameraSetup, LidarSetup
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
    def __init__(self, points, lidar_setup: LidarSetup):
        # Transform point cloud from lidar to camera coordinates.
        self._lidar_setup = lidar_setup
        self.global_points = copy.deepcopy(points)
        self.points = self._to_camera_coordinates(points)
        self.transform = lidar_setup.get_transform()

    @classmethod
    def from_simulator_point_cloud(cls, simulator_pc, lidar_setup: LidarSetup):
        """Creates a pylot point cloud from a simulator point cloud.

        Returns:
          :py:class:`.PointCloud`: A point cloud.
        """
        points = np.frombuffer(simulator_pc.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        if lidar_setup.legacy:
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
        else:
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            # Remove the intensity component of the point cloud.
            points = points[:, :3]
        return cls(points, lidar_setup)

    def merge(self, point_cloud):
        """Merges this point cloud with another point cloud.

        Note:
            The method modifies the point cloud inplace.
        """
        self.global_points = np.concatenate(
            (self.global_points, point_cloud.global_points), 0)
        self.points = np.concatenate((self.points, point_cloud.points), 0)

    def _to_camera_coordinates(self, points):
        # Converts points in lidar coordinates to points in camera coordinates.
        # See CameraSetup in pylot/drivers/sensor_setup.py for coordinate
        # axis orientations.
        #
        # The Velodyne coordinate space is defined as:
        # +x into the screen, +y to the left, and +z up.
        #
        # Note: We're using the ROS velodyne driver coordinate
        # system, not the one specified in the Velodyne manual.
        # Link to the ROS coordinate system:
        # https://www.ros.org/reps/rep-0103.html#axis-orientation
        if self._lidar_setup.lidar_type == 'sensor.lidar.ray_cast':
            if self._lidar_setup.legacy:
                # The legacy CARLA Lidar coordinate space is defined as:
                # +x to right, +y out of the screen, +z down.
                to_camera_transform = Transform(matrix=np.array(
                    [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]))
            else:
                # The latest coordiante space is the unreal space.
                to_camera_transform = Transform(matrix=np.array(
                    [[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]))
        elif self._lidar_setup.lidar_type == 'velodyne':
            to_camera_transform = Transform(matrix=np.array(
                [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]))
        else:
            raise ValueError('Unexpected lidar type {}'.format(
                self._lidar_setup.lidar_type))
        transformed_points = to_camera_transform.transform_points(points)
        return transformed_points

    def get_pixel_location(self, pixel, camera_setup: CameraSetup):
        """ Gets the 3D world location from pixel coordinates.

        Args:
            pixel (:py:class:`~pylot.utils.Vector2D`): Pixel coordinates.
            camera_setup (:py:class:`~pylot.drivers.sensor_setup.CameraSetup`):
                The setup of the camera with its transform in the world frame
                of reference.

        Returns:
            :py:class:`~pylot.utils.Location`: The 3D world location, or None
            if all the point cloud points are behind.
        """
        # Select only points that are in front.
        # Setting the threshold to 0.1 because super close points cause
        # floating point errors.
        fwd_points = self.points[np.where(self.points[:, 2] > 0.1)]
        if len(fwd_points) == 0:
            return None
        intrinsic_mat = camera_setup.get_intrinsic_matrix()
        # Project our 2D pixel location into 3D space, onto the z=1 plane.
        p3d = np.dot(inv(intrinsic_mat), np.array([[pixel.x], [pixel.y],
                                                   [1.0]]))

        if self._lidar_setup.lidar_type == 'sensor.lidar.ray_cast':
            location = PointCloud.get_closest_point_in_point_cloud(
                fwd_points, Vector2D(p3d[0], p3d[1]), normalized=True)
            # Use the depth from the retrieved location.
            p3d *= np.array([location.z])
            p3d = p3d.transpose()
            # Convert from camera to unreal coordinates if the lidar type is
            # sensor.lidar.ray_cast
            to_world_transform = camera_setup.get_unreal_transform()
            camera_point_cloud = to_world_transform.transform_points(p3d)[0]
            pixel_location = Location(camera_point_cloud[0],
                                      camera_point_cloud[1],
                                      camera_point_cloud[2])
        elif self._lidar_setup.lidar_type == 'velodyne':
            location = PointCloud.get_closest_point_in_point_cloud(
                fwd_points, Vector2D(p3d[0], p3d[1]), normalized=False)
            # Use the depth from the retrieved location.
            p3d[2] = location.z
            p3d = p3d.transpose()
            pixel_location = Location(p3d[0, 0], p3d[0, 1], p3d[0, 2])
        return pixel_location

    @staticmethod
    def get_closest_point_in_point_cloud(fwd_points,
                                         pixel,
                                         normalized: bool = False):
        """Finds the closest point in the point cloud to the given point.

        Args:
            pixel (:py:class:`~pylot.utils.Vector2D`): Camera coordinates.

        Returns:
            :py:class:`~pylot.utils.Location`: Closest point cloud point.
        """
        # Select x and y.
        pc_xy = fwd_points[:, 0:2]
        # Create an array from the x, y coordinates of the point.
        xy = np.array([pixel.x, pixel.y]).transpose()

        # Compute distance
        if normalized:
            # Select z
            pc_z = fwd_points[:, 2]
            # Divize x, y by z
            normalized_pc = pc_xy / pc_z[:, None]
            dist = np.sum((normalized_pc - xy)**2, axis=1)
        else:
            dist = np.sum((pc_xy - xy)**2, axis=1)

        # Select index of the closest point.
        closest_index = np.argmin(dist)

        # Return the closest point.
        return Location(fwd_points[closest_index][0],
                        fwd_points[closest_index][1],
                        fwd_points[closest_index][2])

    def save(self, timestamp: int, data_path: str, file_base: str):
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

    def visualize(self, pygame_display, display_width: int,
                  display_height: int):
        """Visualizes the point cloud on a pygame display."""
        import pygame
        # Transform point cloud to top down view.
        lidar_data = np.array(self.global_points[:, :2])
        lidar_data *= (min(display_width, display_height) /
                       (2.0 * self._lidar_setup.get_range_in_meters()))
        lidar_data += (0.5 * display_width, 0.5 * display_height)
        lidar_data = np.fabs(lidar_data)
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (display_width, display_height, 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
        pygame.surfarray.blit_array(pygame_display, lidar_img)
        pygame.display.flip()

    def __repr__(self):
        return 'PointCloud(lidar setup: {}, points: {})'.format(
            self._lidar_setup, self.points)

    def __str__(self):
        return 'PointCloud(transform: {}, number of points: {})'.format(
            self.transform, len(self.points))
