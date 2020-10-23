import numpy as np

from pylot.drivers.sensor_setup import CameraSetup, LidarSetup
from pylot.perception.depth_frame import DepthFrame
from pylot.perception.point_cloud import PointCloud
from pylot.utils import Location, Rotation, Transform, Vector2D

import pytest

# Depth Frame Tests


@pytest.mark.parametrize("x, y, z, threshold, expected",
                         [(1, 0, 150, 100, True), (1, 0, 150, 25, False),
                          (2, 1, 300, 250, True), (2, 1, 300, 150, False)])
def test_pixel_has_same_depth(x, y, z, threshold, expected):
    """Tests if the pixel at (x,y) has a depth within the specified
       threshold of z."""
    camera_setup = None
    depth_frame = DepthFrame([[0, 0.1, 0], [0, 0, 0.5]], camera_setup)
    assert depth_frame.pixel_has_same_depth(x, y, z, threshold) is expected, \
        "Depth thresholding did not work correctly."


@pytest.mark.parametrize("depth_frame, expected", [
    (np.array([[0.4, 0.3], [0.2, 0.1]]), \
        np.array([[400, -400, 400], [300, 300, 300], \
                  [200, -200, -200], [100, 100, -100]])),
    (np.array([[0.1, 0.2]]),
        np.array([[100, -100, 0], [200, 200, 0]])),
    (0.01 * np.ones((3,3)), \
        np.array([[10, -10, 10], [10, 0, 10], [10, 10, 10],
                  [10, -10, 0], [10, 0, 0], [10, 10, 0],
                  [10, -10, -10], [10, 0, -10], [10, 10, -10]]))
])
def test_depth_to_point_cloud(depth_frame, expected):
    height, width = depth_frame.shape
    camera_setup = CameraSetup('test_setup',
                               'sensor.camera.depth',
                               width,
                               height,
                               Transform(location=Location(0, 0, 0),
                                         rotation=Rotation(0, 0, 0)),
                               fov=90)
    depth_frame = DepthFrame(depth_frame, camera_setup)
    # Resulting unreal coordinates.
    point_cloud = depth_frame.as_point_cloud()
    for i in range(width * height):
        assert np.isclose(point_cloud[i][0], expected[i][0]), \
            'Returned x value is not the same as expected'
        assert np.isclose(point_cloud[i][1], expected[i][1]), \
            'Returned y value is not the same as expected'
        assert np.isclose(point_cloud[i][2], expected[i][2]), \
            'Returned z value is not the same as expected'


@pytest.mark.parametrize(
    "depth_frame, expected",
    [(np.array([[0.1, 0.1]]), np.array([[110, -80, 30], [110, 120, 30]]))])
def test_depth_to_point_cloud_nonzero_camera_loc(depth_frame, expected):
    height, width = depth_frame.shape
    camera_setup = CameraSetup('test_setup',
                               'sensor.camera.depth',
                               width,
                               height,
                               Transform(location=Location(10, 20, 30),
                                         rotation=Rotation(0, 0, 0)),
                               fov=90)
    depth_frame = DepthFrame(depth_frame, camera_setup)
    # Resulting unreal coordinates.
    point_cloud = depth_frame.as_point_cloud()
    print(point_cloud)
    for i in range(width * height):
        assert np.isclose(point_cloud[i][0], expected[i][0]), 'Returned x '
        'value is not the same as expected'
        assert np.isclose(point_cloud[i][1], expected[i][1]), 'Returned y '
        'value is not the same as expected'
        assert np.isclose(point_cloud[i][2], expected[i][2]), 'Returned z '
        'value is not the same as expected'


@pytest.mark.parametrize("depth_frame, pixels, expected", [
    (np.array([[0.4, 0.3], [0.2, 0.1]]), \
     [Vector2D(0,1), Vector2D(1,0)], \
     [Location(200, -200, -200), Location(300, 300, 300)])
])
def test_get_pixel_locations(depth_frame, pixels, expected):
    height, width = depth_frame.shape
    camera_setup = CameraSetup('test_setup',
                               'sensor.camera.depth',
                               width,
                               height,
                               Transform(location=Location(0, 0, 0),
                                         rotation=Rotation(0, 0, 0)),
                               fov=90)
    depth_frame = DepthFrame(depth_frame, camera_setup)
    locations = depth_frame.get_pixel_locations(pixels)
    for i in range(len(pixels)):
        assert np.isclose(locations[i].x, expected[i].x), 'Returned x '
        'value is not the same as expected'
        assert np.isclose(locations[i].y, expected[i].y), 'Returned y '
        'value is not the same as expected'
        assert np.isclose(locations[i].z, expected[i].z), 'Returned z '
        'value is not the same as expected'

# Point Cloud Tests


@pytest.mark.parametrize(
    "points, expected",
    [(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 2, 3]
                ]), [[1, 0, 0], [0, 0, -1], [0, 1, 0], [1, 3, -2]])])
def test_initialize_point_cloud(points, expected):
    lidar_setup = LidarSetup('lidar', 'sensor.lidar.ray_cast',
                             Transform(Location(), Rotation()))
    point_cloud = PointCloud(points, lidar_setup)
    for i in range(len(expected)):
        assert all(np.isclose(point_cloud.points[i], expected[i]))


@pytest.mark.parametrize(
    "lidar_points, pixel, expected",
    [

        # In this test for getting pixel locations, lidar points
        # are first converted to camera coordinates, when constructing the
        # PointCloud. Then, get_pixel_location finds the closest point in
        # the point cloud, normalizes our query to have the same depth as
        # this closest point, and converts to unreal coordinates.
        #
        # For example, in the first test case, the lidar points in camera
        # coordinates are (-1,0,1),(1,0,1), and the query pixel is
        # (-0.5, 0, 1). The closest lidar point is (-1,0,1), so the
        # normalization step has no effect. Finally, converting the query
        # pixel to unreal coordinates gives (1, -0.5, 0).

        # Lidar Points are left middle and right middle, same depth.
        (np.array([[-1, -1, 0], [1, -1, 0]]), Vector2D(
            200, 300), Location(1, -0.5, 0)),
        (np.array([[-1, -1, 0], [1, -1, 0]]), Vector2D(
            600, 300), Location(1, 0.5, 0)),
        # Lidar points are left middle and right middle, different depth.
        (np.array([[-2, -2, 0], [1, -1, 0]]), Vector2D(
            200, 300), Location(2, -1, 0)),
        (np.array([[-2, -2, 0], [1, -1, 0]]), Vector2D(
            600, 300), Location(1, 0.5, 0)),
        # Lidar points are top left and bottom right, same depth.
        (np.array([[-2, -2, -1.5], [2, -2, 1.5]]), Vector2D(
            200, 150), Location(2, -1, 0.75)),
    ])
def test_point_cloud_get_pixel_location(lidar_points, pixel, expected):
    camera_setup = CameraSetup(
        'test_setup',
        'sensor.camera.depth',
        801,
        601,  # width, height
        Transform(location=Location(0, 0, 0), rotation=Rotation(0, 0, 0)),
        fov=90)
    lidar_setup = LidarSetup('lidar', 'sensor.lidar.ray_cast',
                             Transform(Location(), Rotation()))
    point_cloud = PointCloud(lidar_points, lidar_setup)
    location = point_cloud.get_pixel_location(pixel, camera_setup)
    assert np.isclose(location.x,
                      expected.x), 'Returned x value is not the same '
    'as expected'
    assert np.isclose(location.y,
                      expected.y), 'Returned y value is not the same '
    'as expected'
    assert np.isclose(location.z,
                      expected.z), 'Returned z value is not the same '
    'as expected'
