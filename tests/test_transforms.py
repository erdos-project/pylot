import pytest

from collections import namedtuple
import numpy as np

import carla
from pylot.utils import Location, Rotation, Transform, Vector2D
from pylot.perception.depth_frame import DepthFrame
from pylot.perception.point_cloud import PointCloud
from pylot.simulation.sensor_setup import CameraSetup

## Location Tests


def test_empty_location():
    """ Test that an empty Location is initializes at (0, 0, 0) """
    empty_location = Location()
    assert np.isclose(empty_location.x, 0), "X value is not zero"
    assert np.isclose(empty_location.y, 0), "Y value is not zero"
    assert np.isclose(empty_location.z, 0), "Z value is not zero"


@pytest.mark.parametrize("x, y, z", [(10, 20, 30), (-10, -20, -30)])
def test_location_creation(x, y, z):
    """ Test that the Location is initialized correctly. """
    location = Location(x, y, z)
    assert np.isclose(location.x, x), "X values are not the same."
    assert np.isclose(location.y, y), "Y values are not the same."
    assert np.isclose(location.z, z), "Z values are not the same."


@pytest.mark.parametrize("x, y, z", [(10, 20, 30), (-10, -20, -30)])
def test_location_from_carla(x, y, z):
    """ Test that the Location is initialized correctly from a carla.Location
    instance """
    location = Location.from_carla_location(carla.Location(x, y, z))
    assert np.isclose(location.x, x), "X values are not the same."
    assert np.isclose(location.y, y), "Y values are not the same."
    assert np.isclose(location.z, z), "Z values are not the same."


def test_negative_location_from_carla():
    """ Test that Location throws a ValueError if incorrect carla_loc argument
    is passed. """
    DummyType = namedtuple("DummyType", "x, y, z")
    dummy_instance = DummyType(10, 20, 30)
    with pytest.raises(ValueError):
        Location.from_carla_location(dummy_instance)


@pytest.mark.parametrize("point_a, point_b, expected",
                         [((1, 2, 3), (1, 2, 3), 0),
                          ((10, 20, 30), (40, 50, 60), 51.961524227)])
def test_distance(point_a, point_b, expected):
    """ Test the distance computed between two points is the same as expected"""
    location_a, location_b = Location(*point_a), Location(*point_b)
    assert np.isclose(location_a.distance(location_b), expected), "Distance "
    "between point_a and point_b is not the same as the expected distance."
    assert np.isclose(location_b.distance(location_a), expected), "Distance "
    "between point_b and point_a is not the same as the expected distance."


def test_as_carla_location():
    """ Test the as_carla_location instance method of Location """
    location = Location(x=1, y=2, z=3)
    carla_location = location.as_carla_location()
    assert isinstance(carla_location, carla.Location), "Returned instance is "
    "not of the type carla.Location"
    assert np.isclose(carla_location.x, location.x), "Returned instance x "
    "value is not the same as the one in location."
    assert np.isclose(carla_location.y, location.y), "Returned instance y "
    "value is not the same as the one in location."
    assert np.isclose(carla_location.z, location.z), "Returned instance z "
    "value is not the same as the one in location."


def test_as_numpy_array():
    """ Test the as_carla_location instance method of Location """
    location = Location(x=1, y=2, z=3)
    np_array = location.as_numpy_array()
    assert isinstance(np_array, np.ndarray), "Returned instance is "
    "not of the type np.ndarray"
    assert all(np.isclose(np_array, [1, 2, 3])), "Returned instance x, y, z "
    "values are not the same as the one in location."


@pytest.mark.parametrize("point_a, point_b, expected", [
    ((1, 2, 3), (1, 2, 3), (2, 4, 6)),
    ((1, 2, 3), (-1, -2, -3), (0, 0, 0)),
])
def test_addition(point_a, point_b, expected):
    """ Test the addition of the two locations. """
    location_a, location_b = Location(*point_a), Location(*point_b)
    sum_location = location_a + location_b
    assert isinstance(sum_location, Location), "The sum was not of the type "
    "Location"
    assert np.isclose(expected[0], sum_location.x), "The x value of the sum "
    "was not the same as the expected value."
    assert np.isclose(expected[1], sum_location.y), "The y value of the sum "
    "was not the same as the expected value."
    assert np.isclose(expected[2], sum_location.z), "The z value of the sum "
    "was not the same as the expected value."


@pytest.mark.parametrize("point_a, point_b, expected", [
    ((1, 2, 3), (1, 2, 3), (0, 0, 0)),
    ((1, 2, 3), (-1, -2, -3), (2, 4, 6)),
])
def test_subtraction(point_a, point_b, expected):
    """ Test the addition of the two locations. """
    location_a, location_b = Location(*point_a), Location(*point_b)
    diff_location = location_a - location_b
    assert isinstance(diff_location, Location), "The sum was not of the type "
    "Location"
    assert np.isclose(expected[0], diff_location.x), "The x value of the sum "
    "was not the same as the expected value."
    assert np.isclose(expected[1], diff_location.y), "The y value of the sum "
    "was not the same as the expected value."
    assert np.isclose(expected[2], diff_location.z), "The z value of the sum "
    "was not the same as the expected value."


# TODO (Sukrit):: Write tests for to_camera_view after the CameraSetup tests.

## Rotation Tests


def test_empty_rotation():
    """ Test that an empty Location is initializes at (0, 0, 0) """
    empty_rotation = Rotation()
    assert np.isclose(empty_rotation.pitch, 0), "pitch value is not zero"
    assert np.isclose(empty_rotation.yaw, 0), "yaw value is not zero"
    assert np.isclose(empty_rotation.roll, 0), "roll value is not zero"


@pytest.mark.parametrize("pitch, yaw, roll", [(90, 90, 90), (0, 0, 0)])
def test_rotation(pitch, yaw, roll):
    """ Test the creation of Rotation from pitch, yaw, roll. """
    rotation = Rotation(pitch, yaw, roll)
    assert np.isclose(rotation.pitch, pitch), "The pitch was not the same."
    assert np.isclose(rotation.yaw, yaw), "The yaw was not the same."
    assert np.isclose(rotation.roll, roll), "The roll was not the same."


@pytest.mark.parametrize("pitch, yaw, roll", [(90, 90, 90), (0, 0, 0)])
def test_rotation_from_carla(pitch, yaw, roll):
    """ Test that the Rotation is initialized correctly from a carla.Rotation
    instance """
    carla_rotation = carla.Rotation(pitch, yaw, roll)
    rotation = Rotation.from_carla_rotation(carla_rotation)
    assert np.isclose(rotation.pitch, pitch), "pitch values are not the same."
    assert np.isclose(rotation.yaw, yaw), "yaw values are not the same."
    assert np.isclose(rotation.roll, roll), "roll values are not the same."


def test_negative_rotation_from_carla():
    """ Test that Rotation throws a ValueError if incorrect carla_rot argument
    is passed. """
    DummyType = namedtuple("DummyType", "pitch, yaw, roll")
    dummy_instance = DummyType(10, 20, 30)
    with pytest.raises(ValueError):
        Rotation.from_carla_rotation(dummy_instance)


def test_as_carla_rotation():
    """ Test the as_carla_rotation instance method of Rotation """
    rotation = Rotation(pitch=1, yaw=2, roll=3)
    carla_rotation = rotation.as_carla_rotation()
    assert isinstance(carla_rotation, carla.Rotation), "Returned instance is "
    "not of the type carla.Rotation"
    assert np.isclose(carla_rotation.pitch, rotation.pitch), "Returned "
    "instance pitch value is not the same as the one in rotation."
    assert np.isclose(carla_rotation.yaw, rotation.yaw), "Returned instance "
    "yaw value is not the same as the one in rotation."
    assert np.isclose(carla_rotation.roll, rotation.roll), "Returned instance "
    "roll value is not the same as the one in location."

## Depth Frame Tests

@pytest.mark.parametrize("x, y, z, threshold, expected", [
    (1, 0, 150, 100, True),
    (1, 0, 150, 25, False),
    (2, 1, 300, 250, True),
    (2, 1, 300, 150, False)
])
def test_pixel_has_same_depth(x, y, z, threshold, expected):
    """Tests if the pixel at (x,y) has a depth within the specified
       threshold of z."""
    camera_setup = None
    depth_frame = DepthFrame([[0, 0.1, 0],
                              [0, 0, 0.5]],
                               camera_setup)
    assert depth_frame.pixel_has_same_depth(x, y, z, threshold) is expected, \
           "Depth thresholding did not work correctly."


@pytest.mark.parametrize("depth_frame, expected", [
    (np.array([[0.4, 0.3], [0.2, 0.1]]), \
        [Location(400, -400, 400), Location(300, 300, 300), \
         Location(200, -200, -200), Location(100, 100, -100)]),
    (np.array([[0.1, 0.2]]), [Location(100, -100, 0), Location(200, 200, 0)]),
    (0.01 * np.ones((3,3)), \
        [Location(10, -10, 10), Location(10, 0, 10), Location(10, 10, 10),
         Location(10, -10, 0), Location(10, 0, 0), Location(10, 10, 0),
         Location(10, -10, -10), Location(10, 0, -10), Location(10, 10, -10)])
])
def test_depth_to_point_cloud(depth_frame, expected):
    height, width = depth_frame.shape
    camera_setup = CameraSetup('test_setup', 'test_type',
                               width, height,
                               Transform(location=Location(0, 0, 0),
                                         rotation=Rotation(0, 0, 0)),
                               fov=90)
    depth_frame = DepthFrame(depth_frame, camera_setup)
    # Resulting unreal coordinates.
    point_cloud = depth_frame.as_point_cloud()
    for i in range(width * height):
        assert np.isclose(point_cloud[i].x, expected[i].x), 'Returned x '
        'value is not the same as expected'
        assert np.isclose(point_cloud[i].y, expected[i].y), 'Returned y '
        'value is not the same as expected'
        assert np.isclose(point_cloud[i].z, expected[i].z), 'Returned z '
        'value is not the same as expected'


@pytest.mark.parametrize("depth_frame, expected", [
    (np.array([[0.1, 0.1]]), [Location(110, -80, 30), Location(110, 120, 30)])
])
def test_depth_to_point_cloud_nonzero_camera_loc(depth_frame, expected):
    height, width = depth_frame.shape
    camera_setup = CameraSetup('test_setup', 'test_type',
                               width, height,
                               Transform(location=Location(10, 20, 30),
                                         rotation=Rotation(0, 0, 0)),
                               fov=90)
    depth_frame = DepthFrame(depth_frame, camera_setup)
    # Resulting unreal coordinates.
    point_cloud = depth_frame.as_point_cloud()
    print (point_cloud)
    for i in range(width * height):
        assert np.isclose(point_cloud[i].x, expected[i].x), 'Returned x '
        'value is not the same as expected'
        assert np.isclose(point_cloud[i].y, expected[i].y), 'Returned y '
        'value is not the same as expected'
        assert np.isclose(point_cloud[i].z, expected[i].z), 'Returned z '
        'value is not the same as expected'


@pytest.mark.parametrize("depth_frame, pixels, expected", [
    (np.array([[0.4, 0.3], [0.2, 0.1]]), \
     [Vector2D(0,1), Vector2D(1,0)], \
     [Location(200, -200, -200), Location(300, 300, 300)])
])
def test_get_pixel_locations(depth_frame, pixels, expected):
    height, width = depth_frame.shape
    camera_setup = CameraSetup('test_setup', 'test_type',
                               width, height,
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


## Point Cloud Tests

@pytest.mark.parametrize("points, expected", [
    ([Location(1,0,0), Location(0,1,0), Location(0,0,1), Location(1,2,3)],
     [[1,0,0],[0,0,-1],[0,1,0],[1,3,-2]])
])
def test_initialize_point_cloud(points, expected):
    point_cloud = PointCloud(points, Transform(Location(), Rotation()))
    for i in range(len(expected)):
        assert all(np.isclose(point_cloud.points[i], expected[i]))


@pytest.mark.parametrize("lidar_points, pixel, expected", [
    # Lidar Points are left middle and right middle, same depth.
    ([Location(-1,-1,0),Location(1,-1,0)], Vector2D(200, 300), Location(1, -0.5, 0)),
    ([Location(-1,-1,0),Location(1,-1,0)], Vector2D(600, 300), Location(1, 0.5, 0)),
    # Lidar points are left middle and right middle, different depth.
    ([Location(-2,-2,0),Location(1,-1,0)], Vector2D(200, 300), Location(2, -1, 0)),
    ([Location(-2,-2,0),Location(1,-1,0)], Vector2D(600, 300), Location(1, 0.5, 0)),
    # Lidar points are top left and bottom right, same depth.
    ([Location(-2,-2,-1.5),Location(2,-2,1.5)], Vector2D(200, 150), Location(2,-1,0.75)),
])
def test_point_cloud_get_pixel_location(lidar_points, pixel, expected):
    camera_setup = CameraSetup('test_setup', 'test_type',
                               801, 601, # width, height
                               Transform(location=Location(0,0,0),
                                         rotation=Rotation(0,0,0)),
                               fov=90)
    point_cloud = PointCloud(lidar_points, Transform(Location(), Rotation()))
    location = point_cloud.get_pixel_location(pixel, camera_setup)
    assert np.isclose(location.x, expected.x), 'Returned x value is not the same '
    'as expected'
    assert np.isclose(location.y, expected.y), 'Returned y value is not the same '
    'as expected'
    assert np.isclose(location.z, expected.z), 'Returned z value is not the same '
    'as expected'
