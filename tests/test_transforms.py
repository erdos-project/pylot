import pytest

# Mock the import of carla.
import sys
sys.modules['carla'] = __import__('mocked_carla')

from collections import namedtuple
import numpy as np

# Import the mocked_carla as carla for the purposes of this test file.
import mocked_carla as carla
from pylot.utils import Location, Rotation, Transform, Vector3D

## Location Tests


def test_empty_location():
    """Test that an empty Location is initialized at (0, 0, 0) """
    empty_location = Location()
    assert np.isclose(empty_location.x, 0), "X value is not zero"
    assert np.isclose(empty_location.y, 0), "Y value is not zero"
    assert np.isclose(empty_location.z, 0), "Z value is not zero"


@pytest.mark.parametrize("x, y, z", [(10, 20, 30), (-10, -20, -30)])
def test_location_creation(x, y, z):
    """Test that the Location is initialized correctly. """
    location = Location(x, y, z)
    assert np.isclose(location.x, x), "X values are not the same."
    assert np.isclose(location.y, y), "Y values are not the same."
    assert np.isclose(location.z, z), "Z values are not the same."


@pytest.mark.parametrize("x, y, z", [(10, 20, 30), (-10, -20, -30)])
def test_location_from_simulator(x, y, z):
    """Test that the Location is initialized correctly from a simulator
    Location instance """
    location = Location.from_simulator_location(carla.Location(x, y, z))
    assert np.isclose(location.x, x), "X values are not the same."
    assert np.isclose(location.y, y), "Y values are not the same."
    assert np.isclose(location.z, z), "Z values are not the same."


def test_negative_location_from_simulator():
    """Test that Location throws a ValueError if incorrect simulator_loc
    argument is passed. """
    DummyType = namedtuple("DummyType", "x, y, z")
    dummy_instance = DummyType(10, 20, 30)
    with pytest.raises(ValueError):
        Location.from_simulator_location(dummy_instance)


@pytest.mark.parametrize("point_a, point_b, expected",
                         [((1, 2, 3), (1, 2, 3), 0),
                          ((10, 20, 30), (40, 50, 60), 51.961524227)])
def test_distance(point_a, point_b, expected):
    """Test the distance computed between two points is the same as expected"""
    location_a, location_b = Location(*point_a), Location(*point_b)
    assert np.isclose(location_a.distance(location_b), expected), "Distance "
    "between point_a and point_b is not the same as the expected distance."
    assert np.isclose(location_b.distance(location_a), expected), "Distance "
    "between point_b and point_a is not the same as the expected distance."


@pytest.mark.parametrize("point_a, point_b, expected",
                         [((1, 2, 3), (1, 2, 3), 0),
                          ((10, 20, 30), (40, 50, 60), 90)])
def test_l1_distance(point_a, point_b, expected):
    """Test the L1 distance computed between the two points. """
    location_a, location_b = Location(*point_a), Location(*point_b)
    assert np.isclose(location_a.l1_distance(location_b), expected), "L1 "
    "Distance between point_a and point_b is not the same as expected."
    assert np.isclose(location_b.l1_distance(location_a), expected), "L1 "
    "Distance between point_a and point_b is not the same as expected."


def test_as_simulator_location():
    """Test the as_simulator_location instance method of Location """
    location = Location(x=1, y=2, z=3)
    simulator_location = location.as_simulator_location()
    assert isinstance(simulator_location, carla.Location), "Returned instance is "
    "not of the type carla.Location"
    assert np.isclose(simulator_location.x, location.x), "Returned instance x "
    "value is not the same as the one in location."
    assert np.isclose(simulator_location.y, location.y), "Returned instance y "
    "value is not the same as the one in location."
    assert np.isclose(simulator_location.z, location.z), "Returned instance z "
    "value is not the same as the one in location."


def test_location_as_numpy_array():
    """ Test the as_simulator_location instance method of Location """
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


## Rotation Tests


def test_empty_rotation():
    """ Test that an empty Location is initialized at (0, 0, 0) """
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
def test_rotation_from_simulator(pitch, yaw, roll):
    """ Test that the Rotation is initialized correctly from a carla.Rotation
    instance """
    simulator_rotation = carla.Rotation(pitch, yaw, roll)
    rotation = Rotation.from_simulator_rotation(simulator_rotation)
    assert np.isclose(rotation.pitch, pitch), "pitch values are not the same."
    assert np.isclose(rotation.yaw, yaw), "yaw values are not the same."
    assert np.isclose(rotation.roll, roll), "roll values are not the same."


def test_negative_rotation_from_simulator():
    """ Test that Rotation throws a ValueError if incorrect simulator_rot argument
    is passed. """
    DummyType = namedtuple("DummyType", "pitch, yaw, roll")
    dummy_instance = DummyType(10, 20, 30)
    with pytest.raises(ValueError):
        Rotation.from_simulator_rotation(dummy_instance)


def test_as_simulator_rotation():
    """ Test the as_simulator_rotation instance method of Rotation """
    rotation = Rotation(pitch=1, yaw=2, roll=3)
    simulator_rotation = rotation.as_simulator_rotation()
    assert isinstance(simulator_rotation, carla.Rotation), "Returned instance is "
    "not of the type carla.Rotation"
    assert np.isclose(simulator_rotation.pitch, rotation.pitch), "Returned "
    "instance pitch value is not the same as the one in rotation."
    assert np.isclose(simulator_rotation.yaw, rotation.yaw), "Returned instance "
    "yaw value is not the same as the one in rotation."
    assert np.isclose(simulator_rotation.roll, rotation.roll), "Returned instance "
    "roll value is not the same as the one in location."


## Vector3D Tests


def test_empty_vector3d():
    """ Test that an empty Vector3D is initialized at (0, 0, 0) """
    empty_vector = Vector3D()
    assert np.isclose(empty_vector.x, 0), "X value is not zero"
    assert np.isclose(empty_vector.y, 0), "Y value is not zero"
    assert np.isclose(empty_vector.z, 0), "Z value is not zero"


@pytest.mark.parametrize("x, y, z", [(1, 2, 3), (-1, -2, -3)])
def test_from_simulator_vector(x, y, z):
    """ Tests the creation of Vector3D. """
    simulator_vector3d = carla.Vector3D(x, y, z)
    vector3d = Vector3D.from_simulator_vector(simulator_vector3d)
    assert isinstance(vector3d, Vector3D), "The returned object is not of type"
    "Vector3D"
    assert np.isclose(simulator_vector3d.x, vector3d.x), "X value is not the same"
    assert np.isclose(simulator_vector3d.y, vector3d.y), "Y value is not the same"
    assert np.isclose(simulator_vector3d.z, vector3d.z), "Z value is not the same"


def test_negative_vector_from_simulator():
    """ Tests that Vector3D throws a ValueError if incorrect vector is
    passed. """
    DummyType = namedtuple("DummyType", "x, y, z")
    dummy_instance = DummyType(0, 0, 0)
    with pytest.raises(ValueError):
        Vector3D.from_simulator_vector(dummy_instance)


@pytest.mark.parametrize('point_a, point_b, expected',
                         [((1, 1, 1), (2, 2, 2), (3, 3, 3)),
                          ((0, 0, 0), (1, 1, 1), (1, 1, 1)),
                          ((1, 1, 1), (-1, -1, -1), (0, 0, 0))])
def test_add_vector(point_a, point_b, expected):
    first_vector, second_vector = Vector3D(*point_a), Vector3D(*point_b)
    sum_vector = first_vector + second_vector
    expected_vector = Vector3D(*expected)
    assert isinstance(sum_vector, Vector3D), "The sum is not an instance of "
    "Vector3D"
    assert np.isclose(sum_vector.x, expected_vector.x), "The x coordinate of "
    "the sum is not the same as the expected x coordinate"
    assert np.isclose(sum_vector.y, expected_vector.y), "The y coordinate of "
    "the sum is not the same as the expected y coordinate"
    assert np.isclose(sum_vector.z, expected_vector.z), "The z coordinate of "
    "the sum is not the same as the expected z coordinate"


@pytest.mark.parametrize('point_a, point_b, expected',
                         [((1, 1, 1), (2, 2, 2), (-1, -1, -1)),
                          ((0, 0, 0), (1, 1, 1), (-1, -1, -1)),
                          ((1, 1, 1), (-1, -1, -1), (2, 2, 2))])
def test_subtract_vector(point_a, point_b, expected):
    first_vector, second_vector = Vector3D(*point_a), Vector3D(*point_b)
    sub_vector = first_vector - second_vector
    expected_vector = Vector3D(*expected)
    assert isinstance(sub_vector, Vector3D), "The difference is not an "
    "instance of Vector3D"
    assert np.isclose(sub_vector.x, expected_vector.x), "The x coordinate of "
    "the difference is not the same as the expected x coordinate"
    assert np.isclose(sub_vector.y, expected_vector.y), "The y coordinate of "
    "the difference is not the same as the expected y coordinate"
    assert np.isclose(sub_vector.z, expected_vector.z), "The z coordinate of "
    "the sum is not the same as the expected z coordinate"


def test_vector_as_numpy_array():
    vector_np = Vector3D().as_numpy_array()
    assert isinstance(vector_np, np.ndarray), "Returned instance is not of "
    "type numpy.ndarray"
    assert all(vector_np == [0, 0, 0]), "The values returned in the numpy "
    "array are not the expected values."


def test_as_simulator_vector():
    vector = Vector3D().as_simulator_vector()
    assert isinstance(vector, carla.Vector3D), "The returned object "
    "is not of the type carla.Vector3D"
    assert np.isclose(vector.x, 0), "The x value of the returned vector"
    " is not 0"
    assert np.isclose(vector.y, 0), "The y value of the returned vector"
    " is not 0"
    assert np.isclose(vector.z, 0), "The z value of the returned vector"
    " is not 0"


@pytest.mark.parametrize("point, expected", [((1, 2, 3), 3.7416573867739413),
                                             ((0, 0, 0), 0)])
def test_vector_magnitude(point, expected):
    magnitude = Vector3D(*point).magnitude()
    assert np.isclose(magnitude, expected), "The magnitude was not similar to"
    " the expected magnitude."


@pytest.mark.parametrize("location, expected",
                         [((1, 2, 3), (2878.5, -2339, 1.0)),
                          ((10000, 20, 30), (961.419, 536.6215, 10000.0)),
                          ((-1, -1, -1), (1919.0, -420, -1.0))])
def test_vector_to_camera_view(location, expected):
    from pylot.drivers.sensor_setup import CameraSetup
    camera_setup = CameraSetup('test_camera',
                               'sensor.camera.rgb',
                               width=1920,
                               height=1080,
                               transform=Transform(Location(), Rotation()))
    location = Location(*location)
    assert all(
        np.isclose(
            location.to_camera_view(
                camera_setup.get_extrinsic_matrix(),
                camera_setup.get_intrinsic_matrix()).as_numpy_array(),
            expected)), "The camera transformation was not as expected."
