import pytest

from collections import namedtuple
import numpy as np

import carla
from pylot.simulation.utils import Location


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
    carla_loc = carla.Location(x, y, z)
    location = Location(carla_location=carla_loc)
    assert np.isclose(location.x, x), "X values are not the same."
    assert np.isclose(location.y, y), "Y values are not the same."
    assert np.isclose(location.z, z), "Z values are not the same."


def test_negative_location_from_carla():
    """ Test that Location throws a ValueError if incorrect carla_loc argument
    is passed. """
    DummyType = namedtuple("DummyType", "x, y, z")
    dummy_instance = DummyType(10, 20, 30)
    with pytest.raises(ValueError):
        Location(carla_location=dummy_instance)
