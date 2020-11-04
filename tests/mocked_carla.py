# This module provides mocked versions of classes and functions provided
# by CARLA in our runtime environment.


class Location(object):
    """ A mock class for Location. """

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Rotation(object):
    """ A mock class for Rotation. """

    def __init__(self, pitch, yaw, roll):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll


class Vector3D(object):
    """ A mock class for Vector3D. """

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
