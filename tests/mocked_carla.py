# This module provides mocked versions of classes and functions provided
# by Carla in our runtime environment.


class Location(object):
    """ A mock class for carla.Location. """

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Rotation(object):
    """ A mock class for carla.Rotation. """

    def __init__(self, pitch, yaw, roll):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
