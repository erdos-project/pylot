"""This module implements localization related Message classes."""

import erdos

from pylot.utils import Transform, Vector3D


class IMUMessage(erdos.Message):
    """Message class to be used to send IMU measurements.

    Args:
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of the
            message.
        transform (:py:class:`~pylot.utils.Transform`): Transform of the IMU.
        acceleration (:py:class:`~pylot.utils.Vector3D`): Linear acceleration
            measurement in m/s^2.
        gyro (:py:class:`~pylot.utils.Vector3D`): Angular velocity measurement
            in rad/sec.
        compass (:obj:`float`): Orientation measurement w.r.t North direction
            ((0, -1, 0) in Unreal) in radians.

    Attributes:
        transform (:py:class:`~pylot.utils.Transform`): Transform of the IMU.
        acceleration (:py:class:`~pylot.utils.Vector3D`): Linear acceleration
            measurement in m/s^2.
        gyro (:py:class:`~pylot.utils.Vector3D`): Angular velocity measurement
            in rad/sec.
        compass (:obj:`float`): Orientation measurement w.r.t North direction
            ((0, -1, 0) in Unreal) in radians.
    """
    def __init__(self, timestamp: erdos.Timestamp, transform: Transform,
                 acceleration: Vector3D, gyro: Vector3D, compass: float):
        super(IMUMessage, self).__init__(timestamp, None)
        self.transform = transform
        self.acceleration = acceleration
        self.gyro = gyro
        self.compass = compass

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'IMUMessage(timestamp: {}, transform: {}, acceleration: {}, '\
            'gyro: {}, compass: {})'.format(
                self.timestamp, self.transform, self.acceleration, self.gyro,
                self.compass)


class GNSSMessage(erdos.Message):
    """Message class to be used to send GNSS measurements.

    Args:
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of the
            message.
        transform (:py:class:`~pylot.utils.Transform`): Transform of the GNSS.
        altitude (float): Height regarding ground level.
        latitude (float): North/South value of a point on the map.
        longitude (float): West/East value of a point on the map.

    Attributes:
        transform (:py:class:`~pylot.utils.Transform`): Transform of the GNSS.
        altitude (float): Height regarding ground level.
        latitude (float): North/South value of a point on the map.
        longitude (float): West/East value of a point on the map.
    """
    def __init__(self, timestamp: erdos.Timestamp, transform: Transform,
                 altitude: float, latitude: float, longitude: float):
        super(GNSSMessage, self).__init__(timestamp, None)
        self.transform = transform
        self.altitude = altitude
        self.latitude = latitude
        self.longitude = longitude

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'GNSSMessage(timestamp: {}, transform: {}, altitude: {}, ' \
                'latitude: {}, longitude: {})'.format(
                    self.timestamp, self.transform, self.altitude,
                    self.latitude, self.longitude)
