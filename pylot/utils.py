import math
import time
from enum import Enum

import numpy as np


class Rotation(object):
    """Used to represent the rotation of an actor or obstacle.

    Rotations are applied in the order: Roll (X), Pitch (Y), Yaw (Z).
    A 90-degree "Roll" maps the positive Z-axis to the positive Y-axis.
    A 90-degree "Pitch" maps the positive X-axis to the positive Z-axis.
    A 90-degree "Yaw" maps the positive X-axis to the positive Y-axis.

    Args:
        pitch: Rotation about Y-axis.
        yaw:   Rotation about Z-axis.
        roll:  Rotation about X-axis.

    Attributes:
        pitch: Rotation about Y-axis.
        yaw:   Rotation about Z-axis.
        roll:  Rotation about X-axis.
    """
    def __init__(self, pitch=0, yaw=0, roll=0):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll

    @classmethod
    def from_carla_rotation(cls, rotation):
        """Creates a pylot Rotation from a CARLA rotation.

        Args:
            rotation (carla.Rotation): An instance of a CARLA rotation.

        Returns:
            :py:class:`.Rotation`: A pylot rotation.
        """
        import carla
        if not isinstance(rotation, carla.Rotation):
            raise ValueError('rotation should be of type carla.Rotation')
        return cls(rotation.pitch, rotation.yaw, rotation.roll)

    def as_carla_rotation(self):
        """ Retrieves the rotation as an instance of a CARLA rotation.

        Returns:
            carla.Rotation: Instance representing the rotation.
        """
        import carla
        return carla.Rotation(self.pitch, self.yaw, self.roll)

    def as_numpy_array(self):
        """Retrieves the Rotation as a numpy array."""
        return np.array([self.pitch, self.yaw, self.roll])

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Rotation(pitch={}, yaw={}, roll={})'.format(
            self.pitch, self.yaw, self.roll)


class Vector3D(object):
    """Represents a 3D vector and provides useful helper functions.

    Args:
        x: The value of the first axis.
        y: The value of the second axis.
        z: The value of the third axis.

    Attributes:
        x: The value of the first axis.
        y: The value of the second axis.
        z: The value of the third axis.
    """
    def __init__(self, x=0, y=0, z=0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    @classmethod
    def from_carla_vector(cls, vector):
        """Creates a pylot Vector3D from a CARLA 3D vector.

        Args:
            vector (carla.Vector3D): An instance of a CARLA 3D vector.

        Returns:
            :py:class:`.Vector3D`: A pylot 3D vector.
        """
        import carla
        if not isinstance(vector, carla.Vector3D):
            raise ValueError('The vector must be a carla.Vector3D')
        return cls(vector.x, vector.y, vector.z)

    def as_numpy_array(self):
        """Retrieves the 3D vector as a numpy array."""
        return np.array([self.x, self.y, self.z])

    def as_numpy_array_2D(self):
        """Drops the 3rd dimension."""
        return np.array([self.x, self.y])

    def as_carla_vector(self):
        """Retrieves the 3D vector as an instance of CARLA 3D vector.

        Returns:
            carla.Vector3D: Instance representing the 3D vector.
        """
        import carla
        return carla.Vector3D(self.x, self.y, self.z)

    def l1_distance(self, other):
        """Calculates the L1 distance between the point and another point.

        Args:
            other (:py:class:`~.Vector3D`): The other vector used to
                calculate the L1 distance to.

        Returns:
            :obj:`float`: The L1 distance between the two points.
        """
        return abs(self.x - other.x) + abs(self.y - other.y) + abs(self.z -
                                                                   other.z)

    def l2_distance(self, other):
        """Calculates the L2 distance between the point and another point.

        Args:
            other (:py:class:`~.Vector3D`): The other vector used to
                calculate the L2 distance to.

        Returns:
            :obj:`float`: The L2 distance between the two points.
        """
        vec = np.array([self.x - other.x, self.y - other.y, self.z - other.z])
        return np.linalg.norm(vec)

    def magnitude(self):
        """Returns the magnitude of the 3D vector."""
        return np.linalg.norm(self.as_numpy_array())

    def to_camera_view(self, extrinsic_matrix, intrinsic_matrix):
        """Converts the given 3D vector to the view of the camera using
        the extrinsic and the intrinsic matrix.

        Args:
            extrinsic_matrix: The extrinsic matrix of the camera.
            intrinsic_matrix: The intrinsic matrix of the camera.

        Returns:
            :py:class:`.Vector3D`: An instance with the coordinates converted
            to the camera view.
        """
        position_vector = np.array([[self.x], [self.y], [self.z], [1.0]])

        # Transform the points to the camera in 3D.
        transformed_3D_pos = np.dot(np.linalg.inv(extrinsic_matrix),
                                    position_vector)

        # Transform the points to 2D.
        position_2D = np.dot(intrinsic_matrix, transformed_3D_pos[:3])

        # Normalize the 2D points.
        location_2D = type(self)(float(position_2D[0] / position_2D[2]),
                                 float(position_2D[1] / position_2D[2]),
                                 float(position_2D[2]))
        return location_2D

    def rotate(self, angle):
        """Rotate the vector by a given angle.

        Args:
            angle (float): The angle to rotate the Vector by (in degrees).

        Returns:
            :py:class:`.Vector3D`: An instance with the coordinates of the
            rotated vector.
        """
        x_ = math.cos(math.radians(angle)) * self.x - math.sin(
            math.radians(angle)) * self.y
        y_ = math.sin(math.radians(angle)) * self.x - math.cos(
            math.radians(angle)) * self.y
        return type(self)(x_, y_, self.z)

    def __add__(self, other):
        """Adds the two vectors together and returns the result."""
        return type(self)(x=self.x + other.x,
                          y=self.y + other.y,
                          z=self.z + other.z)

    def __sub__(self, other):
        """Subtracts the other vector from self and returns the result."""
        return type(self)(x=self.x - other.x,
                          y=self.y - other.y,
                          z=self.z - other.z)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Vector3D(x={}, y={}, z={})'.format(self.x, self.y, self.z)


class Vector2D(object):
    """Represents a 2D vector and provides helper functions."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def as_numpy_array(self):
        """Retrieves the 2D vector as a numpy array."""
        return np.array([self.x, self.y])

    def get_angle(self, other):
        """Computes the angle between the vector and another vector."""
        angle = math.atan2(self.y, self.x) - math.atan2(other.y, other.x)
        if angle > math.pi:
            angle -= 2 * math.pi
        elif angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def l1_distance(self, other):
        """Calculates the L1 distance between the point and another point.

        Args:
            other (:py:class:`~.Vector2D`): The other vector used to
                calculate the L1 distance to.

        Returns:
            :obj:`float`: The L1 distance between the two points.
        """
        return abs(self.x - other.x) + abs(self.y - other.y)

    def l2_distance(self, other):
        """Calculates the L2 distance between the point and another point.

        Args:
            other (:py:class:`~.Vector2D`): The other vector used to
                calculate the L2 distance to.

        Returns:
            :obj:`float`: The L2 distance between the two points.
        """
        vec = np.array([self.x - other.x, self.y - other.y])
        return np.linalg.norm(vec)

    def magnitude(self):
        """Returns the magnitude of the 2D vector."""
        return np.linalg.norm(self.as_numpy_array())

    def __add__(self, other):
        """Adds the two vectors together and returns the result. """
        return type(self)(x=self.x + other.x, y=self.y + other.y)

    def __sub__(self, other):
        """Subtracts the other vector from self and returns the result. """
        return type(self)(x=self.x - other.x, y=self.y - other.y)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Vector2D(x={}, y={})'.format(self.x, self.y)


class Location(Vector3D):
    """Stores a 3D location, and provides useful helper methods.

    Args:
        x: The value of the x-axis.
        y: The value of the y-axis.
        z: The value of the z-axis.

    Attributes:
        x: The value of the x-axis.
        y: The value of the y-axis.
        z: The value of the z-axis.
    """
    def __init__(self, x=0, y=0, z=0):
        super(Location, self).__init__(x, y, z)

    @classmethod
    def from_carla_location(cls, location):
        """Creates a pylot Location from a CARLA location.

        Args:
            location (carla.Location): An instance of a CARLA location.

        Returns:
            :py:class:`.Location`: A pylot location.
        """
        import carla
        if not isinstance(location, carla.Location):
            raise ValueError('The location must be a carla.Location')
        return cls(location.x, location.y, location.z)

    @classmethod
    def from_gps(cls, latitude, longitude, altitude):
        """Creates Location from GPS (latitude, longitude, altitude).

        This is the inverse of the _location_to_gps method found in
        https://github.com/carla-simulator/scenario_runner/blob/master/srunner/tools/route_manipulation.py
        """
        EARTH_RADIUS_EQUA = 6378137.0
        # The following reference values are applicable for towns 1 through 7,
        # and are taken from the corresponding CARLA OpenDrive map files.
        LAT_REF = 49.0
        LON_REF = 8.0

        scale = math.cos(LAT_REF * math.pi / 180.0)
        basex = scale * math.pi * EARTH_RADIUS_EQUA / 180.0 * LON_REF
        basey = scale * EARTH_RADIUS_EQUA * math.log(
            math.tan((90.0 + LAT_REF) * math.pi / 360.0))

        x = scale * math.pi * EARTH_RADIUS_EQUA / 180.0 * longitude - basex
        y = scale * EARTH_RADIUS_EQUA * math.log(
            math.tan((90.0 + latitude) * math.pi / 360.0)) - basey

        # This wasn't in the original carla method, but seems to be necessary.
        y *= -1

        return cls(x, y, altitude)

    def distance(self, other):
        """Calculates the Euclidean distance between the given point and the
        other point.

        Args:
            other (:py:class:`~.Location`): The other location used to
                calculate the Euclidean distance to.

        Returns:
            :obj:`float`: The Euclidean distance between the two points.
        """
        return (self - other).magnitude()

    def as_vector_2D(self):
        """Transforms the Location into a Vector2D.

        Note:
            The method just drops the z-axis.

        Returns:
            :py:class:`.Vector2D`: A 2D vector.
        """
        return Vector2D(self.x, self.y)

    def as_carla_location(self):
        """Retrieves the location as a carla location instance.

        Returns:
            carla.Location: Instance representing the location.
        """
        import carla
        return carla.Location(self.x, self.y, self.z)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Location(x={}, y={}, z={})'.format(self.x, self.y, self.z)


class Transform(object):
    """A class that stores the location and rotation of an obstacle.

    It can be created from a carla.Transform, defines helper functions needed
    in Pylot, and makes the carla.Transform serializable.

    A transform object is instantiated with either a location and a rotation,
    or using a matrix.

    Args:
        location (:py:class:`.Location`, optional): The location of the object
            represented by the transform.
        rotation (:py:class:`.Rotation`, optional): The rotation  (in degrees)
            of the object represented by the transform.
        matrix: The transformation matrix used to convert points in the 3D
            coordinate space with respect to the location and rotation of the
            given object.

    Attributes:
        location (:py:class:`.Location`): The location of the object
            represented by the transform.
        rotation (:py:class:`.Rotation`): The rotation (in degrees) of the
            object represented by the transform.
        forward_vector (:py:class:`.Vector3D`): The forward vector of the
            object represented by the transform.
        matrix: The transformation matrix used to convert points in the 3D
            coordinate space with respect to the location and rotation of the
            given object.
    """
    def __init__(self, location=None, rotation=None, matrix=None):
        if matrix is not None:
            self.matrix = matrix
            self.location = Location(matrix[0, 3], matrix[1, 3], matrix[2, 3])

            # Forward vector is retrieved from the matrix.
            self.forward_vector = Vector3D(self.matrix[0, 0],
                                           self.matrix[1, 0], self.matrix[2,
                                                                          0])
            pitch_r = math.asin(self.forward_vector.z)
            yaw_r = math.acos(
                np.clip(self.forward_vector.x / math.cos(pitch_r), -1, 1))
            roll_r = math.asin(matrix[2, 1] / (-1 * math.cos(pitch_r)))
            self.rotation = Rotation(math.degrees(pitch_r),
                                     math.degrees(yaw_r), math.degrees(roll_r))
        else:
            self.location, self.rotation = location, rotation
            self.matrix = Transform._create_matrix(self.location,
                                                   self.rotation)

            # Forward vector is retrieved from the matrix.
            self.forward_vector = Vector3D(self.matrix[0, 0],
                                           self.matrix[1, 0], self.matrix[2,
                                                                          0])

    @classmethod
    def from_carla_transform(cls, transform):
        """Creates a pylot transform from a carla transform.

        Args:
            transform (carla.Transform): Carla transform.

        Returns:
            :py:class:`.Transform`: An instance of a pylot transform.
        """
        import carla
        if not isinstance(transform, carla.Transform):
            raise ValueError('transform should be of type carla.Transform')
        return cls(Location.from_carla_location(transform.location),
                   Rotation.from_carla_rotation(transform.rotation))

    @staticmethod
    def _create_matrix(location, rotation):
        """Creates a transformation matrix to convert points in the 3D world
        coordinate space with respect to the object.

        Use the transform_points function to transpose a given set of points
        with respect to the object.

        Args:
            location (:py:class:`.Location`): The location of the object
                represented by the transform.
            rotation (:py:class:`.Rotation`): The rotation of the object
                represented by the transform.

        Returns:
            A 4x4 numpy matrix which represents the transformation matrix.
        """
        matrix = np.identity(4)
        cy = math.cos(np.radians(rotation.yaw))
        sy = math.sin(np.radians(rotation.yaw))
        cr = math.cos(np.radians(rotation.roll))
        sr = math.sin(np.radians(rotation.roll))
        cp = math.cos(np.radians(rotation.pitch))
        sp = math.sin(np.radians(rotation.pitch))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = (cp * cy)
        matrix[0, 1] = (cy * sp * sr - sy * cr)
        matrix[0, 2] = -1 * (cy * sp * cr + sy * sr)
        matrix[1, 0] = (sy * cp)
        matrix[1, 1] = (sy * sp * sr + cy * cr)
        matrix[1, 2] = (cy * sr - sy * sp * cr)
        matrix[2, 0] = (sp)
        matrix[2, 1] = -1 * (cp * sr)
        matrix[2, 2] = (cp * cr)
        return matrix

    def __transform(self, points, matrix):
        """Internal function to transform the points according to the
        given matrix. This function either converts the points from
        coordinate space relative to the transform to the world coordinate
        space (using self.matrix), or from world coordinate space to the
        space relative to the transform (using inv(self.matrix))

        Args:
            points: An n by 3 numpy array, where each row is the
                (x, y, z) coordinates of a point.
            matrix: The matrix of the transformation to apply.

        Returns:
            An n by 3 numpy array of transformed points.
        """
        # Needed format: [[X0,..Xn],[Y0,..Yn],[Z0,..Zn]].
        # So let's transpose the point matrix.
        points = points.transpose()

        # Add 1s row: [[X0..,Xn],[Y0..,Yn],[Z0..,Zn],[1,..1]]
        points = np.append(points, np.ones((1, points.shape[1])), axis=0)

        # Point transformation (depends on the given matrix)
        points = np.dot(matrix, points)

        # Get all but the last row in array form.
        points = np.asarray(points[0:3].transpose()).astype(np.float16)

        return points

    def transform_points(self, points):
        """Transforms the given set of points (specified in the coordinate
        space of the current transform) to be in the world coordinate space.

        For example, if the transform is at location (3, 0, 0) and the
        location passed to the argument is (10, 0, 0), this function will
        return (13, 0, 0) i.e. the location of the argument in the world
        coordinate space.

        Args:
            points: A (number of points) by 3 numpy array, where each row is
                the (x, y, z) coordinates of a point.

        Returns:
            An n by 3 numpy array of transformed points.
        """
        return self.__transform(points, self.matrix)

    def inverse_transform_points(self, points):
        """Transforms the given set of points (specified in world coordinate
        space) to be relative to the given transform.

        For example, if the transform is at location (3, 0, 0) and the location
        passed to the argument is (10, 0, 0), this function will return
        (7, 0, 0) i.e. the location of the argument relative to the given
        transform.

        Args:
            points: A (number of points) by 3 numpy array, where each row is
                the (x, y, z) coordinates of a point.

        Returns:
            An n by 3 numpy array of transformed points.
        """
        return self.__transform(points, np.linalg.inv(self.matrix))

    def transform_locations(self, locations):
        """Transforms the given set of locations (specified in the coordinate
        space of the current transform) to be in the world coordinate space.

        This method has the same functionality as transform_points, and
        is provided for convenience; when dealing with a large number of
        points, it is advised to use transform_points to avoid the slow
        conversion between a numpy array and list of locations.

        Args:
            points (list(:py:class:`.Location`)): List of locations.

        Returns:
            list(:py:class:`.Location`): List of transformed points.
        """
        points = np.array([loc.as_numpy_array() for loc in locations])
        transformed_points = self.__transform(points, self.matrix)
        return [Location(x, y, z) for x, y, z in transformed_points]

    def inverse_transform_locations(self, locations):
        """Transforms the given set of locations (specified in world coordinate
        space) to be relative to the given transform.

        This method has the same functionality as inverse_transform_points,
        and is provided for convenience; when dealing with a large number of
        points, it is advised to use inverse_transform_points to avoid the slow
        conversion between a numpy array and list of locations.

        Args:
            points (list(:py:class:`.Location`)): List of locations.

        Returns:
            list(:py:class:`.Location`): List of transformed points.
        """

        points = np.array([loc.as_numpy_array() for loc in locations])
        transformed_points = self.__transform(points,
                                              np.linalg.inv(self.matrix))
        return [Location(x, y, z) for x, y, z in transformed_points]

    def as_carla_transform(self):
        """Converts the transform to a carla transform.

        Returns:
            carla.Transform: Instance representing the current Transform.
        """
        import carla
        return carla.Transform(
            carla.Location(self.location.x, self.location.y, self.location.z),
            carla.Rotation(pitch=self.rotation.pitch,
                           yaw=self.rotation.yaw,
                           roll=self.rotation.roll))

    def get_angle_and_magnitude(self, target_loc):
        """Computes relative angle between the transform and a target location.

        Args:
            target_loc (:py:class:`.Location`): Location of the target.

        Returns:
            Angle in radians and vector magnitude.
        """
        target_vec = target_loc.as_vector_2D() - self.location.as_vector_2D()
        magnitude = target_vec.magnitude()
        if magnitude > 0:
            forward_vector = Vector2D(
                math.cos(math.radians(self.rotation.yaw)),
                math.sin(math.radians(self.rotation.yaw)))
            angle = target_vec.get_angle(forward_vector)
        else:
            angle = 0
        return angle, magnitude

    def is_within_distance_ahead(self, dst_loc, max_distance):
        """Checks if a location is within a distance.

        Args:
            dst_loc (:py:class:`.Location`): Location to compute distance to.
            max_distance (:obj:`float`): Maximum allowed distance.

        Returns:
            bool: True if other location is within max_distance.
        """
        d_angle, norm_dst = self.get_angle_and_magnitude(dst_loc)
        # Return if the vector is too small.
        if norm_dst < 0.001:
            return True
        # Return if the vector is greater than the distance.
        if norm_dst > max_distance:
            return False
        return d_angle < 90.0

    def inverse_transform(self):
        """Returns the inverse transform of this transform."""
        new_matrix = np.linalg.inv(self.matrix)
        return Transform(matrix=new_matrix)

    def __mul__(self, other):
        new_matrix = np.dot(self.matrix, other.matrix)
        return Transform(matrix=new_matrix)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.location:
            return "Transform(location: {}, rotation: {})".format(
                self.location, self.rotation)
        else:
            return "Transform({})".format(str(self.matrix))


class Pose(object):
    """Class used to wrap ego-vehicle information.

    Args:
        transform (:py:class:`~pylot.utils.Transform`): Transform of the ego
            vehicle.
        forward_speed (:obj:`int`): Forward speed in m/s.
        velocity_vector (:py:class:`~pylot.utils.Vector3D`): Velocity vector
            in world frame

    Attributes:
        transform (:py:class:`~pylot.utils.Transform`): Transform of the ego
            vehicle.
        forward_speed (:obj:`int`): Forward speed in m/s.
        velocity_vector (:py:class:`~pylot.utils.Vector3D`): Velocity vector
            in world frame
    """
    def __init__(self,
                 transform,
                 forward_speed,
                 velocity_vector=None,
                 localization_time=None):
        if not isinstance(transform, Transform):
            raise ValueError(
                'transform should be of type pylot.utils.Transform')
        self.transform = transform
        # Forward speed in m/s.
        self.forward_speed = forward_speed
        self.velocity_vector = velocity_vector
        if localization_time is None:
            self.localization_time = time.time()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Pose(transform: {}, forward speed: {}, velocity vector: {})"\
            .format(self.transform, self.forward_speed, self.velocity_vector)


class LaneMarkingColor(Enum):
    """Enum that defines the lane marking colors according to OpenDrive 1.4.

    The goal of this enum is to make sure that lane colors are correctly
    propogated from Carla to Pylot.
    """
    WHITE = 0
    BLUE = 1
    GREEN = 2
    RED = 3
    YELLOW = 4
    OTHER = 5


class LaneMarkingType(Enum):
    """Enum that defines the lane marking types according to OpenDrive 1.4.

    The goal of this enum is to make sure that lane markings are correctly
    propogated from Carla to Pylot.
    """
    OTHER = 0
    BROKEN = 1
    SOLID = 2
    SOLIDSOLID = 3
    SOLIDBROKEN = 4
    BROKENSOLID = 5
    BROKENBROKEN = 6
    BOTTSDOTS = 7
    GRASS = 8
    CURB = 9
    NONE = 10


class LaneChange(Enum):
    """ Enum that defines the permission to turn either left, right, both or
    none for a given lane.

    The goal of this enum is to make sure that the lane change types are
    correctly propogated from Carla to Pylot.
    """
    NONE = 0
    RIGHT = 1
    LEFT = 2
    BOTH = 3


class LaneType(Enum):
    """Enum that defines the type of the lane according to OpenDrive 1.4.

    The goal of this enum is to make sure that the lane change types are
    correctly propogated from Carla to Pylot.
    """
    NONE = 1
    DRIVING = 2
    STOP = 4
    SHOULDER = 8
    BIKING = 16
    SIDEWALK = 32
    BORDER = 64
    RESTRICTED = 128
    PARKING = 256
    BIDIRECTIONAL = 512
    MEDIAN = 1024
    SPECIAL1 = 2048
    SPECIAL2 = 4096
    SPECIAL3 = 8192
    ROADWORKS = 16384
    TRAM = 32768
    RAIL = 65536
    ENTRY = 131072
    EXIT = 262144
    OFFRAMP = 524288
    ONRAMP = 1048576
    ANY = 4294967294


class LaneMarking(object):
    """Used to represent a lane marking.

    Args:
        marking_color (:py:class:`carla.LaneMarkingColor`): The color of the
            lane marking.
        marking_type (:py:class:`carla.LaneMarkingType`): The type of the lane
            marking.
        lane_change (:py:class:`carla.LaneChange`): The type that defines the
            permission to either turn left, right, both or none.

    Attributes:
        marking_color (:py:class:`.LaneMarkingColor`): The color of the lane
            marking
        marking_type (:py:class:`.LaneMarkingType`): The type of the lane
            marking.
        lane_change (:py:class:`.LaneChange`): The type that defines the
            permission to either turn left, right, both or none.
    """
    def __init__(self, marking_color, marking_type, lane_change):
        self.marking_color = LaneMarkingColor(marking_color)
        self.marking_type = LaneMarkingType(marking_type)
        self.lane_change = LaneChange(lane_change)

    @classmethod
    def from_carla_lane_marking(cls, lane_marking):
        """Creates a pylot LaneMarking from a CARLA lane marking.

        Args:
            lane_marking (:py:class:`carla.LaneMarking`): An instance of a
                CARLA lane marking.

        Returns:
            :py:class:`.LaneMarking`: A pylot lane-marking.
        """
        return cls(lane_marking.color, lane_marking.type,
                   lane_marking.lane_change)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "LaneMarking(color: {}, type: {}, change: {})".format(
            self.marking_color, self.marking_type, self.lane_change)


def add_timestamp(image_np, timestamp):
    """Adds a timestamp text to an image np array.

    Args:
        image_np: A numpy array of the image.
        timestamp (:obj:`int`): The timestamp of the image.
    """
    import cv2
    txt_font = cv2.FONT_HERSHEY_SIMPLEX
    timestamp_txt = '{}'.format(timestamp)
    # Put timestamp text.
    cv2.putText(image_np,
                timestamp_txt, (5, 15),
                txt_font,
                0.5, (0, 0, 0),
                thickness=1,
                lineType=cv2.LINE_AA)


def get_top_down_transform(transform, top_down_camera_altitude):
    # Calculation relies on the fact that the camera's FOV is 90.
    top_down_location = (transform.location +
                         Location(0, 0, top_down_camera_altitude))
    return Transform(top_down_location, Rotation(-90, 0, 0))


def time_epoch_ms():
    """Get current time in milliseconds."""
    return int(time.time() * 1000)


def set_tf_loglevel(level):
    """To be used to suppress TensorFlow logging."""
    import logging
    import os
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)


def run_visualizer_control_loop(control_display_stream):
    """Runs a pygame loop that waits for user commands.

    The user commands are send on the control_display_stream
    to control the pygame visualization window.
    """
    import erdos
    import pygame
    clock = pygame.time.Clock()
    from pygame.locals import K_n
    while True:
        clock.tick_busy_loop(60)
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYUP:
                if event.key == K_n:
                    control_display_stream.send(
                        erdos.Message(erdos.Timestamp(coordinates=[0]),
                                      event.key))
