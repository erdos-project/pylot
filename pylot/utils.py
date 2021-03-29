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
    def __init__(self, pitch: float = 0, yaw: float = 0, roll: float = 0):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll

    @classmethod
    def from_simulator_rotation(cls, rotation):
        """Creates a pylot Rotation from a simulator rotation.

        Args:
            rotation: An instance of a simulator rotation.

        Returns:
            :py:class:`.Rotation`: A pylot rotation.
        """
        from carla import Rotation
        if not isinstance(rotation, Rotation):
            raise ValueError('rotation should be of type Rotation')
        return cls(rotation.pitch, rotation.yaw, rotation.roll)

    def as_simulator_rotation(self):
        """ Retrieves the rotation as an instance of a simulator rotation.

        Returns:
            An instance of a simulator class representing the rotation.
        """
        from carla import Rotation
        return Rotation(self.pitch, self.yaw, self.roll)

    def as_numpy_array(self):
        """Retrieves the Rotation as a numpy array."""
        return np.array([self.pitch, self.yaw, self.roll])

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Rotation(pitch={}, yaw={}, roll={})'.format(
            self.pitch, self.yaw, self.roll)


class Quaternion(object):
    """ Represents the Rotation of an obstacle or vehicle in quaternion
    notation.

    Args:
        w: The real-part of the quaternion.
        x: The x-part (i) of the quaternion.
        y: The y-part (j) of the quaternion.
        z: The z-part (k) of the quaternion.

    Attributes:
        w: The real-part of the quaternion.
        x: The x-part (i) of the quaternion.
        y: The y-part (j) of the quaternion.
        z: The z-part (k) of the quaternion.
        matrix: A 3x3 numpy array that can be used to rotate 3D vectors from
            body frame to world frame.
    """
    def __init__(self, w: float, x: float, y: float, z: float):
        norm = np.linalg.norm([w, x, y, z])
        if norm < 1e-50:
            self.w, self.x, self.y, self.z = 0, 0, 0, 0
        else:
            self.w = w / norm
            self.x = x / norm
            self.y = y / norm
            self.z = z / norm
        self.matrix = Quaternion._create_matrix(self.w, self.x, self.y, self.z)

    @staticmethod
    def _create_matrix(w, x, y, z):
        """Creates a Rotation matrix that can be used to transform 3D vectors
        from body frame to world frame.

        Note that this yields the same matrix as a Transform object with the
        quaternion converted to the Euler rotation except this matrix only does
        rotation and no translation.

        Specifically, this matrix is equivalent to:
            Transform(location=Location(0, 0, 0),
                      rotation=self.as_rotation()).matrix[:3, :3]

        Returns:
            A 3x3 numpy array that can be used to rotate 3D vectors from body
            frame to world frame.
        """
        x2, y2, z2 = x * 2, y * 2, z * 2
        xx, xy, xz = x * x2, x * y2, x * z2
        yy, yz, zz = y * y2, y * z2, z * z2
        wx, wy, wz = w * x2, w * y2, w * z2
        m = np.array([[1.0 - (yy + zz), xy - wz, xz + wy],
                      [xy + wz, 1.0 - (xx + zz), yz - wx],
                      [xz - wy, yz + wx, 1.0 - (xx + yy)]])
        return m

    @classmethod
    def from_rotation(cls, rotation: Rotation):
        """Creates a Quaternion from a rotation including pitch, roll, yaw.

        Args:
            rotation (:py:class:`.Rotation`): A pylot rotation representing
                the rotation of the object in degrees.

        Returns:
            :py:class:`.Quaternion`: The quaternion representation of the
            rotation.
        """
        roll_by_2 = np.radians(rotation.roll) / 2.0
        pitch_by_2 = np.radians(rotation.pitch) / 2.0
        yaw_by_2 = np.radians(rotation.yaw) / 2.0

        cr, sr = np.cos(roll_by_2), np.sin(roll_by_2)
        cp, sp = np.cos(pitch_by_2), np.sin(pitch_by_2)
        cy, sy = np.cos(yaw_by_2), np.sin(yaw_by_2)

        w = cr * cp * cy + sr * sp * sy
        x = cr * sp * sy - sr * cp * cy
        y = -cr * sp * cy - sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return cls(w, x, y, z)

    @classmethod
    def from_angular_velocity(cls, angular_velocity, dt: float):
        """Creates a Quaternion from an angular velocity vector and the time
        delta to apply it for.

        Args:
            angular_velocity (:py:class:`.Vector3D`): The vector representing
                the angular velocity of the object in the body-frame.
            dt (float): The time delta to apply the angular velocity for.

        Returns:
            :py:class:`.Quaternion`: The quaternion representing the rotation
                undergone by the object with the given angular velocity in the
                given delta time.
        """
        angular_velocity_np = angular_velocity.as_numpy_array() * dt
        magnitude = np.linalg.norm(angular_velocity_np)

        w = np.cos(magnitude / 2.0)
        if magnitude < 1e-50:
            # To avoid instabilities and nan.
            x, y, z = 0, 0, 0
        else:
            imaginary = angular_velocity_np / magnitude * np.sin(
                magnitude / 2.0)
            x, y, z = imaginary
        return cls(w, x, y, z)

    def as_rotation(self) -> Rotation:
        """Retrieve the Quaternion as a Rotation in degrees.

        Returns:
            :py:class:`.Rotation`: The euler-angle equivalent of the
                Quaternion in degrees.
        """
        SINGULARITY_THRESHOLD = 0.4999995
        RAD_TO_DEG = (180.0) / np.pi

        singularity_test = self.z * self.x - self.w * self.y
        yaw_y = 2.0 * (self.w * self.z + self.x * self.y)
        yaw_x = (1.0 - 2.0 * (self.y**2 + self.z**2))

        pitch, yaw, roll = None, None, None
        if singularity_test < -SINGULARITY_THRESHOLD:
            pitch = -90.0
            yaw = np.arctan2(yaw_y, yaw_x) * RAD_TO_DEG
            roll = -yaw - (2.0 * np.arctan2(self.x, self.w) * RAD_TO_DEG)
        elif singularity_test > SINGULARITY_THRESHOLD:
            pitch = 90.0
            yaw = np.arctan2(yaw_y, yaw_x) * RAD_TO_DEG
            roll = yaw - (2.0 * np.arctan2(self.x, self.w) * RAD_TO_DEG)
        else:
            pitch = np.arcsin(2.0 * singularity_test) * RAD_TO_DEG
            yaw = np.arctan2(yaw_y, yaw_x) * RAD_TO_DEG
            roll = np.arctan2(-2.0 * (self.w * self.x + self.y * self.z),
                              (1.0 - 2.0 *
                               (self.x**2 + self.y**2))) * RAD_TO_DEG
        return Rotation(pitch, yaw, roll)

    def __mul__(self, other):
        """Returns the product self * other.  The product is NOT commutative.

        The product is defined in Unreal as:
         [ (Q2.w * Q1.x) + (Q2.x * Q1.w) + (Q2.y * Q1.z) - (Q2.z * Q1.y),
           (Q2.w * Q1.y) - (Q2.x * Q1.z) + (Q2.y * Q1.w) + (Q2.z * Q1.x),
           (Q2.w * Q1.z) + (Q2.x * Q1.y) - (Q2.y * Q1.x) + (Q2.z * Q1.w),
           (Q2.w * Q1.w) - (Q2.x * Q1.x) - (Q2.y * Q1.y) - (Q2.z * Q1.z) ]
        Copied from DirectX's XMQuaternionMultiply function.
        """
        q1, q2 = other, self
        x = (q2.w * q1.x) + (q2.x * q1.w) + (q2.y * q1.z) - (q2.z * q1.y)
        y = (q2.w * q1.y) - (q2.x * q1.z) + (q2.y * q1.w) + (q2.z * q1.x)
        z = (q2.w * q1.z) + (q2.x * q1.y) - (q2.y * q1.x) + (q2.z * q1.w)
        w = (q2.w * q1.w) - (q2.x * q1.x) - (q2.y * q1.y) - (q2.z * q1.z)
        return Quaternion(w, x, y, z)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Quaternion (w={}, x={}, y={}, z={})'.format(
            self.w, self.x, self.y, self.z)


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
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    @classmethod
    def from_simulator_vector(cls, vector):
        """Creates a pylot Vector3D from a simulator 3D vector.

        Args:
            vector: An instance of a simulator 3D vector.

        Returns:
            :py:class:`.Vector3D`: A pylot 3D vector.
        """
        from carla import Vector3D
        if not isinstance(vector, Vector3D):
            raise ValueError('The vector must be a Vector3D')
        return cls(vector.x, vector.y, vector.z)

    def as_numpy_array(self):
        """Retrieves the 3D vector as a numpy array."""
        return np.array([self.x, self.y, self.z])

    def as_numpy_array_2D(self):
        """Drops the 3rd dimension."""
        return np.array([self.x, self.y])

    def as_simulator_vector(self):
        """Retrieves the 3D vector as an instance of simulator 3D vector.

        Returns:
            An instance of the simulator class representing the 3D vector.
        """
        from carla import Vector3D
        return Vector3D(self.x, self.y, self.z)

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

    def l2_distance(self, other) -> float:
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

    def rotate(self, angle: float):
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
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def as_numpy_array(self):
        """Retrieves the 2D vector as a numpy array."""
        return np.array([self.x, self.y])

    def get_angle(self, other) -> float:
        """Computes the angle between the vector and another vector
           in radians."""
        angle = math.atan2(self.y, self.x) - math.atan2(other.y, other.x)
        if angle > math.pi:
            angle -= 2 * math.pi
        elif angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def l1_distance(self, other) -> float:
        """Calculates the L1 distance between the point and another point.

        Args:
            other (:py:class:`~.Vector2D`): The other vector used to
                calculate the L1 distance to.

        Returns:
            :obj:`float`: The L1 distance between the two points.
        """
        return abs(self.x - other.x) + abs(self.y - other.y)

    def l2_distance(self, other) -> float:
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
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        super(Location, self).__init__(x, y, z)

    @classmethod
    def from_simulator_location(cls, location):
        """Creates a pylot Location from a simulator location.

        Args:
            location: An instance of a simulator location.

        Returns:
            :py:class:`.Location`: A pylot location.
        """
        from carla import Location, Vector3D
        if not (isinstance(location, Location)
                or isinstance(location, Vector3D)):
            raise ValueError('The location must be a Location or Vector3D')
        return cls(location.x, location.y, location.z)

    @classmethod
    def from_gps(cls, latitude: float, longitude: float, altitude: float):
        """Creates Location from GPS (latitude, longitude, altitude).

        This is the inverse of the _location_to_gps method found in
        https://github.com/carla-simulator/scenario_runner/blob/master/srunner/tools/route_manipulation.py
        """
        EARTH_RADIUS_EQUA = 6378137.0
        # The following reference values are applicable for towns 1 through 7,
        # and are taken from the corresponding OpenDrive map files.
        # LAT_REF = 49.0
        # LON_REF = 8.0
        # TODO: Do not hardcode. Get the references from the open drive file.
        LAT_REF = 0.0
        LON_REF = 0.0

        scale = math.cos(LAT_REF * math.pi / 180.0)
        basex = scale * math.pi * EARTH_RADIUS_EQUA / 180.0 * LON_REF
        basey = scale * EARTH_RADIUS_EQUA * math.log(
            math.tan((90.0 + LAT_REF) * math.pi / 360.0))

        x = scale * math.pi * EARTH_RADIUS_EQUA / 180.0 * longitude - basex
        y = scale * EARTH_RADIUS_EQUA * math.log(
            math.tan((90.0 + latitude) * math.pi / 360.0)) - basey

        # This wasn't in the original method, but seems to be necessary.
        y *= -1

        return cls(x, y, altitude)

    def distance(self, other) -> float:
        """Calculates the Euclidean distance between the given point and the
        other point.

        Args:
            other (:py:class:`~.Location`): The other location used to
                calculate the Euclidean distance to.

        Returns:
            :obj:`float`: The Euclidean distance between the two points.
        """
        return (self - other).magnitude()

    def as_vector_2D(self) -> Vector2D:
        """Transforms the Location into a Vector2D.

        Note:
            The method just drops the z-axis.

        Returns:
            :py:class:`.Vector2D`: A 2D vector.
        """
        return Vector2D(self.x, self.y)

    def as_simulator_location(self):
        """Retrieves the location as a simulator location instance.

        Returns:
            An instance of the simulator class representing the location.
        """
        from carla import Location
        return Location(self.x, self.y, self.z)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Location(x={}, y={}, z={})'.format(self.x, self.y, self.z)


class Transform(object):
    """A class that stores the location and rotation of an obstacle.

    It can be created from a simulator transform, defines helper functions
    needed in Pylot, and makes the simulator transform serializable.

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
    def __init__(self,
                 location: Location = None,
                 rotation: Rotation = None,
                 matrix=None):
        if matrix is not None:
            self.matrix = matrix
            self.location = Location(matrix[0, 3], matrix[1, 3], matrix[2, 3])

            # Forward vector is retrieved from the matrix.
            self.forward_vector = \
                Vector3D(self.matrix[0, 0], self.matrix[1, 0],
                         self.matrix[2, 0])
            pitch_r = math.asin(np.clip(self.forward_vector.z, -1, 1))
            yaw_r = math.acos(
                np.clip(self.forward_vector.x / math.cos(pitch_r), -1, 1))
            roll_r = math.asin(
                np.clip(matrix[2, 1] / (-1 * math.cos(pitch_r)), -1, 1))
            self.rotation = Rotation(math.degrees(pitch_r),
                                     math.degrees(yaw_r), math.degrees(roll_r))
        else:
            self.location, self.rotation = location, rotation
            self.matrix = Transform._create_matrix(self.location,
                                                   self.rotation)

            # Forward vector is retrieved from the matrix.
            self.forward_vector = \
                Vector3D(self.matrix[0, 0], self.matrix[1, 0],
                         self.matrix[2, 0])

    @classmethod
    def from_simulator_transform(cls, transform):
        """Creates a pylot transform from a simulator transform.

        Args:
            transform: A simulator transform.

        Returns:
            :py:class:`.Transform`: An instance of a pylot transform.
        """
        from carla import Transform
        if not isinstance(transform, Transform):
            raise ValueError('transform should be of type Transform')
        return cls(Location.from_simulator_location(transform.location),
                   Rotation.from_simulator_rotation(transform.rotation))

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

    def as_simulator_transform(self):
        """Converts the transform to a simulator transform.

        Returns:
            An instance of the simulator class representing the Transform.
        """
        from carla import Location, Rotation, Transform
        return Transform(
            Location(self.location.x, self.location.y, self.location.z),
            Rotation(pitch=self.rotation.pitch,
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

    def is_within_distance_ahead(self, dst_loc: Location,
                                 max_distance: float) -> bool:
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
                 transform: Transform,
                 forward_speed: float,
                 velocity_vector: Vector3D = None,
                 localization_time: float = None):
        if not isinstance(transform, Transform):
            raise ValueError(
                'transform should be of type pylot.utils.Transform')
        self.transform = transform
        # Forward speed in m/s.
        self.forward_speed = forward_speed
        self.velocity_vector = velocity_vector
        if localization_time is None:
            self.localization_time = time.time()
        else:
            self.localization_time = localization_time

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Pose(transform: {}, forward speed: {}, velocity vector: {})"\
            .format(self.transform, self.forward_speed, self.velocity_vector)


class LaneMarkingColor(Enum):
    """Enum that defines the lane marking colors according to OpenDrive 1.4.

    The goal of this enum is to make sure that lane colors are correctly
    propogated from the simulator to Pylot.
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
    propogated from the simulator to Pylot.
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
    correctly propogated from the simulator to Pylot.
    """
    NONE = 0
    RIGHT = 1
    LEFT = 2
    BOTH = 3


class LaneType(Enum):
    """Enum that defines the type of the lane according to OpenDrive 1.4.

    The goal of this enum is to make sure that the lane change types are
    correctly propogated from the simulator to Pylot.
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


class RoadOption(Enum):
    """Enum that defines the possible high-level route plans.

    RoadOptions are usually attached to waypoints we receive from
    the challenge environment.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANE_FOLLOW = 4
    CHANGE_LANE_LEFT = 5
    CHANGE_LANE_RIGHT = 6

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.name


class LaneMarking(object):
    """Used to represent a lane marking.

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
    def from_simulator_lane_marking(cls, lane_marking):
        """Creates a pylot LaneMarking from a simulator lane marking.

        Args:
            lane_marking: An instance of a simulator lane marking.

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


def time_epoch_ms() -> int:
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
            elif event.type == pygame.QUIT:
                raise KeyboardInterrupt
            elif event.type == pygame.KEYDOWN:
                if (event.key == pygame.K_c
                        and pygame.key.get_mods() & pygame.KMOD_CTRL):
                    raise KeyboardInterrupt


def verify_keys_in_dict(required_keys, arg_dict):
    assert set(required_keys).issubset(set(arg_dict.keys())), \
            "one or more of {} not found in {}".format(required_keys, arg_dict)
