import cv2
import math
import numpy as np
import time


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
        self.x, self.y, self.z = x, y, z

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

    def as_numpy_array(self):
        """Retrieves the 3D vector as a numpy array."""
        return np.array([self.x, self.y, self.z])

    def as_carla_vector(self):
        """Retrieves the 3D vector as an instance of CARLA 3D vector.

        Returns:
            carla.Vector3D: Instance representing the 3D vector.
        """
        import carla
        return carla.Vector3D(self.x, self.y, self.z)

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
            An instance with the coordinates converted to the camera view.
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

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Vector3D(x={}, y={}, z={})'.format(self.x, self.y, self.z)


class Vector2D(object):
    """Represents a 2D vector and provides helper functions."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_angle(self, other):
        """Computes the angle between the vector and another vector."""
        angle = math.atan2(self.y, self.x) - math.atan2(other.y, other.x)
        if angle > math.pi:
            angle -= 2 * math.pi
        elif angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def get_vector_and_magnitude(self, other):
        """Calculates vector and magnitude between two vectors.

        Args:
            other (:py:class:`.Vector2D`): The other vector to be used to
                calculate.

        Returns:
            :py:class:`.Vector2D`, :obj:`float`: A tuple comprising of a 2D
            vector and its magnitude.
        """
        vec = np.array([self.x - other.x, self.y - other.y])
        magnitude = np.linalg.norm(vec)
        if magnitude > 0.00001:
            vec = vec / magnitude
        return Vector2D(vec[0], vec[1]), magnitude

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
        """Creates a pylot location from a CARLA location.

        Args:
            location (carla.Location): An instance of a CARLA location.

        Returns:
            :py:class:`.Location`: A pylot location.
        """
        import carla
        if not isinstance(location, carla.Location):
            raise ValueError('The location must be a carla.Location')
        return cls(location.x, location.y, location.z)

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

    def l1_distance(self, other):
        """Calculates the L1 distance between the given point and the other
        point.

        Args:
            other (:py:class:`~.Location`): The other location used to
                calculate the L1 distance to.

        Returns:
            :obj:`float`: The L1 distance between the two points.
        """
        return abs(self.x - other.x) + abs(self.y - other.y) + abs(self.z -
                                                                   other.z)

    def get_vector_and_magnitude(self, other):
        """Calculates vector and magnitude between two locations.

        Args:
            other (:py:class:`~.Location`): The other location to used to
                calculate.

        Returns:
            :py:class:`.Vector2D`, :obj:`float`: A tuple comprising of a 2D
            vector and its magnitude.
        """
        vec = Vector2D(self.x, self.y)
        other_vec = Vector2D(other.x, other.y)
        return vec.get_vector_and_magnitude(other_vec)

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
        rotation (:py:class:`.Rotation`, optional): The rotation  (in degreers)
            of the object represented by the transform.
        matrix: The transformation matrix used to convert points in the 3D
            coordinate space with respect to the location and rotation of the
            given object.

    Attributes:
        location (:py:class:`.Location`): The location of the object
            represented by the transform.
        rotation (:py:class:`.Rotation`): The rotation (in degreers) of the
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
            yaw_r = math.acos(self.forward_vector.x / math.cos(pitch_r))
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
            points (list(:py:class:`.Location`)): list of points.
            matrix: The matrix of the transformation to apply.

        Returns:
            list(:py:class:`.Location`): List of transformed points.
        """
        # Retrieve the locations as numpy arrays.
        points = np.array([loc.as_numpy_array() for loc in points])

        # Needed format: [[X0,..Xn],[Y0,..Yn],[Z0,..Zn]].
        # So let's transpose the point matrix.
        points = points.transpose()

        # Add 1s row: [[X0..,Xn],[Y0..,Yn],[Z0..,Zn],[1,..1]]
        points = np.append(points, np.ones((1, points.shape[1])), axis=0)

        # Point transformation (depends on the given matrix)
        points = np.dot(matrix, points)

        # Get all but the last row in array form.
        points = np.asarray(points[0:3].transpose())

        return [Location(x, y, z) for x, y, z in points]

    def transform_points(self, points):
        """Transforms the given set of locations (specified in the coordinate
        space of the current transform) to be in the world coordinate space.

        For example, if the transform is at location (3, 0, 0) and the
        location passed to the argument is (10, 0, 0), this function will
        return (13, 0, 0) i.e. the location of the argument in the world
        coordinate space.

        Args:
            points (list(:py:class:`.Location`)): list of points.

        Returns:
            list(:py:class:`.Location`): List of transformed points.
        """
        return self.__transform(points, self.matrix)

    def inverse_transform_points(self, points):
        """Transforms the given set of locations (specified in world coordinate
        space) to be relative to the given transform.

        For example, if the transform is at location (3, 0, 0) and the location
        passed to the argument is (10, 0, 0), this function will return
        (7, 0, 0) i.e. the location of the argument relative to the given
        transform.

        Args:
            points (list(:py:class:`.Location`)): list of points.

        Returns:
            list(:py:class:`.Location`): List of transformed points.
        """
        return self.__transform(points, np.linalg.inv(self.matrix))

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

    def get_vector_magnitude_angle(self, target_loc):
        """Computes distance and relative angle between the transform and a
        target location.

        Args:
            target_loc (:py:class:`.Location`): Location of the target.

        Returns:
            Tuple of distance to the target and the angle
        """
        target_vec, magnitude = target_loc.get_vector_and_magnitude(
            self.location)
        if magnitude > 0:
            forward_vector = Vector2D(
                math.cos(math.radians(self.rotation.yaw)),
                math.sin(math.radians(self.rotation.yaw)))
            angle = target_vec.get_angle(forward_vector)
        else:
            angle = 0
        return (target_vec, magnitude, angle)

    def is_within_distance_ahead(self, dst_loc, max_distance):
        """Checks if a location is within a distance.

        Args:
            dst_loc (:py:class:`.Location`): Location to compute distance to.
            max_distance (:obj:`float`): Maximum allowed distance.

        Returns:
            bool: True if other location is within max_distance.
        """
        norm_dst, d_angle = self.get_vector_magnitude_angle(dst_loc)
        # Return if the vector is too small.
        if norm_dst < 0.001:
            return True
        # Return if the vector is greater than the distance.
        if norm_dst > max_distance:
            return False
        return d_angle < 90.0

    def __mul__(self, other):
        new_matrix = np.dot(self.matrix, other.matrix)
        return Transform(matrix=new_matrix)

    def __str__(self):
        if self.location:
            return "Transform(location: {}, rotation: {})".format(
                self.location, self.rotation)
        else:
            return "Transform({})".format(str(self.matrix))


class CanBus(object):
    """Class used to wrap ego-vehicle information.

    Args:
        transform (:py:class:`~pylot.utils.Transform`): Transform of the ego
            vehicle.
        forward_speed (:obj:`int`): Forward speed in m/s.

    Attributes:
        transform (:py:class:`~pylot.utils.Transform`): Transform of the ego
            vehicle.
        forward_speed (:obj:`int`): Forward speed in m/s.
    """
    def __init__(self, transform, forward_speed):
        if not isinstance(transform, Transform):
            raise ValueError(
                'transform should be of type pylot.utils.Transform')
        self.transform = transform
        # Forward speed in m/s.
        self.forward_speed = forward_speed

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "CanBus(transform: {}, forward speed: {})".format(
            self.transform, self.forward_speed)


def add_timestamp(image_np, timestamp):
    """Adds a timestamp text to an image np array.

    Args:
        image_np: A numpy array of the image.
        timestamp (:obj:`int`): The timestamp of the image.
    """
    txt_font = cv2.FONT_HERSHEY_SIMPLEX
    timestamp_txt = '{}'.format(timestamp)
    # Put timestamp text.
    cv2.putText(image_np,
                timestamp_txt, (5, 15),
                txt_font,
                0.5, (0, 0, 0),
                thickness=1,
                lineType=cv2.LINE_AA)


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
