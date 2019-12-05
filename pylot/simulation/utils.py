import carla
from collections import namedtuple
from operator import attrgetter
import math
import numpy as np
from numpy.linalg import inv
from numpy.matlib import repmat

from pylot.perception.detection.utils import DetectedObject,\
    DetectedSpeedLimit, get_bounding_boxes_from_segmented
from pylot.perception.segmentation.utils import get_traffic_sign_pixels

Vehicle = namedtuple('Vehicle', 'id, transform, bounding_box, forward_speed')
Pedestrian = namedtuple('Pedestrian',
                        'id, transform, bounding_box, forward_speed')
SpeedLimitSign = namedtuple('SpeedLimitSign', 'transform, limit')
StopSign = namedtuple('StopSign', 'transform, bounding_box')
DetectedLane = namedtuple('DetectedLane', 'left_marking, right_marking')
LocationGeo = namedtuple('LocationGeo', 'latitude, longitude, altitude')


class CameraSetup(object):
    def __init__(self,
                 name,
                 camera_type,
                 width,
                 height,
                 transform,
                 fov=90):
        self.name = name
        self.camera_type = camera_type
        self.width = width
        self.height = height
        self.transform = transform
        self.fov = fov
        self.intrinsic_mat = create_intrinsic_matrix(
            self.width, self.height, self.fov)

    def get_intrinsic(self):
        return self.intrinsic_mat

    def get_transform(self):
        return self.transform

    def get_unreal_transform(self):
        return camera_to_unreal_transform(self.transform)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'CameraSetup(name: {}, type: {}, width: {}, height: {}, '\
            'transform: {}, fov: {}'.format(
                self.name, self.camera_type, self.width, self.height,
                self.transform, self.fov)


class LidarSetup(object):
    def __init__(self,
                 name,
                 lidar_type,
                 transform,
                 range,
                 rotation_frequency,
                 channels,
                 upper_fov,
                 lower_fov,
                 points_per_second):
        self.name = name
        self.lidar_type = lidar_type
        self.transform = transform
        self.range = range
        self.rotation_frequency = rotation_frequency
        self.channels = channels
        self.upper_fov = upper_fov
        self.lower_fov = lower_fov
        self.points_per_second = points_per_second

    def get_transform(self):
        return self.transform

    def get_unreal_transform(self):
        return lidar_to_unreal_transform(self.transform)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'LidarSetup(name: {}, type: {}, transform: {}, range: {}, '\
            'rotation freq: {}, channels: {}, upper_fov: {}, lower_fov: {}, '\
            'points_per_second: {}'.format(
                self.name, self.lidar_type, self.transform, self.range,
                self.rotation_frequency, self.channels, self.upper_fov,
                self.lower_fov, self.points_per_second)


class IMUSetup(object):
    def __init__(self, name, imu_type, transform):
        self.name = name
        self.imu_type = imu_type
        self.transform = transform

    def get_transform(self):
        return self.transform

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "IMUSetup(name: {}, type: {}, transform: {})".format(
            self.name, self.imu_type, self.transform)


class CanBus(object):
    def __init__(self, transform, forward_speed):
        self.transform = transform
        # Forward speed in m/s.
        self.forward_speed = forward_speed

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "CanBus(transform: {}, forward speed: {})".format(
            self.transform, self.forward_speed)


class BoundingBox(object):
    """ The Pylot version of the carla.BoundingBox instance that defines helper
    functions needed in Pylot, and makes the class serializable.

    Attributes:
        transform: The transform of the bounding box (rotation is (0, 0, 0))
        extent: The extent of the bounding box.
    """

    def __init__(self, carla_bb):
        """ Initializes the BoundingBox instance from the given
        carla.BoundingBox instance.

        Args:
            carla_bb: The carla.BoundingBox instance to initialize from.
        """
        loc = Location(carla_location=carla_bb.location)
        rot = Rotation(0, 0, 0)
        self.transform = Transform(loc, rot)
        self.extent = Vector3D(carla_bb.extent.x, carla_bb.extent.y,
                               carla_bb.extent.z)

    def as_carla_bounding_box(self):
        """ Retrieves the current BoundingBox as an instance of
        carla.BoundingBox

        Returns:
            A carla.BoundingBox instance that represents the current bounding
            box.
        """
        bb_loc = self.transform.location.as_carla_location()
        bb_extent = self.extent.as_carla_vector()
        return carla.BoundingBox(bb_loc, bb_extent)

    def visualize(self, world, actor_transform, time_between_frames=100):
        """ Visualize the given bounding box on the world.

        Args:
            world: The world instance to visualize the bounding box on.
            actor_transform: The current transform of the actor that the
                bounding box is of.
            time_between_frames: Time in ms to show the bounding box for.
        """
        bb = self.as_carla_bounding_box()
        bb.location += actor_transform.location()
        world.debug.draw_box(bb,
                             actor_transform.rotation.as_carla_rotation(),
                             life_time=time_between_frames / 1000.0)

    def to_camera_view(self, object_transform, extrinsic_matrix,
                       intrinsic_matrix):
        """ Converts the coordinates of the bounding box for the given object
        to the coordinates in the view of the camera.

        This method retrieves the extent of the bounding box, transforms them
        to coordinates relative to the bounding box origin, then converts those
        to coordinates relative to the object.

        These coordinates are then considered to be in the world coordinate
        system, which is mapped into the camera view. A negative z-value
        signifies that the bounding box is behind the camera plane.

        Note that this function does not cap the coordinates to be within the
        size of the camera image.

        Args:
            object_transform: The transform of the object that the bounding
                box is associated with.
            extrinsic_matrix: The extrinsic matrix of the camera.
            intrinsic_matrix: The intrinsic matrix of the camera.

        Returns:
            A list of 8 Location instances specifying the 8 corners of the
            bounding box.
        """
        # Retrieve the eight coordinates of the bounding box with respect to
        # the origin of the bounding box.
        import numpy as np
        extent = self.extent
        bbox = np.array([
            Location(x=+extent.x, y=+extent.y, z=-extent.z),
            Location(x=-extent.x, y=+extent.y, z=-extent.z),
            Location(x=-extent.x, y=-extent.y, z=-extent.z),
            Location(x=+extent.x, y=-extent.y, z=-extent.z),
            Location(x=+extent.x, y=+extent.y, z=+extent.z),
            Location(x=-extent.x, y=+extent.y, z=+extent.z),
            Location(x=-extent.x, y=-extent.y, z=+extent.z),
            Location(x=+extent.x, y=-extent.y, z=+extent.z),
        ])

        # Transform the vertices with respect to the bounding box transform.
        bbox = self.transform.transform_points(bbox)

        # Convert the bounding box relative to the world.
        bbox = object_transform.transform_points(bbox)

        # Object's transform is relative to the world. Thus, the bbox contains
        # the 3D bounding box vertices relative to the world.
        camera_coordinates = []
        for vertex in bbox:
            location_2D = vertex.to_camera_view(extrinsic_matrix,
                                                intrinsic_matrix)

            # Add the points to the image.
            camera_coordinates.append(location_2D)

        return camera_coordinates

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "BoundingBox(transform: {}, extent: {})".format(
            self.transform, self.extent)


class Vector3D(object):
    """ Represents a 3D vector and provides useful helper functions.

    Attributes:
        x: The value of the first axis.
        y: The value of the second axis.
        z: The value of the third axis.
    """

    def __init__(self, x=None, y=None, z=None, carla_vector=None):
        """ Initializes the Vector3D instance from the given x, y and z values.
        Alternatively, pass in a carla_vector (carla.vector3D).

        Args:
            x: The value of the first axis.
            y: The value of the second axis.
            z: The value of the third axis.
            carla_vector: carla.vector3D
        """
        if carla_vector is not None:
            x = carla_vector.x
            y = carla_vector.y
            z = carla_vector.z
        self.x, self.y, self.z = x, y, z

    def __add__(self, other):
        """ Adds the two vectors together and returns the result. """
        return type(self)(x=self.x + other.x,
                          y=self.y + other.y,
                          z=self.z + other.z)

    def __sub__(self, other):
        """ Subtracts the other vector from self and returns the result. """
        return type(self)(x=self.x - other.x,
                          y=self.y - other.y,
                          z=self.z - other.z)

    def as_numpy_array(self):
        """ Retrieves the given vector as a numpy array. """
        import numpy as np
        return np.array([self.x, self.y, self.z])

    def as_carla_vector(self):
        """ Retrieves the given vector as an instance of carla.Vector3D. """
        return carla.Vector3D(self.x, self.y, self.z)

    def magnitude(self):
        """ Returns the magnitude of the Vector3D instance. """
        import numpy as np
        return np.linalg.norm(self.as_numpy_array())

    def to_camera_view(self, extrinsic_matrix, intrinsic_matrix):
        """ Converts the given 3D vector to the view of the camera using
        the extrinsic and the intrinsic matrix.

        Args:
            extrinsic_matrix: The extrinsic matrix of the camera.
            intrinsic_matrix: The intrinsic matrix of the camera.

        Returns:
            An instance with the coordinates converted to the camera view.
        """
        import numpy as np
        position_vector = np.array([[self.x], [self.y], [self.z], [1.0]])

        # Transform the points to the camera in 3D.
        transformed_3D_pos = np.dot(np.linalg.inv(extrinsic_matrix),
                                    position_vector)

        # Transform the points to 2D.
        position_2D = np.dot(intrinsic_matrix, transformed_3D_pos[:3])

        # Normalize the 2D points.
        location_2D = type(self)(float(position_2D[0] / position_2D[2]),
                                 float(position_2D[1] / position_2D[2]),
                                 position_2D[2])
        return location_2D

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Vector3D(x={}, y={}, z={})'.format(self.x, self.y, self.z)


class Rotation(object):
    """ The Pylot version of the carla.Rotation instance that defines helper
    functions needed in Pylot, and makes the class serializable.

    Attributes:
        pitch: Rotation about Y-axis.
        yaw:   Rotation about Z-axis.
        roll:  Rotation about X-axis.
    """

    def __init__(self, pitch=0, yaw=0, roll=0, carla_rotation=None):
        """ Initializes the Rotation instance with either the given pitch,
        roll and yaw values or from the carla.Rotation instance, if specified.

        The carla.Rotation instance, if provided, takes precedence over the
        pitch, roll, yaw values provided in the constructor.

        Args:
            pitch: Rotation about Y-axis.
            yaw:   Rotation about Z-axis.
            roll:  Rotation about X-axis.
            carla_rotation: The carla.Rotation instance to instantiate this
                Rotation instance from.
        """
        if carla_rotation is not None:
            self.pitch = carla_rotation.pitch
            self.yaw = carla_rotation.yaw
            self.roll = carla_rotation.roll
        else:
            self.pitch = pitch
            self.yaw = yaw
            self.roll = roll

    def as_carla_rotation(self):
        """ Retrieves the current rotation as an instance of carla.Rotation.

        Returns:
            A carla.Rotation instance representing the current rotation.
        """
        return carla.Rotation(self.pitch, self.yaw, self.roll)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Rotation(pitch={}, yaw={}, roll={})'.format(
            self.pitch, self.yaw, self.roll)


class Location(Vector3D):
    """ The Pylot version of the carla.Location instance that defines helper
    functions needed in Pylot, and makes the class serializable.

    Attributes:
        x: The value of the x-axis.
        y: The value of the y-axis.
        z: The value of the z-axis.
    """

    def __init__(self, x=0, y=0, z=0, carla_location=None):
        """ Initializes the Location instance with either the given x, y, z
        values or from the carla.Location instance if specified.

        The carla.Location instance, if provided, takes precedence over the
        x, y, z values provided in the constructor.

        Args:
            x: The value of the x-axis.
            y: The value of the y-axis.
            z: The value of the z-axis.
            carla_location: The carla.Location instance to instantiate this
                Location instance from.
        """
        if carla_location is not None:
            super(Location, self).__init__(carla_location.x, carla_location.y,
                                           carla_location.z)
        else:
            super(Location, self).__init__(x, y, z)

    def distance(self, other):
        """ Calculates the Euclidean distance between the given point and the
        other point.

        Args:
            other: The other Location instance to calculate the distance to.

        Returns:
            The Euclidean distance between the two points.
        """
        return (self - other).magnitude()

    def as_carla_location(self):
        """ Retrieves the current location as an instance of carla.Location.

        Returns:
            A carla.Location instance representing the current location.
        """
        return carla.Location(self.x, self.y, self.z)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Location(x={}, y={}, z={})'.format(self.x, self.y, self.z)


class Transform(object):
    """ The Pylot version of the carla.Transform instance that defines helper
    functions needed in Pylot, and makes the class serializable.

    Attributes:
        location: The location of the object represented by the transform.
        rotation: The rotation of the object represented by the transform.
        matrix: The transformation matrix used to convert points in the 3D
            coordinate space with respect to the location and rotation of the
            given object.
    """

    # Rotations are applied in the order: Roll (X), Pitch (Y), Yaw (Z).
    # A 90-degree "Roll" maps the positive Z-axis to the positive Y-axis.
    # A 90-degree "Pitch" maps the positive X-axis to the positive Z-axis.
    # A 90-degree "Yaw" maps the positive X-axis to the positive Y-axis.

    def __init__(self,
                 location=None,
                 rotation=None,
                 forward_vector=None,
                 matrix=None,
                 carla_transform=None):
        """ Instantiates a Transform object with either the given location
        and rotation, or using the given matrix.

        Rotation is assumed in degrees.

        First precedence is for carla_transform and then for matrix.

        Args:
            location: The location of the object represented by the transform.
            rotation: The rotation of the object represented by the transform.
            forward_vector: The forward vector of the object represented by
                the transform.
            matrix: The transformation matrix used to convert points in the
                3D coordinate space with respect to the object.
            carla_transform: The carla.Transform to use to initialize the
                transform instance.
        """
        if carla_transform:
            self.location = Location(carla_location=carla_transform.location)
            self.rotation = Rotation(carla_transform.rotation.pitch,
                                     carla_transform.rotation.yaw,
                                     carla_transform.rotation.roll)
            fwd_vec = carla_transform.get_forward_vector()
            self.forward_vector = Vector3D(fwd_vec.x, fwd_vec.y, fwd_vec.z)
            self.matrix = Transform._create_matrix(self.location,
                                                   self.rotation)
        elif matrix is not None:
            import numpy as np
            self.matrix = matrix
            self.location = Location(matrix[0, 3], matrix[1, 3], matrix[2, 3])

            # Forward vector is retrieved from the matrix.
            self.forward_vector = Vector3D(self.matrix[0, 0],
                                           self.matrix[1, 0],
                                           self.matrix[2, 0])
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
                                           self.matrix[1, 0],
                                           self.matrix[2, 0])

    @staticmethod
    def _create_matrix(location, rotation):
        """ Creates a transformation matrix to convert points in the 3D world
        coordinate space with respect to the object.

        Use the transform_points function to transpose a given set of points
        with respect to the object.

        Args:
            location: The location of the object represented by the transform.
            rotation: The rotation of the object represented by the transform.
        Returns:
            a 4x4 numpy matrix which represents the transformation matrix.
        """
        matrix = np.matrix(np.identity(4))
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

    def transform_points(self, points):
        """ Transforms a given set of points in the 3D world coordinate space
        with respect to the object represented by the transform.

        Expected point format:
            List of Location values.

        Args:
            points: Points in the format [Location,..Location]

        Returns:
            Transformed points in the format [Location,..Location]
        """
        # Retrieve the locations as numpy arrays.
        points = np.matrix([loc.as_numpy_array() for loc in points])

        # Needed format: [[X0,..Xn],[Y0,..Yn],[Z0,..Zn]].
        # So let's transpose the point matrix.
        points = points.transpose()

        # Add 1s row: [[X0..,Xn],[Y0..,Yn],[Z0..,Zn],[1,..1]]
        points = np.append(points, np.ones((1, points.shape[1])), axis=0)

        # Point transformation
        points = np.dot(self.matrix, points)

        # Get all but the last row in array form.
        points = np.asarray(points[0:3].transpose())

        return [Location(x, y, z) for x, y, z in points]

    def inverse_transform(self):
        """ Returns the inverse of the given transform. """
        import numpy as np
        return Transform(matrix=np.linalg.inv(self.matrix))

    def as_carla_transform(self):
        """ Convert the transform to a carla.Transform instance.

        Returns:
            A carla.Transform instance representing the current Transform.
        """
        return carla.Transform(
            carla.Location(self.location.x, self.location.y, self.location.z),
            carla.Rotation(pitch=self.rotation.pitch,
                           yaw=self.rotation.yaw,
                           roll=self.rotation.roll))

    def __mul__(self, other):
        new_matrix = np.dot(self.matrix, other.matrix)
        return Transform(matrix=new_matrix)

    def __str__(self):
        if self.location:
            return "Transform(location: {}, rotation: {})".format(
                self.location, self.rotation)
        else:
            return "Transform({})".format(str(self.matrix))


def depth_to_array(image):
    """
    Convert an image containing CARLA encoded depth-map to a 2D array
    containing the depth value of each pixel normalized between [0.0, 1.0].
    """
    array = to_bgra_array(image)
    array = array.astype(np.float32)
    # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
    normalized_depth = np.dot(array[:, :, :3], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
    return normalized_depth


def to_bgra_array(image):
    """Convert a CARLA raw image to a BGRA np array."""
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    return array


def labels_to_array(image):
    """
    Convert an image containing CARLA semantic segmentation labels to a 2D
    array containing the label of each pixel.
    """
    return to_bgra_array(image)[:, :, 2]


def get_speed(velocity_vector):
    """ Compute the speed of the vehicle in m/s.

    Args:
        vehicle: A velocity vector.

    Returns:
        The speed of the given vehicle as a float in m/s.
    """
    speed = math.sqrt(velocity_vector.x**2 + velocity_vector.y**2 +
                      velocity_vector.z**2)
    return speed


def create_intrinsic_matrix(width, height, fov=90.0):
    # (Intrinsic) K Matrix
    k = np.identity(3)
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0
    k[0, 0] = k[1, 1] = width / (2.0 * math.tan(fov * math.pi / 360.0))
    return k


def depth_to_local_point_cloud(depth_frame, width, height, fov, max_depth=0.9):
    """
    Convert a CARLA-encoded depth-map to a 2D array containing
    the 3D position (relative to the camera) of each pixel.
    "max_depth" is used to omit the points that are far enough.

    Args:
        depth_frame: the normalized depth frame
        width: frame width
        height: frame height
        fov: camera field of view
    """
    far = 1000.0  # max depth in meters.
    normalized_depth = depth_frame
    intrinsic_mat = create_intrinsic_matrix(width, height, fov)
    # 2d pixel coordinates
    pixel_length = width * height
    u_coord = repmat(np.r_[0:width:1], height, 1).reshape(pixel_length)
    v_coord = repmat(np.c_[0:height:1], 1, width).reshape(pixel_length)
    normalized_depth = np.reshape(normalized_depth, pixel_length)

    # Search for pixels where the depth is greater than max_depth to
    # delete them
    max_depth_indexes = np.where(normalized_depth > max_depth)
    normalized_depth = np.delete(normalized_depth, max_depth_indexes)
    u_coord = np.delete(u_coord, max_depth_indexes)
    v_coord = np.delete(v_coord, max_depth_indexes)

    # p2d = [u,v,1]
    p2d = np.array([u_coord, v_coord, np.ones_like(u_coord)])

    # P = [X,Y,Z]
    p3d = np.dot(inv(intrinsic_mat), p2d)
    p3d *= normalized_depth * far

    # [[X1,Y1,Z1],[X2,Y2,Z2], ... [Xn,Yn,Zn]]
    # Return the points as location,
    locations = [
        Location(x, y, z) for x, y, z in np.asarray(np.transpose(p3d))
    ]
    return locations


def camera_to_unreal_transform(transform):
    """
    Takes in a Transform that occurs in unreal coordinates,
    and converts it into a Transform that goes from camera
    coordinates to unreal coordinates.
    """
    to_unreal_transform = Transform(matrix=np.array(
        [[0, 0, 1, 0],
         [1, 0, 0, 0],
         [0, -1, 0, 0],
         [0, 0, 0, 1]]))
    return transform * to_unreal_transform


def lidar_to_unreal_transform(transform):
    """
    Takes in a Transform that occurs in unreal coordinates,
    and converts it into a Transform that goes from lidar
    coordinates to unreal coordinates.
    """
    to_unreal_transform = Transform(matrix=np.array(
        [[0, -1, 0, 0],
         [-1, 0, 0, 0],
         [0, 0, -1, 0],
         [0, 0, 0, 1]]))
    return transform * to_unreal_transform


def lidar_to_camera_transform(transform):
    """
    Takes in a Transform that occurs in camera coordinates,
    and converts it into a Transform that goes from lidar
    coordinates to camera coordinates.
    """
    to_camera_transform = Transform(matrix=np.array(
        [[1, 0, 0, 0],
         [0, 0, 1, 0],
         [0, -1, 0, 0],
         [0, 0, 0, 1]]))
    return transform * to_camera_transform


def get_3d_world_position_with_depth_map(
        x, y, depth_frame, width, height, fov, camera_transform):
    """ Gets the 3D world position from pixel coordinates using a depth frame.

        Args:
            x: Pixel x coordinate.
            y: Pixel y coordinate.
            depth_frame: Normalized depth frame.
            width: frame width
            height: frame height
            fov: camera field of view
            camera_transform: Camera transform relative to the world.

       Returns:
            3D world location.
    """
    far = 1.0
    point_cloud = depth_to_local_point_cloud(
        depth_frame, width, height, fov, max_depth=far)
    # Transform the points in 3D world coordinates.
    to_world_transform = camera_to_unreal_transform(camera_transform)
    point_cloud = to_world_transform.transform_points(point_cloud)
    return point_cloud[y * width + x]


def batch_get_3d_world_position_with_depth_map(
        xs, ys, depth_frame, width, height, fov, camera_transform):
    """ Gets the 3D world positions from pixel coordinates using a depth frame.

        Args:
            xs: List of pixel x coordinate.
            ys: List of pixel y coordinate.
            depth_frame: Normalized depth frame.
            width: frame width
            height: frame height
            fov: camera field of view
            camera_transform: Camera transform relative to the world.

       Returns:
            List of 3D world locations.
    """
    assert len(xs) == len(ys)
    far = 1.0
    point_cloud = depth_to_local_point_cloud(
        depth_frame, width, height, fov, max_depth=far)
    # Transform the points in 3D world coordinates.
    to_world_transform = camera_to_unreal_transform(camera_transform)
    point_cloud = to_world_transform.transform_points(point_cloud)
    return [point_cloud[ys[i] * width + xs[i]] for i in range(len(xs))]


def find_point_depth(x, y, point_cloud):
    """ Finds the closest depth normalized point cloud point to x, y."""
    if len(point_cloud) == 0:
        return None
    # Select only points that are in front.
    point_cloud = point_cloud[np.where(point_cloud[:, 2] > 0.0)]
    # Select x and y.
    pc_xy = point_cloud[:, 0:2]
    # Select z
    pc_z = point_cloud[:, 2]
    # Divize x, y by z
    normalized_pc = pc_xy / pc_z[:, None]
    xy = np.array([x, y]).transpose()
    # Compute distance
    dist = np.sum((normalized_pc - xy)**2, axis=1)
    # Select index of the closest point.
    closest_index = np.argmin(dist)
    # Return the closest point.
    return tuple(point_cloud[closest_index])


def lidar_point_cloud_to_camera_coordinates(point_cloud):
    """ Transforms a point cloud from lidar to camera coordinates."""
    point_cloud = [Location(x, y, z) for x, y, z in np.asarray(point_cloud)]
    identity_transform = Transform(
        matrix=np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]]))
    transform = lidar_to_camera_transform(identity_transform)
    transformed_points = transform.transform_points(point_cloud)
    return [[loc.x, loc.y, loc.z] for loc in transformed_points]



def get_3d_world_position_with_point_cloud(
        u, v, pc, camera_transform, width, height, fov):
    """ Gets the 3D world position from pixel coordiantes using a Lidar
        point cloud.

        Args:
            u: Pixel x coordinate.
            v: Pixel y coordinate.
            pc: Point cloud in camera coordinates.
            camera_transform: Camera transform relative to the world.
            width: frame width
            height: frame height
            fov: camera field of view
       Returns:
            3D world location or None if it could not be computed.
    """
    intrinsic_mat = create_intrinsic_matrix(width, height, fov)
    # Project our 2D pixel location into 3D space, onto the z=1 plane.
    p3d = np.dot(inv(intrinsic_mat), np.array([[u], [v], [1.0]]))
    depth = find_point_depth(p3d[0], p3d[1], np.array(pc))
    if depth:
        # Normalize our point to have the same depth as our closest point.
        p3d *= np.array([depth[2]])
        p3d_locations = [
            Location(x, y, z) for x, y, z in np.asarray(p3d.transpose())
        ]
        # Convert from camera to unreal coordinates.
        to_world_transform = camera_to_unreal_transform(camera_transform)
        point_cloud = to_world_transform.transform_points(p3d_locations)
        return point_cloud[0]
    else:
        return None


def get_depth(vehicle_transform, obj_transform):
    """ Retrieves the depth of the object in front of the vehicle.

    This function aims to provide an estimate of the depth returned by the
    depth camera sensor of the given object.

    Args:
        vehicle_transform: The transform of the ego vehicle in world
            coordinates.
        obj_transform: The transform of the object in the world coordinates.

    Returns:
        The depth of the object.
    """
    # Get location of the ego vehicle.
    ego_vehicle_location = vehicle_transform.location.as_numpy_array()

    # Get location of the other object.
    obj_location = obj_transform.location.as_numpy_array()

    # Calculate the vector from the ego vehicle to the object.
    # Scale it by the forward vector, and calculate the norm.
    relative_vector = ego_vehicle_location - obj_location
    return np.linalg.norm(relative_vector *
                          vehicle_transform.forward_vector.as_numpy_array())


def get_bounding_box_in_camera_view(bb_coordinates, image_width, image_height):
    """ Creates the bounding box in the view of the camera image using the
    coordinates generated with respect to the camera transform.

    Args:
        bb_coordinates: The coordinates of the bounding box relative to the
            camera transform.
        image_width: The width of the image being published by the camera.
        image_height: The height of the image being published by the camera.

    Returns:
        None, if the bounding box does not fall into the view of the camera.
        (x1, x2, y1, y2) otherwise, which depict the bottom left and the top
        right point of the bounding box.
    """
    # Make sure that atleast 2 of the bounding box coordinates are in front.
    z_vals = [z for _, _, z in bb_coordinates if z >= 0]
    if len(z_vals) < 2:
        return None

    # Create the thresholding line segments of the camera view.
    from shapely.geometry import LineString
    left = LineString(((0, 0), (0, image_height)))
    bottom = LineString(((0, image_height), (image_width, image_height)))
    right = LineString(((image_width, image_height), (image_width, 0)))
    top = LineString(((image_width, 0), (0, 0)))
    camera_thresholds = [left, bottom, right, top]

    def threshold(p1, p2):
        points = []

        # If the points are themselves within the image, add them to the
        # set of thresholded points.
        if p1[0] >= 0 and p1[0] < image_width and p1[1] >= 0 and p1[
                1] < image_height:
            points.append(p1)

        if p2[0] >= 0 and p2[0] < image_width and p2[1] >= 0 and p2[
                1] < image_height:
            points.append(p2)

        # Compute the intersection of the line segment formed by p1 -- p2
        # with all the thresholds of the camera image.
        p12 = LineString((p1, p2))
        for camera_threshold in camera_thresholds:
            p = p12.intersection(camera_threshold)
            if not p.is_empty:
                if p.geom_type == 'Point':
                    points.append((p.x, p.y))
                elif p.geom_type == 'LineString':
                    for coord in p.coords:
                        points.append((coord[0], coord[1]))
        return points

    # Go over each of the segments of the bounding box and threshold it to
    # be inside the image.
    thresholded_points = []

    points_2D = [(int(x), int(y)) for x, y, _ in bb_coordinates]

    # Bottom plane thresholded.
    thresholded_points.extend(threshold(points_2D[0], points_2D[1]))
    thresholded_points.extend(threshold(points_2D[1], points_2D[2]))
    thresholded_points.extend(threshold(points_2D[2], points_2D[3]))
    thresholded_points.extend(threshold(points_2D[3], points_2D[0]))

    # Top plane thresholded.
    thresholded_points.extend(threshold(points_2D[4], points_2D[5]))
    thresholded_points.extend(threshold(points_2D[5], points_2D[6]))
    thresholded_points.extend(threshold(points_2D[6], points_2D[7]))
    thresholded_points.extend(threshold(points_2D[7], points_2D[4]))

    # Remaining segments thresholded.
    thresholded_points.extend(threshold(points_2D[0], points_2D[4]))
    thresholded_points.extend(threshold(points_2D[1], points_2D[5]))
    thresholded_points.extend(threshold(points_2D[2], points_2D[6]))
    thresholded_points.extend(threshold(points_2D[3], points_2D[7]))

    if len(thresholded_points) == 0:
        return None
    else:
        x = [int(x) for x, _ in thresholded_points]
        y = [int(y) for _, y in thresholded_points]
        return min(x), max(x), min(y), max(y)


def get_2d_bbox_from_3d_box(vehicle_transform,
                            obj_transform,
                            obj_bounding_box,
                            rgb_transform,
                            rgb_intrinsic,
                            rgb_image_size,
                            depth_array,
                            segmented_image,
                            segmentation_class,
                            segmentation_threshold=0.20,
                            depth_threshold=5,
                            max_depth=125):
    """ Retrieves the 2D bounding box with respect to the camera view from the
    given 3D bounding box.

    Args:
        vehicle_transform: The transform in world coordinates of the ego
            vehicle.
        obj_transform: The transform in world coordinates of the object.
        obj_bounding_box: The bounding box in 3D coordinates of the object.
        rgb_transform: The transform of the RGB camera respective to the
            ego vehicle.
        rgb_image_size: The (width, height) of the images produced by the
            camera.
        depth_array: The sensor data returned by the depth camera.
        segmented_image: The sensor data returned by the semantic segmentation
            camera.
        segmentation_class: The segmentation class of the object.
        segmentation_threshold: The amount of pixels that the given
            segmentation class should occupy in the bounding box for a positive
            detection. (default=0.20)
        depth_threshold: The error to tolerate when comparing the calculated
            depth to the object and the depth returned by the sensor.
            (default=5 metres)
        max_depth: The max depth of the object after which it is no longer
            classified as a positive detection. (default = 125 metres)
    """
    # Calculate the depth of the object from the given transforms.
    # Return None if the object is farther than the threshold.
    depth = get_depth(vehicle_transform, obj_transform)
    if depth > max_depth:
        return None

    # Convert the bounding box of the object to the camera coordinates.
    extrinsic_matrix = (vehicle_transform * rgb_transform).matrix
    bb_coordinates = obj_bounding_box.to_camera_view(obj_transform,
                                                     extrinsic_matrix,
                                                     rgb_intrinsic)

    # Threshold the bounding box to be within the camera view.
    bb_coordinates = [(bb.x, bb.y, bb.z) for bb in bb_coordinates]
    thresholded_coordinates = get_bounding_box_in_camera_view(
        bb_coordinates, *rgb_image_size)
    if not thresholded_coordinates:
        return None

    # Retrieve the bottom left and the top right points of the bounding
    # box.
    xmin, xmax, ymin, ymax = thresholded_coordinates

    # Crop the segmented and depth image to the given bounding box.
    cropped_image = segmented_image[ymin:ymax, xmin:xmax]
    cropped_depth = depth_array[ymin:ymax, xmin:xmax]

    # If the size of the bounding box is greater than 0, ensure that the
    # bounding box contains more than a threshold of pixels corresponding
    # to the required segmentation class.
    if cropped_image.size > 0:
        masked_image = np.zeros_like(cropped_image)
        masked_image[np.where(cropped_image == segmentation_class)] = 1
        if np.sum(masked_image) >= segmentation_threshold * masked_image.size:
            # The bounding box contains the required number of pixels that
            # belong to the required class. Ensure that the depth of the
            # object is the depth in the image.
            masked_depth = cropped_depth[np.where(masked_image == 1)]
            mean_depth = np.mean(masked_depth) * 1000
            if depth - depth_threshold <= mean_depth <= depth + depth_threshold:
                return xmin, xmax, ymin, ymax
    return None


def transform_traffic_light_bboxes(light, points):
    """ Transforms the bounding box specified in the points relative to the
    light.

    Args:
        light: TrafficLight object representing the light.
        points: An array of length 4 representing the 4 points of the
            rectangle.
    """

    def rotate(yaw, location):
        """ Rotate a given 3D vector around the Z-axis. """
        rotation_matrix = np.identity(3)
        rotation_matrix[0, 0] = np.cos(yaw)
        rotation_matrix[0, 1] = -np.sin(yaw)
        rotation_matrix[1, 0] = np.sin(yaw)
        rotation_matrix[1, 1] = np.cos(yaw)
        location_vector = np.array([[location.x], [location.y], [location.z]])
        transformed = np.dot(rotation_matrix, location_vector)
        return Location(x=transformed[0, 0],
                        y=transformed[1, 0],
                        z=transformed[2, 0])

    transformed_points = [
        rotate(np.radians(light.transform.rotation.yaw), point)
        for point in points
    ]
    base_relative_points = [
        light.transform.location + point for point in transformed_points
    ]
    return base_relative_points


def is_traffic_light_visible(camera_transform,
                             tl,
                             town_name=None,
                             distance_threshold=70):
    # We dot product the forward vectors (i.e., orientation).
    # Note: we have to rotate the traffic light forward vector
    # so that it's pointing out from the traffic light in the
    # opposite direction in which the ligth is beamed.
    prod = np.dot([
        tl.transform.forward_vector.y, -tl.transform.forward_vector.x,
        tl.transform.forward_vector.z
    ], [
        camera_transform.forward_vector.x, camera_transform.forward_vector.y,
        camera_transform.forward_vector.z
    ])
    if tl.transform.location.distance(
            camera_transform.location) > distance_threshold:
        return prod > 0.4

    if town_name is None:
        return prod > -0.80
    else:
        if town_name == 'Town01' or town_name == 'Town02':
            return prod > 0.3
        return prod > -0.80


def get_traffic_lights_bbox_state(camera_transform, traffic_lights, town_name):
    bbox_state = []
    # Carla has differing placemnts for different towns.
    if town_name == 'Town01' or town_name == 'Town02':
        points = [
            # Back Plane
            Location(x=-0.5, y=-0.1, z=2),
            Location(x=+0.1, y=-0.1, z=2),
            Location(x=+0.1, y=-0.1, z=3),
            Location(x=-0.5, y=-0.1, z=3),

            # Front Plane
            Location(x=-0.5, y=0.5, z=2),
            Location(x=+0.1, y=0.5, z=2),
            Location(x=+0.1, y=0.5, z=3),
            Location(x=-0.5, y=0.5, z=3),
        ]
        for light in traffic_lights:
            bbox_state.append(
                (transform_traffic_light_bboxes(light, points), light.state))
    elif town_name == 'Town03':
        for light in traffic_lights:
            if light.trigger_volume_extent.x > 2 or light.id in [
                    66, 67, 68, 71, 72, 73, 75, 81,
            ]:
                points = [
                    # Back Plane
                    Location(x=-5.2, y=-0.2, z=5.5),
                    Location(x=-4.8, y=-0.2, z=5.5),
                    Location(x=-4.8, y=-0.2, z=6.5),
                    Location(x=-5.2, y=-0.2, z=6.5),

                    # Front Plane
                    Location(x=-5.2, y=0.4, z=5.5),
                    Location(x=-4.8, y=0.4, z=5.5),
                    Location(x=-4.8, y=0.4, z=6.5),
                    Location(x=-5.2, y=0.4, z=6.5),

                ]
                bbox_state.append(
                    (transform_traffic_light_bboxes(light, points),
                     light.state))
                right_points = [
                    point + Location(x=-3.0) for point in points
                ]
                bbox_state.append(
                    (transform_traffic_light_bboxes(light, right_points),
                     light.state))
                if light.id not in [51, 52, 53]:
                    left_points = [
                        point + Location(x=-6.5) for point in points
                    ]
                    bbox_state.append(
                        (transform_traffic_light_bboxes(light, left_points),
                         light.state))

            else:
                points = [
                    # Back Plane
                    Location(x=-0.5, y=-0.1, z=2),
                    Location(x=+0.1, y=-0.1, z=2),
                    Location(x=+0.1, y=-0.1, z=3),
                    Location(x=-0.5, y=-0.1, z=3),

                    # Front Plane
                    Location(x=-0.5, y=0.5, z=2),
                    Location(x=+0.1, y=0.5, z=2),
                    Location(x=+0.1, y=0.5, z=3),
                    Location(x=-0.5, y=0.5, z=3),
                ]
                bbox_state.append(
                    (transform_traffic_light_bboxes(light, points),
                     light.state))
    elif town_name == 'Town04':
        points = [
            # Back Plane
            Location(x=-5.2, y=-0.2, z=5.5),
            Location(x=-4.8, y=-0.2, z=5.5),
            Location(x=-4.8, y=-0.2, z=6.5),
            Location(x=-5.2, y=-0.2, z=6.5),

            # Front Plane
            Location(x=-5.2, y=0.4, z=5.5),
            Location(x=-4.8, y=0.4, z=5.5),
            Location(x=-4.8, y=0.4, z=6.5),
            Location(x=-5.2, y=0.4, z=6.5),
        ]
        middle_points = [  # Light in the middle of the pole.
            # Back Plane
            Location(x=-0.5, y=-0.1, z=2.5),
            Location(x=+0.1, y=-0.1, z=2.5),
            Location(x=+0.1, y=-0.1, z=3.5),
            Location(x=-0.5, y=-0.1, z=3.5),

            # Front Plane
            Location(x=-0.5, y=0.5, z=2.5),
            Location(x=+0.1, y=0.5, z=2.5),
            Location(x=+0.1, y=0.5, z=3.5),
            Location(x=-0.5, y=0.5, z=3.5),
        ]
        right_points = [point + Location(x=-3.0) for point in points]
        left_points = [point + Location(x=-5.5) for point in points]
        for light in traffic_lights:
            bbox_state.append(
                (transform_traffic_light_bboxes(light, points),
                 light.state))
            if light.trigger_volume_extent.x > 5:
                # This is a traffic light with 4 signs, we need to come up with
                # more bounding boxes.
                bbox_state.append(
                    (transform_traffic_light_bboxes(light, middle_points),
                     light.state))
                bbox_state.append(
                    (transform_traffic_light_bboxes(light, right_points),
                     light.state))
                bbox_state.append(
                    (transform_traffic_light_bboxes(light, left_points),
                     light.state))
    elif town_name == 'Town05':
        points = [
            # Back Plane
            Location(x=-5.2, y=-0.2, z=5.5),
            Location(x=-4.8, y=-0.2, z=5.5),
            Location(x=-4.8, y=-0.2, z=6.5),
            Location(x=-5.2, y=-0.2, z=6.5),

            # Front Plane
            Location(x=-5.2, y=0.4, z=5.5),
            Location(x=-4.8, y=0.4, z=5.5),
            Location(x=-4.8, y=0.4, z=6.5),
            Location(x=-5.2, y=0.4, z=6.5),
        ]
        middle_points = [  # Light in the middle of the pole.
            # Back Plane
            Location(x=-0.4, y=-0.1, z=2.55),
            Location(x=+0.2, y=-0.1, z=2.55),
            Location(x=+0.2, y=-0.1, z=3.55),
            Location(x=-0.4, y=-0.1, z=3.55),

            # Front Plane
            Location(x=-0.4, y=0.5, z=2.55),
            Location(x=+0.2, y=0.5, z=2.55),
            Location(x=+0.2, y=0.5, z=3.55),
            Location(x=-0.5, y=0.5, z=3.55),
        ]
        right_points = [point + Location(x=-3.0) for point in points]
        left_points = [point + Location(x=-5.5) for point in points]

        # Town05 randomizes the identifiers for the traffic light at each
        # reload of the world. We cannot depend on static identifiers for
        # figuring out which lights only have a single traffic light.
        single_light = filter(lambda light: light.trigger_volume_extent.x < 2,
                              traffic_lights)
        if len(single_light) != 1:
            raise ValueError(
                "Expected a single traffic light with a trigger "
                "volume less than 2 in Town05. Received {}".format(
                    len(single_light)))
        single_light = single_light[0]
        single_light_ids = map(attrgetter('id'),
                               single_light.get_group_traffic_lights())
        for light in traffic_lights:
            bbox_state.append(
                (transform_traffic_light_bboxes(light, points),
                 light.state))
            if light.id not in single_light_ids:
                # This is a traffids light with 4 signs, we need to come up
                # with more bounding boxes.
                bbox_state.append(
                    (transform_traffic_light_bboxes(light, middle_points),
                     light.state))
                bbox_state.append(
                    (transform_traffic_light_bboxes(light, right_points),
                     light.state))
                bbox_state.append(
                    (transform_traffic_light_bboxes(light, left_points),
                     light.state))
    else:
        raise ValueError('Could not find a town named {}'.format(town_name))
    return bbox_state


def get_traffic_light_det_objs(traffic_lights,
                               camera_transform,
                               depth_array,
                               segmented_image,
                               frame_width,
                               frame_height,
                               town_name,
                               fov=90):
    """ Get the traffic lights that are within the camera frame.
    Note: This method should be used with Carla 0.9.*
    """
    # Create the extrinsic and intrinsic matrices for the given camera.
    extrinsic_matrix = camera_to_unreal_transform(camera_transform).matrix
    intrinsic_matrix = create_intrinsic_matrix(frame_width, frame_height, fov)

    # Iterate over all the traffic lights, and figure out which ones are
    # facing us and are visible in the camera view.
    detected = []
    for light in traffic_lights:

        if not is_traffic_light_visible(camera_transform, light, town_name):
            continue

        bboxes = get_traffic_lights_bbox_state(camera_transform, [light],
                                               town_name)

        # Convert the returned bounding boxes to 2D and check if the
        # light is occluded. If not, add it to the detected object list.
        for box, color in bboxes:
            bounding_box = [
                loc.to_camera_view(extrinsic_matrix, intrinsic_matrix)
                for loc in box
            ]
            bounding_box = [(bb.x, bb.y, bb.z) for bb in bounding_box]
            thresholded_coordinates = get_bounding_box_in_camera_view(
                bounding_box, frame_width, frame_height)
            if not thresholded_coordinates:
                continue

            xmin, xmax, ymin, ymax = thresholded_coordinates

            # Crop the segmented and depth image to the given bounding box.
            cropped_image = segmented_image[ymin:ymax, xmin:xmax]
            cropped_depth = depth_array[ymin:ymax, xmin:xmax]

            if cropped_image.size > 0:
                masked_image = np.zeros_like(cropped_image)
                masked_image[np.where(cropped_image == 12)] = 1
                if np.sum(masked_image) >= 0.20 * masked_image.size:
                    masked_depth = cropped_depth[np.where(masked_image == 1)]
                    mean_depth = np.mean(masked_depth) * 1000
                    if abs(mean_depth -
                           bounding_box[0][-1]) <= 2 and mean_depth < 150:
                        detected.append(
                            DetectedObject((xmin, xmax, ymin, ymax), 1.0,
                                           color.get_label()))
    return detected


def get_speed_limit_det_objs(
        speed_signs, vehicle_transform, camera_transform, depth_frame,
        frame_width, frame_height, fov, segmented_frame):
    """ Get the speed limit signs that are withing the camera frame.

    Args:
        speed_signs: List of speed limit signs in the world.
        vehicle_transform: Ego-vehicle transform in world coordinates.
        camera_transform: Camera transform in world coordinates.
        fov: Camera field of view.
        segmented_frame: Segmented frame.
    """
    # Compute the bounding boxes.
    traffic_signs_frame = get_traffic_sign_pixels(segmented_frame)
    bboxes = get_bounding_boxes_from_segmented(
        traffic_signs_frame, min_width=8, min_height=9)

    # Get the positions of the bounding box centers.
    x_mids = [(bbox[0] + bbox[1]) / 2 for bbox in bboxes]
    y_mids = [(bbox[2] + bbox[3]) / 2 for bbox in bboxes]
    pos_3d = batch_get_3d_world_position_with_depth_map(
        x_mids, y_mids, depth_frame, frame_width, frame_height,
        fov, camera_transform)
    pos_and_bboxes = zip(pos_3d, bboxes)
    ts_bboxes = _match_bboxes_with_speed_signs(
        vehicle_transform, pos_and_bboxes, speed_signs)

    det_objs = [DetectedSpeedLimit(bbox, limit, 1.0, 'speed limit')
                for (bbox, limit) in ts_bboxes]
    return det_objs


def _match_bboxes_with_speed_signs(vehicle_transform, pos_bboxes, speed_signs):
    result = []
    for pos, bbox in pos_bboxes:
        best_ts = None
        best_dist = 1000000
        for ts in speed_signs:
            dist = ((pos.x - ts.transform.location.x)**2 +
                    (pos.y - ts.transform.location.y)**2)
            if (dist < best_dist):
                best_dist = dist
                best_ts = ts
        if not best_ts:
            continue
        # Check that the sign is facing the ego vehicle.
        yaw_diff = (best_ts.transform.rotation.yaw -
                    vehicle_transform.rotation.yaw)
        if yaw_diff < 0:
            yaw_diff += 360
        elif yaw_diff >= 360:
            yaw_diff -= 360
        if best_dist < 5 ** 2 and yaw_diff > 30 and yaw_diff < 150:
            result.append((bbox, best_ts.limit))
    return result


def _get_stop_markings_bbox(
        bbox3d,
        depth_frame,
        camera_transform,
        camera_intrinsic,
        frame_width,
        frame_height):
    """ Gets a 2D stop marking bouding box from a 3D bounding box."""
    # Offset trigger_volume by -0.85 so that the top plane is on the ground.
    ext_z_value = bbox3d.extent.z - 0.85
    ext = [
        Location(x=+bbox3d.extent.x, y=+bbox3d.extent.y, z=ext_z_value),
        Location(x=+bbox3d.extent.x, y=-bbox3d.extent.y, z=ext_z_value),
        Location(x=-bbox3d.extent.x, y=+bbox3d.extent.y, z=ext_z_value),
        Location(x=-bbox3d.extent.x, y=-bbox3d.extent.y, z=ext_z_value),
    ]
    bbox = bbox3d.transform.transform_points(ext)
    coords = []
    for loc in bbox:
        loc_view = loc.to_camera_view(camera_transform.matrix,
                                      camera_intrinsic)
        if (loc_view.z >= 0 and loc_view.x >= 0 and loc_view.y >= 0 and
            loc_view.x < frame_width and loc_view.y < frame_height):
            coords.append(loc_view)
    if len(coords) == 4:
        xmin = min(coords[0].x, coords[1].x, coords[2].x, coords[3].x)
        xmax = max(coords[0].x, coords[1].x, coords[2].x, coords[3].x)
        ymin = min(coords[0].y, coords[1].y, coords[2].y, coords[3].y)
        ymax = max(coords[0].y, coords[1].y, coords[2].y, coords[3].y)
        # Check if the bbox is not obstructed and if it's sufficiently
        # big for the text to be readable.
        if (ymax - ymin > 15 and
            have_same_depth(int(coords[0].x),
                            int(coords[0].y),
                            coords[0].z,
                            depth_frame,
                            0.4)):
            return (int(xmin), int(xmax), int(ymin), int(ymax))
    return None


def have_same_depth(x, y, z, depth_array, threshold):
    x, y = int(x), int(y)
    return abs(depth_array[y][x] * 1000 - z) < threshold


def get_traffic_stop_det_objs(
        traffic_stops,
        camera_transform,
        depth_frame,
        frame_width,
        frame_height,
        fov):
    """ Get traffic stop lane markings that are withing the camera frame.

    Args:
        traffic_stops: List of traffic stop actors in the world.
        camera_transform: Camera transform in world coordinates.
        fov: Camera field of view.

    Returns:
        List of DetectedObjects.
    """
    det_objs = []
    bgr_intrinsic = create_intrinsic_matrix(frame_width, frame_height, fov)
    for transform, bbox in traffic_stops:
        bbox2d = _get_stop_markings_bbox(
            bbox, depth_frame, camera_transform, bgr_intrinsic,
            frame_width, frame_height)
        if bbox2d:
            det_objs.append(DetectedObject(bbox2d, 1.0, 'stop marking'))
    return det_objs


class TrafficLight(object):
    """ The Pylot version of a carla TrafficLight that defines helper
    functions needed in Pylot, and makes the class serializable.

    Args:
        id: The identifier of the TrafficLight.
        transform: The transform of the TrafficLight.
        trigger_volume_extent: The extent of the trigger volume of the light.
        state: The state of the light. (Green/Yellow/Red/Off)
    """

    def __init__(self, traffic_light):
        """ Initializes the TrafficLight instance with the given carla
        TrafficLight instance.

        Args:
            traffic_light: The carla.TrafficLight instance to initialize this
                instance with.
        """
        # Retrieve the ID of the TrafficLight.
        self.id = traffic_light.id

        # Retrieve the Transform of the TrafficLight.
        self.transform = Transform(
            carla_transform=traffic_light.get_transform())

        # Retrieve the Trigger Volume of the TrafficLight.
        self.trigger_volume_extent = Vector3D(
            traffic_light.trigger_volume.extent.x,
            traffic_light.trigger_volume.extent.y,
            traffic_light.trigger_volume.extent.z)

        # Retrieve the State of the TrafficLight (Red/Yellow/Green)
        from pylot.perception.detection.utils import TrafficLightColor
        traffic_light_state = traffic_light.get_state()
        if traffic_light_state == carla.TrafficLightState.Red:
            self.state = TrafficLightColor.RED
        elif traffic_light_state == carla.TrafficLightState.Yellow:
            self.state = TrafficLightColor.YELLOW
        elif traffic_light_state == carla.TrafficLightState.Green:
            self.state = TrafficLightColor.GREEN
        else:
            self.state = TrafficLightColor.OFF


def kalman_step(y, A, B, c, D, e, u, Q, R, init_x, init_V):
    '''
    Args: 
        y - observations
        A, B, C, d:  x(:,t+1) = A x(:,t) + B u(:,t) + c + w(:,t) 
                     y(:,t)   = D x(:,t) + e      + v(:,t)
        Q - covariance matrix of system x(t+1)=A*x(t)+w(t) , w(t)~N(0,Q)
        R - covariance matrix of output y(t)=C*x(t)+v(t) , v(t)~N(0,R)
        init_x - initial/prev mean
        init_V - initial/prev covariance
    Return: 
        xfilt - Next State
        Vfilt - Estimate Covariance
    '''

    error = np.zeros([y.shape[0]])
    xpred = np.zeros([init_x.shape[0]])
    xfilt = np.zeros_like(xpred)
    Vpred = np.zeros([init_V.shape[0], init_V.shape[1]])
    Vfilt = np.zeros_like(Vpred)
  
    xpred = A.dot(init_x) + B.dot(u) + c
    Vpred = A.dot(init_V).dot(A.T) + Q
    error = y - (D.dot(xpred) + e)
    S = D.dot(Vpred).dot(D.T) + R 
    K = Vpred.dot(D.T).dot( np.linalg.inv(S)) 

    xfilt = xpred + K.dot(error)
    Vfilt = (np.eye(K.shape[0]) - K.dot(D)).dot(Vpred)

    return xfilt,Vfilt
