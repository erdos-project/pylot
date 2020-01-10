import copy
import cv2
import math
import numpy as np
from numpy.linalg import inv
from numpy.matlib import repmat
import os
import PIL.Image as Image
import time

import pylot.perception.detection.utils


class Rotation(object):
    """ Used to represent the rotation of an actor or obstacle.

    Attributes:
        pitch: Rotation about Y-axis.
        yaw:   Rotation about Z-axis.
        roll:  Rotation about X-axis.
    """
    def __init__(self, pitch=0, yaw=0, roll=0):
        """ Initializes the Rotation instance with either the given pitch,
        roll and yaw values.

        Args:
            pitch: Rotation about Y-axis.
            yaw:   Rotation about Z-axis.
            roll:  Rotation about X-axis.
        """
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll

    @classmethod
    def from_carla_rotation(cls, rotation):
        """ Creates a Rotation from a carla.Rotation."""
        import carla
        if not isinstance(rotation, carla.Rotation):
            raise ValueError('rotation should be of type carla.Rotation')
        return cls(rotation.pitch, rotation.yaw, rotation.roll)

    def as_carla_rotation(self):
        """ Retrieves the current rotation as an instance of carla.Rotation.

        Returns:
            A carla.Rotation instance representing the current rotation.
        """
        import carla
        return carla.Rotation(self.pitch, self.yaw, self.roll)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Rotation(pitch={}, yaw={}, roll={})'.format(
            self.pitch, self.yaw, self.roll)


class Vector3D(object):
    """ Represents a 3D vector and provides useful helper functions.

    Attributes:
        x: The value of the first axis.
        y: The value of the second axis.
        z: The value of the third axis.
    """
    def __init__(self, x, y, z):
        """ Initializes the Vector3D instance from the given x, y and z values.

        Args:
            x: The value of the first axis.
            y: The value of the second axis.
            z: The value of the third axis.
        """
        self.x, self.y, self.z = x, y, z

    @classmethod
    def from_carla_vector(cls, vector):
        """ Creates a Vector3D from a carla.vector3d."""
        import carla
        if not isinstance(vector, carla.Vector3D):
            raise ValueError('The vector must be a carla.Vector3D')
        return cls(vector.x, vector.y, vector.z)

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
        return np.array([self.x, self.y, self.z])

    def as_carla_vector(self):
        """ Retrieves the given vector as an instance of carla.Vector3D. """
        import carla
        return carla.Vector3D(self.x, self.y, self.z)

    def magnitude(self):
        """ Returns the magnitude of the Vector3D instance. """
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


class Vector2D(object):
    """ Represents a 2D vector and provides helper functions."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        """ Adds the two vectors together and returns the result. """
        return type(self)(x=self.x + other.x, y=self.y + other.y)

    def __sub__(self, other):
        """ Subtracts the other vector from self and returns the result. """
        return type(self)(x=self.x - other.x, y=self.y - other.y)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Vector2D(x={}, y={})'.format(self.x, self.y)


class Location(Vector3D):
    """ Stores a 3D location, and provides useful helper methods.

    Attributes:
        x: The value of the x-axis.
        y: The value of the y-axis.
        z: The value of the z-axis.
    """
    def __init__(self, x=0, y=0, z=0):
        """ Initializes the Location instance with either the given x, y, z
        values.

        Args:
            x: The value of the x-axis.
            y: The value of the y-axis.
            z: The value of the z-axis.
        """
        super(Location, self).__init__(x, y, z)

    @classmethod
    def from_carla_location(cls, location):
        """ Creates a Location from a carla.Location."""
        import carla
        if not isinstance(location, carla.Location):
            raise ValueError('The location must be a carla.Location')
        return cls(location.x, location.y, location.z)

    def distance(self, other):
        """ Calculates the Euclidean distance between the given point and the
        other point.

        Args:
            other: The other Location instance to calculate the distance to.

        Returns:
            The Euclidean distance between the two points.
        """
        return (self - other).magnitude()

    def l1_distance(self, other):
        """ Calculates the L1 distance between the given point and the other
        point.

        Args:
            other: The other Location instance to calculate the L1 distance to.

        Returns:
            The L1 distance between the two points.
        """
        return abs(self.x - other.x) + abs(self.y - other.y) + abs(self.z -
                                                                   other.z)

    def as_carla_location(self):
        """ Retrieves the current location as an instance of carla.Location.

        Returns:
            A carla.Location instance representing the current location.
        """
        import carla
        return carla.Location(self.x, self.y, self.z)

    def is_within_distance_ahead(self, dst_loc, max_distance):
        """
        Check if a location is within a distance.

        Args:
            dst_loc: The location to compute distance for.
            max_distance: Maximum allowed distance.
        Returns:
            True if other location is within max_distance.
        """
        target_vector = np.array([dst_loc.x - self.x, dst_loc.y - self.y])
        norm_dst = np.linalg.norm(target_vector)
        # Return if the vector is too small.
        if norm_dst < 0.001:
            return True
        if norm_dst > max_distance:
            return False
        forward_vector = np.array([
            math.cos(math.radians(self.rotation.yaw)),
            math.sin(math.radians(self.rotation.yaw))
        ])
        d_angle = math.degrees(
            math.acos(np.dot(forward_vector, target_vector) / norm_dst))
        return d_angle < 90.0

    def compute_magnitude_angle(self, target_loc):
        """
        Computes relative angle and distance between the location and  a target
        location.

        Args:
            target_loc: Location of the target.

        Returns:
            Tuple of distance to the target and the angle
        """
        target_vector = np.array(
            [target_loc.x - self.x, target_loc.y - self.y])
        norm_target = np.linalg.norm(target_vector)

        forward_vector = np.array([
            math.cos(math.radians(self.rotation.yaw)),
            math.sin(math.radians(self.rotation.yaw))
        ])
        d_angle = math.degrees(
            math.acos(np.dot(forward_vector, target_vector) / norm_target))

        return (norm_target, d_angle)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Location(x={}, y={}, z={})'.format(self.x, self.y, self.z)


class Transform(object):
    """ A class that stores the location and rotation of an obstacle.

    It can be created from a carla.Transform, defines helper functions needed
    in Pylot, and makes the carla.Transform serializable.

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
                 matrix=None):
        """ Instantiates a Transform object with either the given location
        and rotation, or using the given matrix.

        Rotation is assumed in degrees.

        Args:
            location: The location of the object represented by the transform.
            rotation: The rotation of the object represented by the transform.
            forward_vector: The forward vector of the object represented by
                the transform.
            matrix: The transformation matrix used to convert points in the
                3D coordinate space with respect to the object.
        """
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
        """ Creates a Transform from a carla.Transform."""
        import carla
        if not isinstance(transform, carla.Transform):
            raise ValueError('transform should be of type carla.Transform')
        return cls(Location.from_carla_location(transform.location),
                   Rotation.from_carla_rotation(transform.rotation))

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

    def __transform(self, points, matrix):
        """ Internal function to transform the points according to the
        given matrix. This function either converts the points from
        coordinate space relative to the transform to the world coordinate
        space (using self.matrix), or from world coordinate space to the
        space relative to the transform (using inv(self.matrix))

        Args:
            points: Points in the format [Location, ... Location]
            matrix: The matrix of the transformation to apply.

        Returns:
            Transformed points in the format [Location, ... Location]
        """
        # Retrieve the locations as numpy arrays.
        points = np.matrix([loc.as_numpy_array() for loc in points])

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
        """ Transforms the given set of locations (specified in the coordinate
        space of the current transform) to be in the world coordinate space.

        For example, if the transform is at location (3, 0, 0) and the
        location passed to the argument is (10, 0, 0), this function will
        return (13, 0, 0) i.e. the location of the argument in the world
        coordinate space.

        Args:
            points: Points in the format [Location,..Location]

        Returns:
            Transformed points in the format [Location,..Location]
        """
        return self.__transform(points, self.matrix)

    def inverse_transform_points(self, points):
        """ Transforms the given set of locations (specified in world
        coordinate space) to be relative to the given transform.

        For example, if the transform is at location (3, 0, 0) and the
        location passed to the argument is (10, 0, 0), this function will
        return (7, 0, 0) i.e. the location of the argument relative to the
        given transform.

        Args:
            points: Points in the format [Location, ... Location]

        Returns:
            Transformed points in the format [Location, ... Location]
        """
        return self.__transform(points, np.linalg.inv(self.matrix))

    def as_carla_transform(self):
        """ Convert the transform to a carla.Transform instance.

        Returns:
            A carla.Transform instance representing the current Transform.
        """
        import carla
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


class CanBus(object):
    """ Class used to wrap ego-vehicle information."""
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


class CameraFrame(object):
    def __init__(self, frame, encoding):
        self.frame = frame
        if encoding != 'BGR' and encoding != 'RGB':
            raise ValueError('Unsupported encoding {}'.format(encoding))
        self.encoding = encoding
        self.width = self.frame.shape[1]
        self.height = self.frame.shape[0]

    @classmethod
    def from_carla_frame(cls, carla_frame):
        # Convert it to a BGR image and store it as a frame with the BGR
        # encoding.
        import carla
        if not isinstance(carla_frame, carla.Image):
            raise ValueError('carla_frame should be of type carla.Image')
        _frame = np.frombuffer(carla_frame.raw_data, dtype=np.dtype("uint8"))
        _frame = np.reshape(_frame, (carla_frame.height, carla_frame.width, 4))
        return cls(_frame[:, :, :3], 'BGR')

    def as_numpy_array(self):
        return self.frame.astype(np.uint8)

    def as_bgr_numpy_array(self):
        if self.encoding == 'RGB':
            return self.frame[:, :, ::-1]
        else:
            return self.frame

    def as_rgb_numpy_array(self):
        if self.encoding == 'BGR':
            return self.frame[:, :, ::-1]
        else:
            return self.frame

    def annotate_with_bounding_boxes(
        self,
        timestamp,
        detected_obstacles,
        bbox_color_map=pylot.perception.detection.utils.GROUND_COLOR_MAP):
        add_timestamp(self.frame, timestamp)
        for obstacle in detected_obstacles:
            obstacle.visualize_on_img(self.frame, bbox_color_map)

    def visualize(self, window_name, timestamp=None):
        """ Creates a cv2 window to visualize the camera frame."""
        if self.encoding != 'BGR':
            image_np = self.as_bgr_numpy_array()
        else:
            image_np = self.frame
        if timestamp is not None:
            add_timestamp(image_np, timestamp)
        cv2.imshow(window_name, image_np)
        cv2.waitKey(1)

    def save(self, timestamp, data_path, file_base):
        if self.encoding != 'RGB':
            image_np = self.as_rgb_numpy_array()
        else:
            image_np = self.frame
        file_name = os.path.join(data_path,
                                 '{}-{}.png'.format(file_base, timestamp))
        img = Image.fromarray(image_np)
        img.save(file_name)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'CameraFrame(width: {}, height: {}, encoding: {})'.format(
            self.width, self.height, self.encoding)


class DepthFrame(object):
    def __init__(self, frame, camera_setup):
        """ Initializes a depth frame.

        Args:
            frame: A numpy array storring the depth frame.
            camera_setup: The camera setup used to generate this frame.
        """
        self.frame = frame
        self.camera_setup = camera_setup

    @classmethod
    def from_carla_frame(cls, frame, camera_setup):
        """ Creates a DepthFrame from a carla depth frame."""
        # Convert an image containing CARLA encoded depth-map to a 2D
        # array containing the depth value of each pixel normalized
        # between [0.0, 1.0]
        _frame = np.frombuffer(frame.raw_data, dtype=np.dtype("uint8"))
        _frame = np.reshape(_frame, (frame.height, frame.width, 4))
        frame = _frame.astype(np.float32)
        # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
        frame = np.dot(frame[:, :, :3], [65536.0, 256.0, 1.0])
        frame /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
        return cls(frame, camera_setup)

    def as_numpy_array(self):
        return self.frame

    def pixel_has_same_depth(self, x, y, z, threshold):
        return abs(self.frame[int(y)][int(x)] * 1000 - z) < threshold

    def as_point_cloud(self):
        """
        Converts a CARLA-encoded depth-map to a 2D array containing
        the 3D position (relative to the camera) of each pixel.
        """
        far = 1000.0  # max depth in meters.
        intrinsic_mat = self.camera_setup.get_intrinsic_matrix()
        width, height = self.camera_setup.width, self.camera_setup.height
        # 2d pixel coordinates
        pixel_length = width * height
        u_coord = repmat(np.r_[0:width:1], height, 1).reshape(pixel_length)
        v_coord = repmat(np.c_[0:height:1], 1, width).reshape(pixel_length)
        normalized_depth = np.reshape(self.frame, pixel_length)

        # p2d = [u,v,1]
        p2d = np.array([u_coord, v_coord, np.ones_like(u_coord)])

        # P = [X,Y,Z]
        p3d = np.dot(inv(intrinsic_mat), p2d)
        p3d *= normalized_depth * far

        # [[X1,Y1,Z1],[X2,Y2,Z2], ... [Xn,Yn,Zn]]
        locations = [
            Location(x, y, z) for x, y, z in np.asarray(np.transpose(p3d))
        ]
        # Transform the points in 3D world coordinates.
        to_world_transform = self.camera_setup.get_unreal_transform()
        point_cloud = to_world_transform.transform_points(locations)
        return point_cloud

    def get_pixel_locations(self, pixels):
        """ Gets the 3D world locations from pixel coordinates.

        Args:
            pixels: List of pylot.utils.Vector2D pixel coordinates.
        Returns:
            List of pylot.utils.Locations
        """
        point_cloud = self.as_point_cloud()
        return [
            point_cloud[pixel.y * self.camera_setup.width + pixel.x]
            for pixel in pixels
        ]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'DepthFrame(camera_setup: {})'.format(self.camera_setup)


class PointCloud(object):
    def __init__(self, points, transform):
        """ Initializes the point cloud.

        Args:
            points: A list of pylot.utils.Location.
            transform: utils.Transform of the point cloud, relative to the
                ego-vehicle.
        """
        # Transform point cloud from lidar to camera coordinates.
        self.points = self._to_camera_coordinates(points)
        self.transform = transform

    @classmethod
    def from_carla_point_cloud(cls, carla_pc, transform):
        # Transform the raw_data into a point cloud.
        points = np.frombuffer(carla_pc.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 3), 3))
        point_cloud = [Location(x, y, z) for x, y, z in np.asarray(points)]
        return cls(point_cloud, transform)

    def _to_camera_coordinates(self, points):
        identity_transform = Transform(matrix=np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
        # Converts a transform that occurs in camera coordinates into a
        # transform that goes from lidar coordinates to camera coordinates.
        to_camera_transform = Transform(matrix=np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]))
        transform = identity_transform * to_camera_transform
        transformed_points = transform.transform_points(points)
        return [[loc.x, loc.y, loc.z] for loc in transformed_points]

    def get_pixel_location(self, pixel, camera_setup):
        """ Gets the 3D world location from pixel coordinates.

        Args:
            pixel: A pylot.utils.Vector2D denoting pixel coordinates.
            camera_setup: The setup of the camera.
        Returns:
            A pylot.utils.Location of the 3D world location, or None if all the
            point cloud points are behind.
        """
        if len(self.points) == 0:
            return None
        intrinsic_mat = camera_setup.get_intrinsic_matrix()
        # Project our 2D pixel location into 3D space, onto the z=1 plane.
        p3d = np.dot(inv(intrinsic_mat), np.array([[pixel.x], [pixel.y],
                                                   [1.0]]))
        location = self._get_closest_point_in_point_cloud(
            Vector2D(p3d[0], p3d[1]))
        # Normalize our point to have the same depth as our closest point.
        p3d *= np.array([location.z])
        p3d_locations = [
            Location(px, py, pz) for px, py, pz in np.asarray(p3d.transpose())
        ]
        # Convert from camera to unreal coordinates.
        to_world_transform = camera_setup.get_unreal_transform()
        camera_point_cloud = to_world_transform.transform_points(p3d_locations)
        return camera_point_cloud[0]

    def _get_closest_point_in_point_cloud(self, pixel):
        """ Finds the closest depth normalized point cloud point."""
        # Select only points that are in front.
        fwd_points = self.points[np.where(self.points[:, 2] > 0.0)]
        # Select x and y.
        pc_xy = fwd_points[:, 0:2]
        # Select z
        pc_z = fwd_points[:, 2]
        # Divize x, y by z
        normalized_pc = pc_xy / pc_z[:, None]
        xy = np.array([pixel.x, pixel.y]).transpose()
        # Compute distance
        dist = np.sum((normalized_pc - xy)**2, axis=1)
        # Select index of the closest point.
        closest_index = np.argmin(dist)
        # Return the closest point.
        return Location(fwd_points[closest_index][0],
                        fwd_points[closest_index][1],
                        fwd_points[closest_index][2])

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'PointCloud(transform: {}, number of points: {})'.format(
            self.transform, len(self.points))


def add_timestamp(image_np, timestamp):
    """ Adds a timestamp text to an image np array."""
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
    """ Get current time in milliseconds."""
    return int(time.time() * 1000)


def set_tf_loglevel(level):
    """ To be used to suppress TensorFlow logging."""
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
