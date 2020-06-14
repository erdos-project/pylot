import numpy as np

from pylot.utils import Location, Rotation, Transform


def create_rgb_camera_setup(camera_name,
                            camera_location,
                            width,
                            height,
                            fov=90):
    """ Creates an RGBCameraSetup instance with the given values.

    The Rotation is set to (pitch=0, yaw=0, roll=0).

    Args:
        camera_name (str): The name of the camera instance.
        camera_location (:py:class:`~pylot.utils.Location`): The location of
            the camera with respect to the center of the vehicle.
        width (int): The width of the image returned by the camera.
        height (int): The height of the image returned by the camera.
        fov (float): The field of view of the image returned by the camera.

    Returns:
        :py:class:`~pylot.drivers.sensor_setup.RGBCameraSetup`: A camera
        setup with the given parameters.
    """
    transform = Transform(camera_location, Rotation())
    return RGBCameraSetup(camera_name, width, height, transform, fov)


def create_depth_camera_setup(camera_name_prefix,
                              camera_location,
                              width,
                              height,
                              fov=90):
    """ Creates a DepthCameraSetup instance with the given values.

    The Rotation is set to (pitch=0, yaw=0, roll=0).

    Args:
        camera_name_prefix (str): The name of the camera instance. A suffix
            of "_depth" is appended to the name.
        camera_location (:py:class:`~pylot.utils.Location`): The location of
            the camera with respect to the center of the vehicle.
        width (int): The width of the image returned by the camera.
        height (int): The height of the image returned by the camera.
        fov (float): The field of view of the image returned by the camera.

    Returns:
        :py:class:`~pylot.drivers.sensor_setup.DepthCameraSetup`: A camera
        setup with the given parameters.
    """
    transform = Transform(camera_location, Rotation())
    return DepthCameraSetup(camera_name_prefix + '_depth',
                            width,
                            height,
                            transform,
                            fov=fov)


def create_segmented_camera_setup(camera_name_prefix,
                                  camera_location,
                                  width,
                                  height,
                                  fov=90):
    """ Creates a SegmentedCameraSetup instance with the given values.

    The Rotation is set to (pitch=0, yaw=0, roll=0).

    Args:
        camera_name_prefix (str): The name of the camera instance. A suffix
            of "_segmented" is appended to the name.
        camera_location (:py:class:`~pylot.utils.Location`): The location of
            the camera with respect to the center of the vehicle.
        width (int): The width of the image returned by the camera.
        height (int): The height of the image returned by the camera.
        fov (float): The field of view of the image returned by the camera.

    Returns:
        :py:class:`~pylot.drivers.sensor_setup.SegmentedCameraSetup`: A
        camera setup with the given parameters.
    """
    transform = Transform(camera_location, Rotation())
    return SegmentedCameraSetup(camera_name_prefix + '_segmented',
                                width,
                                height,
                                transform,
                                fov=fov)


def create_left_right_camera_setups(camera_name_prefix,
                                    location,
                                    width,
                                    height,
                                    camera_offset,
                                    fov=90):
    """ Creates a dual-RGB-camera setup with the center at the given location,
    and the two cameras on either side of the center at a distance specified
    by the camera_offset.

    The Rotation is set to (pitch=0, yaw=0, roll=0).

    Args:
        camera_name_prefix (str): The name of the camera instance. A suffix
            of "_left" and "_right" is appended to the name.
        location (:py:class:`~pylot.utils.Location`): The location of the
            center of the cameras with respect to the center of the vehicle.
        width (int): The width of the image returned by the cameras.
        height (int): The height of the image returned by the cameras.
        camera_offset (float): The offset of the two cameras from the center.
        fov (float): The field of view of the image returned by the cameras.

    Returns:
        tuple: A tuple containing two instances of
        :py:class:`~pylot.drivers.sensor_setup.RGBCameraSetup` for the left
        and right camera setups with the given parameters.
    """
    rotation = Rotation()
    left_loc = location + Location(0, -camera_offset, 0)
    right_loc = location + Location(0, camera_offset, 0)
    left_transform = Transform(left_loc, rotation)
    right_transform = Transform(right_loc, rotation)
    left_camera_setup = RGBCameraSetup(camera_name_prefix + '_left',
                                       width,
                                       height,
                                       left_transform,
                                       fov=fov)
    right_camera_setup = RGBCameraSetup(camera_name_prefix + '_right',
                                        width,
                                        height,
                                        right_transform,
                                        fov=fov)
    return (left_camera_setup, right_camera_setup)


def create_center_lidar_setup(location, rotation_frequency=20):
    """ Creates a LidarSetup instance with the given location.

    The Rotation is set to (pitch=0, roll=0, yaw=0).

    Args:
        location (:py:class:`~pylot.utils.Location`): The location of the
            LIDAR with respect to the center of the vehicle.

    Returns:
        :py:class:`~pylot.drivers.sensor_setup.LidarSetup`: A LidarSetup
        with the given location.
    """
    rotation = Rotation()
    # Place the lidar in the same position as the camera.
    lidar_transform = Transform(location, rotation)
    return LidarSetup(
        name='front_center_lidar',
        lidar_type='sensor.lidar.ray_cast',
        transform=lidar_transform,
        range=10000,  # in centimeters
        rotation_frequency=rotation_frequency,
        channels=32,
        upper_fov=15,
        lower_fov=-30,
        points_per_second=250000)


class CameraSetup(object):
    """ CameraSetup stores information about an instance of the camera
    mounted on the vehicle.

    Args:
        name (str): The name of the camera instance.
        camera_type (str): The type of the camera. One of `('sensor.camera.rgb'
            , 'sensor.camera.depth', 'sensor.camera.semantic_segmentation')`
        width (int): The width of the image returned by the camera.
        height (int): The height of the image returned by the camera.
        transform (:py:class:`~pylot.utils.Transform`): The transform
            containing the location and rotation of the camera instance with
            respect to the vehicle.
        fov (float): The field-of-view of the camera.

    Attributes:
        name (str): The name of the camera instance.
        camera_type (str): The type of the camera. One of `('sensor.camera.rgb'
            , 'sensor.camera.depth', 'sensor.camera.semantic_segmentation')`
        width (int): The width of the image returned by the camera.
        height (int): The height of the image returned by the camera.
        transform (:py:class:`~pylot.utils.Transform`): The transform
            containing the location and rotation of the camera instance with
            respect to the vehicle.
        fov (float): The field-of-view of the camera.
    """
    def __init__(self, name, camera_type, width, height, transform, fov=90):
        # Ensure that the name is a string.
        assert isinstance(name, str), "The name should be of type 'str'"
        self.name = name

        # Ensure that the camera type is one of the three that we support.
        assert camera_type in (
            'sensor.camera.rgb', 'sensor.camera.depth',
            'sensor.camera.semantic_segmentation'), "The camera_type " \
            "should belong to ('sensor.camera.rgb', 'sensor.camera.depth', " \
            "'sensor.camera.semantic_segmentation')"
        self.camera_type = camera_type

        # The width of the image produced by the camera should be > 1.
        assert width > 1, "Valid camera setup should have width > 1"
        assert isinstance(width, int) and isinstance(height, int), "The width"
        " and height should be of type 'int'"
        self.width, self.height = width, height

        # Ensure that the transform is of the type pylot.Transform.
        assert isinstance(transform, Transform), "The given transform is not "
        "of the type pylot.utils.Transform"
        self.transform = transform

        # Ensure that the field-of-view is a float.
        fov = float(fov) if isinstance(fov, int) else fov
        assert isinstance(fov, float), "The fov should be of type 'float'"
        self.fov = fov

        # Generate the intrinsic and extrinsic matrices.
        self._intrinsic_mat = CameraSetup.__create_intrinsic_matrix(
            self.width, self.height, self.fov)
        self._unreal_transform = CameraSetup.__create_unreal_transform(
            self.transform)

    @staticmethod
    def __create_intrinsic_matrix(width, height, fov):
        """ Creates the intrinsic matrix for a camera with the given
        parameters.

        Args:
            width (int): The width of the image returned by the camera.
            height (int): The height of the image returned by the camera.
            fov (float): The field-of-view of the camera.

        Returns:
            :py:class:`numpy.ndarray`: A 3x3 intrinsic matrix of the camera.
        """
        import numpy as np
        k = np.identity(3)
        # We use width - 1 and height - 1 to find the center column and row
        # of the image, because the images are indexed from 0.

        # Center column of the image.
        k[0, 2] = (width - 1) / 2.0
        # Center row of the image.
        k[1, 2] = (height - 1) / 2.0
        # Focal length.
        k[0, 0] = k[1, 1] = (width - 1) / (2.0 * np.tan(fov * np.pi / 360.0))
        return k

    @staticmethod
    def __create_unreal_transform(transform):
        """ Converts a Transform from the camera coordinate space to the
        Unreal coordinate space.

        The camera space is defined as:
            +x to right, +y to down, +z into the screen.

        The unreal coordinate space is defined as:
            +x into the screen, +y to right, +z to up.

        Args:
            transform (:py:class:`~pylot.utils.Transform`): The transform to
                convert to Unreal coordinate space.

        Returns:
            :py:class:`~pylot.utils.Transform`: The given transform after
                transforming to the Unreal coordinate space.
        """
        import numpy as np
        to_unreal_transform = Transform(matrix=np.array(
            [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]))
        return transform * to_unreal_transform

    def get_intrinsic_matrix(self):
        """ Get the intrinsic matrix of the camera denoted by the CameraSetup.

        Returns:
            :py:class:`numpy.ndarray`: The 3x3 intrinsic matrix of the camera.
        """
        return self._intrinsic_mat

    def get_extrinsic_matrix(self):
        """ Get the extrinsic matrix of the camera denoted by the transform
        of the camera with respect to the vehicle to which it is attached.

        Returns:
            :py:class:`numpy.ndarray`: The 4x4 extrinsic matrix of the camera.
        """
        return self._unreal_transform.matrix

    def get_name(self):
        """ Get the name of the camera instance.

        Returns:
            str: The name of the camera instance.
        """
        return self.name

    def get_unreal_transform(self):
        """ Get the transform of the camera with respect to the vehicle in
        the Unreal Engine coordinate space.

        Returns:
            :py:class:`~pylot.utils.Transform`: The transform of the camera
            in the Unreal Engine coordinate space.
        """
        return self._unreal_transform

    def get_transform(self):
        """ Get the transform of the camera with respect to the vehicle to
        which it is attached.

        Returns:
            :py:class:`~pylot.utils.Transform`: The transform of the camera
            with respect to the vehicle to which it is attached.
        """
        return self.transform

    def set_transform(self, transform):
        """ Set the transform of the camera with respect to the vehicle to
        which it is attached.

        Args:
            transform (:py:class:`~pylot.utils.Transform`): The new transform
                of the camera with respect to the vehicle to which it is
                attached.
        """
        assert isinstance(transform, Transform), "The given transform is not "
        "of the type pylot.utils.Transform"
        self.transform = transform
        self._unreal_transform = CameraSetup.__create_unreal_transform(
            self.transform)

    def set_resolution(self, width, height):
        self.width = width
        self.height = height
        self._intrinsic_mat = CameraSetup.__create_intrinsic_matrix(
            self.width, self.height, self.fov)

    def get_fov(self):
        """ Get the field of view of the camera.

        Returns:
            float: The field of view of the given camera.
        """
        return self.fov

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'CameraSetup(name: {}, type: {}, width: {}, height: {}, '\
            'transform: {}, fov: {})'.format(
                self.name, self.camera_type, self.width, self.height,
                self.transform, self.fov)


class RGBCameraSetup(CameraSetup):
    """ A CameraSetup that denotes an RGB camera from Carla.

    Args:
        name (str): The name of the camera instance.
        width (int): The width of the image returned by the camera.
        height (int): The height of the image returned by the camera.
        transform (:py:class:`~pylot.utils.Transform`): The transform
            containing the location and rotation of the camera instance
            with respect to the vehicle.
        fov (float): The field-of-view of the camera.

    Attributes:
        name (str): The name of the camera instance.
        camera_type (str): The type of the camera. One of `('sensor.camera.rgb'
            , 'sensor.camera.depth', 'sensor.camera.semantic_segmentation')`
        width (int): The width of the image returned by the camera.
        height (int): The height of the image returned by the camera.
        transform (:py:class:`~pylot.utils.Transform`): The transform
            containing the location and rotation of the camera instance with
            respect to the vehicle.
        fov (float): The field-of-view of the camera.
    """
    def __init__(self, name, width, height, transform, fov=90):
        super(RGBCameraSetup, self).__init__(name, 'sensor.camera.rgb', width,
                                             height, transform, fov)


class DepthCameraSetup(CameraSetup):
    """ A CameraSetup that denotes a Depth camera from Carla.

    Args:
        name (str): The name of the camera instance.
        width (int): The width of the image returned by the camera.
        height (int): The height of the image returned by the camera.
        transform (:py:class:`~pylot.utils.Transform`): The transform
            containing the location and rotation of the camera instance with
            respect to the vehicle.
        fov (float): The field-of-view of the camera.

    Attributes:
        name (str): The name of the camera instance.
        camera_type (str): The type of the camera. One of `('sensor.camera.rgb'
            , 'sensor.camera.depth', 'sensor.camera.semantic_segmentation')`
        width (int): The width of the image returned by the camera.
        height (int): The height of the image returned by the camera.
        transform (:py:class:`~pylot.utils.Transform`): The transform
            containing the location and rotation of the camera instance with
            respect to the vehicle.
        fov (float): The field-of-view of the camera.
    """
    def __init__(self, name, width, height, transform, fov=90):
        super(DepthCameraSetup, self).__init__(name, 'sensor.camera.depth',
                                               width, height, transform, fov)


class SegmentedCameraSetup(CameraSetup):
    """ A CameraSetup that denotes a Semantic Segmentation camera from Carla.

    Args:
        name (str): The name of the camera instance.
        width (int): The width of the image returned by the camera.
        height (int): The height of the image returned by the camera.
        transform (:py:class:`~pylot.utils.Transform`): The transform
            containing the location and rotation of the camera instance with
            respect to the vehicle.
        fov (float): The field-of-view of the camera.

    Attributes:
        name (str): The name of the camera instance.
        camera_type (str): The type of the camera. One of `('sensor.camera.rgb'
            , 'sensor.camera.depth', 'sensor.camera.semantic_segmentation')`
        width (int): The width of the image returned by the camera.
        height (int): The height of the image returned by the camera.
        transform (:py:class:`~pylot.utils.Transform`): The transform
            containing the location and rotation of the camera instance with
            respect to the vehicle.
        fov (float): The field-of-view of the camera.
    """
    def __init__(self, name, width, height, transform, fov=90):
        super(SegmentedCameraSetup,
              self).__init__(name, 'sensor.camera.semantic_segmentation',
                             width, height, transform, fov)


class LidarSetup(object):
    """ LidarSetup stores information about an instance of LIDAR mounted on
    the vehicle.

    Args:
        name (str): The name of the LIDAR instance.
        lidar_type (str): The type of the LIDAR instance. Should be set to
            `'sensor.lidar.ray_cast'` currently.
        transform (:py:class:`~pylot.utils.Transform`): The transform
            containing the location and rotation of the LIDAR instance with
            respect to the vehicle.
        range (float): The range of the LIDAR (in centimeters).
        rotation_frequency (float): The rotation frequency of the LIDAR.
        channels (int): The number of channels output by the LIDAR.
        upper_fov (float): The upper_fov of the data output by the LIDAR.
        lower_fov (float): The lower_fov of the data output by the LIDAR.
        points_per_second (int): The number of points generated by the LIDAR
            per second.

    Attributes:
        name (str): The name of the LIDAR instance.
        lidar_type (str): The type of the LIDAR instance. Should be set to
            `'sensor.lidar.ray_cast'` currently.
        transform (:py:class:`~pylot.utils.Transform`): The transform
            containing the location and rotation of the LIDAR instance with
            respect to the vehicle.
        range (float): The range of the LIDAR (in centimeters).
        rotation_frequency (float): The rotation frequency of the LIDAR.
        channels (int): The number of channels output by the LIDAR.
        upper_fov (float): The upper_fov of the data output by the LIDAR.
        lower_fov (float): The lower_fov of the data output by the LIDAR.
        points_per_second (int): The number of points generated by the LIDAR
            per second.
    """
    def __init__(self,
                 name,
                 lidar_type,
                 transform,
                 range=5000,
                 rotation_frequency=20,
                 channels=32,
                 upper_fov=15,
                 lower_fov=-30,
                 points_per_second=500000):
        # Ensure that the name is a string.
        assert isinstance(name, str), "The name should be of type 'str'"
        self.name = name

        # Ensure that the type of LIDAR is currently supported.
        assert lidar_type == 'sensor.lidar.ray_cast' or lidar_type == 'velodyne', "The LIDAR should be of type 'sensor.lidar.ray_cast' or 'velodyne'"
        self.lidar_type = lidar_type

        # Ensure that the transform is of the correct type.
        assert isinstance(transform, Transform), "The given transform is not "
        "of the type pylot.utils.Transform"
        self.transform = transform

        # Try to coerce the range to float and throw an error if not possible.
        range = float(range) if isinstance(range, int) else range
        assert isinstance(range, float), "The range should be of type 'float'"
        self.range = range

        # Try to coerce the rotation_frequency to float and throw an error,
        # if not possible.
        rotation_frequency = float(rotation_frequency) if \
            isinstance(rotation_frequency, int) else rotation_frequency
        assert isinstance(rotation_frequency, float), "The rotation_frequency"
        " should be of type 'float'"
        self.rotation_frequency = rotation_frequency

        # Ensure that the channels are of correct type.
        assert isinstance(channels, int), "The channels should be of type "
        "'int'"
        self.channels = channels

        # Try to coerce the upper_fov and lower_fov to float, and throw an
        # error if not possible.
        upper_fov = float(upper_fov) if \
            isinstance(upper_fov, int) else upper_fov
        lower_fov = float(lower_fov) if \
            isinstance(lower_fov, int) else lower_fov
        assert isinstance(upper_fov, float) and isinstance(lower_fov, float),\
            "The upper_fov and lower_fov should be of type 'float'"
        self.upper_fov, self.lower_fov = upper_fov, lower_fov

        # Ensure that the points_per_second is of type 'int'
        assert isinstance(points_per_second, int), "The points_per_second"
        " should be of type 'int'"
        self.points_per_second = points_per_second
        self._unreal_transform = LidarSetup.__create_unreal_transform(
            self.transform)

    @staticmethod
    def __create_unreal_transform(transform):
        """ Converts a Transform from the LIDAR coordinate space to the
        Unreal Engine coordinate space.

        The LIDAR space is defined as:
            +x to the right, +y is out of the screen, +z is down.

        The Unreal Engine coordinate space is defined as:
            +x into the screen, +y to the right, +z is up.

        Args:
            transform(:py:class:`~pylot.utils.Transform`): The transform to
                convert to the Unreal Engine coordinate space.

        Returns:
            :py:class:`~pylot.utils.Transform`: The given transform after
                transforming to the Unreal Engine coordinate space.
        """
        to_camera_transform = Transform(matrix=np.array(
            [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]))
        return transform * to_camera_transform

    def get_name(self):
        """ Get the name of the LIDAR instance.

        Returns:
            str: The name of the LIDAR instance.
        """
        return self.name

    def get_transform(self):
        """ Get the transform of the LIDAR with respect to the vehicle to
        which it is attached.

        Returns:
            :py:class:`~pylot.utils.Transform`: The transform of the LIDAR with
            respect to the vehicle to which it is attached.
        """
        return self.transform

    def set_transform(self, transform):
        """ Set the transform of the LIDAR with respect to the vehicle to which
        it is attached.

        Args:
            transform (:py:class:`~pylot.utils.Transform`): The new transform
                of the LIDAR with respect to the vehicle to which it is
                attached.
        """
        assert isinstance(transform, Transform), "The given transform is not "
        "of the type pylot.utils.Transform"
        self.transform = transform
        self._unreal_transform = LidarSetup.__create_unreal_transform(
            self.transform)

    def get_unreal_transform(self):
        """ Get the transform of the LIDAR with respect to the vehicle in the
        Unreal Engine coordinate space.

        Returns:
            :py:class:`~pylot.utils.Transform`: The transform of the LIDAR with
            respect to the vehicle in the Unreal Engine coordinate space.
        """
        return self._unreal_transform

    def get_range_in_meters(self):
        """ Get the range of the LIDAR in metres.

        Returns:
            float: The range of the LIDAR in metres.
        """
        return self.range / 100

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
    """ IMUSetup stores information about an instance of the IMU sensor
    attached to the vehicle.

    Args:
        name (str): The name of the IMU instance.
        transform (:py:class:`pylot.utils.Tranform`): The transform containing
            the location and rotation of the IMU instance with respect to the
            vehicle.

    Attributes:
        name (str): The name of the IMU instance.
        transform (:py:class:`~pylot.utils.Transform`): The transform
            containing the location and rotation of the IMU instance with
            respect to the vehicle.
    """
    def __init__(self, name, transform):
        # Ensure that the name is of the correct type.
        assert isinstance(name, str), "The name should be of type 'str'"
        self.name = name

        # Ensure that the transform is of the correct type.
        assert isinstance(transform, Transform), "The transform should be of "
        "type 'pylot.utils.Transform'"
        self.transform = transform

    def get_name(self):
        """ Get the name of the IMU instance.

        Returns:
            str: The name of the IMU instance.
        """
        return self.name

    def get_transform(self):
        """ Get the transform of the IMU sensor with respect to the vehicle
        to which it is attached.

        Returns:
            :py:class:`~pylot.utils.Transform`: The transform of the IMU sensor
            with respect to the vehicle to which it is attached.
        """
        return self.transform

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "IMUSetup(name: {}, transform: {})".format(
            self.name, self.transform)


class GNSSSetup(object):
    """ GNSSSetup stores information about an instance of the GNSS sensor
    attached to the vehicle.

    Args:
        name (str): The name of the GNSS instance.
        transform (:py:class:`pylot.utils.Transform`): The transform containing
            the location and rotation of the GNSS instance with respect to the
            vehicle

    Attributes:
        name (str): The name of the GNSS instance.
        transform (:py:class:`pylot.utils.Transform`): The transform containing
            the location and rotation of the GNSS instance with respect to the
            vehicle
    """

    def __init__(self, name, transform):
        # Ensure that the name is of the correct type.
        assert isinstance(name, str), "The name should be of type `str`"
        self.name = name

        # Ensure that the transform is of the correct type.
        assert isinstance(transform, Transform), "The transform should be of "
        "type 'pylot.utils.Transform'"
        self.transform = transform

    def get_name(self):
        """ Get the name of the GNSS instance.

        Returns:
            str: The name of the GNSS instance.
        """
        return self.name

    def get_transform(self):
        """ Get the transform of the GNSS sensor with respect to the vehicle
        to which it is attached.

        Returns:
            :py:class:`~pylot.utils.Transform`: The transform of the GNSS
            sensor with respect to the vehicle to which it is attached.
        """
        return self.transform

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "GNSSSetup(name: {}, transform: {})".format(
            self.name, self.transform)
