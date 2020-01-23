import numpy as np

from pylot.utils import Location, Rotation, Transform


def create_rgb_camera_setup(camera_name,
                            camera_location,
                            width,
                            height,
                            fov=90):
    transform = Transform(camera_location, Rotation())
    return RGBCameraSetup(camera_name, width, height, transform, fov)


def create_depth_camera_setup(camera_name_prefix,
                              camera_location,
                              width,
                              height,
                              fov=90):
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


def create_center_lidar_setup(location):
    rotation = Rotation()
    # Place the lidar in the same position as the camera.
    lidar_transform = Transform(location, rotation)
    return LidarSetup(
        name='front_center_lidar',
        lidar_type='sensor.lidar.ray_cast',
        transform=lidar_transform,
        range=5000,  # in centimers
        rotation_frequency=20,
        channels=32,
        upper_fov=15,
        lower_fov=-30,
        points_per_second=500000)


def create_imu_setup(location):
    return IMUSetup(name='imu', transform=Transform(location, Rotation()))


class CameraSetup(object):
    """ CameraSetup stores information about an instance of the camera
    mounted on the vehicle.

    Attributes:
        name: The name of the camera instance.
        camera_type: The type of the camera. One of ('sensor.camera.rgb',
            'sensor.camera.depth', 'sensor.camera.semantic_segmentation')
        width: The width of the image returned by the camera.
        height: The height of the image returned by the camera.
        transform: The transform containing the location and rotation of the
            camera instance with respect to the vehicle.
        fov: The field-of-view of the camera.
    """

    def __init__(self, name, camera_type, width, height, transform, fov=90):
        """ Initializes the CameraSetup with the given parameters.

        Args:
            name: The name of the camera instance.
            camera_type: The type of the camera. One of ('sensor.camera.rgb',
                'sensor.camera.depth', 'sensor.camera.semantic_segmentation')
            width: The width of the image returned by the camera.
            height: The height of the image returned by the camera.
            transform: The transform containing the location and rotation of
                the camera instance with respect to the vehicle.
            fov: The field-of-view of the camera.
        """
        self.name = name

        # Ensure that the camera type is one of the three that we support.
        assert camera_type in ('sensor.camera.rgb', 'sensor.camera.depth', \
                'sensor.camera.semantic_segmentaion'), "The camera_type "
        "should belong to ('sensor.camera.rgb', 'sensor.camera.depth', "
        "'sensor.camera.semantic_segmentation')"
        self.camera_type = camera_type

        # The width of the image produced by the camera should be > 1.
        assert width > 1, "Valid camera setup should have width > 1"
        self.width, self.height = width, height

        # Ensure that the transform is of the type pylot.Transform.
        assert isinstance(transform, Transform), "The given transform is not "
        "of the type pylot.utils.Transform"
        self._transform = transform

        self.fov = fov

        # Generate the intrinsic and extrinsic matrices.
        self._intrinsic_mat = CameraSetup.__create_intrinsic_matrix(
            self.width, self.height, self.fov)
        self._unreal_transform = CameraSetup.__create_unreal_transform(
            self._transform)

    @staticmethod
    def __create_intrinsic_matrix(width, height, fov):
        """ Creates the intrinsic matrix for a camera with the given
        parameters.

        Args:
            width: The width of the image returned by the camera.
            height: The height of the image returned by the camera.
            fov: The field-of-view of the camera.

        Returns:
            A 3x3 numpy matrix which contains the intrinsic matrix of the
            camera.
        """
        import numpy as np
        k = np.identity(3)
        # Center column of the image.
        k[0, 2] = (width - 1) / 2.0
        # Center row of the image.
        k[1, 2] = (height - 1) / 2.0
        # Focal length.
        k[0, 0] = k[1, 1] = (width - 1) / (2.0 * np.tan(fov * np.pi / 360.0))
        return k

    @staticmethod
    def __create_unreal_transform(transform):
        """
        Takes in a Transform that occurs in unreal coordinates,
        and converts it into a Transform that goes from camera
        coordinates to unreal coordinates.

        With no additional rotations:
        Unreal coordinates: +x is into the screen, +y is to the right, +z is up
        Camera coordinates: +x is to the right, +y is down, +z is into the screen
        Lidar coordinates: +x is to the right, +y is out of the screen, +z is down
        """
        import numpy as np
        to_unreal_transform = Transform(matrix=np.array(
            [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]))
        return transform * to_unreal_transform

    def get_intrinsic_matrix(self):
        """ Get the intrinsic matrix of the camera denoted by the CameraSetup.

        Returns:
            A 3x3 numpy matrix containing the intrinsic matrix of the camera.
        """
        return self._intrinsic_mat

    def get_extrinsic_matrix(self):
        """ Get the extrinsic matrix of the camera denoted by the transform
        of the camera with respect to the vehicle to which it is attached.

        Returns:
            A 4x4 matrix containing the extrinsic matrix of the camera.
        """
        return self._unreal_transform.matrix

    def get_name(self):
        """ Get the name of the camera instance.

        Returns:
            The name of the camera instance.
        """
        return self.name

    def get_unreal_transform(self):
        """ Get the transform of the camera with respect to the vehicle in
        the Unreal Engine coordinate space.

        Returns:
            A pylot.utils.Transform instance which contains the transform of
            the camera with respect to the vehicle in the Unreal Engine
            coordinate space.
        """
        return self._unreal_transform

    def get_transform(self):
        """ Get the transform of the camera with respect to the vehicle to
        which it is attached.

        Args:
            A pylot.utils.Transform instance which contains the transform of
            the camera with respect to the vehicle to which it is attached.
        """
        return self._transform

    def set_transform(self, transform):
        """ Set the transform of the camera with respect to the vehicle to
        which it is attached.

        Args:
            A pylot.utils.Transform instance depicting the transform of the
            camera with respect to the vehicle to which it is attached.
        """
        assert isinstance(transform, Transform), "The given transform is not "
        "of the type pylot.utils.Transform"
        self._transform = transform
        self._unreal_transform = CameraSetup.__create_unreal_transform(
            self._transform)

    def get_fov(self):
        """ Get the field of view of the camera.

        Returns:
            The field of view of the given camera.
        """
        return self.fov

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'CameraSetup(name: {}, type: {}, width: {}, height: {}, '\
            'transform: {}, fov: {})'.format(
                self.name, self.camera_type, self.width, self.height,
                self._transform, self.fov)

    transform = property(get_transform, set_transform)


class RGBCameraSetup(CameraSetup):
    def __init__(self, name, width, height, transform, fov=90):
        super(RGBCameraSetup, self).__init__(name, 'sensor.camera.rgb', width,
                                             height, transform, fov)


class DepthCameraSetup(CameraSetup):
    def __init__(self, name, width, height, transform, fov=90):
        super(DepthCameraSetup, self).__init__(name, 'sensor.camera.depth',
                                               width, height, transform, fov)


class SegmentedCameraSetup(CameraSetup):
    def __init__(self, name, width, height, transform, fov=90):
        super(SegmentedCameraSetup,
              self).__init__(name, 'sensor.camera.semantic_segmentation',
                             width, height, transform, fov)


class LidarSetup(object):
    """ A helper class storing infromation about the setup of a Lidar."""

    def __init__(self, name, lidar_type, transform, range, rotation_frequency,
                 channels, upper_fov, lower_fov, points_per_second):
        self.name = name
        self.lidar_type = lidar_type
        self._transform = transform
        self.range = range
        self.rotation_frequency = rotation_frequency
        self.channels = channels
        self.upper_fov = upper_fov
        self.lower_fov = lower_fov
        self.points_per_second = points_per_second
        self._unreal_transform = LidarSetup.__create_unreal_transform(
            self._transform)

    @staticmethod
    def __create_unreal_transform(transform):
        """
        Takes in a Transform that occurs in camera coordinates,
        and converts it into a Transform that goes from lidar
        coordinates to camera coordinates.
        """
        to_camera_transform = Transform(matrix=np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]))
        return transform * to_camera_transform

    def get_name(self):
        return self.name

    def get_transform(self):
        return self._transform

    def set_transform(self, transform):
        self._transform = transform
        self._unreal_transform = LidarSetup.__create_unreal_transform(
            self._transform)

    def get_unreal_transform(self):
        return self._unreal_transform

    def get_range_in_meters(self):
        return self.range / 1000

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'LidarSetup(name: {}, type: {}, transform: {}, range: {}, '\
            'rotation freq: {}, channels: {}, upper_fov: {}, lower_fov: {}, '\
            'points_per_second: {}'.format(
                self.name, self.lidar_type, self._transform, self.range,
                self.rotation_frequency, self.channels, self.upper_fov,
                self.lower_fov, self.points_per_second)

    transform = property(get_transform, set_transform)


class IMUSetup(object):
    def __init__(self, name, transform):
        self.name = name
        self.transform = transform

    def get_name(self):
        return self.name

    def get_transform(self):
        return self.transform

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "IMUSetup(name: {}, transform: {})".format(
            self.name, self.transform)
