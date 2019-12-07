import pylot.simulation.utils
from pylot.simulation.utils import Location, Rotation, Transform


def create_rgb_camera_setup(
        camera_name, camera_location, width, height, fov=90):
    rotation = Rotation(0, 0, 0)
    transform = Transform(camera_location, rotation)
    return RGBCameraSetup(camera_name, width, height, transform, fov)


def create_depth_camera_setup(
        camera_name_prefix, camera_location, width, height, fov=90):
    rotation = Rotation(0, 0, 0)
    transform = Transform(camera_location, rotation)
    return DepthCameraSetup(
        camera_name_prefix + '_depth', width, height, transform, fov=fov)


def create_segmented_camera_setup(
        camera_name_prefix, camera_location, width, height, fov=90):
    rotation = Rotation(0, 0, 0)
    transform = Transform(camera_location, rotation)
    return SegmentedCameraSetup(
        camera_name_prefix + '_segmented', width, height, transform, fov=fov)


def create_left_right_camera_setups(
        camera_name_prefix, location, width, height, camera_offset, fov=90):
    rotation = Rotation(0, 0, 0)
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
    rotation = Rotation(0, 0, 0)
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
        self.intrinsic_mat = pylot.simulation.utils.create_intrinsic_matrix(
            self.width, self.height, self.fov)

    def get_fov(self):
        return self.fov

    def get_intrinsic(self):
        return self.intrinsic_mat

    def get_name(self):
        return self.name

    def get_transform(self):
        return self.transform

    def get_unreal_transform(self):
        return pylot.simulation.utils.camera_to_unreal_transform(
            self.transform)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'CameraSetup(name: {}, type: {}, width: {}, height: {}, '\
            'transform: {}, fov: {}'.format(
                self.name, self.camera_type, self.width, self.height,
                self.transform, self.fov)


class RGBCameraSetup(CameraSetup):
    def __init__(self,
                 name,
                 width,
                 height,
                 transform,
                 fov=90):
        super(RGBCameraSetup, self).__init__(
            name, 'sensor.camera.rgb', width, height, transform, fov)


class DepthCameraSetup(CameraSetup):
    def __init__(self,
                 name,
                 width,
                 height,
                 transform,
                 fov=90):
        super(DepthCameraSetup, self).__init__(
            name, 'sensor.camera.depth', width, height, transform, fov)


class SegmentedCameraSetup(CameraSetup):
    def __init__(self,
                 name,
                 width,
                 height,
                 transform,
                 fov=90):
        super(SegmentedCameraSetup, self).__init__(
            name,
            'sensor.camera.semantic_segmentation',
            width,
            height,
            transform,
            fov)


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

    def get_name(self):
        return self.name

    def get_transform(self):
        return self.transform

    def get_unreal_transform(self):
        return pylot.simulation.utils.lidar_to_unreal_transform(self.transform)

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
    def __init__(self, name, transform):
        self.name = name
        self.transform = transform

     def get_transform(self):
        return self.transform

     def __repr__(self):
        return self.__str__()

     def __str__(self):
        return "IMUSetup(name: {}, transform: {})".format(
            self.name, self.transform)
