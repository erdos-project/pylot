import pytest

from pylot.drivers.sensor_setup import *
from pylot.utils import Location, Rotation, Transform


# CameraSetup tests
def test_camera_setup_failed_initialization():
    """
    Ensure that the CameraSetup constructor fails when wrong values or values
    of the wrong type are provided.
    """
    # Set up the correct values for all the constructor values.
    name, camera_type = 'camera_setup', 'sensor.camera.rgb'
    width, height = 1920, 1080
    transform = Transform(Location(), Rotation())

    # Ensure that a wrong name throws an error.
    with pytest.raises(AssertionError):
        camera_setup = CameraSetup(name=1,
                                   camera_type=camera_type,
                                   width=width,
                                   height=height,
                                   transform=transform)

    # Ensure that a wrong camera_type throws an error.
    with pytest.raises(AssertionError):
        camera_setup = CameraSetup(name=name,
                                   camera_type=None,
                                   width=width,
                                   height=height,
                                   transform=transform)

    # Ensure that a wrong width, height throws an error.
    with pytest.raises(AssertionError):
        camera_setup = CameraSetup(name=name,
                                   camera_type=camera_type,
                                   width=float(width),
                                   height=float(height),
                                   transform=transform)

    # Ensure that a wrong transform throws an error.
    with pytest.raises(AssertionError):
        camera_setup = CameraSetup(name=name,
                                   camera_type=camera_type,
                                   width=width,
                                   height=height,
                                   transform=None)

    # Ensure that a bad fov raises an error.
    with pytest.raises(AssertionError):
        camera_setup = CameraSetup(name=name,
                                   camera_type=camera_type,
                                   width=width,
                                   height=height,
                                   transform=transform,
                                   fov=None)


def test_intrinsic_matrix():
    """
    Ensure that the intrinsic matrix of a CameraSetup is correct.

    Compare the values of the matrix to the ones specified by:
    http://ksimek.github.io/2013/08/13/intrinsic/
    """
    camera_setup = CameraSetup(name='camera_setup',
                               camera_type='sensor.camera.rgb',
                               width=1920,
                               height=1080,
                               transform=Transform(Location(), Rotation()),
                               fov=90.0)

    # Retrieve the intrinsic matrix from the camera_setup.
    intrinsic_matrix = camera_setup.get_intrinsic_matrix()
    assert isinstance(intrinsic_matrix, np.ndarray), "The intrinsic matrix is"
    " not of the type numpy.ndarray"
    assert intrinsic_matrix.shape == (3, 3), "The intrinsic matrix should be "
    "a 3x3 matrix."

    # Ensure that the focal length is properly set.
    fx = (1920 - 1) / (2.0 * np.tan(90 * np.pi / 360))
    assert np.isclose(intrinsic_matrix[0, 0], fx), "The fx value of the "
    "intrinsic matrix is incorrect."
    assert np.isclose(intrinsic_matrix[1, 1], fx), "The fy value of the "
    "intrinsic matrix is incorrect."

    # Ensure that the center points are right.
    x0, y0 = (1920 - 1) / 2.0, (1080 - 1) / 2.0
    assert np.isclose(intrinsic_matrix[0, 2], x0), "The principal point offset"
    " x0 is not correct."
    assert np.isclose(intrinsic_matrix[1, 2], y0), "The principal point offset"
    " y0 is not correct."

    # Ensure that we have no axis skew.
    assert np.isclose(intrinsic_matrix[0, 1], 0), "The axis skew should be 0"


@pytest.mark.parametrize("rotation, expected", [((0, 0, 0), (0, 90, 90)),
                                                ((0, 0, 90), (-90, 90, 90)),
                                                ((0, 90, 0), (0, 180, 90))])
def test_camera_unreal_transform(rotation, expected):
    """
    Ensure that the camera space to unreal engine coordinate space conversion
    is correct.

    The camera space is defined as:
        +x to the right, +y to down, +z into the screen.

    The unreal engine coordinate space is defined as:
        +x into the screen, +y to the right, +z to up.
    """
    camera_rotation = Rotation(*rotation)
    camera_setup = CameraSetup(name='camera_setup',
                               camera_type='sensor.camera.rgb',
                               width=1920,
                               height=1080,
                               transform=Transform(Location(),
                                                   camera_rotation))
    transformed_rotation = camera_setup.get_unreal_transform().rotation
    transformed_rotation = [
        transformed_rotation.pitch, transformed_rotation.yaw,
        transformed_rotation.roll
    ]
    assert all(np.isclose(transformed_rotation, expected)), \
            "The unreal transformation does not match the expected transform."


## LidarSetup tests
def test_lidar_setup_failed_initialization():
    # Set up the required parameters for the initialization.
    name, lidar_type = 'lidar_setup', 'sensor.lidar.ray_cast'
    transform = Transform(Location(), Rotation())
    range, rotation_frequency, channels = 5000, 30, 3
    upper_fov, lower_fov = 90.0, 90.0
    points_per_second = 1000

    # Ensure that a wrong name raises an error.
    with pytest.raises(AssertionError):
        lidar_setup = LidarSetup(name=1,
                                 lidar_type=lidar_type,
                                 transform=transform,
                                 range=range,
                                 rotation_frequency=rotation_frequency,
                                 channels=channels,
                                 upper_fov=upper_fov,
                                 lower_fov=lower_fov,
                                 points_per_second=points_per_second)

    # Ensure that the wrong lidar_type raises an error.
    with pytest.raises(AssertionError):
        lidar_setup = LidarSetup(name=name,
                                 lidar_type=None,
                                 transform=transform,
                                 range=range,
                                 rotation_frequency=rotation_frequency,
                                 channels=channels,
                                 upper_fov=upper_fov,
                                 lower_fov=lower_fov,
                                 points_per_second=points_per_second)

    # Ensure that a wrong transform raises an error.
    with pytest.raises(AssertionError):
        lidar_setup = LidarSetup(name=name,
                                 lidar_type=lidar_type,
                                 transform=None,
                                 range=range,
                                 rotation_frequency=rotation_frequency,
                                 channels=channels,
                                 upper_fov=upper_fov,
                                 lower_fov=lower_fov,
                                 points_per_second=points_per_second)

    # Ensure that a wrong range raises an error.
    with pytest.raises(AssertionError):
        lidar_setup = LidarSetup(name=name,
                                 lidar_type=lidar_type,
                                 transform=transform,
                                 range=None,
                                 rotation_frequency=rotation_frequency,
                                 channels=channels,
                                 upper_fov=upper_fov,
                                 lower_fov=lower_fov,
                                 points_per_second=points_per_second)

    # Ensure that a wrong rotation frequency raises an error.
    with pytest.raises(AssertionError):
        lidar_setup = LidarSetup(name=name,
                                 lidar_type=lidar_type,
                                 transform=transform,
                                 range=range,
                                 rotation_frequency=None,
                                 channels=channels,
                                 upper_fov=upper_fov,
                                 lower_fov=lower_fov,
                                 points_per_second=points_per_second)

    # Ensure that a wrong channel raises an error.
    with pytest.raises(AssertionError):
        lidar_setup = LidarSetup(name=name,
                                 lidar_type=lidar_type,
                                 transform=transform,
                                 range=range,
                                 rotation_frequency=rotation_frequency,
                                 channels=float(channels),
                                 upper_fov=upper_fov,
                                 lower_fov=lower_fov,
                                 points_per_second=points_per_second)

    # Ensure that a wrong fov raises an error.
    with pytest.raises(AssertionError):
        lidar_setup = LidarSetup(name=name,
                                 lidar_type=lidar_type,
                                 transform=transform,
                                 range=range,
                                 rotation_frequency=rotation_frequency,
                                 channels=channels,
                                 upper_fov=None,
                                 lower_fov=None,
                                 points_per_second=points_per_second)

    # Ensure that a wrong points_per_second raises an error.
    with pytest.raises(AssertionError):
        lidar_setup = LidarSetup(name=name,
                                 lidar_type=lidar_type,
                                 transform=transform,
                                 range=range,
                                 rotation_frequency=rotation_frequency,
                                 channels=channels,
                                 upper_fov=upper_fov,
                                 lower_fov=lower_fov,
                                 points_per_second=float(points_per_second))


@pytest.mark.parametrize("rotation, expected", [((0, 0, 0), (0, 90, 0)),
                                                ((0, 90, 0), (0, 180, 00)),
                                                ((90, 0, 0), (0, 90, 90))])
def test_lidar_unreal_transform(rotation, expected):
    """
    Ensure that the LIDAR space to unreal engine coordinate space conversion
    is correct.

    The LIDAR space is defined as:
        +x to the right, +y out of the screen, +z is down.

    The unreal engine coordinate space is defined as:
        +x into the screen, +y to the right, +z to up.
    """
    lidar_rotation = Rotation(*rotation)
    lidar_setup = LidarSetup(name='lidar_setup',
                             lidar_type='sensor.lidar.ray_cast',
                             transform=Transform(Location(), lidar_rotation),
                             range=6000.0,
                             rotation_frequency=3000.0,
                             channels=3,
                             upper_fov=90.0,
                             lower_fov=90.0,
                             points_per_second=100)

    transformed_rotation = lidar_setup.get_unreal_transform().rotation
    transformed_rotation = [
        transformed_rotation.pitch, transformed_rotation.yaw,
        transformed_rotation.roll
    ]
    assert all(np.isclose(transformed_rotation, expected)), \
            "The unreal transformation does not match the expected transform."


## IMUSetup tests
def test_imu_setup_failed_initialization():
    # Set up the required parameters for the initialization.
    name = 'imu_sensor'
    transform = Transform(Location(), Rotation())

    with pytest.raises(AssertionError):
        imu_setup = IMUSetup(name=1, transform=transform)

    with pytest.raises(AssertionError):
        imu_setup = IMUSetup(name=name, transform=None)


def test_imu_setup_initialization():
    # Set up the required parameters for the initialization.
    name = 'imu_sensor'
    transform = Transform(Location(), Rotation())
    imu_setup = IMUSetup(name, transform)
    assert imu_setup.name == name, "The name in the setup is not the same."
    assert imu_setup.transform == transform, "The transform is not the same."
