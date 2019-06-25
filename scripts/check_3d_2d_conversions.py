# Checks methods for getting 3d world locations from depth map and from point cloud.

import copy
import numpy as np
import pptk
import time

import carla

import pylot.utils
import pylot.simulation.messages
import pylot.simulation.utils
from pylot.simulation.utils import depth_to_array, to_bgra_array,\
     camera_to_unreal_transform,\
     get_3d_world_position_with_depth_map,\
     get_3d_world_position_with_point_cloud,\
     lidar_point_cloud_to_camera_coordinates,\
     to_erdos_transform
from matplotlib import pyplot as plt

lidar_pc = None
depth_pc = None
last_frame = None

# Pixels to check for when the target vehicle is set at (20, 2, 0) and
# the sensor position at (2, 8, 1.4)
#pixels_to_check = [(400, 285), (400, 350), (500, 285), (245, 320)]

# Pixels to check for when the target vehicle is set at (242, 131.239990234, 0)
# and the sensor position at (237.699996948, 132.239990234, 1.32062494755).
pixels_to_check = [(200, 370)]

target_vehicle_transform = carla.Transform(
    carla.Location(242, 131.239990234, 0),
    carla.Rotation(pitch=0, yaw=0, roll=0))

print ("Target Vehicle Location:", target_vehicle_transform.location.x,
                                   target_vehicle_transform.location.y,
                                   target_vehicle_transform.location.z)

# target_vehicle_transform = carla.Transform(
#     carla.Location(20, 2, 0),
#     carla.Rotation(pitch=0, yaw=0, roll=0))

# Create the camera, lidar, depth camera position.
sensor_transform = carla.Transform(
    carla.Location(237.699996948, 132.239990234, 1.32062494755),
    carla.Rotation(pitch=0, yaw=0, roll=0))
print ("Our Location:", sensor_transform.location.x,
                        sensor_transform.location.y,
                        sensor_transform.location.z)

# sensor_transform = carla.Transform(
#     carla.Location(2, 8, 1.4),
#     carla.Rotation(pitch=0, yaw=0, roll=0))

vehicle_transform = pylot.simulation.utils.Transform(
    pylot.simulation.utils.Location(0, 0, 0),
    pylot.simulation.utils.Rotation(pitch=0, yaw=0, roll=0))


def get_world(host="localhost", port=2000):
    """ Get a handle to the world running inside the simulation.

    Args:
        host: The host where the simulator is running.
        port: The port to connect to at the given host.

    Returns:
        A tuple of `(client, world)` where the `client` is a connection to the
        simulator and `world` is a handle to the world running inside the
        simulation at the host:port.
    """
    client, world = None, None
    try:
        client = carla.Client(host, port)
        client.set_timeout(10.0)
        world = client.get_world()
    except RuntimeError as r:
        client, world = None, None
        print("Received an error while connecting to the "
              "simulator: {}".format(r))
    return (client, world)


def on_lidar_msg(carla_pc):
    game_time = int(carla_pc.timestamp * 1000)
    print("Received lidar msg {}".format(game_time))
    points = np.frombuffer(carla_pc.raw_data, dtype=np.dtype('f4'))
    points = copy.deepcopy(points)
    points = np.reshape(points, (int(points.shape[0] / 3), 3))

    lidar_transform = to_erdos_transform(carla_pc.transform)

    # Transform lidar points from lidar coordinates to camera coordinates.
    points = lidar_point_cloud_to_camera_coordinates(points)
    for (x, y) in pixels_to_check:
        pos3d_pc = get_3d_world_position_with_point_cloud(
            x, y, points.tolist(), lidar_transform, 800, 600, 90.0)
        print("{} Computed using lidar {}".format((x, y), pos3d_pc))

    global lidar_pc
    lidar_pc = points.tolist()
#    pptk.viewer(points)


def on_camera_msg(carla_image):
    game_time = int(carla_image.timestamp * 1000)
    print("Received camera msg {}".format(game_time))
    global last_frame
    last_frame = pylot.utils.bgra_to_bgr(to_bgra_array(carla_image))


def on_depth_msg(carla_image):
    game_time = int(carla_image.timestamp * 1000)
    print("Received depth camera msg {}".format(game_time))

    depth_camera_transform = to_erdos_transform(carla_image.transform)

    depth_msg = pylot.simulation.messages.DepthFrameMessage(
        depth_to_array(carla_image),
        depth_camera_transform,
        carla_image.fov,
        None)

    for (x, y) in pixels_to_check:
        print("{} Depth at pixel {}".format((x, y), depth_msg.frame[y][x]))
        pos3d_depth = get_3d_world_position_with_depth_map(
            x, y, depth_msg.frame, depth_msg.width, depth_msg.height,
            depth_msg.fov, depth_camera_transform)
        print("{} Computed using depth map {}".format((x, y), pos3d_depth))

    depth_point_cloud = pylot.simulation.utils.depth_to_local_point_cloud(
        depth_msg.frame, depth_msg.width, depth_msg.height,
        depth_msg.fov, max_depth=1.0)
    # Transform the depth cloud to world coordinates.
    transform = camera_to_unreal_transform(depth_camera_transform)
    depth_point_cloud = transform.transform_points(depth_point_cloud)

    global depth_pc
    depth_pc = depth_point_cloud.tolist()
#    pptk.viewer(depth_point_cloud)


def add_lidar(transform, callback):
    lidar_blueprint = world.get_blueprint_library().find(
       'sensor.lidar.ray_cast')
    lidar_blueprint.set_attribute('channels', '32')
    lidar_blueprint.set_attribute('range', '5000')
    lidar_blueprint.set_attribute('points_per_second', '500000')
    lidar_blueprint.set_attribute('rotation_frequency', '20')
    lidar_blueprint.set_attribute('upper_fov', '15')
    lidar_blueprint.set_attribute('lower_fov', '-30')
    lidar = world.spawn_actor(lidar_blueprint, transform)
    # Register callback to be invoked when a new point cloud is received.
    lidar.listen(callback)
    return lidar


def add_depth_camera(transform, callback):
    depth_blueprint = world.get_blueprint_library().find(
        'sensor.camera.depth')
    depth_blueprint.set_attribute('image_size_x', '800')
    depth_blueprint.set_attribute('image_size_y', '600')
    depth_camera = world.spawn_actor(depth_blueprint, transform)
    # Register callback to be invoked when a new frame is received.
    depth_camera.listen(callback)
    return depth_camera


def add_camera(transform, callback):
    camera_blueprint = world.get_blueprint_library().find(
        'sensor.camera.rgb')
    camera_blueprint.set_attribute('image_size_x', '800')
    camera_blueprint.set_attribute('image_size_y', '600')
    camera = world.spawn_actor(camera_blueprint, transform)
    # Register callback to be invoked when a new frame is received.
    camera.listen(callback)
    return camera


def add_vehicle(transform):
    # Location of the vehicle in world coordinates.
    v_blueprint = world.get_blueprint_library().find('vehicle.audi.a2')
    vehicle = world.spawn_actor(v_blueprint, transform)
    return vehicle


# Connect to the Carla simulator.
client, world = get_world()
settings = world.get_settings()
settings.synchronous_mode = True
world.apply_settings(settings)


print("Adding sensors")
target_vehicle = add_vehicle(target_vehicle_transform)
lidar = add_lidar(sensor_transform, on_lidar_msg)
depth_camera = add_depth_camera(sensor_transform, on_depth_msg)
camera = add_camera(sensor_transform, on_camera_msg)

# Move the spectactor view to the camera position.
world.get_spectator().set_transform(sensor_transform)

try:
    # Tick the simulator once to get 1 data reading.
    world.tick()

    while lidar_pc is None or depth_pc is None or last_frame is None:
        time.sleep(0.2)

    plt.imshow(last_frame)
    plt.show()
    # Sleep a bit to give time to inspect the image.
finally:
    # Destroy the actors.
    lidar.destroy()
    depth_camera.destroy()
    target_vehicle.destroy()
    camera.destroy()
