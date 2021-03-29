# Checks methods for getting 3d world locations from depth map and from point
# cloud.
import time

from absl import app
from absl import flags

from carla import Location, Rotation, Transform

import cv2


import pylot.utils
from pylot.drivers.sensor_setup import CameraSetup, LidarSetup
from pylot.perception.camera_frame import CameraFrame
from pylot.perception.depth_frame import DepthFrame
from pylot.perception.point_cloud import PointCloud
from pylot.simulation.utils import get_world

FLAGS = flags.FLAGS

flags.DEFINE_string('simulator_host', 'localhost', 'Simulator host.')
lidar_pc = None
depth_pc = None
last_frame = None


def on_lidar_msg(simulator_pc):
    game_time = int(simulator_pc.timestamp * 1000)
    print("Received lidar msg {}".format(game_time))
    lidar_transform = pylot.utils.Transform.from_simulator_transform(
        simulator_pc.transform)
    lidar_setup = LidarSetup('lidar',
                             lidar_type='sensor.lidar.ray_cast',
                             transform=lidar_transform)
    point_cloud = PointCloud.from_simulator_point_cloud(simulator_pc,
                                                        lidar_setup)
    camera_setup = CameraSetup("lidar_camera",
                               "sensor.camera.depth",
                               800,
                               600,
                               lidar_transform,
                               fov=90.0)
    for (x, y) in pixels_to_check:
        pixel = pylot.utils.Vector2D(x, y)
        location = point_cloud.get_pixel_location(pixel, camera_setup)
        print("{} Computed using lidar {}".format((x, y), location))

    global lidar_pc
    lidar_pc = point_cloud
    # pptk.viewer(point_cloud.points)


def on_camera_msg(simulator_image):
    game_time = int(simulator_image.timestamp * 1000)
    print("Received camera msg {}".format(game_time))

    camera_transform = pylot.utils.Transform.from_simulator_transform(
        simulator_image.transform)

    camera_setup = CameraSetup("rgb_camera",
                               "sensor.camera.rgb",
                               800,
                               600,
                               camera_transform,
                               fov=90.0)
    global last_frame
    last_frame = CameraFrame.from_simulator_frame(simulator_image,
                                                  camera_setup)


def on_depth_msg(simulator_image):
    game_time = int(simulator_image.timestamp * 1000)
    print("Received depth camera msg {}".format(game_time))

    depth_camera_transform = pylot.utils.Transform.from_simulator_transform(
        simulator_image.transform)

    camera_setup = CameraSetup("depth_camera",
                               "sensor.camera.depth",
                               800,
                               600,
                               depth_camera_transform,
                               fov=90.0)
    depth_frame = DepthFrame.from_simulator_frame(simulator_image, camera_setup)

    for (x, y) in pixels_to_check:
        print("{} Depth at pixel {}".format((x, y), depth_frame.frame[y][x]))
        pos3d_depth = depth_frame.get_pixel_locations(
            [pylot.utils.Vector2D(x, y)])[0]
        print("{} Computed using depth map {}".format((x, y), pos3d_depth))

    global depth_pc
    depth_pc = depth_frame.as_point_cloud()
    # pptk.viewer(depth_pc)


def add_lidar(world, transform, callback):
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


def add_depth_camera(world, transform, callback):
    depth_blueprint = world.get_blueprint_library().find('sensor.camera.depth')
    depth_blueprint.set_attribute('image_size_x', '800')
    depth_blueprint.set_attribute('image_size_y', '600')
    depth_camera = world.spawn_actor(depth_blueprint, transform)
    # Register callback to be invoked when a new frame is received.
    depth_camera.listen(callback)
    return depth_camera


def add_camera(world, transform, callback):
    camera_blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_blueprint.set_attribute('image_size_x', '800')
    camera_blueprint.set_attribute('image_size_y', '600')
    camera = world.spawn_actor(camera_blueprint, transform)
    # Register callback to be invoked when a new frame is received.
    camera.listen(callback)
    return camera


def add_vehicle(world, transform):
    # Location of the vehicle in world coordinates.
    v_blueprint = world.get_blueprint_library().find('vehicle.audi.a2')
    vehicle = world.spawn_actor(v_blueprint, transform)
    return vehicle


def run_scenario(target_vehicle_transform, sensor_transform):
    # Reset messages.
    global lidar_pc, last_frame, depth_pc
    lidar_pc, last_frame, depth_pc = None, None, None
    client, world = get_world(host=FLAGS.simulator_host)
    world = client.load_world('Town01')
    settings = world.get_settings()
    settings.synchronous_mode = True
    world.apply_settings(settings)

    print("Adding sensors")
    target_vehicle = add_vehicle(world, target_vehicle_transform)
    lidar = add_lidar(world, sensor_transform, on_lidar_msg)
    depth_camera = add_depth_camera(world, sensor_transform, on_depth_msg)
    camera = add_camera(world, sensor_transform, on_camera_msg)

    # Move the spectactor view to the camera position.
    world.get_spectator().set_transform(sensor_transform)

    print("Target Vehicle Location:", target_vehicle_transform.location.x,
          target_vehicle_transform.location.y,
          target_vehicle_transform.location.z)

    print("Our Location:", sensor_transform.location.x,
          sensor_transform.location.y, sensor_transform.location.z)

    try:
        # Tick the simulator once to get 1 data reading.
        world.tick()

        while lidar_pc is None or depth_pc is None or last_frame is None:
            time.sleep(0.2)

        cv2.imshow('camera view', last_frame.frame)
        cv2.waitKey(0)
    finally:
        # Destroy the actors.
        lidar.destroy()
        depth_camera.destroy()
        target_vehicle.destroy()
        camera.destroy()


def main(argv):
    global pixels_to_check
    target_vehicle_transform = Transform(Location(242, 131.24, 0))
    sensor_transform = Transform(Location(237.7, 132.24, 1.3))
    pixels_to_check = [(200, 370)]
    run_scenario(target_vehicle_transform, sensor_transform)

    target_vehicle_transform = Transform(Location(2, 12, 0))
    sensor_transform = Transform(
        Location(0, 18, 1.4), Rotation(pitch=0, yaw=-90, roll=0))
    pixels_to_check = [(500, 400), (600, 400), (500, 350), (600, 350)]
    run_scenario(target_vehicle_transform, sensor_transform)


if __name__ == '__main__':
    app.run(main)
