import glob
import os
import sys
import threading

try:
    sys.path.append(
        glob.glob('../carla/dist/carla-*%d.%d-%s.egg' %
                  (sys.version_info.major, sys.version_info.minor,
                   'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random
import time

import erdos
import logging
from absl import flags, app

import pylot.flags
import pylot.utils
import pylot.simulation.utils
from pylot.perception.camera_frame import CameraFrame
from pylot.drivers.sensor_setup import RGBCameraSetup
from pylot.perception.messages import FrameMessage

_lock = threading.Lock()

FLAGS = flags.FLAGS

CENTER_CAMERA_LOCATION = pylot.utils.Location(1.0, 0.0, 1.8)


def main(args):
    actor_list = []

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()

        blueprint_library = world.get_blueprint_library()
        bp = world.get_blueprint_library().filter('vehicle.lincoln.mkz2017')[0]

        # Get random spawn position
        transform = random.choice(world.get_map().get_spawn_points())

        # Spawn lincoln vehicle
        vehicle = world.spawn_actor(bp, transform)

        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        # Let's put the vehicle to drive around
        vehicle.set_autopilot(True)

        transform = pylot.utils.Transform(CENTER_CAMERA_LOCATION,
                                          pylot.utils.Rotation())

        rgb_camera_setup = RGBCameraSetup('center_camera',
                                          FLAGS.camera_image_width,
                                          FLAGS.camera_image_height, transform,
                                          FLAGS.camera_fov)

        # Install the camera
        camera_blueprint = world.get_blueprint_library().find(
            rgb_camera_setup.camera_type)
        camera_blueprint.set_attribute('image_size_x',
                                       str(rgb_camera_setup.width))
        camera_blueprint.set_attribute('image_size_y',
                                       str(rgb_camera_setup.height))
        camera_blueprint.set_attribute('fov', str(rgb_camera_setup.fov))
        camera_blueprint.set_attribute('sensor_tick', str(1.0 / 20))

        transform = rgb_camera_setup.get_transform().as_simulator_transform()

        print("Spawning a camera: {}".format(rgb_camera_setup))
        camera = world.spawn_actor(camera_blueprint,
                                   transform,
                                   attach_to=vehicle)

        camera_ingest_stream = erdos.streams.IngestStream(name='camera')
        ttd_ingest_stream = erdos.streams.IngestStream(name='ttd')

        from pylot.perception.detection.detection_operator import DetectionOperator
        detection_op_cfg = erdos.operator.OperatorConfig(name='detection_op')
        obstacles_stream = erdos.connect_two_in_one_out(
            DetectionOperator,
            detection_op_cfg,
            camera_ingest_stream,
            ttd_ingest_stream,
            model_path=FLAGS.obstacle_detection_model_paths[0],
            flags=FLAGS)

        erdos.run_async()

        def process_images(simulator_image):
            """Invoked when an image is received from the simulator."""
            game_time = int(simulator_image.timestamp * 1000)
            timestamp = erdos.Timestamp(coordinates=[game_time])
            watermark_msg = erdos.WatermarkMessage(timestamp)

            # Ensure that the code executes serially
            with _lock:
                msg = None
                if rgb_camera_setup.camera_type == 'sensor.camera.rgb':
                    msg = FrameMessage(
                        timestamp,
                        CameraFrame.from_simulator_frame(
                            simulator_image, rgb_camera_setup))
                    camera_ingest_stream.send(msg)
                    # ttd_ingest_stream.send(erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))  Panics

        # Register camera frame callback
        camera.listen(process_images)

        # Spawn 20 test vehicles
        pylot.simulation.utils.spawn_vehicles(client, world, 8000, 20,
                                              logging.Logger(name="test"))

        time.sleep(5)

    finally:
        print('destroying actors')
        camera.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')


if __name__ == '__main__':
    app.run(main)
