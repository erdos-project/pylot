from absl import app
from absl import flags
import cv2
import time

import carla

import pylot.config
import pylot.utils
from pylot.simulation.carla_utils import get_world
from pylot.simulation.utils import to_bgra_array

FLAGS = flags.FLAGS


def process_images(carla_image):
    game_time = int(carla_image.timestamp * 1000)
    print('Received frame for {}'.format(game_time))
    # frame = pylot.utils.bgra_to_bgr(to_bgra_array(carla_image))
    # cv2.imshow("test", frame)
    # cv2.waitKey(1)


def main(argv):
    client, world = get_world(FLAGS.carla_host,
                              FLAGS.carla_port,
                              FLAGS.carla_timeout)

    # Replayer time factor is only available in > 0.9.5.
    client.set_replayer_time_factor(0.1)
    print(client.replay_file(FLAGS.carla_replay_file,
                             FLAGS.carla_replay_start_time,
                             FLAGS.carla_replay_duration,
                             FLAGS.carla_replay_id))
    # Sleep a bit to allow the server to start the replay.
    time.sleep(1)
    vehicle = world.get_actors().find(FLAGS.carla_replay_id)
    if vehicle is None:
        raise ValueError("There was an issue finding the vehicle.")

    # Install the camera.
    camera_blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_blueprint.set_attribute('image_size_x',
                                   str(FLAGS.carla_camera_image_width))
    camera_blueprint.set_attribute('image_size_y',
                                   str(FLAGS.carla_camera_image_height))

    transform = carla.Transform(carla.Location(2.0, 0.0, 1.4),
                                carla.Rotation(pitch=0, yaw=0, roll=0))

    camera = world.spawn_actor(camera_blueprint,
                               transform,
                               attach_to=vehicle)

    # Register the callback on the camera.
    camera.listen(process_images)

    time.sleep(20)


if __name__ == '__main__':
    app.run(main)
