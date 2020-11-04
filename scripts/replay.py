import time

from absl import app
from absl import flags

from carla import Location, Rotation, Transform

import pylot.flags
from pylot.simulation.utils import get_world

flags.DEFINE_float('replay_start_time', 0.0,
                   'The time at which to start replaying')
flags.DEFINE_float('replay_duration', 0.0,
                   'The duration of the replay run')
flags.DEFINE_integer('replay_id', 0,
                     'The actor id to follow during the replay')
flags.DEFINE_string('replay_file', '', 'Path to the log file')

FLAGS = flags.FLAGS


def process_images(image):
    game_time = int(image.timestamp * 1000)
    print('Received frame for {}'.format(game_time))
    # frame = pylot.utils.bgra_to_bgr(to_bgra_array(image))
    # cv2.imshow("test", frame)
    # cv2.waitKey(1)


def main(argv):
    client, world = get_world(FLAGS.simulator_host, FLAGS.simulator_port,
                              FLAGS.simulator_timeout)

    # Replayer time factor is only available in > 0.9.5.
    client.set_replayer_time_factor(0.1)
    print(
        client.replay_file(FLAGS.replay_file,
                           FLAGS.replay_start_time,
                           FLAGS.replay_duration, FLAGS.replay_id))
    # Sleep a bit to allow the server to start the replay.
    time.sleep(1)
    vehicle = world.get_actors().find(FLAGS.replay_id)
    if vehicle is None:
        raise ValueError("There was an issue finding the vehicle.")

    # Install the camera.
    camera_blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_blueprint.set_attribute('image_size_x',
                                   str(FLAGS.camera_image_width))
    camera_blueprint.set_attribute('image_size_y',
                                   str(FLAGS.camera_image_height))

    transform = Transform(Location(2.0, 0.0, 1.4),
                          Rotation(pitch=0, yaw=0, roll=0))

    camera = world.spawn_actor(camera_blueprint, transform, attach_to=vehicle)

    # Register the callback on the camera.
    camera.listen(process_images)

    time.sleep(20)


if __name__ == '__main__':
    app.run(main)
