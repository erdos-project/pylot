from __future__ import print_function
import sys
import argparse
from time import sleep
from functools import partial

import carla
from pylot.simulation.carla_utils import get_world


def spawn_camera(camera_bp, transform, ego_vehicle, width='1280',
                 height='720'):
    """ Spawns the camera with the provided blueprint, and attaches it to the
    given ego_vehicle at the transform.

    Args:
        camera_bp: The blueprint of the camera to be spawned.
        transform: The transform at which to attach the camera.
        ego_vehicle: The vehicle to which the camera should be attached.
        width: The width of the image to be retrieved from the camera.
        height: The height of the image to be retrieved from the camera.

    Returns:
       The instance of the camera object spawned.
    """
    _world = ego_vehicle.get_world()
    camera_blueprint = _world.get_blueprint_library().find(camera_bp)
    camera_blueprint.set_attribute('image_size_x', width)
    camera_blueprint.set_attribute('image_size_y', height)
    camera = _world.spawn_actor(camera_blueprint,
                                transform,
                                attach_to=ego_vehicle)
    _world.tick()
    return camera


def retrieve_actor(world, bp_regex, role_name):
    """ Retrieves the actor from the world with the given blueprint and the
    role_name.

    Args:
        world: The instance of the simulator to retrieve the actors from.
        bp_regex: The blueprint of the actor to be retrieved from the simulator.
        role_name: The name of the actor to be retrieved.

    Returns:
        The actor retrieved from the given world with the role_name, if exists.
        Otherwise, returns None.
    """
    possible_actors = world.get_actors().filter(bp_regex)
    for actor in possible_actors:
        if actor.attributes['role_name'] == role_name:
            return actor
    return None


def set_synchronous_mode(world, value, delta=None):
    """ Sets the synchronous mode on the world to the given value.

    Args:
        world: The world to set the synchronous mode on.
        value: The value to set the synchronous mode to.
        delta: The delta at which to run the simulator if synchronous mode
        is on.
    """
    settings = world.get_settings()
    settings.synchronous_mode = value
    if value:
        settings.fixed_delta_seconds = delta
    else:
        settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    if value:
        world.tick()


def setup_world(host, port, delta):
    """ Connects to the simulator at the given host:port and sets the mode to
    synchronous.

    Args:
        host: The host where the simulator is running.
        port: The port to connect to at the given host.
        delta: The delta at which to run the simulation.

    Returns:
        The instance of `world` that was returned from the simulator.
    """
    _, world = get_world(host, port)
    if world is None:
        print("Could not connect to the simulator at {}:{}".format(host, port),
              file=sys.stderr)
        sys.exit(1)

    # Turn on synchronous mode.
    set_synchronous_mode(world, True, delta)

    return world

counter = 0

def process_segmentation_images(msg, ego_vehicle, speed, dump=False):
    print("Received a message for the time: {}".format(msg.timestamp))

def main(args):
    """ The main function that orchestrates the setup of the world, connection
    to the scenario and the subsequent logging of the frames for computation
    of the mIoU.

    Args:
        args: The argparse.Namespace instance that is retrieved from parsing
        the arguments.
    """

    # Setup the world.
    world = setup_world(args.host, args.port, args.delta)

    # Retrieve the ego-vehicle from the simulation.
    ego_vehicle = None
    while ego_vehicle is None:
        print("Waiting for the scenario to be ready ...")
        time.sleep(1)
        ego_vehicle = retrieve_actor(world, 'vehicle.*', 'hero')
        world.tick()

    # Connect the segmentation camera to the vehicle.
    segmentation_camera_transform = carla.Transform(
        location=carla.Location(1.5, 0.0, 1.4),
        rotation=carla.Rotation(0, 0, 0))
    segmentation_camera = spawn_camera('sensor.camera.semantic_segmentation',
                                       segmentation_camera_transform,
                                       ego_vehicle, *args.res.split('x'))

    # Register a callback function with the camera.
    segmentation_camera.listen(
        partial(process_segmentation_images,
                ego_vehicle=ego_vehicle,
                speed=args.speed,
                dump=args.dump))

    try:
        while True:
            time.sleep(1)
            world.tick()
    except KeyboardInterrupt:
        # Cleanup the state on the simulator.
        set_synchronous_mode(world, False)
        segmentation_camera.destroy()
        sys.exit(0)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="CARLA mIoU collector")
    argparser.add_argument('-s',
                           '--speed',
                           dest='speed',
                           required=True,
                           help="Set the speed of the vehicle.")
    argparser.add_argument('-d',
                           '--delta',
                           dest='delta',
                           type=float,
                           help='The delta at which to run the simulator.')
    argparser.add_argument('--host',
                           metavar='H',
                           default='127.0.0.1',
                           help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('--port',
                           metavar='P',
                           default=2000,
                           type=int,
                           help='TCP port to listen to (default: 2000)')
    argparser.add_argument('--res',
                           default='1280x720',
                           help='window resolution (default: 1280x720)')
    argparser.add_argument('--dump',
                           action='store_true',
                           help="Dump the images to _out directory")
    main(argparser.parse_args())
