from __future__ import print_function
from time import sleep
import json
import os
import sys
import argparse
import functools
import collections
import carla

from pylot.simulation.carla_utils import get_world
from pylot.planning.scenario_runner_planner import Planner

VEHICLE_DESTINATION = carla.Location(x=387.73 - 370, y=327.07, z=0.5)
SAVED_FRAMES = collections.deque()
CLEANUP_FUNCTION = None


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


def process_segmentation_images(msg,
                                ego_vehicle,
                                planner,
                                speed,
                                data_dir):
    print("Received a message for the time: {}".format(msg.timestamp))

    # If we are in distance to the destination, stop and exit with success.
    if ego_vehicle.get_location().distance(VEHICLE_DESTINATION) <= 5:
        ego_vehicle.set_velocity(carla.Vector3D())
        CLEANUP_FUNCTION()
        sys.exit(0)

    # Planner
    data = {}
    data["time"] = planner.get_time()
    data["ego_accel"] = planner.get_ego_accel()
    data["ego_speed"] = planner.get_ego_speed()
    data["ego_location"] = planner.get_ego_location()
    data["ego_bbox"] = planner.get_ego_bbox()
    data["speed_limit"] = planner.get_speed_limit()
    data["bounds_and_marks"] = planner.get_road_bounds_and_lane_marks()
    data["dynamic_bboxes"] = planner.get_dynamic_bboxes()
    data["static"] = planner.get_static_bboxes()
    with open(os.path.join(data_dir, "{}.json".format(data["time"])), "w") as outfile:
        outfile.write(json.dumps(data, outfile))

    # Move the ego_vehicle according to the given speed.
    ego_vehicle.set_velocity(carla.Vector3D(x=-speed))

    # Move the simulator forward.
    ego_vehicle.get_world().tick()


def cleanup_function(world, cameras):
    """ Cleans up the state of the simulator.

    Args:
        world: The instance of the world to restore the asynchronous mode on.
        cameras: The cameras to destroy.
        csv_file: The open results file to close.
    """
    set_synchronous_mode(world, False)
    for camera in cameras:
        camera.destroy()


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
        sleep(1)
        ego_vehicle = retrieve_actor(world, 'vehicle.*', 'hero')
        world.tick()

    # Retrieve the pedestrian from the simulation.
    pedestrian = None
    while pedestrian is None:
        print("Waiting for the scenario to be ready ...")
        sleep(1)
        pedestrian = retrieve_actor(world, 'walker.*', 'pedestrian')
        world.tick()

    planner = Planner(ego_vehicle, pedestrian)

    # Connect the segmentation camera to the vehicle.
    segmentation_camera_transform = carla.Transform(
        location=carla.Location(1.5, 0.0, 1.4),
        rotation=carla.Rotation(0, 0, 0))
    segmentation_camera = spawn_camera('sensor.camera.semantic_segmentation',
                                       segmentation_camera_transform,
                                       ego_vehicle, *args.res.split('x'))

    # Create the cleanup function.
    global CLEANUP_FUNCTION
    CLEANUP_FUNCTION = functools.partial(
        cleanup_function,
        world=world,
        cameras=[segmentation_camera]
    )

    # Register a callback function with the camera.
    segmentation_camera.listen(
        functools.partial(process_segmentation_images,
                          ego_vehicle=ego_vehicle,
                          planner=planner,
                          speed=args.speed,
                          data_dir=args.data_dir
                          )
    )

    try:
        # To keep the thread alive so that the images can be processed.
        while True:
            pass
    except KeyboardInterrupt:
        CLEANUP_FUNCTION()
        sys.exit(0)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="CARLA mIoU collector")
    argparser.add_argument('-s',
                           '--speed',
                           default=25,
                           dest='speed',
                           type=float,
                           help="Set the speed of the vehicle in m/s.")
    argparser.add_argument('-d',
                           '--delta',
                           default=0.1,
                           dest='delta',
                           type=float,
                           help='The delta at which to run the simulator.')
    argparser.add_argument("--data_dir",
                           default="./data",
                           help="Path to data directory.")
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

    # Verify arguments.
    args = argparser.parse_args()
    if args.delta > 0.1:
        raise ValueError(
            "The CARLA simulator does not work well with frame rates lower "
            "than 10FPS.")

    # Create data dir
    if not os.path.isdir(args.data_dir):
        os.mkdir(args.data_dir)

    main(args)
