from __future__ import print_function

import argparse
import collections
import csv
import functools
import os
import sys
from time import sleep

from carla import Location, Rotation, Transform, Vector3D

import pylot.utils
from pylot.drivers.sensor_setup import SegmentedCameraSetup
from pylot.perception.segmentation.segmented_frame import SegmentedFrame
from pylot.simulation.utils import get_world

VEHICLE_DESTINATION = Location(x=387.73 - 370, y=327.07, z=0.5)
SAVED_FRAMES = collections.deque()
CLEANUP_FUNCTION = None


def spawn_camera(camera_bp,
                 transform,
                 ego_vehicle,
                 width='1280',
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
    camera_setup = SegmentedCameraSetup(
        "segmented_camera", width, height,
        pylot.utils.Transform.from_simulator_transform(transform))

    _world.tick()
    return camera, camera_setup


def retrieve_actor(world, bp_regex, role_name):
    """ Retrieves the actor from the world with the given blueprint and the
    role_name.

    Args:
        world: The instance of the simulator to retrieve the actors from.
        bp_regex: The actor's blueprint to be retrieved from the simulator.
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


def compute_and_log_miou(current_frame, current_timestamp, csv, deadline=210):
    """ Computes the mIOU for the given frame relative to the previous frames
    and logs it to the given csv file.

    Args:
        current_frame: The frame to compute the mIOU for.
        current_timestamp: The timestamp associated with the frame.
        csv: The csv file to write the results to.
    """
    SAVED_FRAMES.append((current_timestamp, current_frame))

    # Remove data older than the deadline that we don't need anymore.
    while (current_timestamp - SAVED_FRAMES[0][0]) * 1000 > deadline:
        SAVED_FRAMES.popleft()

    # Go over each of the saved frames, compute the difference in the
    # timestamp, and the mIOU and log both of them.
    for old_timestamp, old_frame in SAVED_FRAMES:
        (mean_iou, class_iou) = \
            current_frame.compute_semantic_iou_using_masks(old_frame)
        time_diff = current_timestamp - old_timestamp

        # Format of the CSV file: (latency_in_ms, class, mean IOU)
        csv.writerow([time_diff * 1000, "Scene", mean_iou])

        # Insert the results for the person.
        person_key = 4
        if person_key in class_iou:
            csv.writerow([time_diff * 1000, "Person", class_iou[person_key]])


def process_segmentation_images(msg,
                                camera_setup,
                                ego_vehicle,
                                speed,
                                csv,
                                dump=False):
    print("Received a message for the time: {}".format(msg.timestamp))

    # If we are in distance to the destination, stop and exit with success.
    if ego_vehicle.get_location().distance(VEHICLE_DESTINATION) <= 5:
        ego_vehicle.set_velocity(Vector3D())
        CLEANUP_FUNCTION()
        sys.exit(0)

    # Compute the segmentation mIOU.
    frame = SegmentedFrame.from_simulator_image(msg, camera_setup)
    compute_and_log_miou(frame, msg.timestamp, csv)

    # Visualize the run.
    if dump:
        frame.save(int(msg.timestamp * 1000), './_out/', 'seg')

    # Move the ego_vehicle according to the given speed.
    ego_vehicle.set_velocity(Vector3D(x=-speed))

    # Move the simulator forward.
    ego_vehicle.get_world().tick()


def cleanup_function(world, cameras, csv_file):
    """ Cleans up the state of the simulator.

    Args:
        world: The instance of the world to restore the asynchronous mode on.
        cameras: The cameras to destroy.
        csv_file: The open results file to close.
    """
    set_synchronous_mode(world, False)
    for camera in cameras:
        camera.destroy()
    csv_file.close()


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

    # Connect the segmentation camera to the vehicle.
    segmentation_camera_transform = Transform(
        location=Location(1.0, 0.0, 1.8),
        rotation=Rotation(0, 0, 0))
    segmentation_camera, camera_setup = spawn_camera(
        'sensor.camera.semantic_segmentation', segmentation_camera_transform,
        ego_vehicle, *args.res.split('x'))

    # Open the CSV file for writing.
    csv_file = open(args.output, 'w')
    csv_writer = csv.writer(csv_file)

    # Create the cleanup function.
    global CLEANUP_FUNCTION
    CLEANUP_FUNCTION = functools.partial(cleanup_function,
                                         world=world,
                                         cameras=[segmentation_camera],
                                         csv_file=csv_file)

    # Register a callback function with the camera.
    segmentation_camera.listen(
        functools.partial(process_segmentation_images,
                          camera_setup=camera_setup,
                          ego_vehicle=ego_vehicle,
                          speed=args.speed,
                          csv=csv_writer,
                          dump=args.dump))

    try:
        # To keep the thread alive so that the images can be processed.
        while True:
            pass
    except KeyboardInterrupt:
        CLEANUP_FUNCTION()
        sys.exit(0)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="mIoU data collector")
    argparser.add_argument('-s',
                           '--speed',
                           dest='speed',
                           required=True,
                           type=float,
                           help="Set the speed of the vehicle in m/s.")
    argparser.add_argument('-d',
                           '--delta',
                           dest='delta',
                           type=float,
                           required=True,
                           help='The delta at which to run the simulator.')
    argparser.add_argument('-o',
                           '--output',
                           default='results.csv',
                           help='The CSV file to output the results to.')
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

    # Verify arguments.
    args = argparser.parse_args()
    if args.delta > 0.1:
        raise ValueError(
            "The simulator does not work well with frame rates lower "
            "than 10FPS.")

    if not args.output.endswith('csv'):
        raise ValueError("The output should be a CSV file.")

    if args.dump and not os.path.exists('./_out'):
        os.mkdir('./_out')

    main(args)
