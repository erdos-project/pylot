from __future__ import print_function

import argparse
import collections
import csv
import functools
import sys
from time import sleep
try:
    import queue
except ImportError:
    import Queue as queue

from carla import Location, Rotation, Transform, Vector3D

from miou_scenario_runner import cleanup_function, retrieve_actor, \
    setup_world, spawn_camera

import numpy as np

import pygame

import pylot.utils
from pylot.drivers.sensor_setup import DepthCameraSetup
from pylot.perception.depth_frame import DepthFrame
from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.detection.utils import get_precision_recall_at_iou
from pylot.perception.segmentation.segmented_frame import SegmentedFrame

VEHICLE_DESTINATION = Location(x=387.73 - 370, y=327.07, z=0.5)
SAVED_DETECTIONS = collections.deque()
CLEANUP_FUNCTION = None
RGB_IMAGES = queue.Queue()
SEMANTIC_IMAGES = queue.Queue()
BB_COLOR = (248, 64, 24)


def process_rgb_images(rgb_image_msg):
    """ The callback function for the RGB camera. Just saves the images to the
    global queue to be retrieved by the other function for analysis.

    Args:
        rgb_image_msg: The RGB image sent by the sensor from the simulation.
    """
    global RGB_IMAGES
    RGB_IMAGES.put(rgb_image_msg)


def process_semantic_images(semantic_image_msg):
    """ The callback function for the semantic camera. Just saves the images
    to the global queue to be retrieved by the other function for analysis.

    Args:
        semantic_image_msg: The semantic image sent by the sensor from the
        simulation.
    """
    global SEMANTIC_IMAGES
    SEMANTIC_IMAGES.put(semantic_image_msg)


def retrieve_rgb_image(timestamp):
    """ Retrieve the RGB image from the global queue that is populated by the
    image messages from the sensor.

    Args:
        timestamp: The timestamp to retrieve the RGB image for.

    Returns:
        The image message from the queue with the same timestamp as the given
        timestamp.
    """
    while True:
        rgb_image_msg = RGB_IMAGES.get()
        if rgb_image_msg.timestamp == timestamp:
            return rgb_image_msg


def retrieve_semantic_image(timestamp):
    """ Retrieve the semantic image from the global queue that is populated by
    the image messages from the sensor.

    Args:
        timestamp: The timestamp to retrieve the RGB image for.

    Returns:
        The image message from the queue with the same timestamp as the given
        timestamp.
    """
    while True:
        semantic_image_msg = SEMANTIC_IMAGES.get()
        if semantic_image_msg.timestamp == timestamp:
            return semantic_image_msg


def draw_image(image, surface, blend=False):
    """ Draw the given image on the surface.

    Args:
        image: The image to draw.
        surface: The pygame surface on which to draw the image.
        blend: Blend the image into the surface.
    """
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def draw_bounding_box(bbox, surface, color=BB_COLOR):
    """ Draw the given bounding box on the given surface.

    Args:
        bbox: The bounding box to draw on the given surface.
        surface: The surface to draw the bounding box on.
    """
    (x_min, x_max, y_min, y_max) = bbox
    pygame.draw.line(surface, color, (x_min, y_min), (x_max, y_min))
    pygame.draw.line(surface, color, (x_min, y_min), (x_min, y_max))
    pygame.draw.line(surface, color, (x_max, y_min), (x_max, y_max))
    pygame.draw.line(surface, color, (x_min, y_max), (x_max, y_max))


def compute_and_log_map(current_people,
                        current_timestamp,
                        csv,
                        deadline=210,
                        base_iou=0.5,
                        step=0.05):
    """ Computes the AP from the given IOU for the obstacles. Note that,
    since we use a perfect detector, our confidence values for each detection
    is 1.0 and so we can't vary the recall. Thus, we cannot calculate the
    area under the precision-recall curve, and default to using AP50 as our
    metric for mAP.

    Args:
        current_people: List of bboxes for the detected people
        current_timestamp: The timestamp associated with the current frame.
        csv: The csv file to write the results to.
        deadline: The oldest frame to compare the results to.
        base_iou: The IOU to start from.
        step: The step to take from the base_IOU to reach 1.0
    """
    SAVED_DETECTIONS.append((current_timestamp, current_people))

    # Remove data older than the deadline that we don't need anymore.
    while (current_timestamp - SAVED_DETECTIONS[0][0]) * 1000 > deadline:
        SAVED_DETECTIONS.popleft()

    # Go over each of the saved frames, compute the difference in the
    # timestamp, the AP at the given IOU and log them.
    for old_timestamp, old_detections in SAVED_DETECTIONS:
        for iou in np.arange(base_iou, 1.0, step):
            precision, _ = get_precision_recall_at_iou(current_people,
                                                       old_detections, iou)
            time_diff = current_timestamp - old_timestamp

            # Format of the CSV file: (latency_in_ms, AP{IOU}, {IOU})
            csv.writerow([time_diff * 1000, precision, iou])


def process_depth_images(msg,
                         depth_camera_setup,
                         ego_vehicle,
                         speed,
                         csv,
                         surface,
                         visualize=False):
    print("Received a message for the time: {}".format(msg.timestamp))

    # If we are in distance to the destination, stop and exit with success.
    if ego_vehicle.get_location().distance(VEHICLE_DESTINATION) <= 5:
        ego_vehicle.set_velocity(Vector3D())
        CLEANUP_FUNCTION()
        sys.exit(0)

    # Get the RGB image corresponding to the given depth image timestamp.
    rgb_image = retrieve_rgb_image(msg.timestamp)

    # Get the semantic image corresponding to the given depth image timestamp.
    semantic_image = retrieve_semantic_image(msg.timestamp)
    semantic_frame = SegmentedFrame.from_simulator_image(semantic_image,
                                                     depth_frame.camera_setup)

    # Visualize the image and the bounding boxes if needed.
    if visualize:
        draw_image(rgb_image, surface)

    # Transform people into obstacles relative to the current frame.
    bb_surface = None
    resolution = (depth_camera_setup.width, depth_camera_setup.height)
    if visualize:
        bb_surface = pygame.Surface(resolution)
        bb_surface.set_colorkey((0, 0, 0))

    vehicle_transform = pylot.utils.Transform.from_simulator_transform(
        ego_vehicle.get_transform())

    depth_frame = DepthFrame.from_simulator_frame(msg, depth_camera_setup)
    # Transform the static camera setup with respect to the location of the
    # vehicle in the world.
    depth_frame.camera_setup.set_transform(vehicle_transform *
                                           depth_frame.camera_setup.transform)

    detected_people = []
    for person in ego_vehicle.get_world().get_actors().filter('walker.*'):
        obstacle = Obstacle.from_simulator_actor(person)
        if obstacle._distance(vehicle_transform) > 125:
            bbox = None
        else:
            bbox = obstacle.populate_bounding_box_2D(depth_frame,
                                                     semantic_frame.frame)
        if bbox is not None:
            detected_people.append(bbox)
            if visualize:
                draw_bounding_box(bbox, bb_surface)

    # We have drawn all the bounding boxes on the bb_surface, now put it on
    # the RGB image surface.
    if visualize:
        surface.blit(bb_surface, (0, 0))
        pygame.display.flip()

    # Compute the mAP.
    print("We detected a total of {} people.".format(len(detected_people)))
    compute_and_log_map(detected_people, msg.timestamp, csv)

    # Move the ego_vehicle according to the given speed.
    ego_vehicle.set_velocity(Vector3D(x=-speed))

    ego_vehicle.get_world().tick()


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

    # Transform of the cameras.
    camera_transform = Transform(location=Location(1.0, 0.0, 1.8),
                                 rotation=Rotation(0, 0, 0))

    # Connect the RGB camera to the vehicle.
    rgb_camera = spawn_camera('sensor.camera.rgb', camera_transform,
                              ego_vehicle, *args.res.split('x'))

    # Connect the Semantic segmentation camera to the vehicle.
    semantic_camera = spawn_camera('sensor.camera.semantic_segmentation',
                                   camera_transform, ego_vehicle,
                                   *args.res.split('x'))

    # Connect the depth camera to the vehicle.
    depth_camera = spawn_camera('sensor.camera.depth', camera_transform,
                                ego_vehicle, *args.res.split('x'))

    # Open the CSV file for writing.
    csv_file = open(args.output, 'w')
    csv_writer = csv.writer(csv_file)

    # Create the cleanup function.
    global CLEANUP_FUNCTION
    CLEANUP_FUNCTION = functools.partial(
        cleanup_function,
        world=world,
        cameras=[rgb_camera, semantic_camera, depth_camera],
        csv_file=csv_file)

    # Create a PyGame surface for debugging purposes.
    width, height = map(int, args.res.split('x'))
    surface = None
    if args.visualize:
        surface = pygame.display.set_mode((width, height),
                                          pygame.HWSURFACE | pygame.DOUBLEBUF)

    # Register a callback function with the camera.
    rgb_camera.listen(process_rgb_images)
    semantic_camera.listen(process_semantic_images)

    depth_camera_setup = DepthCameraSetup(
        "depth_camera", width, height,
        pylot.utils.Transform.from_simulator_transform(camera_transform))
    depth_camera.listen(
        functools.partial(process_depth_images,
                          depth_camera_setup=depth_camera_setup,
                          ego_vehicle=ego_vehicle,
                          speed=args.speed,
                          csv=csv_writer,
                          surface=surface,
                          visualize=args.visualize))

    try:
        # To keep the thread alive so that the images can be processed.
        while True:
            pass
    except KeyboardInterrupt:
        CLEANUP_FUNCTION()
        sys.exit(0)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="mAP data collector")
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
    argparser.add_argument('--visualize',
                           action='store_true',
                           help="Visualize the bounding boxes and the "
                           "RGB images.")

    # Verify arguments.
    args = argparser.parse_args()
    if args.delta > 0.1:
        raise ValueError(
            "The simulator does not work well with frame rates lower "
            "than 10FPS.")

    if not args.output.endswith('csv'):
        raise ValueError("The output should be a CSV file.")

    main(args)
