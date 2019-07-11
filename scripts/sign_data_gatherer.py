from absl import app
from absl import flags
import copy
import carla
import json
import numpy as np
import PIL.Image as Image
import time

from pylot.map.hd_map import HDMap
from pylot.perception.detection.utils import visualize_ground_bboxes
from pylot.simulation.carla_utils import convert_speed_limit_actors,\
    convert_traffic_stop_actors, convert_traffic_light_actors,\
    get_world, to_carla_location, to_carla_transform
from pylot.simulation.utils import depth_to_array, labels_to_array,\
    to_bgra_array, to_pylot_transform
from pylot.utils import bgra_to_bgr, bgr_to_rgb
import pylot.simulation.utils

FLAGS = flags.FLAGS
CARLA_IMAGE = None
DEPTH_FRAME = None
SEGMENTED_FRAME = None
TRAFFIC_LIGHTS = []
TRAFFIC_STOPS = []
SPEED_SIGNS = []

flags.DEFINE_string('data_path', 'data/', 'Path where data will be saved')
flags.DEFINE_integer('frame_width', 800, 'Camera frame width')
flags.DEFINE_integer('frame_height', 600, 'Camera frame height')
flags.DEFINE_bool('visualize_bboxes', False,
                  'True to enable bbox visualizer')


def on_camera_msg(image):
    global CARLA_IMAGE
    CARLA_IMAGE = image


def on_depth_msg(carla_image):
    global DEPTH_FRAME
    DEPTH_FRAME = depth_to_array(carla_image)


def on_segmented_msg(carla_image):
    global SEGMENTED_FRAME
    SEGMENTED_FRAME = labels_to_array(carla_image)


def add_camera(world, transform, callback):
    camera_blueprint = world.get_blueprint_library().find(
        'sensor.camera.rgb')
    camera_blueprint.set_attribute('image_size_x', str(FLAGS.frame_width))
    camera_blueprint.set_attribute('image_size_y', str(FLAGS.frame_height))
    camera = world.spawn_actor(camera_blueprint, transform)
    # Register callback to be invoked when a new frame is received.
    camera.listen(callback)
    return camera


def add_depth_camera(world, transform, callback):
    depth_blueprint = world.get_blueprint_library().find(
        'sensor.camera.depth')
    depth_blueprint.set_attribute('image_size_x', str(FLAGS.frame_width))
    depth_blueprint.set_attribute('image_size_y', str(FLAGS.frame_height))
    depth_camera = world.spawn_actor(depth_blueprint, transform)
    # Register callback to be invoked when a new frame is received.
    depth_camera.listen(callback)
    return depth_camera


def add_segmented_camera(world, transform, callback):
    segmented_blueprint = world.get_blueprint_library().find(
        'sensor.camera.semantic_segmentation')
    segmented_blueprint.set_attribute('image_size_x', str(FLAGS.frame_width))
    segmented_blueprint.set_attribute('image_size_y', str(FLAGS.frame_height))
    segmented_camera = world.spawn_actor(segmented_blueprint, transform)
    segmented_camera.listen(callback)
    return segmented_camera


def setup_world():
    # Connect to the Carla simulator.
    client, world = get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    world.apply_settings(settings)
    return world


def wait_for_data(world):
    world.tick()
    global CARLA_IMAGE
    global DEPTH_FRAME
    global SEGMENTED_FRAME
    while (CARLA_IMAGE is None or
           DEPTH_FRAME is None or
           SEGMENTED_FRAME is None):
        time.sleep(0.1)


def reset_frames():
    global DEPTH_FRAME
    global SEGMENTED_FRAME
    global CARLA_IMAGE
    # Reset frames.
    DEPTH_FRAME = None
    SEGMENTED_FRAME = None
    CARLA_IMAGE = None


def log_bounding_boxes(
        carla_image, depth_frame, segmented_frame,
        traffic_lights, speed_signs, stop_signs):
    game_time = int(carla_image.timestamp * 1000)
    print("Processing game time {}".format(game_time))
    frame = bgra_to_bgr(to_bgra_array(carla_image))
    # Copy the frame to ensure its on the heap.
    frame = copy.deepcopy(frame)
    transform = to_pylot_transform(carla_image.transform)

    speed_limit_det_objs = pylot.simulation.utils.get_speed_limit_det_objs(
        speed_signs, transform, transform, depth_frame, FLAGS.frame_width,
        FLAGS.frame_height, 90, segmented_frame)
    traffic_stop_det_objs = pylot.simulation.utils.get_traffic_stop_det_objs(
        stop_signs, transform, transform)
    traffic_light_det_objs = pylot.simulation.utils.get_traffic_light_det_objs(
        traffic_lights, transform, transform, depth_frame, FLAGS.frame_width,
        FLAGS.frame_height, 90, segmented_frame)

    det_objs = (speed_limit_det_objs +
                traffic_stop_det_objs +
                traffic_light_det_objs)

    if FLAGS.visualize_bboxes:
        visualize_ground_bboxes('bboxes', game_time, frame, det_objs)

    # Log the frame.
    frame = bgr_to_rgb(frame)
    file_name = '{}signs-{}.png'.format(FLAGS.data_path, game_time)
    rgb_img = Image.fromarray(np.uint8(frame))
    rgb_img.save(file_name)

    # Log the bounding boxes.
    bboxes = [det_obj.get_bbox_label() for det_obj in det_objs]
    file_name = '{}bboxes-{}.json'.format(FLAGS.data_path, game_time)
    with open(file_name, 'w') as outfile:
        json.dump(bboxes, outfile)


def main(argv):
    world = setup_world()
    world_map = world.get_map()
    wps = world_map.generate_waypoints(1.0)
    hd_map = HDMap(world_map, None)
    world.tick()
    # Sleep a bit to ensure the simulator actually ticks.
    time.sleep(3)
    actor_list = world.get_actors()
    tl_actors = actor_list.filter('traffic.traffic_light*')
    global TRAFFIC_LIGHTS
    TRAFFIC_LIGHTS = convert_traffic_light_actors(tl_actors)
    traffic_stop_actors = actor_list.filter('traffic.stop')
    global TRAFFIC_STOPS
    TRAFFIC_STOPS = convert_traffic_stop_actors(traffic_stop_actors)
    speed_limit_actors = actor_list.filter('traffic.speed_limit*')
    global SPEED_SIGNS
    SPEED_SIGNS = convert_speed_limit_actors(speed_limit_actors)

    transforms_of_interest = []
    # Add transforms that are close to traffic lights.
    for w in wps:
        w_loc = w.transform.location
        intersection_dist = hd_map.distance_to_intersection(w_loc)
        if intersection_dist:
            camera_transform = to_pylot_transform(w.transform)
            camera_transform.location.z += 2.0
            transform = to_carla_transform(camera_transform)
            transforms_of_interest.append(transform)

    # Add transforms that are close to speed limit signs.
    for speed_sign in SPEED_SIGNS:
        for offset in range(3, 25):
            offset_loc = pylot.simulation.utils.Location(x=0, y=offset, z=2.0)
            transform = to_carla_transform(speed_sign.transform)
            location = transform.transform(to_carla_location(offset_loc))
            w = world_map.get_waypoint(
                location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving)
            camera_transform = to_pylot_transform(w.transform)
            camera_transform.location.z += 2.0
            transform = to_carla_transform(camera_transform)
            transforms_of_interest.append(transform)

    # Add transforms that are close to stop signs.
    for stop_sign in TRAFFIC_STOPS:
        for offset in range(5, 15, 3):
            offset_loc = pylot.simulation.utils.Location(x=offset, y=0, z=2.0)
            transform = to_carla_transform(stop_sign.transform)
            location = transform.transform(to_carla_location(offset_loc))
            w = world_map.get_waypoint(
                location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving)
            camera_transform = to_pylot_transform(w.transform)
            camera_transform.location.z += 2.0
            transform = to_carla_transform(camera_transform)
            transforms_of_interest.append(transform)

    for transform in transforms_of_interest:
        camera = add_camera(world, transform, on_camera_msg)
        depth_camera = add_depth_camera(world, transform, on_depth_msg)
        segmented_camera = add_segmented_camera(
            world, transform, on_segmented_msg)
        wait_for_data(world)
        global CARLA_IMAGE
        global DEPTH_FRAME
        global SEGMENTED_FRAME
        log_bounding_boxes(
            CARLA_IMAGE,
            DEPTH_FRAME,
            SEGMENTED_FRAME,
            TRAFFIC_LIGHTS,
            SPEED_SIGNS,
            TRAFFIC_STOPS)
        reset_frames()
        segmented_camera.destroy()
        depth_camera.destroy()
        camera.destroy()


if __name__ == '__main__':
    app.run(main)
