from absl import app
from absl import flags
import copy
import carla
import json
import numpy as np
import PIL.Image as Image
import time

from pylot.map.hd_map import HDMap
from pylot.perception.detection.utils import DetectedObject,\
    TrafficLightColor, annotate_image_with_bboxes,\
    get_bounding_boxes_from_segmented, visualize_ground_bboxes
from pylot.perception.segmentation.utils import get_traffic_sign_pixels
from pylot.simulation.carla_utils import convert_speed_limit_actors,\
    convert_traffic_stop_actors, convert_traffic_light_actors,\
    get_world, to_carla_location, to_carla_transform
from pylot.simulation.utils import batch_get_3d_world_position_with_depth_map,\
    depth_to_array, labels_to_array, match_bboxes_with_traffic_lights,\
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


def log_bounding_boxes(carla_image, depth_frame, segmented_frame):
    game_time = int(carla_image.timestamp * 1000)
    print("Processing game time {}".format(game_time))
    # Save the camera frame.
    frame = bgra_to_bgr(to_bgra_array(carla_image))
    # Copy the frame to ensure its on the heap.
    frame = copy.deepcopy(frame)

    # Compute the bounding boxes.
    transform = to_pylot_transform(carla_image.transform)
    traffic_signs_frame = get_traffic_sign_pixels(SEGMENTED_FRAME)
    bboxes = get_bounding_boxes_from_segmented(traffic_signs_frame)

    # Get the positions of the bounding box centers.
    x_mids = [(bbox[0] + bbox[1]) / 2 for bbox in bboxes]
    y_mids = [(bbox[2] + bbox[3]) / 2 for bbox in bboxes]
    pos_3d = batch_get_3d_world_position_with_depth_map(
        x_mids, y_mids, DEPTH_FRAME, FLAGS.frame_width, FLAGS.frame_height,
        90, transform)
    pos_and_bboxes = zip(pos_3d, bboxes)

    global TRAFFIC_LIGHTS
    # Map traffic lights to bounding boxes based on 3d world position.
    tl_bboxes = match_bboxes_with_traffic_lights(
        transform, pos_and_bboxes, TRAFFIC_LIGHTS)
    det_objs = []
    for bbox, color in tl_bboxes:
        if color == TrafficLightColor.GREEN:
            det_objs.append(DetectedObject(bbox, 1.0, 'green traffic light'))
        elif color == TrafficLightColor.YELLOW:
            det_objs.append(DetectedObject(bbox, 1.0, 'yellow traffic light'))
        elif color == TrafficLightColor.RED:
            det_objs.append(DetectedObject(bbox, 1.0, 'red traffic light'))
        else:
            det_objs.append(DetectedObject(bbox, 1.0, 'off traffic light'))

    if FLAGS.visualize_bboxes:
        tl_bboxes = [(det_obj.corners, det_obj.label)
                     for det_obj in det_objs]
        visualize_ground_bboxes(
            'bboxes', game_time, frame, [], [], [], tl_bboxes)

    # color_map = {'red traffic light': [0, 0, 255],
    #              'yellow traffic light': [0, 255, 255],
    #              'green traffic light': [0, 255, 0]}
    # annotate_image_with_bboxes(game_time, frame, det_objs, color_map)

    frame = bgr_to_rgb(frame)
    file_name = '{}traffic-light-{}.png'.format(FLAGS.data_path, game_time)
    rgb_img = Image.fromarray(np.uint8(frame))
    rgb_img.save(file_name)

    bboxes = []
    for det_obj in det_objs:
        (xmin, xmax, ymin, ymax) = det_obj.corners
        bboxes.append((det_obj.label, ((xmin, ymin), (xmax, ymax))))
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

    for w in wps:
        w_loc = w.transform.location
        intersection_dist = hd_map.distance_to_intersection(w_loc)
        if intersection_dist:
            camera_transform = to_pylot_transform(w.transform)
            camera_transform.location.z += 2.0
            transform = to_carla_transform(camera_transform)
            camera = add_camera(world, transform, on_camera_msg)
            depth_camera = add_depth_camera(world, transform, on_depth_msg)
            segmented_camera = add_segmented_camera(
                world, transform, on_segmented_msg)
            wait_for_data(world)
            global CARLA_IMAGE
            global DEPTH_FRAME
            global SEGMENTED_FRAME
            log_bounding_boxes(CARLA_IMAGE, DEPTH_FRAME, SEGMENTED_FRAME)
            reset_frames()
            segmented_camera.destroy()
            depth_camera.destroy()
            camera.destroy()


if __name__ == '__main__':
    app.run(main)
