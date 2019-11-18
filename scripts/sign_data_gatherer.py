from absl import app
from absl import flags
import carla
import copy
import json
import numpy as np
import PIL.Image as Image
import time

from pylot.perception.detection.utils import annotate_image_with_bboxes,\
    visualize_ground_bboxes
from pylot.simulation.carla_utils import convert_speed_limit_actors,\
    convert_traffic_light_actor, convert_traffic_light_actors,\
    convert_traffic_stop_actors, get_world
from pylot.simulation.utils import depth_to_array, labels_to_array,\
    to_bgra_array, to_pylot_transform
from pylot.utils import bgra_to_bgr, bgr_to_rgb
import pylot.simulation.utils


FLAGS = flags.FLAGS
CARLA_IMAGE = None
DEPTH_FRAME = None
SEGMENTED_FRAME = None

flags.DEFINE_string('data_path', 'data/', 'Path where data will be saved')
flags.DEFINE_integer('frame_width', 1920, 'Camera frame width')
flags.DEFINE_integer('frame_height', 1080, 'Camera frame height')
flags.DEFINE_bool('visualize_bboxes', False,
                  'True to enable bbox visualizer')
flags.DEFINE_bool('log_bbox_images', False,
                  'True to enable logging of bbox annodated images')
flags.DEFINE_integer('camera_fov', 45, 'Camera fov')


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
    camera_blueprint.set_attribute('fov', str(FLAGS.camera_fov))
    camera = world.spawn_actor(camera_blueprint, transform)
    # Register callback to be invoked when a new frame is received.
    camera.listen(callback)
    return camera


def add_depth_camera(world, transform, callback):
    depth_blueprint = world.get_blueprint_library().find(
        'sensor.camera.depth')
    depth_blueprint.set_attribute('image_size_x', str(FLAGS.frame_width))
    depth_blueprint.set_attribute('image_size_y', str(FLAGS.frame_height))
    depth_blueprint.set_attribute('fov', str(FLAGS.camera_fov))
    depth_camera = world.spawn_actor(depth_blueprint, transform)
    # Register callback to be invoked when a new frame is received.
    depth_camera.listen(callback)
    return depth_camera


def add_segmented_camera(world, transform, callback):
    segmented_blueprint = world.get_blueprint_library().find(
        'sensor.camera.semantic_segmentation')
    segmented_blueprint.set_attribute('image_size_x', str(FLAGS.frame_width))
    segmented_blueprint.set_attribute('image_size_y', str(FLAGS.frame_height))
    segmented_blueprint.set_attribute('fov', str(FLAGS.camera_fov))
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


def get_traffic_light_objs(traffic_lights, camera_transform, depth_frame,
                           width, height, color, town_name):
    det_objs = pylot.simulation.utils.get_traffic_light_det_objs(
        traffic_lights, camera_transform, depth_frame,
        width, height, town_name, FLAGS.camera_fov)
    # Overwrite traffic light color because we control it without refreshing
    # the agents.
    if color == carla.TrafficLightState.Yellow:
        label = 'yellow'
    elif color == carla.TrafficLightState.Green:
        label = 'green'
    elif color == carla.TrafficLightState.Red:
        label = 'red'
    elif color == carla.TrafficLightState.Off:
        label = 'off'
    else:
        raise ValueError('Unknown traffic light color')
    label += ' traffic light'

    for det_obj in det_objs:
        det_obj.label = label
    return det_objs


def log_bounding_boxes(
        carla_image, depth_frame, segmented_frame,
        traffic_lights, tl_color, speed_signs, stop_signs):
    game_time = int(carla_image.timestamp * 1000)
    print("Processing game time {}".format(game_time))
    frame = bgra_to_bgr(to_bgra_array(carla_image))
    # Copy the frame to ensure its on the heap.
    frame = copy.deepcopy(frame)
    transform = to_pylot_transform(carla_image.transform)
    _, world = get_world()
    town_name = world.get_map().name

    speed_limit_det_objs = pylot.simulation.utils.get_speed_limit_det_objs(
        speed_signs, transform, transform, depth_frame, FLAGS.frame_width,
        FLAGS.frame_height, FLAGS.camera_fov, segmented_frame)
    traffic_stop_det_objs = pylot.simulation.utils.get_traffic_stop_det_objs(
        stop_signs, transform, depth_frame, FLAGS.frame_width,
        FLAGS.frame_height, FLAGS.camera_fov)
    traffic_light_det_objs = get_traffic_light_objs(
        traffic_lights, transform, depth_frame,
        FLAGS.frame_width, FLAGS.frame_height, tl_color, town_name)

    det_objs = (speed_limit_det_objs +
                traffic_stop_det_objs +
                traffic_light_det_objs)

    if FLAGS.visualize_bboxes:
        visualize_ground_bboxes('bboxes', game_time, frame, det_objs)

    # Log the frame.
    rgb_frame = bgr_to_rgb(frame)
    file_name = '{}signs-{}.png'.format(FLAGS.data_path, game_time)
    rgb_img = Image.fromarray(np.uint8(rgb_frame))
    rgb_img.save(file_name)

    if FLAGS.log_bbox_images:
        annotate_image_with_bboxes(game_time, frame, det_objs)
        rgb_frame = bgr_to_rgb(frame)
        file_name = '{}annotated-signs-{}.png'.format(
            FLAGS.data_path, game_time)
        rgb_img = Image.fromarray(np.uint8(rgb_frame))
        rgb_img.save(file_name)

    # Log the bounding boxes.
    bboxes = [det_obj.get_bbox_label() for det_obj in det_objs]
    file_name = '{}bboxes-{}.json'.format(FLAGS.data_path, game_time)
    with open(file_name, 'w') as outfile:
        json.dump(bboxes, outfile)


def change_traffic_light_colors(world, color):
    actor_list = world.get_actors()
    tl_actors = actor_list.filter('traffic.traffic_light*')
    for tl in tl_actors:
        tl.set_state(color)
        tl.freeze(True)
    world.tick()


def get_actors(world):
    actor_list = world.get_actors()
    tl_actors = actor_list.filter('traffic.traffic_light*')
    traffic_lights = convert_traffic_light_actors(tl_actors)
    traffic_stop_actors = actor_list.filter('traffic.stop')
    traffic_stops = convert_traffic_stop_actors(traffic_stop_actors)
    speed_limit_actors = actor_list.filter('traffic.speed_limit*')
    speed_signs = convert_speed_limit_actors(speed_limit_actors)
    return (tl_actors, traffic_lights, traffic_stops, speed_signs)


def log_obstacles(world,
                  transforms_of_interest,
                  traffic_lights,
                  tl_color,
                  speed_signs,
                  traffic_stops):
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
            traffic_lights,
            tl_color,
            speed_signs,
            traffic_stops)
        reset_frames()
        segmented_camera.destroy()
        depth_camera.destroy()
        camera.destroy()


def check_lights_opposite(light_a, light_b):
    """ Checks if the two given lights are opposite to each other or not. """
    def get_forward_vector(light):
        light_vector = light.get_transform().get_forward_vector()
        return [light_vector.x, light_vector.y, light_vector.z]
    light_a_vector = get_forward_vector(light_a)
    light_b_vector = get_forward_vector(light_b)
    return np.dot(light_a_vector, light_b_vector) < -0.98


def log_traffic_lights(world):
    world_map = world.get_map()
    (traffic_lights, _, traffic_stops, speed_signs) = get_actors(world)
    tl_colors = [
        carla.TrafficLightState.Yellow, carla.TrafficLightState.Green,
        carla.TrafficLightState.Red
    ]
    for light in traffic_lights:
        print("Working for traffic light {}".format(light.id))
        # For every traffic light, get the neighbouring lights except the one
        # directly opposite.
        group_lights = []
        for n_light in light.get_group_traffic_lights():
            if not check_lights_opposite(light, n_light):
                group_lights.append(convert_traffic_light_actor(n_light))

        transforms_of_interest = []
        for offset in range(10, 40, 5):
            # Traffic lights have different coordinate systems, hence
            # we need to offset y, instead of x and add that to the trigger
            # volume location.
            offset_loc = pylot.simulation.utils.Location(
                x=light.trigger_volume.location.x,
                y=light.trigger_volume.location.y + offset,
                z=light.trigger_volume.location.z)
            offset_rot = pylot.simulation.utils.Rotation(pitch=0,
                                                         yaw=0,
                                                         roll=0)
            offset_trans = pylot.simulation.utils.Transform(
                offset_loc, offset_rot)

            # Transform the offset relative to the traffic light.
            transform = to_pylot_transform(
                light.get_transform()) * offset_trans
            location = transform.location.as_carla_location()

            # Get the waypoint nearest to the transform.
            w = world_map.get_waypoint(location,
                                       project_to_road=True,
                                       lane_type=carla.LaneType.Driving)
            w_rotation = w.transform.rotation
            camera_transform = to_pylot_transform(w.transform)
            camera_transform.location.z += 2.0
            transform = camera_transform.as_carla_transform()
            transforms_of_interest.append(transform)

            # Get the right lanes.
            wp_right = w.get_right_lane()
            while wp_right and wp_right.lane_type == carla.LaneType.Driving \
                    and w_rotation == wp_right.transform.rotation:
                camera_transform = to_pylot_transform(wp_right.transform)
                camera_transform.location.z += 2.0
                transform = camera_transform.as_carla_transform()
                transforms_of_interest.append(transform)
                wp_right = wp_right.get_right_lane()

            # Get the left lanes.
            wp_left = w.get_left_lane()
            while wp_left and wp_left.lane_type == carla.LaneType.Driving and \
                    w_rotation == wp_left.transform.rotation:
                camera_transform = to_pylot_transform(wp_left.transform)
                camera_transform.location.z += 2.0
                transform = camera_transform.as_carla_transform()
                transforms_of_interest.append(transform)
                wp_left = wp_left.get_left_lane()

        print("The total number of transforms were: {}".format(
            len(transforms_of_interest)))
        for tl_color in tl_colors:
            change_traffic_light_colors(world, tl_color)
            world.tick()
            time.sleep(1)
            log_obstacles(world, transforms_of_interest, group_lights,
                          tl_color, speed_signs, traffic_stops)


def log_speed_limits(world):
    world_map = world.get_map()
    (_, traffic_lights, traffic_stops, speed_signs) = get_actors(world)
    transforms_of_interest = []
    # Add transforms that are close to speed limit signs.
    for speed_sign in speed_signs:
        for offset in range(10, 25, 5):
            # Speed signs have different coordinate systems, hence
            # we need to offset y, instead of x.
            offset_loc = pylot.simulation.utils.Location(x=0, y=offset, z=0)
            offset_rot = pylot.simulation.utils.Rotation(
                pitch=0, yaw=0, roll=0)
            offset_trans = pylot.simulation.utils.Transform(
                offset_loc, offset_rot)
            transform = speed_sign.transform * offset_trans
            location = transform.location.as_carla_location()
            w = world_map.get_waypoint(
                location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving)
            camera_transform = to_pylot_transform(w.transform)
            camera_transform.location.z += 2.0
            transform = camera_transform.as_carla_transform()
            transforms_of_interest.append(transform)
    # Ensure all traffic lights are red.
    change_traffic_light_colors(world, carla.TrafficLightState.Red)
    world.tick()
    time.sleep(1)
    (_, traffic_lights, traffic_stops, speed_signs) = get_actors(world)
    log_obstacles(world,
                  transforms_of_interest,
                  traffic_lights,
                  carla.TrafficLightState.Red,
                  speed_signs,
                  traffic_stops)


def log_stop_signs(world):
    world_map = world.get_map()
    (_, traffic_lights, traffic_stops, speed_signs) = get_actors(world)
    transforms_of_interest = []
    # Add transforms that are close to stop signs.
    for stop_sign in traffic_stops:
        for offset in range(10, 25, 5):
            offset_loc = pylot.simulation.utils.Location(x=-offset, y=0, z=0)
            offset_rot = pylot.simulation.utils.Rotation(
                pitch=0, yaw=0, roll=0)
            offset_trans = pylot.simulation.utils.Transform(
                offset_loc, offset_rot)
            transform = stop_sign.transform * offset_trans
            location = transform.location.as_carla_location()
            w = world_map.get_waypoint(
                location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving)
            camera_transform = to_pylot_transform(w.transform)
            camera_transform.location.z += 2.0
            transform = camera_transform.as_carla_transform()
            transforms_of_interest.append(transform)
    # Ensure all traffic lights are red.
    change_traffic_light_colors(world, carla.TrafficLightState.Red)
    world.tick()
    time.sleep(1)
    (_, traffic_lights, traffic_stops, speed_signs) = get_actors(world)
    log_obstacles(world,
                  transforms_of_interest,
                  traffic_lights,
                  carla.TrafficLightState.Red,
                  speed_signs,
                  traffic_stops)


def main(argv):
    world = setup_world()
    world.tick()
    # Sleep a bit to ensure the simulator actually ticks.
    time.sleep(1)
    log_traffic_lights(world)
    log_speed_limits(world)
    log_stop_signs(world)


if __name__ == '__main__':
    app.run(main)
