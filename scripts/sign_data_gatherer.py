from absl import app
from absl import flags
import carla
import copy
import json
import numpy as np
import PIL.Image as Image
import time
import re

from pylot.perception.detection.utils import annotate_image_with_bboxes,\
    visualize_ground_bboxes
from pylot.perception.segmentation.segmented_frame import SegmentedFrame
from pylot.simulation.carla_utils import convert_speed_limit_actors,\
    convert_traffic_light_actors, convert_traffic_stop_actors, get_world
from pylot.simulation.utils import Transform, TrafficLight
from pylot.utils import bgr_to_rgb
import pylot.simulation.utils
from pylot.simulation.sensor_setup import DepthCameraSetup
from pylot.simulation.messages import FrameMessage, DepthFrameMessage

FLAGS = flags.FLAGS
CARLA_IMAGE = None
DEPTH_FRAME = None
SEGMENTED_FRAME = None

flags.DEFINE_string('data_path', 'data/', 'Path where data will be saved')
flags.DEFINE_integer('frame_width', 1920, 'Camera frame width')
flags.DEFINE_integer('frame_height', 1080, 'Camera frame height')
flags.DEFINE_bool('visualize_bboxes', False, 'True to enable bbox visualizer')
flags.DEFINE_bool('log_bbox_images', False,
                  'True to enable logging of bbox annodated images')
flags.DEFINE_integer('camera_fov', 45, 'Camera fov')


def on_camera_msg(image):
    global CARLA_IMAGE
    CARLA_IMAGE = FrameMessage(image,
                               int(image.timestamp * 1000),
                               encoding='carla')


def on_depth_msg(carla_image):
    global DEPTH_FRAME
    transform = Transform(carla_transform=carla_image.transform)
    camera_setup = DepthCameraSetup("depth_camera", FLAGS.frame_width,
                                    FLAGS.camera_height, transform,
                                    FLAGS.camera_fov)
    DEPTH_FRAME = DepthFrameMessage(carla_image,
                                    camera_setup,
                                    int(carla_image.timestamp * 1000),
                                    encoding='carla')


def on_segmented_msg(carla_image):
    global SEGMENTED_FRAME
    SEGMENTED_FRAME = SegmentedFrame(carla_image, encoding='carla')


def add_camera(world, transform, callback):
    camera_blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_blueprint.set_attribute('image_size_x', str(FLAGS.frame_width))
    camera_blueprint.set_attribute('image_size_y', str(FLAGS.frame_height))
    camera_blueprint.set_attribute('fov', str(FLAGS.camera_fov))
    camera = world.spawn_actor(camera_blueprint, transform)
    # Register callback to be invoked when a new frame is received.
    camera.listen(callback)
    return camera


def add_depth_camera(world, transform, callback):
    depth_blueprint = world.get_blueprint_library().find('sensor.camera.depth')
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
    settings.fixed_delta_seconds = 0.02
    world.apply_settings(settings)
    return world


def wait_for_data(world):
    world.tick()
    global CARLA_IMAGE
    global DEPTH_FRAME
    global SEGMENTED_FRAME
    while (CARLA_IMAGE is None or DEPTH_FRAME is None
           or SEGMENTED_FRAME is None):
        time.sleep(0.1)


def reset_frames():
    global DEPTH_FRAME
    global SEGMENTED_FRAME
    global CARLA_IMAGE
    # Reset frames.
    DEPTH_FRAME = None
    SEGMENTED_FRAME = None
    CARLA_IMAGE = None


def get_traffic_light_objs(traffic_lights, camera_setup, depth_frame,
                           segmented_frame, color, town_name):
    det_objs = pylot.simulation.utils.get_traffic_light_det_objs(
        traffic_lights, depth_frame, segmented_frame.as_numpy_array(),
        town_name, camera_setup)
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


def log_bounding_boxes(carla_image, depth_frame, segmented_frame,
                       traffic_lights, tl_color, speed_signs, stop_signs,
                       weather, town):
    game_time = int(carla_image.timestamp * 1000)
    print("Processing game time {} in {} with weather {}".format(
        game_time, town, weather))
    frame = carla_image.frame
    # Copy the frame to ensure its on the heap.
    frame = copy.deepcopy(frame)
    transform = Transform(carla_transform=carla_image.transform)
    _, world = get_world()
    town_name = world.get_map().name

    camera_setup = depth_frame.camera_setup
    speed_limit_det_objs = []
    if speed_signs:
        speed_limit_det_objs = pylot.simulation.utils.get_speed_limit_det_objs(
            speed_signs, transform, depth_frame.frame, segmented_frame,
            camera_setup)

    traffic_stop_det_objs = []
    if stop_signs:
        traffic_stop_det_objs = pylot.simulation.utils.get_traffic_stop_det_objs(
            stop_signs, depth_frame.frame, camera_setup)

    traffic_light_det_objs = []
    if traffic_lights:
        traffic_light_det_objs = get_traffic_light_objs(
            traffic_lights, camera_setup, depth_frame.frame, segmented_frame,
            tl_color, town_name)

    det_objs = (speed_limit_det_objs + traffic_stop_det_objs +
                traffic_light_det_objs)

    if FLAGS.visualize_bboxes:
        visualize_ground_bboxes('bboxes', game_time, frame, det_objs)

    # Log the frame.
    rgb_frame = bgr_to_rgb(frame)
    file_name = '{}signs-{}_{}_{}.png'.format(FLAGS.data_path, game_time,
                                              weather, town)
    rgb_img = Image.fromarray(np.uint8(rgb_frame))
    rgb_img.save(file_name)

    if FLAGS.log_bbox_images:
        annotate_image_with_bboxes(game_time, frame, det_objs)
        rgb_frame = bgr_to_rgb(frame)
        file_name = '{}annotated-signs-{}_{}_{}.png'.format(
            FLAGS.data_path, game_time, weather, town)
        rgb_img = Image.fromarray(np.uint8(rgb_frame))
        rgb_img.save(file_name)

    # Log the bounding boxes.
    bboxes = [det_obj.get_bbox_label() for det_obj in det_objs]
    file_name = '{}bboxes-{}_{}_{}.json'.format(FLAGS.data_path, game_time,
                                                weather, town)
    with open(file_name, 'w') as outfile:
        json.dump(bboxes, outfile)


def change_traffic_light_colors(world, color):
    actor_list = world.get_actors()
    tl_actors = actor_list.filter('traffic.traffic_light*')
    for tl in tl_actors:
        tl.set_state(color)
        tl.freeze(True)
        if color == carla.TrafficLightState.Green:
            tl.set_green_time(99999999999999999999999999999999999999)
            tl.set_yellow_time(0)
            tl.set_red_time(0)
        elif color == carla.TrafficLightState.Yellow:
            tl.set_green_time(0)
            tl.set_yellow_time(99999999999999999999999999999999999999)
            tl.set_red_time(0)
        else:
            tl.set_green_time(0)
            tl.set_yellow_time(0)
            tl.set_red_time(99999999999999999999999999999999999999)
    world.tick()
    time.sleep(1)

    # Make sure that everything is frozen and in the correct state.
    tl_actors = actor_list.filter('traffic.traffic_light*')
    for tl in tl_actors:
        assert tl.is_frozen() and tl.state == color, \
                "Traffic Light {}: Frozen: {}, Color: {}, Expected: {}".format(
                            tl.id, tl.is_frozen(), tl.state, color)


def test_traffic_light_colors(world, color):
    actor_list = world.get_actors()
    tl_actors = actor_list.filter('traffic.traffic_light*')
    for tl in tl_actors:
        if tl.state != color:
            raise ValueError("The expected color was {}, got {}".format(
                color, tl.state))


def get_actors(world):
    actor_list = world.get_actors()
    tl_actors = actor_list.filter('traffic.traffic_light*')
    traffic_lights = convert_traffic_light_actors(tl_actors)
    traffic_stop_actors = actor_list.filter('traffic.stop')
    traffic_stops = convert_traffic_stop_actors(traffic_stop_actors)
    speed_limit_actors = actor_list.filter('traffic.speed_limit*')
    speed_signs = convert_speed_limit_actors(speed_limit_actors)
    return (tl_actors, traffic_lights, traffic_stops, speed_signs)


def log_obstacles(world, transforms_of_interest, traffic_lights, tl_color,
                  speed_signs, traffic_stops, weather, town):
    for transform in transforms_of_interest:
        camera = add_camera(world, transform, on_camera_msg)
        depth_camera = add_depth_camera(world, transform, on_depth_msg)
        segmented_camera = add_segmented_camera(world, transform,
                                                on_segmented_msg)
        wait_for_data(world)
        global CARLA_IMAGE
        global DEPTH_FRAME
        global SEGMENTED_FRAME

        # Ensure that the traffic lights are of the given color.
        test_traffic_light_colors(world, tl_color)

        log_bounding_boxes(CARLA_IMAGE, DEPTH_FRAME, SEGMENTED_FRAME,
                           traffic_lights, tl_color, speed_signs,
                           traffic_stops, weather, town)
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
    (traffic_lights, _, _, _) = get_actors(world)
    tl_colors = [
        carla.TrafficLightState.Yellow, carla.TrafficLightState.Green,
        carla.TrafficLightState.Red
    ]
    transforms_of_interest = []
    for light in traffic_lights:
        print("Working for traffic light {}".format(light.id))
        # For every traffic light, get the neighbouring lights except the one
        # directly opposite.
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
            transform = Transform(
                carla_transform=light.get_transform()) * offset_trans
            location = transform.location.as_carla_location()

            # Get the waypoint nearest to the transform.
            w = world_map.get_waypoint(location,
                                       project_to_road=True,
                                       lane_type=carla.LaneType.Driving)
            w_rotation = w.transform.rotation
            camera_transform = Transform(carla_transform=w.transform)
            camera_transform.location.z += 2.0
            transform = camera_transform.as_carla_transform()
            transforms_of_interest.append(transform)

            # Get the right lanes.
            wp_right = w.get_right_lane()
            while wp_right and wp_right.lane_type == carla.LaneType.Driving \
                    and w_rotation == wp_right.transform.rotation:
                camera_transform = Transform(
                    carla_transform=wp_right.transform)
                camera_transform.location.z += 2.0
                transform = camera_transform.as_carla_transform()
                transforms_of_interest.append(transform)
                wp_right = wp_right.get_right_lane()

            # Get the left lanes.
            wp_left = w.get_left_lane()
            while wp_left and wp_left.lane_type == carla.LaneType.Driving and \
                    w_rotation == wp_left.transform.rotation:
                camera_transform = Transform(carla_transform=wp_left.transform)
                camera_transform.location.z += 2.0
                transform = camera_transform.as_carla_transform()
                transforms_of_interest.append(transform)
                wp_left = wp_left.get_left_lane()

    print("The total number of transforms were: {}".format(
        len(transforms_of_interest)))

    traffic_lights = [TrafficLight(light) for light in traffic_lights]
    for weather in find_weather_presets():
        change_weather(world, weather)
        time.sleep(1)
        for tl_color in tl_colors:
            change_traffic_light_colors(world, tl_color)
            world.tick()
            time.sleep(1)
            log_obstacles(world, transforms_of_interest, traffic_lights,
                          tl_color, None, None, weather, world_map.name)


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
            offset_rot = pylot.simulation.utils.Rotation(pitch=0,
                                                         yaw=0,
                                                         roll=0)
            offset_trans = pylot.simulation.utils.Transform(
                offset_loc, offset_rot)
            transform = speed_sign.transform * offset_trans
            location = transform.location.as_carla_location()
            w = world_map.get_waypoint(location,
                                       project_to_road=True,
                                       lane_type=carla.LaneType.Driving)
            camera_transform = Transform(carla_transform=w.transform)
            camera_transform.location.z += 2.0
            transform = camera_transform.as_carla_transform()
            transforms_of_interest.append(transform)
    # Ensure all traffic lights are red.
    change_traffic_light_colors(world, carla.TrafficLightState.Red)
    world.tick()
    time.sleep(1)
    (_, traffic_lights, traffic_stops, speed_signs) = get_actors(world)
    for weather in find_weather_presets():
        change_weather(world, weather)
        log_obstacles(world, transforms_of_interest, traffic_lights,
                      carla.TrafficLightState.Red, speed_signs, traffic_stops,
                      weather, world_map.name)


def log_stop_signs(world):
    world_map = world.get_map()
    (_, traffic_lights, traffic_stops, speed_signs) = get_actors(world)
    transforms_of_interest = []
    # Add transforms that are close to stop signs.
    for stop_sign in traffic_stops:
        for offset in range(10, 25, 5):
            offset_loc = pylot.simulation.utils.Location(x=-offset, y=0, z=0)
            offset_rot = pylot.simulation.utils.Rotation(pitch=0,
                                                         yaw=0,
                                                         roll=0)
            offset_trans = pylot.simulation.utils.Transform(
                offset_loc, offset_rot)
            transform = stop_sign.transform * offset_trans
            location = transform.location.as_carla_location()
            w = world_map.get_waypoint(location,
                                       project_to_road=True,
                                       lane_type=carla.LaneType.Driving)
            camera_transform = Transform(carla_transform=w.transform)
            camera_transform.location.z += 2.0
            transform = camera_transform.as_carla_transform()
            transforms_of_interest.append(transform)
    # Ensure all traffic lights are red.
    change_traffic_light_colors(world, carla.TrafficLightState.Red)
    world.tick()
    time.sleep(1)
    (_, traffic_lights, traffic_stops, speed_signs) = get_actors(world)
    for weather in find_weather_presets():
        change_weather(world, weather)
        log_obstacles(world, transforms_of_interest, traffic_lights,
                      carla.TrafficLightState.Red, speed_signs, traffic_stops,
                      weather, world_map.name)


def change_weather(world, weather):
    world.set_weather(getattr(carla.WeatherParameters, weather))


def find_weather_presets():
    presets = [
        x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)
    ]
    return presets


def main(argv):
    world = setup_world()
    world.tick()
    # Sleep a bit to ensure the simulator actually ticks.
    time.sleep(1)
    log_traffic_lights(world)
    # log_speed_limits(world)
    # log_stop_signs(world)


if __name__ == '__main__':
    app.run(main)
