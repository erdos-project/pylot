import json
import re
import time

import PIL.Image as Image

from absl import app
from absl import flags

from carla import LaneType, TrafficLightState, WeatherParameters

import numpy as np

import pylot.simulation.utils
import pylot.utils
from pylot.drivers.sensor_setup import DepthCameraSetup, RGBCameraSetup, \
    SegmentedCameraSetup
from pylot.perception.camera_frame import CameraFrame
from pylot.perception.depth_frame import DepthFrame
from pylot.perception.detection.speed_limit_sign import SpeedLimitSign
from pylot.perception.detection.stop_sign import StopSign
from pylot.perception.detection.traffic_light import TrafficLight
from pylot.perception.segmentation.segmented_frame import SegmentedFrame
from pylot.simulation.utils import get_world


FLAGS = flags.FLAGS
SIMULATOR_IMAGE = None
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
    global SIMULATOR_IMAGE
    SIMULATOR_IMAGE = image


def on_depth_msg(simulator_image):
    global DEPTH_FRAME
    transform = pylot.utils.Transform.from_simulator_transform(
        simulator_image.transform)
    camera_setup = DepthCameraSetup("depth_camera", FLAGS.frame_width,
                                    FLAGS.frame_height, transform,
                                    FLAGS.camera_fov)

    DEPTH_FRAME = DepthFrame.from_simulator_frame(simulator_image,
                                                  camera_setup)


def on_segmented_msg(simulator_image):
    global SEGMENTED_FRAME
    transform = pylot.utils.Transform.from_simulator_transform(
        simulator_image.transform)
    camera_setup = SegmentedCameraSetup("segmented_camera", FLAGS.frame_width,
                                        FLAGS.frame_height, transform,
                                        FLAGS.camera_fov)
    SEGMENTED_FRAME = SegmentedFrame.from_simulator_image(simulator_image,
                                                          camera_setup)


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
    client, world = get_world(timeout=20)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.02
    world.apply_settings(settings)
    return world


def wait_for_data(world):
    world.tick()
    global SIMULATOR_IMAGE
    global DEPTH_FRAME
    global SEGMENTED_FRAME
    while (SIMULATOR_IMAGE is None or DEPTH_FRAME is None
           or SEGMENTED_FRAME is None):
        time.sleep(0.1)


def reset_frames():
    global DEPTH_FRAME
    global SEGMENTED_FRAME
    global SIMULATOR_IMAGE
    # Reset frames.
    DEPTH_FRAME = None
    SEGMENTED_FRAME = None
    SIMULATOR_IMAGE = None


def get_traffic_light_obstacles(traffic_lights, depth_frame, segmented_frame,
                                color, town_name):
    det_obstacles = pylot.simulation.utils.get_traffic_lights_obstacles(
        traffic_lights, depth_frame, segmented_frame, town_name)
    # Overwrite traffic light color because we control it without refreshing
    # the agents.
    if color == TrafficLightState.Yellow:
        label = 'yellow'
    elif color == TrafficLightState.Green:
        label = 'green'
    elif color == TrafficLightState.Red:
        label = 'red'
    elif color == TrafficLightState.Off:
        label = 'off'
    else:
        raise ValueError('Unknown traffic light color')
    label += ' traffic light'

    for obstacle in det_obstacles:
        obstacle.label = label
    return det_obstacles


def log_bounding_boxes(simulator_image, depth_frame, segmented_frame,
                       traffic_lights, tl_color, speed_signs, stop_signs,
                       weather, town):
    game_time = int(simulator_image.timestamp * 1000)
    print("Processing game time {} in {} with weather {}".format(
        game_time, town, weather))
    transform = pylot.utils.Transform.from_simulator_transform(
        simulator_image.transform)
    camera_setup = RGBCameraSetup("rgb_camera", FLAGS.frame_width,
                                  FLAGS.frame_height, transform,
                                  FLAGS.camera_fov)
    frame = CameraFrame.from_simulator_frame(simulator_image, camera_setup)

    speed_limit_det_obstacles = []
    if speed_signs:
        speed_limit_det_obstacles = \
            pylot.simulation.utils.get_detected_speed_limits(
                speed_signs, depth_frame, segmented_frame)

    traffic_stop_det_obstacles = []
    if stop_signs:
        traffic_stop_det_obstacles = \
            pylot.simulation.utils.get_detected_traffic_stops(
                stop_signs, depth_frame)

    traffic_light_det_obstacles = []
    if traffic_lights:
        traffic_light_det_obstacles = get_traffic_light_obstacles(
            traffic_lights, depth_frame, segmented_frame, tl_color, town)

    det_obstacles = (speed_limit_det_obstacles + traffic_stop_det_obstacles +
                     traffic_light_det_obstacles)
    # Log the frame.
    file_name = '{}signs-{}_{}_{}.png'.format(FLAGS.data_path, game_time,
                                              weather, town)
    rgb_img = Image.fromarray(frame.as_rgb_numpy_array())
    rgb_img.save(file_name)

    if FLAGS.log_bbox_images:
        frame.annotate_with_bounding_boxes(game_time, det_obstacles)
        file_name = '{}annotated-signs-{}_{}_{}.png'.format(
            FLAGS.data_path, game_time, weather, town)
        rgb_img = Image.fromarray(frame.as_rgb_numpy_array())
        rgb_img.save(file_name)

    # Log the bounding boxes.
    bboxes = [obstacle.get_in_log_format() for obstacle in det_obstacles]
    file_name = '{}bboxes-{}_{}_{}.json'.format(FLAGS.data_path, game_time,
                                                weather, town)
    with open(file_name, 'w') as outfile:
        json.dump(bboxes, outfile)

    if FLAGS.visualize_bboxes:
        frame.annotate_with_bounding_boxes(game_time, det_obstacles)
        frame.visualize('bboxes')


def change_traffic_light_colors(world, color):
    actor_list = world.get_actors()
    tl_actors = actor_list.filter('traffic.traffic_light*')
    for tl in tl_actors:
        tl.set_state(color)
        tl.freeze(True)
        if color == TrafficLightState.Green:
            tl.set_green_time(99999999999999999999999999999999999999)
            tl.set_yellow_time(0)
            tl.set_red_time(0)
        elif color == TrafficLightState.Yellow:
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
    traffic_lights = [
        TrafficLight.from_simulator_actor(tl_actor) for tl_actor in tl_actors
    ]
    traffic_stop_actors = actor_list.filter('traffic.stop')
    traffic_stops = [
        StopSign.from_simulator_actor(ts_actor)
        for ts_actor in traffic_stop_actors
    ]
    speed_limit_actors = actor_list.filter('traffic.speed_limit*')
    speed_signs = [
        SpeedLimitSign.from_simulator_actor(ts_actor)
        for ts_actor in speed_limit_actors
    ]
    return (tl_actors, traffic_lights, traffic_stops, speed_signs)


def log_obstacles(world, transforms_of_interest, traffic_lights, tl_color,
                  speed_signs, traffic_stops, weather, town):
    for transform in transforms_of_interest:
        camera = add_camera(world, transform, on_camera_msg)
        depth_camera = add_depth_camera(world, transform, on_depth_msg)
        segmented_camera = add_segmented_camera(world, transform,
                                                on_segmented_msg)
        wait_for_data(world)
        global SIMULATOR_IMAGE
        global DEPTH_FRAME
        global SEGMENTED_FRAME

        # Ensure that the traffic lights are of the given color.
        test_traffic_light_colors(world, tl_color)

        log_bounding_boxes(SIMULATOR_IMAGE, DEPTH_FRAME, SEGMENTED_FRAME,
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
        TrafficLightState.Yellow, TrafficLightState.Green,
        TrafficLightState.Red
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
            offset_loc = pylot.utils.Location(
                x=light.trigger_volume.location.x,
                y=light.trigger_volume.location.y + offset,
                z=light.trigger_volume.location.z)
            offset_trans = pylot.utils.Transform(offset_loc,
                                                 pylot.utils.Rotation())

            # Transform the offset relative to the traffic light.
            transform = pylot.utils.Transform.from_simulator_transform(
                light.get_transform()) * offset_trans
            location = transform.location.as_simulator_location()

            # Get the waypoint nearest to the transform.
            w = world_map.get_waypoint(location,
                                       project_to_road=True,
                                       lane_type=LaneType.Driving)
            w_rotation = w.transform.rotation
            camera_transform = pylot.utils.Transform.from_simulator_transform(
                w.transform)
            camera_transform.location.z += 2.0
            transform = camera_transform.as_simulator_transform()
            transforms_of_interest.append(transform)

            # Get the right lanes.
            wp_right = w.get_right_lane()
            while wp_right and wp_right.lane_type == LaneType.Driving \
                    and w_rotation == wp_right.transform.rotation:
                camera_transform = \
                    pylot.utils.Transform.from_simulator_transform(
                        wp_right.transform)
                camera_transform.location.z += 2.0
                transform = camera_transform.as_simulator_transform()
                transforms_of_interest.append(transform)
                wp_right = wp_right.get_right_lane()

            # Get the left lanes.
            wp_left = w.get_left_lane()
            while wp_left and wp_left.lane_type == LaneType.Driving and \
                    w_rotation == wp_left.transform.rotation:
                camera_transform = \
                    pylot.utils.Transform.from_simulator_transform(
                        wp_left.transform)
                camera_transform.location.z += 2.0
                transform = camera_transform.as_simulator_transform()
                transforms_of_interest.append(transform)
                wp_left = wp_left.get_left_lane()

    print("The total number of transforms were: {}".format(
        len(transforms_of_interest)))

    traffic_lights = [
        TrafficLight.from_simulator_actor(light) for light in traffic_lights
    ]
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
            offset_loc = pylot.utils.Location(x=0, y=offset, z=0)
            offset_trans = pylot.utils.Transform(offset_loc,
                                                 pylot.utils.Rotation())
            transform = speed_sign.transform * offset_trans
            location = transform.location.as_simulator_location()
            w = world_map.get_waypoint(location,
                                       project_to_road=True,
                                       lane_type=LaneType.Driving)
            camera_transform = pylot.utils.Transform.from_simulator_transform(
                w.transform)
            camera_transform.location.z += 2.0
            transform = camera_transform.as_simulator_transform()
            transforms_of_interest.append(transform)
    # Ensure all traffic lights are red.
    change_traffic_light_colors(world, TrafficLightState.Red)
    world.tick()
    time.sleep(1)
    (_, traffic_lights, traffic_stops, speed_signs) = get_actors(world)
    for weather in find_weather_presets():
        change_weather(world, weather)
        log_obstacles(world, transforms_of_interest, traffic_lights,
                      TrafficLightState.Red, speed_signs, traffic_stops,
                      weather, world_map.name)


def log_stop_signs(world):
    world_map = world.get_map()
    (_, traffic_lights, traffic_stops, speed_signs) = get_actors(world)
    transforms_of_interest = []
    # Add transforms that are close to stop signs.
    for stop_sign in traffic_stops:
        for offset in range(10, 25, 5):
            offset_loc = pylot.utils.Location(x=-offset, y=0, z=0)
            offset_trans = pylot.utils.Transform(offset_loc,
                                                 pylot.utils.Rotation())
            transform = stop_sign.transform * offset_trans
            location = transform.location.as_simulator_location()
            w = world_map.get_waypoint(location,
                                       project_to_road=True,
                                       lane_type=LaneType.Driving)
            camera_transform = pylot.utils.Transform.from_simulator_transform(
                w.transform)
            camera_transform.location.z += 2.0
            transform = camera_transform.as_simulator_transform()
            transforms_of_interest.append(transform)
    # Ensure all traffic lights are red.
    change_traffic_light_colors(world, TrafficLightState.Red)
    world.tick()
    time.sleep(1)
    (_, traffic_lights, traffic_stops, speed_signs) = get_actors(world)
    for weather in find_weather_presets():
        change_weather(world, weather)
        log_obstacles(world, transforms_of_interest, traffic_lights,
                      TrafficLightState.Red, speed_signs, traffic_stops,
                      weather, world_map.name)


def change_weather(world, weather):
    world.set_weather(getattr(WeatherParameters, weather))


def find_weather_presets():
    presets = [
        x for x in dir(WeatherParameters) if re.match('[A-Z].+', x)
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
