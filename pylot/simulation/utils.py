import carla
from collections import namedtuple
import re

import pylot.utils
from pylot.perception.depth_frame import DepthFrame
from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.detection.speed_limit_sign import SpeedLimitSign
from pylot.perception.detection.stop_sign import StopSign
from pylot.perception.detection.traffic_light import TrafficLight
from pylot.perception.detection.utils import BoundingBox2D, DetectedObstacle, \
    DetectedSpeedLimit

# Type used to send location info from Carla.
LocationGeo = namedtuple('LocationGeo', 'latitude, longitude, altitude')


def get_world(host="localhost", port=2000, timeout=10):
    """ Get a handle to the world running inside the simulation.

    Args:
        host: The host where the simulator is running.
        port: The port to connect to at the given host.
        timeout: The timeout of the connection.

    Returns:
        A tuple of `(client, world)` where the `client` is a connection to the
        simulator and `world` is a handle to the world running inside the
        simulation at the host:port.
    """
    client, world = None, None
    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        world = client.get_world()
    except RuntimeError as r:
        client, world = None, None
        print("Received an error while connecting to the "
              "simulator: {}".format(r))
    return (client, world)


def get_map(host="localhost", port=2000, timeout=10):
    """ Get a handle to the Carla map.

    Args:
        host: The host where the simulator is running.
        port: The port to connect to at the given host.
        timeout: The timeout of the connection.

    Returns:
        A map of the Carla city.
    """
    _, world = get_world(host, port, timeout)
    if world is None:
        raise ValueError("There was an issue connecting to the simulator.")
    return world.get_map()


def get_weathers():
    """ Gets the list of weathers that the simulation supports.

    Returns:
        A dictionary of {name: weather} that we can set the simulation to.
    """
    names = [
        name for name in dir(carla.WeatherParameters)
        if re.match('[A-Z].+', name)
    ]
    weathers = {x: getattr(carla.WeatherParameters, x) for x in names}
    return weathers


def set_synchronous_mode(world, fps):
    """ Sets Carla to run in synchronous mode.

    Args:
        world: A handle to the world running inside the simulation.
        fps: The frames per second rate the simulation should tick at.
    """
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / fps
    world.apply_settings(settings)


def reset_world(world):
    """ Resets the simulation to the original state. Removes all the vehicles,
    sensors and other actors from the environment.

    Args:
        world: A handle to the world running inside the simulation.
    """
    actors = world.get_actors()
    for actor in actors:
        if actor.type_id == 'spectator' or actor.type_id.startswith('traffic'):
            pass
        else:
            actor.destroy()


def extract_data_in_pylot_format(actor_list):
    """ Extracts actor information in ERDOS format from an actor list.

    Args:
        actor_list: A Carla actor list object with all the simulation actors.

    Returns:
        A tuple that contains objects for all different types of actors.
    """
    # Note: the output will include the ego vehicle as well.
    vec_actors = actor_list.filter('vehicle.*')
    vehicles = [
        Obstacle.from_carla_actor(vec_actor) for vec_actor in vec_actors
    ]

    person_actors = actor_list.filter('walker.pedestrian.*')
    people = [
        Obstacle.from_carla_actor(ped_actor) for ped_actor in person_actors
    ]

    tl_actors = actor_list.filter('traffic.traffic_light*')
    traffic_lights = [
        TrafficLight.from_carla_actor(tl_actor) for tl_actor in tl_actors
    ]

    speed_limit_actors = actor_list.filter('traffic.speed_limit*')
    speed_limits = [
        SpeedLimitSign.from_carla_actor(ts_actor)
        for ts_actor in speed_limit_actors
    ]

    traffic_stop_actors = actor_list.filter('traffic.stop')
    traffic_stops = [
        StopSign.from_carla_actor(ts_actor) for ts_actor in traffic_stop_actors
    ]

    return (vehicles, people, traffic_lights, speed_limits, traffic_stops)


def draw_trigger_volume(world, actor):
    """ Draws the trigger volume of an actor."""
    transform = actor.get_transform()
    tv = transform.transform(actor.trigger_volume.location)
    bbox = carla.BoundingBox(tv, actor.trigger_volume.extent)
    world.debug.draw_box(bbox, transform.rotation, life_time=1000)


def get_top_down_transform(transform, top_down_lateral_view):
    # Height calculation relies on the fact that the camera's FOV is 90.
    top_down_location = (transform.location +
                         pylot.utils.Location(0, 0, top_down_lateral_view))
    return pylot.utils.Transform(top_down_location,
                                 pylot.utils.Rotation(-90, 0, 0))


def get_traffic_lights_obstacles(traffic_lights, depth_frame, segmented_frame,
                                 town_name):
    """ Get the traffic lights that are within the camera frame.

    Args:
        traffic_lights: List of perception.detection.traffic_light.TrafficLight.
        depth_frame: A perception.depth_frame.DepthFrame.
        segmented_frame: A perception.segmentation.segmented_frame.SegmentedFrame.
    """
    camera_transform = depth_frame.camera_setup.get_transform()
    # Iterate over all the traffic lights, and figure out which ones are
    # facing us and are visible in the camera view.
    detected = []
    for light in traffic_lights:
        if not light.is_traffic_light_visible(camera_transform, town_name):
            continue
        detected.extend(
            light.get_all_detected_traffic_light_boxes(
                town_name, depth_frame, segmented_frame.as_numpy_array()))
    return detected


def get_detected_speed_limits(speed_signs, depth_frame, segmented_frame):
    """ Get the speed limit signs that are withing the camera frame.

    Args:
        speed_signs: List of speed limit signs in the world.
        depth_frame: A perception.depth_frame.DepthFrame, with a
            camera_setup relative to the world.
        segmented_frame: A perception.segmentation.segmented_frame.SegmentedFrame.

    Returns:
        A list of pylot.perception.detection.DetectedSpeedLimit.
    """
    def match_bboxes_with_speed_signs(camera_transform, loc_bboxes,
                                      speed_signs):
        result = []
        for location, bbox in loc_bboxes:
            best_ts = None
            best_dist = 1000000
            for ts in speed_signs:
                dist = location.distance(ts.transform.location)
                if dist < best_dist:
                    best_dist = dist
                    best_ts = ts
            if not best_ts:
                continue
            # Check that the sign is facing the ego vehicle.
            yaw_diff = (best_ts.transform.rotation.yaw -
                        camera_transform.rotation.yaw)
            if yaw_diff < 0:
                yaw_diff += 360
            elif yaw_diff >= 360:
                yaw_diff -= 360
            if best_dist < 5**2 and yaw_diff > 30 and yaw_diff < 150:
                result.append(
                    DetectedSpeedLimit(bbox, best_ts.limit, 1.0,
                                       'speed limit'))
        return result

    if not isinstance(depth_frame, DepthFrame):
        raise ValueError(
            'depth_frame should be of type perception.depth_frame.DepthFrame')
    # Compute the 2D bounding boxes.
    bboxes_2d = segmented_frame.get_traffic_sign_bounding_boxes(min_width=8,
                                                                min_height=9)
    # Transform the centers of 2D bounding boxes to 3D locations.
    coordinates = [bbox.get_center_point() for bbox in bboxes_2d]
    locations = depth_frame.get_pixel_locations(coordinates)
    loc_and_bboxes = zip(locations, bboxes_2d)
    det_speed_limits = match_bboxes_with_speed_signs(
        depth_frame.camera_setup.transform, loc_and_bboxes, speed_signs)
    return det_speed_limits


def get_detected_traffic_stops(traffic_stops, depth_frame):
    """ Get traffic stop lane markings that are withing the camera frame.

    Args:
        traffic_stops: List of traffic stop actors in the world.
        depth_frame: A pylot.perception.depth_frame.DepthFrame, with a
            camera_setup relative to the world.

    Returns:
        List of DetectedObstacles.
    """
    def get_stop_markings_bbox(bbox3d, depth_frame):
        """ Gets a 2D stop marking bounding box from a 3D bounding box."""
        # Move trigger_volume by -0.85 so that the top plane is on the ground.
        ext_z_value = bbox3d.extent.z - 0.85
        ext = [
            pylot.utils.Location(x=+bbox3d.extent.x,
                                 y=+bbox3d.extent.y,
                                 z=ext_z_value),
            pylot.utils.Location(x=+bbox3d.extent.x,
                                 y=-bbox3d.extent.y,
                                 z=ext_z_value),
            pylot.utils.Location(x=-bbox3d.extent.x,
                                 y=+bbox3d.extent.y,
                                 z=ext_z_value),
            pylot.utils.Location(x=-bbox3d.extent.x,
                                 y=-bbox3d.extent.y,
                                 z=ext_z_value),
        ]
        bbox = bbox3d.transform.transform_points(ext)
        camera_transform = depth_frame.camera_setup.get_transform()
        coords = []
        for loc in bbox:
            loc_view = loc.to_camera_view(
                camera_transform.matrix,
                depth_frame.camera_setup.get_intrinsic_matrix())
            if (loc_view.z >= 0 and loc_view.x >= 0 and loc_view.y >= 0
                    and loc_view.x < depth_frame.camera_setup.width
                    and loc_view.y < depth_frame.camera_setup.height):
                coords.append(loc_view)
        if len(coords) == 4:
            xmin = min(coords[0].x, coords[1].x, coords[2].x, coords[3].x)
            xmax = max(coords[0].x, coords[1].x, coords[2].x, coords[3].x)
            ymin = min(coords[0].y, coords[1].y, coords[2].y, coords[3].y)
            ymax = max(coords[0].y, coords[1].y, coords[2].y, coords[3].y)
            # Check if the bbox is not obstructed and if it's sufficiently
            # big for the text to be readable.
            if (ymax - ymin > 15 and depth_frame.pixel_has_same_depth(
                    int(coords[0].x), int(coords[0].y), coords[0].z, 0.4)):
                return BoundingBox2D(int(xmin), int(xmax), int(ymin),
                                     int(ymax))
        return None

    if not isinstance(depth_frame, DepthFrame):
        raise ValueError(
            'depth_frame should be of type perception.depth_frame.DepthFrame')
    det_obstacles = []
    for transform, bbox in traffic_stops:
        bbox_2d = get_stop_markings_bbox(bbox, depth_frame)
        if bbox_2d is not None:
            det_obstacles.append(DetectedObstacle(bbox_2d, 1.0,
                                                  'stop marking'))
    return det_obstacles
