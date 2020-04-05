import carla
from collections import namedtuple
from enum import Enum
import random
import re
import time

import pylot.utils
from pylot.perception.depth_frame import DepthFrame
from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.detection.speed_limit_sign import SpeedLimitSign
from pylot.perception.detection.stop_sign import StopSign
from pylot.perception.detection.traffic_light import TrafficLight
from pylot.perception.detection.utils import BoundingBox2D

# Type used to send location info from Carla.
LocationGeo = namedtuple('LocationGeo', 'latitude, longitude, altitude')


def get_world(host="localhost", port=2000, timeout=10):
    """Get a handle to the world running inside the simulation.

    Args:
        host (:obj:`str`): The host where the simulator is running.
        port (:obj:`int`): The port to connect to at the given host.
        timeout (:obj:`int`): The timeout of the connection (in seconds).

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
    """Get a handle to the Carla map.

    Args:
        host (:obj:`str`): The host where the simulator is running.
        port (:obj:`int`): The port to connect to at the given host.
        timeout (:obj:`int`): The timeout of the connection (in seconds).

    Returns:
        carla.Map: A map of the Carla city.
    """
    _, world = get_world(host, port, timeout)
    if world is None:
        raise ValueError("There was an issue connecting to the simulator.")
    return world.get_map()


def set_weather(world, weather):
    """Sets the simulation weather."""
    names = [
        name for name in dir(carla.WeatherParameters)
        if re.match('[A-Z].+', name)
    ]
    weathers = {x: getattr(carla.WeatherParameters, x) for x in names}
    world.set_weather(weathers[weather])
    return weathers


def set_simulation_mode(world, flags):
    # Turn on the synchronous mode so we can control the simulation.
    if (flags.carla_mode == 'synchronous'
            or flags.carla_mode == 'pseudo-asynchronous'):
        set_synchronous_mode(world, flags.carla_fps)
    elif flags.carla_mode == 'asynchronous-fixed-time-step':
        set_asynchronous_fixed_time_step_mode(world, flags.carla_fps)
    elif flags.carla_mode == 'asynchronous':
        set_asynchronous_mode(world)
    else:
        raise ValueError('Unexpected simulation mode {}'.format(
            flags.carla_mode))


def set_synchronous_mode(world, fps):
    """Sets Carla to run in synchronous mode.

    Args:
        world (carla.World): A handle to the world running inside the
            simulator.
        fps (:obj:`int`): Frames per second rate the simulation should tick at.
    """
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / fps
    world.apply_settings(settings)


def set_asynchronous_fixed_time_step_mode(world, fps):
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = 1.0 / fps
    world.apply_settings(settings)


def set_asynchronous_mode(world):
    """Sets the simulator to asynchronous mode.

    Args:
        world (carla.World): A handle to the world running inside the
            simulation.
    """
    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)


def reset_world(world):
    """Resets the simulation to the original state.

    Removes all the vehicles, sensors and other actors from the environment.

    Args:
        world (carla.World): A handle to the world running inside the
            simulation.
    """
    actors = world.get_actors()
    for actor in actors:
        if actor.type_id == 'spectator' or actor.type_id.startswith('traffic'):
            pass
        else:
            actor.destroy()


def spawn_actors(client, world, carla_version, ego_spawn_point_index,
                 auto_pilot, num_people, num_vehicles, logger):
    vehicle_ids = spawn_vehicles(client, world, num_vehicles, logger)
    ego_vehicle = spawn_ego_vehicle(world, ego_spawn_point_index, auto_pilot)
    people = []
    if (carla_version == '0.9.6' or carla_version == '0.9.7'
            or carla_version == '0.9.8'):
        # People do not move in versions older than 0.9.6.
        (people, people_control_ids) = spawn_people(client, world, num_people,
                                                    logger)
        people_actors = world.get_actors(people_control_ids)
        for i, ped_control_id in enumerate(people_control_ids):
            # Start person.
            people_actors[i].start()
            people_actors[i].go_to_location(
                world.get_random_location_from_navigation())
    return ego_vehicle, vehicle_ids, people


def spawn_ego_vehicle(world, spawn_point_index, auto_pilot):
    # Set our vehicle to be the one used in the CARLA challenge.
    v_blueprint = world.get_blueprint_library().filter(
        'vehicle.lincoln.mkz2017')[0]

    ego_vehicle = None

    while not ego_vehicle:
        if spawn_point_index == -1:
            # Pick a random spawn point.
            start_pose = random.choice(world.get_map().get_spawn_points())
        else:
            spawn_points = world.get_map().get_spawn_points()
            assert spawn_point_index < len(spawn_points), \
                'Spawn point index is too big. ' \
                'Town does not have sufficient spawn points.'
            start_pose = spawn_points[spawn_point_index]

        ego_vehicle = world.try_spawn_actor(v_blueprint, start_pose)
    if auto_pilot:
        ego_vehicle.set_autopilot(True)
    return ego_vehicle


def spawn_people(client, world, num_people, logger):
    """Spawns people at random locations inside the world.

    Args:
        num_people: The number of people to spawn.
    """
    p_blueprints = world.get_blueprint_library().filter('walker.pedestrian.*')
    unique_locs = set([])
    spawn_points = []
    # Get unique spawn points.
    for i in range(num_people):
        attempt = 0
        while attempt < 10:
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if loc is not None:
                # Transform to tuple so that location is comparable.
                p_loc = (loc.x, loc.y, loc.z)
                if p_loc not in unique_locs:
                    spawn_point.location = loc
                    spawn_points.append(spawn_point)
                    unique_locs.add(p_loc)
                    break
            attempt += 1
        if attempt == 10:
            logger.error('Could not find unique person spawn point')
    # Spawn the people.
    batch = []
    for spawn_point in spawn_points:
        p_blueprint = random.choice(p_blueprints)
        if p_blueprint.has_attribute('is_invincible'):
            p_blueprint.set_attribute('is_invincible', 'false')
        batch.append(carla.command.SpawnActor(p_blueprint, spawn_point))
    # Apply the batch and retrieve the identifiers.
    ped_ids = []
    for response in client.apply_batch_sync(batch, True):
        if response.error:
            logger.info('Received an error while spawning a person: {}'.format(
                response.error))
        else:
            ped_ids.append(response.actor_id)
    # Spawn the person controllers
    ped_controller_bp = world.get_blueprint_library().find(
        'controller.ai.walker')
    batch = []
    for ped_id in ped_ids:
        batch.append(
            carla.command.SpawnActor(ped_controller_bp, carla.Transform(),
                                     ped_id))
    ped_control_ids = []
    for response in client.apply_batch_sync(batch, True):
        if response.error:
            logger.info('Error while spawning a person controller: {}'.format(
                response.error))
        else:
            ped_control_ids.append(response.actor_id)

    return (ped_ids, ped_control_ids)


def spawn_vehicles(client, world, num_vehicles, logger):
    """ Spawns vehicles at random locations inside the world.

    Args:
        num_vehicles: The number of vehicles to spawn.
    """
    logger.debug('Trying to spawn {} vehicles.'.format(num_vehicles))
    # Get the spawn points and ensure that the number of vehicles
    # requested are less than the number of spawn points.
    spawn_points = world.get_map().get_spawn_points()
    if num_vehicles >= len(spawn_points):
        logger.warning(
            'Requested {} vehicles but only found {} spawn points'.format(
                num_vehicles, len(spawn_points)))
        num_vehicles = len(spawn_points)
    else:
        random.shuffle(spawn_points)

    # Get all the possible vehicle blueprints inside the world.
    v_blueprints = world.get_blueprint_library().filter('vehicle.*')

    # Construct a batch message that spawns the vehicles.
    batch = []
    for transform in spawn_points[:num_vehicles]:
        blueprint = random.choice(v_blueprints)

        # Change the color of the vehicle.
        if blueprint.has_attribute('color'):
            color = random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # Let the vehicle drive itself.
        blueprint.set_attribute('role_name', 'autopilot')

        batch.append(
            carla.command.SpawnActor(blueprint, transform).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True)))

    # Apply the batch and retrieve the identifiers.
    vehicle_ids = []
    for response in client.apply_batch_sync(batch, True):
        if response.error:
            logger.info(
                'Received an error while spawning a vehicle: {}'.format(
                    response.error))
        else:
            vehicle_ids.append(response.actor_id)
    return vehicle_ids


def set_vehicle_physics(vehicle, moi, mass):
    if moi and mass:
        physics_control = vehicle.get_physics_control()
        physics_control.moi = moi
        physics_control.mass = mass
        vehicle.apply_physics_control(physics_control)


def wait_for_ego_vehicle(world):
    """Loops until a hero vehicle is spawned.

    Note: The loop ticks the simulation.
    """
    # Connect to the ego-vehicle spawned by the scenario runner.
    while True:
        time.sleep(1)
        possible_actors = world.get_actors().filter('vehicle.*')
        for actor in possible_actors:
            if actor.attributes['role_name'] == 'hero':
                return actor
        world.tick()


def extract_data_in_pylot_format(actor_list):
    """Extracts actor information in pylot format from an actor list.

    Args:
        actor_list (carla.ActorList): An actor list object with all the
            simulation actors.

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
    """Draws the trigger volume of an actor.

    Args:
        world (carla.World): A handle to the world running inside the
            simulation.
        actor (carla.Actor): A CARLA actor.
    """
    transform = actor.get_transform()
    tv = transform.transform(actor.trigger_volume.location)
    bbox = carla.BoundingBox(tv, actor.trigger_volume.extent)
    world.debug.draw_box(bbox, transform.rotation, life_time=1000)


def get_traffic_lights_obstacles(traffic_lights, depth_frame, segmented_frame,
                                 town_name):
    """Get the traffic lights that are within the camera frame.

    Warning:
        The depth frame must have a camera setup transform relative to the
        world.

    Args:
        traffic_lights (list(:py:class:`~pylot.perception.detection.traffic_light.TrafficLight`)):
            List of traffic lights.
        depth_frame (:py:class:`~pylot.perception.depth_frame.DepthFrame`):
            Depth frame captured from the same position as the camera frame.
        segmented_frame (:py:class:`~pylot.perception.segmentation.segmented_frame.SegmentedFrame`):
            Segmented frame captured from the same position as the camera
            frame.

    Returns:
        list(:py:class:`~pylot.perception.detection.traffic_light.TrafficLight`):
        List of detected traffic light obstacles.
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
    """Get the speed limit signs that are withing the camera frame.

    Warning:
        The depth frame must have a camera setup transform relative to the
        world.

    Args:
        speed_signs (list(:py:class:`~pylot.perception.detection.speed_limit_sign.SpeedLimitSign`)):
            List of speed limit signs in the world.
        depth_frame (:py:class:`~pylot.perception.depth_frame.DepthFrame`):
            Depth frame captured from the same position as the camera frame.
        segmented_frame (:py:class:`~pylot.perception.segmentation.segmented_frame.SegmentedFrame`):
            Segmented frame captured from the same position as the camera
            frame.

    Returns:
        list(:py:class:`~pylot.perception.detection.speed_limit_sign.SpeedLimitSign`):
        List of detected speed limits with 2D bounding boxes set.
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
                best_ts.bounding_box = bbox
                result.append(best_ts)
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
    """Gets traffic stop lane markings that are withing the camera frame.

    Warning:
        The depth frame must have a camera setup transform relative to the
        world.

    Args:
        traffic_stops (list(:py:class:`~pylot.perception.detection.stop_sign.StopSign`)):
            List of traffic stops in the world.
        depth_frame (:py:class:`~pylot.perception.depth_frame.DepthFrame`):
            Depth frame captured from the same position as the camera frame.

    Returns:
        list(:py:class:`~pylot.perception.detection.stop_sign.StopSign`):
        List of detected traffic stops with 2D bounding boxes set.
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
        bbox = bbox3d.transform.transform_locations(ext)
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
    det_stop_signs = []
    for stop_sign in traffic_stops:
        bbox_2d = get_stop_markings_bbox(stop_sign.bounding_box_3d,
                                         depth_frame)
        if bbox_2d is not None:
            stop_sign.bounding_box = bbox_2d
            det_stop_signs.append(stop_sign)
    return det_stop_signs


def get_vehicle_handle(world, vehicle_id):
    num_tries = 0
    while num_tries < 30:
        vehicle = world.get_actors().find(vehicle_id)
        if vehicle:
            return vehicle
        time.sleep(0.2)
        num_tries += 1
    raise ValueError("There was an issue finding the vehicle.")


class TrafficInfractionType(Enum):
    """ An enumeration of the different infractions that Pylot supports. """
    RED_LIGHT_INVASION = 1
