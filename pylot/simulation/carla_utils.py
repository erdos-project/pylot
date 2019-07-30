import numpy as np
import carla

from pylot.perception.detection.utils import TrafficLightColor
from pylot.simulation.utils import to_pylot_transform, to_pylot_location
import pylot.simulation.utils
from pylot.simulation.utils import Location


def get_world(host="localhost", port=2000, timeout=10):
    """ Get a handle to the world running inside the simulation.

    Args:
        host: The host where the simulator is running.
        port: The port to connect to at the given host.

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
    _, world = get_world(host, port, timeout)
    if world is None:
        raise ValueError("There was an issue connecting to the simulator.")
    return world.get_map()


def get_weathers():
    """ Gets the list of weathers that the simulation supports.

    Returns:
        A list of all (weather, name) tuples that we can set the simulation to.
    """
    weathers = [
        (carla.WeatherParameters.ClearNoon, "ClearNoon"),
        (carla.WeatherParameters.CloudyNoon, "CloudyNoon"),
        (carla.WeatherParameters.WetNoon, "WetNoon"),
        (carla.WeatherParameters.WetCloudyNoon, "WetCloudyNoon"),
        (carla.WeatherParameters.MidRainyNoon, "MidRainyNoon"),
        (carla.WeatherParameters.HardRainNoon, "HardRainNoon"),
        (carla.WeatherParameters.SoftRainNoon, "SoftRainNoon"),
        (carla.WeatherParameters.ClearSunset, "ClearSunset"),
        (carla.WeatherParameters.CloudySunset, "CloudySunset"),
        (carla.WeatherParameters.WetSunset, "WetSunset"),
        (carla.WeatherParameters.WetCloudySunset, "WetCloudySunet"),
        (carla.WeatherParameters.MidRainSunset, "MidRainSunset"),
        (carla.WeatherParameters.HardRainSunset, "HardRainSunset"),
        (carla.WeatherParameters.SoftRainSunset, "SoftRainSunset"),
    ]
    return weathers


def set_synchronous_mode(world, fps):
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


def to_carla_location(location):
    return carla.Location(
        location.x, location.y, location.z)


def to_carla_transform(transform):
    """ Converts an Pylot transform object to a Carla transform object."""
    return carla.Transform(
        carla.Location(transform.location.x,
                       transform.location.y,
                       transform.location.z),
        carla.Rotation(pitch=transform.rotation.pitch,
                       yaw=transform.rotation.yaw,
                       roll=transform.rotation.roll))


def render_bounding_boxes_in_world(
        world, actor_list, time_between_frames_ms=100):
    """ Render the 3D ground bboxes on the images.

    Args:
        world: Connection to the simulator world.
        actor_list: List of actors in the world.
        time_between_frames_ms: The life time of the rendered bboxes.
    """
    vehicles = actor_list.filter('vehicle.*')
    for vehicle in vehicles:
        transform = vehicle.get_transform()
        bounding_box = vehicle.bounding_box
        bounding_box.location += transform.location
        world.debug.draw_box(bounding_box,
                             transform.rotation,
                             life_time=time_between_frames_ms / 1000.0,
                             persistent_lines=True)


def extract_data_in_pylot_format(actor_list):
    """ Extracts actor information in ERDOS format from an actor list."""
    vec_actors = actor_list.filter('vehicle.*')
    vehicles = convert_vehicle_actors(vec_actors)

    pedestrian_actors = actor_list.filter('walker.pedestrian.*')
    pedestrians = convert_pedestrian_actors(pedestrian_actors)

    tl_actors = actor_list.filter('traffic.traffic_light*')
    traffic_lights = convert_traffic_light_actors(tl_actors)

    speed_limit_actors = actor_list.filter('traffic.speed_limit*')
    speed_limits = convert_speed_limit_actors(speed_limit_actors)

    traffic_stop_actors = actor_list.filter('traffic.stop')
    traffic_stops = convert_traffic_stop_actors(traffic_stop_actors)

    return (vehicles, pedestrians, traffic_lights, speed_limits, traffic_stops)


def convert_vehicle_actors(vec_actors):
    """ Converts a Carla vehicle actor into a Pylot Vehicle object."""
    vehicles = []
    # TODO(ionel): Handle hero vehicle!
    for vec_actor in vec_actors:
        transform = to_pylot_transform(vec_actor.get_transform())
        bounding_box = pylot.simulation.utils.BoundingBox(
            vec_actor.bounding_box)
        speed = pylot.simulation.utils.get_speed(vec_actor.get_velocity())
        vehicle = pylot.simulation.utils.Vehicle(
            transform, bounding_box, speed)
        vehicles.append(vehicle)
    return vehicles


def convert_pedestrian_actors(pedestrian_actors):
    """ Converts a Carla pedestrian actor into a Pylot pedestrian object."""
    pedestrians = []
    for ped_actor in pedestrian_actors:
        transform = to_pylot_transform(ped_actor.get_transform())
        bounding_box = pylot.simulation.utils.BoundingBox(
            ped_actor.bounding_box)
        speed = pylot.simulation.utils.get_speed(ped_actor.get_velocity())
        pedestrian = pylot.simulation.utils.Pedestrian(
            ped_actor.id, transform, bounding_box, speed)
        pedestrians.append(pedestrian)
    return pedestrians


def convert_to_pylot_traffic_light(tl_actor):
    transform = to_pylot_transform(tl_actor.get_transform())
    tl_state = tl_actor.get_state()
    erdos_tl_state = None
    if tl_state == carla.TrafficLightState.Red:
        erdos_tl_state = TrafficLightColor.RED
    elif tl_state == carla.TrafficLightState.Yellow:
        erdos_tl_state = TrafficLightColor.YELLOW
    elif tl_state == carla.TrafficLightState.Green:
        erdos_tl_state = TrafficLightColor.GREEN
    else:
        erdos_tl_state = TrafficLightColor.OFF
    extent = pylot.simulation.utils.Extent(
        tl_actor.trigger_volume.extent.x,
        tl_actor.trigger_volume.extent.y,
        tl_actor.trigger_volume.extent.z)
    traffic_light = pylot.simulation.utils.TrafficLight(
        tl_actor.id, transform, erdos_tl_state, extent)
    return traffic_light


def convert_traffic_light_actors(tl_actors):
    """ Converts a Carla traffic light actor into a Pylot tl object."""
    traffic_lights = []
    for tl_actor in tl_actors:
        traffic_lights.append(convert_to_pylot_traffic_light(tl_actor))
    return traffic_lights


def convert_speed_limit_actors(speed_limit_actors):
    """ Converts a Carla speed limit actor into a Pylot speed limit object."""
    speed_limits = []
    for ts_actor in speed_limit_actors:
        transform = to_pylot_transform(ts_actor.get_transform())
        speed_limit = int(ts_actor.type_id.split('.')[-1])
        speed_sign = pylot.simulation.utils.SpeedLimitSign(
            transform, speed_limit)
        speed_limits.append(speed_sign)
    return speed_limits


def convert_traffic_stop_actors(traffic_stop_actors):
    """ Converts a Carla traffic stop actor into a Pylot stop sign object."""
    stop_signs = []
    for ts_actor in traffic_stop_actors:
        transform = to_pylot_transform(ts_actor.get_transform())
        world_trigger_volume = ts_actor.get_transform().transform(
            ts_actor.trigger_volume.location)
        bbox = pylot.simulation.utils.BoundingBox(
            carla.BoundingBox(world_trigger_volume,
                              ts_actor.trigger_volume.extent))
        stop_sign = pylot.simulation.utils.StopSign(transform, bbox)
        stop_signs.append(stop_sign)
    return stop_signs


def draw_trigger_volume(world, actor):
    transform = actor.get_transform()
    tv = transform.transform(actor.trigger_volume.location)
    bbox = carla.BoundingBox(tv, actor.trigger_volume.extent)
    world.debug.draw_box(bbox, transform.rotation, life_time=1000)
