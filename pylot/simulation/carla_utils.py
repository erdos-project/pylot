import re
import carla

import pylot.simulation.utils
import pylot.utils
from pylot.perception.detection.traffic_light import TrafficLight


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
    """ Converts a Carla vehicle object into a pylot.simulation.utils.Obstacle
        object.

    Args:
        vec_actors: A list of Carla vehicle actors.

    Returns:
        A list of obstacles.
    """
    vehicles = []
    for vec_actor in vec_actors:
        vehicles.append(pylot.simulation.utils.Obstacle(vec_actor))
    return vehicles


def convert_pedestrian_actors(pedestrian_actors):
    """ Converts a Carla pedestrian actor into a pylot.simulation.util.Obstacle
        object.

    Args:
        pedestrian_actors: A list of Carla pedestrian actors.

    Returns:
        A list of obstacles.
    """
    pedestrians = []
    for ped_actor in pedestrian_actors:
        pedestrians.append(pylot.simulation.utils.Obstacle(ped_actor))
    return pedestrians


def convert_traffic_light_actors(tl_actors):
    """ Converts a Carla traffic light actor into a Pylot tl object.

    Args:
        tl_actors: A list of Carla traffic light actors.

    Returns:
        A list of  pylot.perception.detection.traffic_light.TrafficLights.
    """
    traffic_lights = []
    for tl_actor in tl_actors:
        traffic_lights.append(TrafficLight.from_carla_actor(tl_actor))
    return traffic_lights


def convert_speed_limit_actors(speed_limit_actors):
    """ Converts a Carla speed limit actor into a Pylot speed limit object.

    Args:
        speed_limit_actors: A list of Carla speed limit actors.

    Returns:
        A list of Pylot SpeedLimitSigns.
    """
    speed_limits = []
    for ts_actor in speed_limit_actors:
        transform = pylot.utils.Transform.from_carla_transform(
            ts_actor.get_transform())
        speed_limit = int(ts_actor.type_id.split('.')[-1])
        speed_sign = pylot.simulation.utils.SpeedLimitSign(
            transform, speed_limit)
        speed_limits.append(speed_sign)
    return speed_limits


def convert_traffic_stop_actors(traffic_stop_actors):
    """ Converts a Carla traffic stop actor into a Pylot stop sign object.

    Args:
        traffic_stop_actors: A list of Carla traffic stop sign actors.

    Returns:
        A list of Pylot StopSigns.
    """
    stop_signs = []
    for ts_actor in traffic_stop_actors:
        transform = pylot.utils.Transform.from_carla_transform(
            ts_actor.get_transform())
        world_trigger_volume = ts_actor.get_transform().transform(
            ts_actor.trigger_volume.location)
        bbox = pylot.simulation.utils.BoundingBox(
            carla.BoundingBox(world_trigger_volume,
                              ts_actor.trigger_volume.extent))
        stop_sign = pylot.simulation.utils.StopSign(transform, bbox)
        stop_signs.append(stop_sign)
    return stop_signs


def draw_trigger_volume(world, actor):
    """ Draws the trigger volume of an actor."""
    transform = actor.get_transform()
    tv = transform.transform(actor.trigger_volume.location)
    bbox = carla.BoundingBox(tv, actor.trigger_volume.extent)
    world.debug.draw_box(bbox, transform.rotation, life_time=1000)
