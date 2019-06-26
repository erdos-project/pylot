import carla

from pylot.simulation.utils import to_erdos_transform
import pylot.simulation.utils


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


def to_carla_transform(transform):
    return carla.Transform(
        carla.Location(transform.location.x,
                       transform.location.y,
                       transform.location.z),
        carla.Rotation(pitch=transform.rotation.pitch,
                       yaw=transform.rotation.yaw,
                       roll=transform.rotation.roll))


def get_ground_data(actor_list):
    vec_actors = actor_list.filter('vehicle.*')
    vehicles = convert_vehicle_actors(vec_actors)

    pedestrian_actors = actor_list.filter('*walker*')
    pedestrians = convert_pedestrian_actors(pedestrian_actors)

    tl_actors = actor_list.filter('traffic.traffic_light*')
    traffic_lights = convert_traffic_light_actors(tl_actors)

    speed_limit_actors = actor_list.filter('traffic.speed_limit*')
    speed_limits = convert_speed_limit_actors(speed_limit_actors)

    traffic_stop_actors = actor_list.filter('traffic.stop')
    traffic_stops = convert_traffic_stop_actors(traffic_stop_actors)

    return (vehicles, pedestrians, traffic_lights, speed_limits, traffic_stops)


def convert_vehicle_actors(vec_actors):
    vehicles = []
    # TODO(ionel): Handle hero vehicle!
    for vec_actor in vec_actors:
        loc = vec_actor.get_location()
        pos = pylot.simulation.utils.Location(loc.x, loc.y, loc.z)
        transform = to_erdos_transform(vec_actor.get_transform())
        bounding_box = pylot.simulation.utils.BoundingBox(
            vec_actor.bounding_box)
        speed = pylot.simulation.utils.get_speed(vec_actor.get_velocity())
        vehicle = pylot.simulation.utils.Vehicle(
            pos, transform, bounding_box, speed)
        vehicles.append(vehicle)
    return vehicles


def convert_pedestrian_actors(pedestrian_actors):
    pedestrians = []
    for ped_actor in pedestrian_actors:
        loc = ped_actor.get_location()
        pos = pylot.simulation.utils.Location(loc.x, loc.y, loc.z)
        transform = to_erdos_transform(ped_actor.get_transform())
        speed = pylot.simulation.utils.get_speed(ped_actor.get_velocity())
        # TODO(ionel): Pedestrians do not have a bounding box in 0.9.5.
        pedestrian = pylot.simulation.utils.Pedestrian(
            ped_actor.id, pos, transform, None, speed)
        pedestrians.append(pedestrian)
    return pedestrians


def convert_traffic_light_actors(tl_actors):
    traffic_lights = []
    for tl_actor in tl_actors:
        loc = tl_actor.get_location()
        pos = pylot.simulation.utils.Location(loc.x, loc.y, loc.z)
        transform = to_erdos_transform(tl_actor.get_transform())
        traffic_light = pylot.simulation.utils.TrafficLight(
            pos, transform, tl_actor.get_state())
        traffic_lights.append(traffic_light)
    return traffic_lights


def convert_speed_limit_actors(speed_limit_actors):
    speed_limits = []
    for ts_actor in speed_limit_actors:
        loc = ts_actor.get_location()
        pos = pylot.simulation.utils.Location(loc.x, loc.y, loc.z)
        transform = to_erdos_transform(ts_actor.get_transform())
        speed_limit = int(ts_actor.type_id.split('.')[-1])
        speed_sign = pylot.simulation.utils.SpeedLimitSign(
            pos, transform, speed_limit)
        speed_limits.append(speed_sign)
    return speed_limits


def convert_traffic_stop_actors(traffic_stop_actors):
    stop_signs = []
    for ts_actor in traffic_stop_actors:
        transform = to_erdos_transform(ts_actor.get_transform())
        stop_signs.append(transform)
    return stop_signs
