import carla


def get_world(host="localhost", port=2000):
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
        client.set_timeout(10.0)
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
