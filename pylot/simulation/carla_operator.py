import random
import sys
import time
import carla

# ERDOS specific imports.
from erdos.op import Op
from erdos.timestamp import Timestamp
from erdos.data_stream import DataStream
from erdos.utils import frequency, setup_logging, setup_csv_logging
from erdos.message import Message, WatermarkMessage

import pylot.utils
from pylot.simulation.carla_utils import get_weathers, get_world, reset_world
import pylot.simulation.messages
from pylot.simulation.utils import to_erdos_transform
import pylot.simulation.utils


class CarlaOperator(Op):
    """ CarlaOperator initializes and controls the simulation.

    This operator connects to the simulation, sets the required weather in the
    simulation world, initializes the required number of actors, and the
    vehicle that the rest of the pipeline drives.

    Attributes:
        _client: A connection to the simulator.
        _world: A handle to the world running inside the simulation.
        _vehicles: A list of identifiers of the vehicles inside the simulation.
    """

    def __init__(self, name, flags, log_file_name=None, csv_file_name=None):
        """ Initializes the CarlaOperator with the given name.

        Args:
            name: The unique name of the operator.
            flags: A handle to the global flags instance to retrieve the
                configuration.
            log_file_name: The file to log the required information to.
            csv_file_name: The file to log info to in csv format.
        """
        super(CarlaOperator, self).__init__(name)
        self._flags = flags
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)

        # Connect to CARLA and retrieve the world running.
        self._client, self._world = get_world(self._flags.carla_host,
                                              self._flags.carla_port,
                                              self._flags.carla_timeout)
        if self._client is None or self._world is None:
            raise ValueError('There was an issue connecting to the simulator.')

        # TODO (Sukrit) :: ERDOS provides no way to retrieve handles to the
        # class objects to do garbage collection. Hence, objects from
        # previous runs of the simulation may persist. We need to clean them
        # up right now. In future, move this logic to a seperate destroy
        # function.
        reset_world(self._world)

        # Set the weather.
        weather, name = get_weathers()[self._flags.carla_weather - 1]
        self._logger.info('Setting the weather to {}'.format(name))
        self._world.set_weather(weather)
        # Turn on the synchronous mode so we can control the simulation.
        self._set_synchronous_mode(self._flags.carla_synchronous_mode)

        # Spawn the required number of vehicles.
        self._vehicles = self._spawn_vehicles(self._flags.carla_num_vehicles)

        # Spawn the vehicle that the pipeline has to drive and send it to
        # the downstream operators.
        self._driving_vehicle = self._spawn_driving_vehicle()

        # Tick once to ensure that the actors are spawned before the data-flow
        # starts.
        self._world.tick()

    @staticmethod
    def setup_streams(input_streams):
        # Register callback on control stream.
        input_streams.filter(pylot.utils.is_control_stream).add_callback(
            CarlaOperator.on_control_msg)
        ground_agent_streams = [
            pylot.utils.create_can_bus_stream(),
            pylot.utils.create_ground_traffic_lights_stream(),
            pylot.utils.create_ground_vehicles_stream(),
            pylot.utils.create_ground_pedestrians_stream(),
            pylot.utils.create_ground_traffic_signs_stream()]
        return ground_agent_streams + [pylot.utils.create_vehicle_id_stream()]

    def on_control_msg(self, msg):
        """ Invoked when a ControlMessage is received.

        Args:
            msg: A control.messages.ControlMessage message.
        """
        # Transform the message to a carla control cmd.
        vec_control = carla.VehicleControl(
            throttle=msg.throttle,
            steer=msg.steer,
            brake=msg.brake,
            hand_brake=msg.hand_brake,
            reverse=msg.reverse)
        self._driving_vehicle.apply_control(vec_control)

    def _set_synchronous_mode(self, value):
        """ Sets the synchronous mode to the desired value.

        Args:
            value: The boolean value to turn synchronous mode on/off.
        """
        self._logger.debug('Setting the synchronous mode to {}'.format(value))
        settings = self._world.get_settings()
        settings.synchronous_mode = value
        self._world.apply_settings(settings)

    def _spawn_vehicles(self, num_vehicles):
        """ Spawns the required number of vehicles at random locations inside
        the world.

        Args:
            num_vehicles: The number of vehicles to spawn.
        """
        self._logger.debug('Trying to spawn {} vehicles.'.format(num_vehicles))

        # Get the spawn points and ensure that the number of vehicles
        # requested are less than the number of spawn points.
        spawn_points = self._world.get_map().get_spawn_points()
        if num_vehicles >= len(spawn_points):
            self._logger.warning(
                'Requested {} vehicles but only found {} spawn points'.format(
                    num_vehicles, len(spawn_points)))
            num_vehicles = len(spawn_points)
        else:
            random.shuffle(spawn_points)

        # Get all the possible vehicle blueprints inside the world.
        v_blueprints = self._world.get_blueprint_library().filter('vehicle.*')

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
                    carla.command.SetAutopilot(carla.command.FutureActor,
                                               True)))

        # Apply the batch and retrieve the identifiers.
        vehicle_ids = []
        for response in self._client.apply_batch_sync(batch):
            if response.error:
                self._logger.info(
                    'Received an error while spawning a vehicle: {}'.format(
                        response.error))
            else:
                vehicle_ids.append(response.actor_id)
        return vehicle_ids

    def _spawn_driving_vehicle(self):
        """ Spawns the vehicle that the rest of the pipeline drives.

        Returns:
            A handle to the vehicle being driven around.
        """
        self._logger.debug('Spawning the vehicle to be driven around.')

        # Set our vehicle to be the one used in the CARLA challenge.
        v_blueprint = self._world.get_blueprint_library().filter(
            'vehicle.lincoln.mkz2017')[0]

        driving_vehicle = None
        while not driving_vehicle:
            start_pose = random.choice(self._world.get_map().get_spawn_points())
            driving_vehicle = self._world.try_spawn_actor(v_blueprint, start_pose)
        return driving_vehicle

    def on_world_tick(self, msg):
        """ Callback function that gets called when the world is ticked.
        This function sends a WatermarkMessage to the downstream operators as
        a signal that they need to release data to the rest of the pipeline.

        Args:
            msg: Data recieved from the simulation at a tick.
        """
        game_time = int(msg.elapsed_seconds * 1000)
        self._logger.info('The world is at the timestamp {}'.format(game_time))
        # Create a timestamp and send a WatermarkMessage on the output stream.
        timestamp = Timestamp(coordinates=[game_time])
        watermark_msg = WatermarkMessage(timestamp)
        self.__publish_hero_vehicle_data(timestamp, watermark_msg)
        self.__publish_ground_actors_data(timestamp, watermark_msg)
        # XXX(ionel): We tick after we send data. Otherwise, we may fall
        # behind.
        self._world.tick()

#    @frequency(10)
    def tick_at_frequency(self):
        """ This function ticks the world at the desired frequency. """
        self._world.tick()

    def execute(self):
        # Register a callback function and a function that ticks the world.
        # TODO(ionel): We do not currently have a top message.
        timestamp = Timestamp(coordinates=[sys.maxint])
        vehicle_id_msg = Message(self._driving_vehicle.id, timestamp)
        self.get_output_stream('vehicle_id_stream').send(vehicle_id_msg)
        self.get_output_stream('vehicle_id_stream').send(
            WatermarkMessage(timestamp))

        self._world.on_tick(self.on_world_tick)
        self.tick_at_frequency()
        self.spin()

    def __publish_hero_vehicle_data(self, timestamp, watermark_msg):
        vec_transform = to_erdos_transform(
            self._driving_vehicle.get_transform())
        forward_speed = pylot.simulation.utils.get_speed(
            self._driving_vehicle.get_velocity())
        can_bus = pylot.simulation.utils.CanBus(vec_transform, forward_speed)
        self.get_output_stream('can_bus').send(
            Message(can_bus, timestamp))
        self.get_output_stream('can_bus').send(watermark_msg)

        # Set the world simulation view with respect to the vehicle.
        v_pose = self._driving_vehicle.get_transform()
        v_pose.location -= 10 * carla.Location(v_pose.get_forward_vector())
        v_pose.location.z = 5
        self._world.get_spectator().set_transform(v_pose)

    def __publish_ground_actors_data(self, timestamp, watermark_msg):
        # Get all the actors in the simulation.
        actor_list = self._world.get_actors()

        vec_actors = actor_list.filter('vehicle.*')
        vehicles = self.__convert_vehicle_actors(vec_actors)

        pedestrian_actors = actor_list.filter('*walker*')
        pedestrians = self.__convert_pedestrian_actors(pedestrian_actors)

        tl_actors = actor_list.filter('traffic.traffic_light*')
        traffic_lights = self.__convert_traffic_light_actors(tl_actors)

        speed_limit_actors = actor_list.filter('traffic.speed_limit*')
        speed_limits = self.__convert_speed_limit_actors(speed_limit_actors)

        traffic_stop_actors = actor_list.filter('traffic.stop')
        traffic_stops = self.__convert_traffic_stop_actors(traffic_stop_actors)
        # TODO(ionel): Send traffic stops.

        vehicles_msg = pylot.simulation.messages.GroundVehiclesMessage(
            vehicles, timestamp)
        self.get_output_stream('vehicles').send(vehicles_msg)
        self.get_output_stream('vehicles').send(watermark_msg)
        pedestrians_msg = pylot.simulation.messages.GroundPedestriansMessage(
            pedestrians, timestamp)
        self.get_output_stream('pedestrians').send(pedestrians_msg)
        self.get_output_stream('pedestrians').send(watermark_msg)
        traffic_lights_msg = pylot.simulation.messages.GroundTrafficLightsMessage(
            traffic_lights, timestamp)
        self.get_output_stream('traffic_lights').send(traffic_lights_msg)
        self.get_output_stream('traffic_lights').send(watermark_msg)
        traffic_signs_msg = pylot.simulation.messages.GroundSpeedSignsMessage(
            speed_limits, timestamp)
        self.get_output_stream('traffic_signs').send(traffic_signs_msg)
        self.get_output_stream('traffic_signs').send(watermark_msg)

    def __convert_vehicle_actors(self, vec_actors):
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

    def __convert_pedestrian_actors(self, pedestrian_actors):
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

    def __convert_traffic_light_actors(self, tl_actors):
        traffic_lights = []
        for tl_actor in tl_actors:
            loc = tl_actor.get_location()
            pos = pylot.simulation.utils.Location(loc.x, loc.y, loc.z)
            transform = to_erdos_transform(tl_actor.get_transform())
            traffic_light = pylot.simulation.utils.TrafficLight(
                pos, transform, tl_actor.get_state())
            traffic_lights.append(traffic_light)
        return traffic_lights

    def __convert_speed_limit_actors(self, speed_limit_actors):
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

    def __convert_traffic_stop_actors(self, traffic_stop_actors):
        for ts_actor in traffic_stop_actors:
            loc = ts_actor.get_location()
            pos = pylot.simulation.utils.Location(loc.x, loc.y, loc.z)
            transform = to_erdos_transform(ts_actor.get_transform())
