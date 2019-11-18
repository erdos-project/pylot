import random
import sys
import time
import carla

# ERDOS specific imports.
from erdos.op import Op
from erdos.timestamp import Timestamp
from erdos.utils import setup_logging, setup_csv_logging
from erdos.message import Message, WatermarkMessage

import pylot.utils
from pylot.simulation.carla_utils import extract_data_in_pylot_format,\
    get_weathers, get_world, reset_world, set_synchronous_mode
import pylot.simulation.messages
from pylot.simulation.utils import Transform
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

    def __init__(self,
                 name,
                 auto_pilot,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        """ Initializes the CarlaOperator with the given name.

        Args:
            name: The unique name of the operator.
            auto_pilot: True to enable auto_pilot for ego vehicle.
            flags: A handle to the global flags instance to retrieve the
                configuration.
            log_file_name: The file to log the required information to.
            csv_file_name: The file to log info to in csv format.
        """
        super(CarlaOperator, self).__init__(name)
        self._flags = flags
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._auto_pilot = auto_pilot
        # Connect to CARLA and retrieve the world running.
        self._client, self._world = get_world(self._flags.carla_host,
                                              self._flags.carla_port,
                                              self._flags.carla_timeout)
        if self._client is None or self._world is None:
            raise ValueError('There was an issue connecting to the simulator.')

        if self._flags.carla_version == '0.9.6':
            self._world = self._client.load_world(
                'Town{:02d}'.format(self._flags.carla_town))
        else:
            # TODO (Sukrit) :: ERDOS provides no way to retrieve handles to the
            # class objects to do garbage collection. Hence, objects from
            # previous runs of the simulation may persist. We need to clean
            # them up right now. In future, move this logic to a seperate
            # destroy function.
            reset_world(self._world)

        # Set the weather.
        weather = get_weathers()[self._flags.carla_weather]
        self._logger.info('Setting the weather to {}'.format(
            self._flags.carla_weather))
        self._world.set_weather(weather)
        # Turn on the synchronous mode so we can control the simulation.
        if self._flags.carla_synchronous_mode:
            set_synchronous_mode(self._world, self._flags.carla_fps)

        # Spawn the required number of vehicles.
        self._vehicles = self._spawn_vehicles(self._flags.carla_num_vehicles)

        # Spawn the vehicle that the pipeline has to drive and send it to
        # the downstream operators.
        self._driving_vehicle = self._spawn_driving_vehicle()

        if self._flags.carla_version == '0.9.6':
            # Pedestrians are do not move in versions older than 0.9.6.
            (self._pedestrians, ped_control_ids) = self._spawn_pedestrians(
                self._flags.carla_num_pedestrians)

        # Tick once to ensure that the actors are spawned before the data-flow
        # starts.
        self._tick_at = time.time()
        self._tick_simulator()

        # Start pedestrians
        if self._flags.carla_version == '0.9.6':
            self._start_pedestrians(ped_control_ids)

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
            pylot.utils.create_ground_speed_limit_signs_stream(),
            pylot.utils.create_ground_stop_signs_stream()]
        return ground_agent_streams + [pylot.utils.create_vehicle_id_stream()]

    def on_control_msg(self, msg):
        """ Invoked when a ControlMessage is received.

        Args:
            msg: A control.messages.ControlMessage message.
        """
        # If auto pilot is enabled for the ego vehicle we do not apply the
        # control, but we still want to tick in this method to ensure that
        # all operators finished work before the world ticks.
        if not self._auto_pilot:
            # Transform the message to a carla control cmd.
            vec_control = carla.VehicleControl(
                throttle=msg.throttle,
                steer=msg.steer,
                brake=msg.brake,
                hand_brake=msg.hand_brake,
                reverse=msg.reverse)
            self._driving_vehicle.apply_control(vec_control)
        # Tick the world after the operator received a control command.
        # This usually indicates that all the operators have completed
        # processing the previous timestamp. However, this is not always
        # true (e.g., logging operators that are not part of the main loop).
        self._tick_simulator()

    def _tick_simulator(self):
        if (not self._flags.carla_synchronous_mode or
            self._flags.carla_step_frequency == -1):
            # Run as fast as possible.
            self._world.tick()
            return
        time_until_tick = self._tick_at - time.time()
        if time_until_tick > 0:
            time.sleep(time_until_tick)
        else:
            self._logger.error(
                'Cannot tick Carla at frequency {}'.format(
                    self._flags.carla_step_frequency))
        self._tick_at += 1.0 / self._flags.carla_step_frequency
        self._world.tick()

    def _spawn_pedestrians(self, num_pedestrians):
        """ Spawns pedestrians at random locations inside the world.

        Args:
            num_pedestrians: The number of pedestrians to spawn.
        """
        p_blueprints = self._world.get_blueprint_library().filter(
            'walker.pedestrian.*')
        unique_locs = set([])
        spawn_points = []
        # Get unique spawn points.
        for i in range(num_pedestrians):
            attempt = 0
            while attempt < 10:
                spawn_point = carla.Transform()
                loc = self._world.get_random_location_from_navigation()
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
                self._logger.error(
                    'Could not find unique pedestrian spawn point')
        # Spawn the pedestrians.
        batch = []
        for spawn_point in spawn_points:
            p_blueprint = random.choice(p_blueprints)
            if p_blueprint.has_attribute('is_invincible'):
                p_blueprint.set_attribute('is_invincible', 'false')
            batch.append(carla.command.SpawnActor(p_blueprint, spawn_point))
        # Apply the batch and retrieve the identifiers.
        ped_ids = []
        for response in self._client.apply_batch_sync(batch, True):
            if response.error:
                self._logger.info(
                    'Received an error while spawning a pedestrian: {}'.format(
                        response.error))
            else:
                ped_ids.append(response.actor_id)
        # Spawn the pedestrian controllers
        ped_controller_bp = self._world.get_blueprint_library().find(
            'controller.ai.walker')
        batch = []
        for ped_id in ped_ids:
            batch.append(carla.command.SpawnActor(ped_controller_bp,
                                                  carla.Transform(),
                                                  ped_id))
        ped_control_ids = []
        for response in self._client.apply_batch_sync(batch, True):
            if response.error:
                self._logger.info(
                    'Error while spawning a pedestrian controller: {}'.format(
                        response.error))
            else:
                ped_control_ids.append(response.actor_id)

        return (ped_ids, ped_control_ids)

    def _start_pedestrians(self, ped_control_ids):
        ped_actors = self._world.get_actors(ped_control_ids)
        for i, ped_control_id in enumerate(ped_control_ids):
            # Start pedestrian.
            ped_actors[i].start()
            ped_actors[i].go_to_location(
                self._world.get_random_location_from_navigation())

    def _spawn_vehicles(self, num_vehicles):
        """ Spawns vehicles at random locations inside the world.

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
        """ Spawns the ego vehicle.

        Returns:
            A handle to the ego vehicle.
        """
        self._logger.debug('Spawning the vehicle to be driven around.')

        # Set our vehicle to be the one used in the CARLA challenge.
        v_blueprint = self._world.get_blueprint_library().filter(
            'vehicle.lincoln.mkz2017')[0]

        driving_vehicle = None

        while not driving_vehicle:
            if self._flags.carla_spawn_point_index == -1:
                # Pick a random spawn point.
                start_pose = random.choice(
                    self._world.get_map().get_spawn_points())
            else:
                spawn_points = self._world.get_map().get_spawn_points()
                assert self._flags.carla_spawn_point_index < len(spawn_points), \
                    'Spawn point index is too big. ' \
                    'Town does not have sufficient spawn points.'
                start_pose = spawn_points[self._flags.carla_spawn_point_index]

            driving_vehicle = self._world.try_spawn_actor(v_blueprint,
                                                          start_pose)
        if self._auto_pilot:
            driving_vehicle.set_autopilot(self._auto_pilot)
        return driving_vehicle

    def publish_world_data(self, msg):
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

    def execute(self):
        # Register a callback function and a function that ticks the world.
        # TODO(ionel): We do not currently have a top message.
        timestamp = Timestamp(coordinates=[sys.maxint])
        vehicle_id_msg = Message(self._driving_vehicle.id, timestamp)
        self.get_output_stream('vehicle_id_stream').send(vehicle_id_msg)
        self.get_output_stream('vehicle_id_stream').send(
            WatermarkMessage(timestamp))

        # XXX(ionel): Hack to fix a race condition. Driver operators
        # register a carla listen callback only after they've received
        # the vehicle id value. We miss frames if we tick before
        # they register a listener. Thus, we sleep here a bit to
        # give them sufficient time to register a callback.
        time.sleep(10)
        self._tick_simulator()
        time.sleep(5)
        self._world.on_tick(self.publish_world_data)
        self._tick_simulator()
        self.spin()

    def __publish_hero_vehicle_data(self, timestamp, watermark_msg):
        vec_transform = Transform(
            carla_transform=self._driving_vehicle.get_transform())
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

        (vehicles,
         pedestrians,
         traffic_lights,
         speed_limits,
         traffic_stops) = extract_data_in_pylot_format(actor_list)

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
        speed_limit_signs_msg = pylot.simulation.messages.GroundSpeedSignsMessage(
            speed_limits, timestamp)
        self.get_output_stream('speed_limit_signs').send(speed_limit_signs_msg)
        self.get_output_stream('speed_limit_signs').send(watermark_msg)
        stop_signs_msg = pylot.simulation.messages.GroundStopSignsMessage(
            traffic_stops, timestamp)
        self.get_output_stream('stop_signs').send(stop_signs_msg)
        self.get_output_stream('stop_signs').send(watermark_msg)
