import carla
import erdos
import random
import sys
import time

import pylot.utils
from pylot.perception.messages import ObstaclesMessage, SpeedSignsMessage, \
    StopSignsMessage, TrafficLightsMessage
from pylot.simulation.utils import extract_data_in_pylot_format, \
    get_weathers, get_world, reset_world, set_simulation_mode


class CarlaOperator(erdos.Operator):
    """ CarlaOperator initializes and controls the simulation.

    This operator connects to the simulation, sets the required weather in the
    simulation world, initializes the required number of actors, and the
    vehicle that the rest of the pipeline drives.

    Args:
        flags: A handle to the global flags instance to retrieve the
            configuration.

    Attributes:
        _client: A connection to the simulator.
        _world: A handle to the world running inside the simulation.
        _vehicles: A list of identifiers of the vehicles inside the simulation.
    """
    def __init__(self, control_stream, can_bus_stream,
                 ground_traffic_lights_stream, ground_obstacles_stream,
                 ground_speed_limit_signs_stream, ground_stop_signs_stream,
                 vehicle_id_stream, open_drive_stream,
                 global_trajectory_stream, flags):
        if flags.random_seed:
            random.seed(flags.random_seed)
        # Register callback on control stream.
        control_stream.add_callback(self.on_control_msg)
        self.can_bus_stream = can_bus_stream
        self.ground_traffic_lights_stream = ground_traffic_lights_stream
        self.ground_obstacles_stream = ground_obstacles_stream
        self.ground_speed_limit_signs_stream = ground_speed_limit_signs_stream
        self.ground_stop_signs_stream = ground_stop_signs_stream
        self.vehicle_id_stream = vehicle_id_stream
        self.open_drive_stream = open_drive_stream
        self.global_trajectory_stream = global_trajectory_stream

        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        # Connect to CARLA and retrieve the world running.
        self._client, self._world = get_world(self._flags.carla_host,
                                              self._flags.carla_port,
                                              self._flags.carla_timeout)
        if self._client is None or self._world is None:
            raise ValueError('There was an issue connecting to the simulator.')

        if not self._flags.carla_scenario_runner:
            # Load the appropriate town.
            self._initialize_world()

        # Save the spectator handle so that we don't have to repeteadly get the
        # handle (which is slow).
        self._spectator = self._world.get_spectator()
        self._send_world_messages()

        set_simulation_mode(self._world, self._flags)

        if self._flags.carla_scenario_runner:
            # Waits until the ego vehicle is spawned by the scenario runner.
            self._wait_for_ego_vehicle()
        else:
            # Spawns the person and vehicle actors.
            self._spawn_actors()

    @staticmethod
    def connect(control_stream):
        can_bus_stream = erdos.WriteStream()
        ground_traffic_lights_stream = erdos.WriteStream()
        ground_obstacles_stream = erdos.WriteStream()
        ground_speed_limit_signs_stream = erdos.WriteStream()
        ground_stop_signs_stream = erdos.WriteStream()
        vehicle_id_stream = erdos.WriteStream()
        open_drive_stream = erdos.WriteStream()
        global_trajectory_stream = erdos.WriteStream()
        return [
            can_bus_stream, ground_traffic_lights_stream,
            ground_obstacles_stream, ground_speed_limit_signs_stream,
            ground_stop_signs_stream, vehicle_id_stream, open_drive_stream,
            global_trajectory_stream
        ]

    @erdos.profile_method()
    def on_control_msg(self, msg):
        """ Invoked when a ControlMessage is received.

        Args:
            msg: A control.messages.ControlMessage message.
        """
        self._logger.debug('@{}: received control message'.format(
            msg.timestamp))
        # If auto pilot is enabled for the ego vehicle we do not apply the
        # control, but we still want to tick in this method to ensure that
        # all operators finished work before the world ticks.
        if self._flags.control_agent != 'carla_auto_pilot':
            # Transform the message to a carla control cmd.
            vec_control = carla.VehicleControl(throttle=msg.throttle,
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

    def _send_world_messages(self):
        """ Sends initial open drive and trajectory messages."""
        # Send open drive string.
        top_timestamp = erdos.Timestamp(coordinates=[sys.maxsize])
        self.open_drive_stream.send(
            erdos.Message(top_timestamp,
                          self._world.get_map().to_opendrive()))
        top_watermark = erdos.WatermarkMessage(top_timestamp)
        self.open_drive_stream.send(top_watermark)
        self.global_trajectory_stream.send(top_watermark)

    def _initialize_world(self):
        """ Setups the world town, and activates the desired weather."""
        if self._flags.carla_version == '0.9.5':
            # TODO (Sukrit) :: ERDOS provides no way to retrieve handles to the
            # class objects to do garbage collection. Hence, objects from
            # previous runs of the simulation may persist. We need to clean
            # them up right now. In future, move this logic to a seperate
            # destroy function.
            reset_world(self._world)
        else:
            self._world = self._client.load_world('Town{:02d}'.format(
                self._flags.carla_town))
        # Set the weather.
        weather = get_weathers()[self._flags.carla_weather]
        self._logger.info('Setting the weather to {}'.format(
            self._flags.carla_weather))
        self._world.set_weather(weather)

    def _spawn_actors(self):
        # Spawn the required number of vehicles.
        self._vehicles = self._spawn_vehicles(self._flags.carla_num_vehicles)

        # Spawn the vehicle that the pipeline has to drive and send it to
        # the downstream operators.
        self._driving_vehicle = self._spawn_driving_vehicle()

        if (self._flags.carla_version == '0.9.6'
                or self._flags.carla_version == '0.9.7'
                or self._flags.carla_version == '0.9.8'):
            # People are do not move in versions older than 0.9.6.
            (self._people, ped_control_ids) = self._spawn_people(
                self._flags.carla_num_people)

        # Tick once to ensure that the actors are spawned before the data-flow
        # starts.
        self._tick_at = time.time()
        self._tick_simulator()

        # Start people
        if (self._flags.carla_version == '0.9.6'
                or self._flags.carla_version == '0.9.7'
                or self._flags.carla_version == '0.9.8'):
            self._start_people(ped_control_ids)

    def _wait_for_ego_vehicle(self):
        # Connect to the ego-vehicle spawned by the scenario runner.
        self._driving_vehicle = None
        while self._driving_vehicle is None:
            self._logger.info("Waiting for the scenario to be ready ...")
            time.sleep(1)
            possible_actors = self._world.get_actors().filter('vehicle.*')
            for actor in possible_actors:
                if actor.attributes['role_name'] == 'hero':
                    self._driving_vehicle = actor
                    break
            self._world.tick()
        if self._flags.carla_vehicle_moi and self._flags.carla_vehicle_mass:
            # Fix the physics of the vehicle to increase the max speed.
            physics_control = self._driving_vehicle.get_physics_control()
            physics_control.moi = 0.1
            physics_control.mass = 100
            self._driving_vehicle.apply_physics_control(physics_control)

    def _tick_simulator(self):
        if (self._flags.carla_mode == 'asynchronous'
                or self._flags.carla_mode == 'asynchronous-fixed-time-step'):
            # No need to tick when running in these modes.
            return
        if self._flags.carla_step_frequency == -1:
            # Run as fast as possible.
            self._world.tick()
            return
        time_until_tick = self._tick_at - time.time()
        if time_until_tick > 0:
            time.sleep(time_until_tick)
        else:
            self._logger.error('Cannot tick Carla at frequency {}'.format(
                self._flags.carla_step_frequency))
        self._tick_at += 1.0 / self._flags.carla_step_frequency
        self._world.tick()

    def _spawn_people(self, num_people):
        """ Spawns people at random locations inside the world.

        Args:
            num_people: The number of people to spawn.
        """
        p_blueprints = self._world.get_blueprint_library().filter(
            'walker.pedestrian.*')
        unique_locs = set([])
        spawn_points = []
        # Get unique spawn points.
        for i in range(num_people):
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
                self._logger.error('Could not find unique person spawn point')
        # Spawn the people.
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
                    'Received an error while spawning a person: {}'.format(
                        response.error))
            else:
                ped_ids.append(response.actor_id)
        # Spawn the person controllers
        ped_controller_bp = self._world.get_blueprint_library().find(
            'controller.ai.walker')
        batch = []
        for ped_id in ped_ids:
            batch.append(
                carla.command.SpawnActor(ped_controller_bp, carla.Transform(),
                                         ped_id))
        ped_control_ids = []
        for response in self._client.apply_batch_sync(batch, True):
            if response.error:
                self._logger.info(
                    'Error while spawning a person controller: {}'.format(
                        response.error))
            else:
                ped_control_ids.append(response.actor_id)

        return (ped_ids, ped_control_ids)

    def _start_people(self, ped_control_ids):
        ped_actors = self._world.get_actors(ped_control_ids)
        for i, ped_control_id in enumerate(ped_control_ids):
            # Start person.
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
        for response in self._client.apply_batch_sync(batch, True):
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

            driving_vehicle = self._world.try_spawn_actor(
                v_blueprint, start_pose)
        if self._flags.control_agent == 'carla_auto_pilot':
            driving_vehicle.set_autopilot(True)
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
        with erdos.profile(self.config.name + '.publish_world_data',
                           self,
                           event_data={'timestamp': str(game_time)}):
            timestamp = erdos.Timestamp(coordinates=[game_time])
            watermark_msg = erdos.WatermarkMessage(timestamp)
            self.__publish_hero_vehicle_data(timestamp, watermark_msg)
            self.__publish_ground_actors_data(timestamp, watermark_msg)

    def run(self):
        # Register a callback function and a function that ticks the world.
        # TODO(ionel): We do not currently have a top message.
        timestamp = erdos.Timestamp(coordinates=[sys.maxsize])
        self.vehicle_id_stream.send(
            erdos.Message(timestamp, self._driving_vehicle.id))
        self.vehicle_id_stream.send(erdos.WatermarkMessage(timestamp))

        # XXX(ionel): Hack to fix a race condition. Driver operators
        # register a carla listen callback only after they've received
        # the vehicle id value. We miss frames if we tick before
        # they register a listener. Thus, we sleep here a bit to
        # give them sufficient time to register a callback.
        time.sleep(3)
        self._tick_simulator()
        time.sleep(5)
        self._world.on_tick(self.publish_world_data)
        self._tick_simulator()

    def __publish_hero_vehicle_data(self, timestamp, watermark_msg):
        vec_transform = pylot.utils.Transform.from_carla_transform(
            self._driving_vehicle.get_transform())
        velocity_vector = pylot.utils.Vector3D.from_carla_vector(
            self._driving_vehicle.get_velocity())
        forward_speed = velocity_vector.magnitude()
        can_bus = pylot.utils.CanBus(vec_transform, forward_speed,
                                     velocity_vector)
        self.can_bus_stream.send(erdos.Message(timestamp, can_bus))
        self.can_bus_stream.send(erdos.WatermarkMessage(timestamp))

        # Set the world simulation view with respect to the vehicle.
        v_pose = self._driving_vehicle.get_transform()
        v_pose.location -= 10 * carla.Location(v_pose.get_forward_vector())
        v_pose.location.z = 5
        self._spectator.set_transform(v_pose)

    def __publish_ground_actors_data(self, timestamp, watermark_msg):
        # Get all the actors in the simulation.
        actor_list = self._world.get_actors()

        (vehicles, people, traffic_lights, speed_limits,
         traffic_stops) = extract_data_in_pylot_format(actor_list)

        # Send ground people and vehicles.
        self.ground_obstacles_stream.send(
            ObstaclesMessage(timestamp, vehicles + people))
        self.ground_obstacles_stream.send(erdos.WatermarkMessage(timestamp))
        # Send ground traffic lights.
        self.ground_traffic_lights_stream.send(
            TrafficLightsMessage(timestamp, traffic_lights))
        self.ground_traffic_lights_stream.send(
            erdos.WatermarkMessage(timestamp))
        # Send ground speed signs.
        self.ground_speed_limit_signs_stream.send(
            SpeedSignsMessage(timestamp, speed_limits))
        self.ground_speed_limit_signs_stream.send(
            erdos.WatermarkMessage(timestamp))
        # Send stop signs.
        self.ground_stop_signs_stream.send(
            StopSignsMessage(timestamp, traffic_stops))
        self.ground_stop_signs_stream.send(erdos.WatermarkMessage(timestamp))
