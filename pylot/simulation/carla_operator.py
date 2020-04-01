import carla
import erdos
import random
import time

import pylot.simulation.utils
import pylot.utils
from pylot.perception.messages import ObstaclesMessage, SpeedSignsMessage, \
    StopSignsMessage, TrafficLightsMessage


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
    def __init__(self, control_stream, pose_stream,
                 ground_traffic_lights_stream, ground_obstacles_stream,
                 ground_speed_limit_signs_stream, ground_stop_signs_stream,
                 vehicle_id_stream, open_drive_stream,
                 global_trajectory_stream, flags):
        if flags.random_seed:
            random.seed(flags.random_seed)
        # Register callback on control stream.
        control_stream.add_callback(self.on_control_msg)
        self.pose_stream = pose_stream
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
        self._client, self._world = pylot.simulation.utils.get_world(
            self._flags.carla_host, self._flags.carla_port,
            self._flags.carla_timeout)
        if self._client is None or self._world is None:
            raise ValueError('There was an issue connecting to the simulator.')

        if not self._flags.carla_scenario_runner:
            # Load the appropriate town if the operator is not running in a
            # scenario setup.
            self._initialize_world()

        # Save the spectator handle so that we don't have to repeteadly get the
        # handle (which is slow).
        self._spectator = self._world.get_spectator()

        pylot.simulation.utils.set_simulation_mode(self._world, self._flags)

        if self._flags.carla_scenario_runner:
            # Wait until the ego vehicle is spawned by the scenario runner.
            self._logger.info("Waiting for the scenario to be ready ...")
            self._ego_vehicle = pylot.simulation.utils.wait_for_ego_vehicle()
            self._logger.info("Found ego vehicle")
        else:
            # Spawn ego vehicle, people and vehicles.
            (self._ego_vehicle, self._vehicle_ids,
             self._people) = pylot.simulation.utils.spawn_actors(
                 self._client, self._world, self._flags.carla_version,
                 self._flags.carla_spawn_point_index,
                 self._flags.control_agent == 'carla_auto_pilot',
                 self._flags.carla_num_people, self._flags.carla_num_vehicles,
                 self._logger)

        pylot.simulation.utils.set_vehicle_physics(
            self._ego_vehicle, self._flags.carla_vehicle_moi,
            self._flags.carla_vehicle_mass)

    @staticmethod
    def connect(control_stream):
        pose_stream = erdos.WriteStream()
        ground_traffic_lights_stream = erdos.WriteStream()
        ground_obstacles_stream = erdos.WriteStream()
        ground_speed_limit_signs_stream = erdos.WriteStream()
        ground_stop_signs_stream = erdos.WriteStream()
        vehicle_id_stream = erdos.WriteStream()
        open_drive_stream = erdos.WriteStream()
        global_trajectory_stream = erdos.WriteStream()
        return [
            pose_stream, ground_traffic_lights_stream, ground_obstacles_stream,
            ground_speed_limit_signs_stream, ground_stop_signs_stream,
            vehicle_id_stream, open_drive_stream, global_trajectory_stream
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
            self._ego_vehicle.apply_control(vec_control)
        # Tick the world after the operator received a control command.
        # This usually indicates that all the operators have completed
        # processing the previous timestamp. However, this is not always
        # true (e.g., logging operators that are not part of the main loop).
        self._tick_simulator()

    def send_actor_data(self, msg):
        """ Callback function that gets called when the world is ticked.
        This function sends a WatermarkMessage to the downstream operators as
        a signal that they need to release data to the rest of the pipeline.

        Args:
            msg: Data recieved from the simulation at a tick.
        """
        game_time = int(msg.elapsed_seconds * 1000)
        self._logger.info('The world is at the timestamp {}'.format(game_time))
        with erdos.profile(self.config.name + '.send_actor_data',
                           self,
                           event_data={'timestamp': str(game_time)}):
            timestamp = erdos.Timestamp(coordinates=[game_time])
            watermark_msg = erdos.WatermarkMessage(timestamp)
            self.__send_hero_vehicle_data(timestamp, watermark_msg)
            self.__send_ground_actors_data(timestamp, watermark_msg)
            self.__update_spectactor_pose()

    def run(self):
        self.__send_world_data()
        # Tick here once to ensure that the driver operators can get a handle
        # to the ego vehicle.
        self._tick_at = time.time()
        self._tick_simulator()
        # XXX(ionel): Hack to fix a race condition. Driver operators
        # register a carla listen callback only after they've received
        # the vehicle id value. We miss frames if we tick before
        # they register a listener. Thus, we sleep here a bit to
        # give them sufficient time to register a callback.
        time.sleep(3)
        self._world.on_tick(self.send_actor_data)
        self._tick_simulator()

    def _initialize_world(self):
        """ Setups the world town, and activates the desired weather."""
        if self._flags.carla_version == '0.9.5':
            # TODO (Sukrit) :: ERDOS provides no way to retrieve handles to the
            # class objects to do garbage collection. Hence, objects from
            # previous runs of the simulation may persist. We need to clean
            # them up right now. In future, move this logic to a seperate
            # destroy function.
            pylot.simulation.utils.reset_world(self._world)
        else:
            self._world = self._client.load_world('Town{:02d}'.format(
                self._flags.carla_town))
        self._logger.info('Setting the weather to {}'.format(
            self._flags.carla_weather))
        pylot.simulation.utils.set_weather(self._world,
                                           self._flags.carla_weather)

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

    def __send_hero_vehicle_data(self, timestamp, watermark_msg):
        vec_transform = pylot.utils.Transform.from_carla_transform(
            self._ego_vehicle.get_transform())
        velocity_vector = pylot.utils.Vector3D.from_carla_vector(
            self._ego_vehicle.get_velocity())
        forward_speed = velocity_vector.magnitude()
        pose = pylot.utils.Pose(vec_transform, forward_speed, velocity_vector)
        self.pose_stream.send(erdos.Message(timestamp, pose))
        self.pose_stream.send(erdos.WatermarkMessage(timestamp))

    def __send_ground_actors_data(self, timestamp, watermark_msg):
        # Get all the actors in the simulation.
        actor_list = self._world.get_actors()

        (vehicles, people, traffic_lights, speed_limits, traffic_stops
         ) = pylot.simulation.utils.extract_data_in_pylot_format(actor_list)

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

    def __send_world_data(self):
        """ Sends ego vehicle id, open drive and trajectory messages."""
        # Send the id of the ego vehicle. This id is used by the CARLA driver
        # operators to get a handle to the ego vehicle, which they use to
        # attach sensors.
        self.vehicle_id_stream.send(
            erdos.Message(erdos.Timestamp(coordinates=[0]),
                          self._ego_vehicle.id))
        self.vehicle_id_stream.send(
            erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))

        # Send open drive string.
        self.open_drive_stream.send(
            erdos.Message(erdos.Timestamp(coordinates=[0]),
                          self._world.get_map().to_opendrive()))
        top_watermark = erdos.WatermarkMessage(erdos.Timestamp(is_top=True))
        self.open_drive_stream.send(top_watermark)
        self.global_trajectory_stream.send(top_watermark)

    def __update_spectactor_pose(self):
        # Set the world simulation view with respect to the vehicle.
        v_pose = self._ego_vehicle.get_transform()
        v_pose.location -= 10 * carla.Location(v_pose.get_forward_vector())
        v_pose.location.z = 5
        self._spectator.set_transform(v_pose)
