import carla
import enum
import erdos
import heapq
import random
import time
from functools import total_ordering

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

    def __init__(self, control_stream, notify_stream, pose_stream,
                 pose_stream_for_control, ground_traffic_lights_stream,
                 ground_obstacles_stream, ground_speed_limit_signs_stream,
                 ground_stop_signs_stream, vehicle_id_stream,
                 open_drive_stream, global_trajectory_stream,
                 release_sensor_stream, flags):
        if flags.random_seed:
            random.seed(flags.random_seed)
        # Register callback on control stream.
        control_stream.add_callback(self.on_control_msg)
        erdos.add_watermark_callback([notify_stream], [release_sensor_stream],
                                     self.on_sensor_notify)
        self.pose_stream = pose_stream
        self.pose_stream_for_control = pose_stream_for_control
        self.ground_traffic_lights_stream = ground_traffic_lights_stream
        self.ground_obstacles_stream = ground_obstacles_stream
        self.ground_speed_limit_signs_stream = ground_speed_limit_signs_stream
        self.ground_stop_signs_stream = ground_stop_signs_stream
        self.vehicle_id_stream = vehicle_id_stream
        self.open_drive_stream = open_drive_stream
        self.global_trajectory_stream = global_trajectory_stream
        self.release_sensor_stream = release_sensor_stream

        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            self.config.name + '-csv', self.config.csv_log_file_name)
        # Connect to CARLA and retrieve the world running.
        self._client, self._world = pylot.simulation.utils.get_world(
            self._flags.carla_host, self._flags.carla_port,
            self._flags.carla_timeout)
        if self._client is None or self._world is None:
            raise ValueError('There was an issue connecting to the simulator.')

        if not self._flags.carla_scenario_runner and \
                self._flags.control_agent != "manual":
            # Load the appropriate town.
            self._initialize_world()

        # Save the spectator handle so that we don't have to repeteadly get the
        # handle (which is slow).
        self._spectator = self._world.get_spectator()

        if self._flags.carla_version == '0.9.8':
            # Create a traffic manager to that auto pilot works.
            self._traffic_manager = self._client.get_trafficmanager(8000)
            self._traffic_manager.set_synchronous_mode(
                self._flags.carla_mode == 'synchronous')

        pylot.simulation.utils.set_simulation_mode(self._world, self._flags)

        if self._flags.carla_scenario_runner or \
                self._flags.control_agent == "manual":
            # Wait until the ego vehicle is spawned by the scenario runner.
            self._logger.info("Waiting for the scenario to be ready ...")
            self._ego_vehicle = pylot.simulation.utils.wait_for_ego_vehicle(
                self._world)
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

        # Dictionary that stores the processing times when sensors are ready
        # to realease data. This info is used to calculate the real processing
        # time of our pipeline without including CARLA-induced sensor delays.
        self._next_localization_sensor_reading = None
        self._next_control_sensor_reading = None
        self._simulator_in_sync = False
        self._tick_events = []
        self._control_msgs = {}

    @staticmethod
    def connect(control_stream, notify_stream):
        pose_stream = erdos.WriteStream()
        pose_stream_for_control = erdos.WriteStream()
        ground_traffic_lights_stream = erdos.WriteStream()
        ground_obstacles_stream = erdos.WriteStream()
        ground_speed_limit_signs_stream = erdos.WriteStream()
        ground_stop_signs_stream = erdos.WriteStream()
        vehicle_id_stream = erdos.WriteStream()
        open_drive_stream = erdos.WriteStream()
        global_trajectory_stream = erdos.WriteStream()
        release_sensor_stream = erdos.WriteStream()
        return [
            pose_stream, pose_stream_for_control, ground_traffic_lights_stream,
            ground_obstacles_stream, ground_speed_limit_signs_stream,
            ground_stop_signs_stream, vehicle_id_stream, open_drive_stream,
            global_trajectory_stream, release_sensor_stream
        ]

    @erdos.profile_method()
    def on_control_msg(self, msg):
        """ Invoked when a ControlMessage is received.

        Args:
            msg: A control.messages.ControlMessage message.
        """
        self._logger.debug('@{}: received control message'.format(
            msg.timestamp))
        if self._flags.carla_mode == 'pseudo-asynchronous':
            heapq.heappush(
                self._tick_events,
                (msg.timestamp.coordinates[0], TickEvent.CONTROL_CMD))
            self._control_msgs[msg.timestamp.coordinates[0]] = msg
            # Tick until the next sensor read game time to ensure that the
            # data-flow has a new round of sensor inputs. Apply control
            # commands if they must be applied before the next sensor read.
            self._consume_next_event()
        else:
            # If auto pilot or manual mode is enabled then we do not apply the
            # control, but we still want to tick in this method to ensure that
            # all operators finished work before the world ticks.
            if self._flags.control_agent not in ['carla_auto_pilot', 'manual']:
                self._apply_control_msg(msg)
            # Tick the world after the operator received a control command.
            # This usually indicates that all the operators have completed
            # processing the previous timestamp (with the exception of logging
            # operators that are not part of the main loop).
            self._tick_simulator()

    def _consume_next_event(self):
        while True:
            (sim_time, event_type) = heapq.heappop(self._tick_events)
            if event_type == TickEvent.SENSOR_READ:
                self._tick_simulator_until(sim_time)
                break
            elif event_type == TickEvent.CONTROL_CMD:
                self._tick_simulator_until(sim_time)
                control_msg = self._control_msgs[sim_time]
                self._apply_control_msg(control_msg)

    def on_sensor_notify(self, timestamp, release_sensor_stream):
        # The first sensor reading needs to be discarded because it might
        # not be correctly spaced out.
        if not self._simulator_in_sync:
            self._simulator_in_sync = True
        release_sensor_stream.send(erdos.WatermarkMessage(timestamp))

    def send_actor_data(self, msg):
        """ Callback function that gets called when the world is ticked.
        This function sends a WatermarkMessage to the downstream operators as
        a signal that they need to release data to the rest of the pipeline.

        Args:
            msg: Data recieved from the simulation at a tick.
        """
        game_time = int(msg.elapsed_seconds * 1000)
        self._logger.info('The world is at the timestamp {}'.format(game_time))
        timestamp = erdos.Timestamp(coordinates=[game_time])
        watermark_msg = erdos.WatermarkMessage(timestamp)
        with erdos.profile(self.config.name + '.send_actor_data',
                           self,
                           event_data={'timestamp': str(timestamp)}):
            localization_update, control_update = False, False
            if (self._flags.carla_localization_frequency == -1
                    or self._next_localization_sensor_reading is None
                    or game_time == self._next_localization_sensor_reading):
                if self._flags.carla_mode == 'pseudo-asynchronous':
                    self._update_next_localization_pseudo_asynchronous_ticks(
                        game_time)
                self.__send_hero_vehicle_data(self.pose_stream, timestamp,
                                              watermark_msg)
                self.__send_ground_actors_data(timestamp, watermark_msg)
                self.__update_spectactor_pose()
                localization_update = True

            if self._flags.carla_mode == "pseudo-asynchronous" and (
                    self._flags.carla_control_frequency == -1
                    or self._next_control_sensor_reading is None
                    or game_time == self._next_control_sensor_reading):
                self._update_next_control_pseudo_asynchronous_ticks(game_time)
                self.__send_hero_vehicle_data(self.pose_stream_for_control,
                                              timestamp, watermark_msg)
                self.__update_spectactor_pose()
                control_update = True

            # If we sent a localization message, but did not have a message
            # through control for this time, then we need to tick the simulator
            # forward. (only if we are in the pseudo-asynchronous mode)
            if self._flags.carla_mode == "pseudo-asynchronous" and (
                    localization_update and not control_update):
                self._consume_next_event()

    def _update_next_localization_pseudo_asynchronous_ticks(self, game_time):
        if self._flags.carla_localization_frequency > -1:
            self._next_localization_sensor_reading = (
                game_time +
                int(1000 / self._flags.carla_localization_frequency))
            if not self._simulator_in_sync:
                # If this is the first sensor reading, then tick
                # one more time because the second sensor reading
                # is sometimes delayed by 1 tick.
                self._next_localization_sensor_reading += int(
                    1000 / self._flags.carla_fps)
        else:
            self._next_localization_sensor_reading = (
                game_time + int(1000 / self._flags.carla_fps))
        heapq.heappush(
            self._tick_events,
            (self._next_localization_sensor_reading, TickEvent.SENSOR_READ))

    def _update_next_control_pseudo_asynchronous_ticks(self, game_time):
        if self._flags.carla_control_frequency > -1:
            self._next_control_sensor_reading = (
                game_time + int(1000 / self._flags.carla_control_frequency))
        else:
            self._next_control_sensor_reading = (
                game_time + int(1000 / self._flags.carla_fps))
        heapq.heappush(
            self._tick_events,
            (self._next_control_sensor_reading, TickEvent.SENSOR_READ))

    def run(self):
        self.__send_world_data()
        # Tick here once to ensure that the driver operators can get a handle
        # to the ego vehicle.
        # XXX(ionel): Hack to fix a race condition. Driver operators
        # register a carla listen callback only after they've received
        # the vehicle id value. We miss frames if we tick before
        # they register a listener. Thus, we sleep here a bit to
        # give them sufficient time to register a callback.
        time.sleep(4)
        self._tick_simulator()
        time.sleep(4)
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
        self._world.tick()

    def _tick_simulator_until(self, goal_time):
        while True:
            snapshot = self._world.get_snapshot()
            sim_time = int(snapshot.timestamp.elapsed_seconds * 1000)
            if sim_time < goal_time:
                self._world.tick()
            else:
                return

    def _apply_control_msg(self, msg):
        # Transform the message to a carla control cmd.
        vec_control = carla.VehicleControl(throttle=msg.throttle,
                                           steer=msg.steer,
                                           brake=msg.brake,
                                           hand_brake=msg.hand_brake,
                                           reverse=msg.reverse)
        self._ego_vehicle.apply_control(vec_control)

    def __send_hero_vehicle_data(self, stream, timestamp, watermark_msg):
        vec_transform = pylot.utils.Transform.from_carla_transform(
            self._ego_vehicle.get_transform())
        velocity_vector = pylot.utils.Vector3D.from_carla_vector(
            self._ego_vehicle.get_velocity())
        forward_speed = velocity_vector.magnitude()
        print("Forward speed is {}".format(forward_speed))
        pose = pylot.utils.Pose(vec_transform, forward_speed, velocity_vector)
        stream.send(erdos.Message(timestamp, pose))
        stream.send(erdos.WatermarkMessage(timestamp))

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
        if self._flags.carla_mode != 'pseudo-asynchronous':
            # Send top watermark to ensure that drivers send sensor data ASAP.
            self.release_sensor_stream.send(top_watermark)

    def __update_spectactor_pose(self):
        # Set the world simulation view with respect to the vehicle.
        v_pose = self._ego_vehicle.get_transform()
        v_pose.location -= 10 * carla.Location(v_pose.get_forward_vector())
        v_pose.location.z = 5
        self._spectator.set_transform(v_pose)


@total_ordering
class TickEvent(enum.Enum):
    CONTROL_CMD = 1
    SENSOR_READ = 2

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented
