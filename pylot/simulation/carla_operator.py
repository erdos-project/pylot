import enum
import heapq
import random
import threading
import time
from functools import total_ordering

from carla import Location, VehicleControl, command

import erdos
from erdos import ReadStream, Timestamp, WriteStream

import pylot.simulation.utils
import pylot.utils
from pylot.control.messages import ControlMessage
from pylot.perception.messages import ObstaclesMessage, SpeedSignsMessage, \
    StopSignsMessage, TrafficLightsMessage


class CarlaOperator(erdos.Operator):
    """Initializes and controls a CARLA simulation.

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
    def __init__(
            self, control_stream: ReadStream,
            release_sensor_stream: ReadStream,
            pipeline_finish_notify_stream: ReadStream,
            pose_stream: WriteStream, pose_stream_for_control: WriteStream,
            ground_traffic_lights_stream: WriteStream,
            ground_obstacles_stream: WriteStream,
            ground_speed_limit_signs_stream: WriteStream,
            ground_stop_signs_stream: WriteStream,
            vehicle_id_stream: WriteStream, open_drive_stream: WriteStream,
            global_trajectory_stream: WriteStream, flags):
        if flags.random_seed:
            random.seed(flags.random_seed)
        # Register callback on control stream.
        control_stream.add_callback(self.on_control_msg)
        erdos.add_watermark_callback([release_sensor_stream], [],
                                     self.on_sensor_ready)
        if flags.simulator_mode == "pseudo-asynchronous":
            erdos.add_watermark_callback([pipeline_finish_notify_stream], [],
                                         self.on_pipeline_finish)
        self.pose_stream = pose_stream
        self.pose_stream_for_control = pose_stream_for_control
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
        self._csv_logger = erdos.utils.setup_csv_logging(
            self.config.name + '-csv', self.config.csv_log_file_name)
        # Connect to simulator and retrieve the world running.
        self._client, self._world = pylot.simulation.utils.get_world(
            self._flags.simulator_host, self._flags.simulator_port,
            self._flags.simulator_timeout)
        self._simulator_version = self._client.get_client_version()

        if not self._flags.scenario_runner and \
                self._flags.control != "manual":
            # Load the appropriate town.
            self._initialize_world()

        # Save the spectator handle so that we don't have to repeteadly get the
        # handle (which is slow).
        self._spectator = self._world.get_spectator()

        if pylot.simulation.utils.check_simulator_version(
                self._simulator_version, required_minor=9, required_patch=8):
            # Any simulator version after 0.9.7.
            # Create a traffic manager to that auto pilot works.
            self._traffic_manager = self._client.get_trafficmanager(
                self._flags.carla_traffic_manager_port)
            self._traffic_manager.set_synchronous_mode(
                self._flags.simulator_mode == 'synchronous')

        if self._flags.scenario_runner:
            # Tick until 4.0 seconds time so that all synchronous scenario runs
            # start at exactly the same game time.
            pylot.simulation.utils.set_synchronous_mode(self._world, 1000)
            self._tick_simulator_until(4000)

        pylot.simulation.utils.set_simulation_mode(self._world, self._flags)

        if self._flags.scenario_runner or self._flags.control == "manual":
            # Wait until the ego vehicle is spawned by the scenario runner.
            self._logger.info("Waiting for the scenario to be ready ...")
            self._ego_vehicle = pylot.simulation.utils.wait_for_ego_vehicle(
                self._world)
            self._logger.info("Found ego vehicle")
        else:
            # Spawn ego vehicle, people and vehicles.
            (self._ego_vehicle, self._vehicle_ids,
             self._people) = pylot.simulation.utils.spawn_actors(
                 self._client, self._world,
                 self._flags.carla_traffic_manager_port,
                 self._simulator_version,
                 self._flags.simulator_spawn_point_index,
                 self._flags.control == 'simulator_auto_pilot',
                 self._flags.simulator_num_people,
                 self._flags.simulator_num_vehicles, self._logger)

        pylot.simulation.utils.set_vehicle_physics(
            self._ego_vehicle, self._flags.simulator_vehicle_moi,
            self._flags.simulator_vehicle_mass)

        # Lock used to ensure that simulator callbacks are not executed
        # concurrently.
        self._lock = threading.Lock()

        # Dictionary that stores the processing times when sensors are ready
        # to realease data. This info is used to calculate the real processing
        # time of our pipeline without including simulator-induced sensor
        # delays.
        self._next_localization_sensor_reading = None
        self._next_control_sensor_reading = None
        self._simulator_in_sync = False
        self._tick_events = []
        self._control_msgs = {}

    @staticmethod
    def connect(control_stream: ReadStream, release_sensor_stream: ReadStream,
                pipeline_finish_notify_stream: ReadStream):
        pose_stream = erdos.WriteStream()
        pose_stream_for_control = erdos.WriteStream()
        ground_traffic_lights_stream = erdos.WriteStream()
        ground_obstacles_stream = erdos.WriteStream()
        ground_speed_limit_signs_stream = erdos.WriteStream()
        ground_stop_signs_stream = erdos.WriteStream()
        vehicle_id_stream = erdos.WriteStream()
        open_drive_stream = erdos.WriteStream()
        global_trajectory_stream = erdos.WriteStream()
        return [
            pose_stream,
            pose_stream_for_control,
            ground_traffic_lights_stream,
            ground_obstacles_stream,
            ground_speed_limit_signs_stream,
            ground_stop_signs_stream,
            vehicle_id_stream,
            open_drive_stream,
            global_trajectory_stream,
        ]

    @erdos.profile_method()
    def on_control_msg(self, msg: ControlMessage):
        """ Invoked when a ControlMessage is received.

        Args:
            msg: A control.messages.ControlMessage message.
        """
        self._logger.debug('@{}: received control message'.format(
            msg.timestamp))
        if self._flags.simulator_mode == 'pseudo-asynchronous':
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
            if self._flags.control not in ['simulator_auto_pilot', 'manual']:
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

    def on_pipeline_finish(self, timestamp: Timestamp):
        self._logger.debug("@{}: Received pipeline finish.".format(timestamp))
        game_time = timestamp.coordinates[0]
        if (self._flags.simulator_control_frequency == -1
                or self._next_control_sensor_reading is None
                or game_time == self._next_control_sensor_reading):
            # There was supposed to be a control message for this timestamp
            # too. Send the Pose message and continue after the control message
            # is received.
            self._update_next_control_pseudo_asynchronous_ticks(game_time)
            self.__send_hero_vehicle_data(self.pose_stream_for_control,
                                          timestamp)
            self.__update_spectactor_pose()
        else:
            # No pose message was supposed to be sent for this timestamp, we
            # need to consume the next event to move the dataflow forward.
            self._consume_next_event()

    def on_sensor_ready(self, timestamp: Timestamp):
        # The first sensor reading needs to be discarded because it might
        # not be correctly spaced out.
        if not self._simulator_in_sync:
            self._simulator_in_sync = True

    def send_actor_data(self, msg):
        """ Callback function that gets called when the world is ticked.
        This function sends a WatermarkMessage to the downstream operators as
        a signal that they need to release data to the rest of the pipeline.

        Args:
            msg: Data recieved from the simulation at a tick.
        """
        # Ensure that the callback executes serially.
        with self._lock:
            game_time = int(msg.elapsed_seconds * 1000)
            self._logger.info(
                'The world is at the timestamp {}'.format(game_time))
            timestamp = erdos.Timestamp(coordinates=[game_time])
            with erdos.profile(self.config.name + '.send_actor_data',
                               self,
                               event_data={'timestamp': str(timestamp)}):
                if (self._flags.simulator_localization_frequency == -1
                        or self._next_localization_sensor_reading is None or
                        game_time == self._next_localization_sensor_reading):
                    if self._flags.simulator_mode == 'pseudo-asynchronous':
                        self._update_next_localization_pseudo_async_ticks(
                            game_time)
                    self.__send_hero_vehicle_data(self.pose_stream, timestamp)
                    self.__send_ground_actors_data(timestamp)
                    self.__update_spectactor_pose()

                if self._flags.simulator_mode == "pseudo-asynchronous" and (
                        self._flags.simulator_control_frequency == -1
                        or self._next_control_sensor_reading is None
                        or game_time == self._next_control_sensor_reading):
                    self._update_next_control_pseudo_asynchronous_ticks(
                        game_time)
                    self.__send_hero_vehicle_data(self.pose_stream_for_control,
                                                  timestamp)
                    self.__update_spectactor_pose()

    def _update_next_localization_pseudo_async_ticks(self, game_time: int):
        if self._flags.simulator_localization_frequency > -1:
            self._next_localization_sensor_reading = (
                game_time +
                int(1000 / self._flags.simulator_localization_frequency))
            if not self._simulator_in_sync:
                # If this is the first sensor reading, then tick
                # one more time because the second sensor reading
                # is sometimes delayed by 1 tick.
                self._next_localization_sensor_reading += int(
                    1000 / self._flags.simulator_fps)
        else:
            self._next_localization_sensor_reading = (
                game_time + int(1000 / self._flags.simulator_fps))
        heapq.heappush(
            self._tick_events,
            (self._next_localization_sensor_reading, TickEvent.SENSOR_READ))

    def _update_next_control_pseudo_asynchronous_ticks(self, game_time: int):
        if self._flags.simulator_control_frequency > -1:
            self._next_control_sensor_reading = (
                game_time +
                int(1000 / self._flags.simulator_control_frequency))
        else:
            self._next_control_sensor_reading = (
                game_time + int(1000 / self._flags.simulator_fps))
        if (self._next_control_sensor_reading !=
                self._next_localization_sensor_reading):
            heapq.heappush(
                self._tick_events,
                (self._next_control_sensor_reading, TickEvent.SENSOR_READ))

    def run(self):
        self.__send_world_data()
        # Tick here once to ensure that the driver operators can get a handle
        # to the ego vehicle.
        # XXX(ionel): Hack to fix a race condition. Driver operators
        # register a simulator listen callback only after they've received
        # the vehicle id value. We miss frames if we tick before
        # they register a listener. Thus, we sleep here a bit to
        # give them sufficient time to register a callback.
        time.sleep(4)
        self._tick_simulator()
        time.sleep(4)
        # The older CARLA versions require an additional tick to sync
        # sensors.
        self._world.on_tick(self.send_actor_data)
        self._tick_simulator()

    def _initialize_world(self):
        """ Setups the world town, and activates the desired weather."""
        if self._simulator_version == '0.9.5':
            # TODO (Sukrit) :: ERDOS provides no way to retrieve handles to the
            # class objects to do garbage collection. Hence, objects from
            # previous runs of the simulation may persist. We need to clean
            # them up right now. In future, move this logic to a seperate
            # destroy function.
            pylot.simulation.utils.reset_world(self._world)
        else:
            self._world = self._client.load_world('Town{:02d}'.format(
                self._flags.simulator_town))
        self._logger.info('Setting the weather to {}'.format(
            self._flags.simulator_weather))
        pylot.simulation.utils.set_weather(self._world,
                                           self._flags.simulator_weather)

    def _tick_simulator(self):
        if (self._flags.simulator_mode == 'asynchronous-fixed-time-step'
                or self._flags.simulator_mode == 'asynchronous'):
            # No need to tick when running in these modes.
            return
        self._world.tick()

    def _tick_simulator_until(self, goal_time: int):
        while True:
            snapshot = self._world.get_snapshot()
            sim_time = int(snapshot.timestamp.elapsed_seconds * 1000)
            if sim_time < goal_time:
                self._world.tick()
            else:
                return

    def _apply_control_msg(self, msg: ControlMessage):
        # Transform the message to a simulator control cmd.
        vec_control = VehicleControl(throttle=msg.throttle,
                                     steer=msg.steer,
                                     brake=msg.brake,
                                     hand_brake=msg.hand_brake,
                                     reverse=msg.reverse)
        self._client.apply_batch_sync(
            [command.ApplyVehicleControl(self._ego_vehicle.id, vec_control)])

    def __send_hero_vehicle_data(self, stream: WriteStream,
                                 timestamp: Timestamp):
        vec_transform = pylot.utils.Transform.from_simulator_transform(
            self._ego_vehicle.get_transform())
        velocity_vector = pylot.utils.Vector3D.from_simulator_vector(
            self._ego_vehicle.get_velocity())
        forward_speed = velocity_vector.magnitude()
        pose = pylot.utils.Pose(vec_transform, forward_speed, velocity_vector,
                                timestamp.coordinates[0])
        stream.send(erdos.Message(timestamp, pose))
        stream.send(erdos.WatermarkMessage(timestamp))

    def __send_ground_actors_data(self, timestamp: Timestamp):
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
        # Send the id of the ego vehicle. This id is used by the driver
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
        v_pose.location -= 10 * Location(v_pose.get_forward_vector())
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
