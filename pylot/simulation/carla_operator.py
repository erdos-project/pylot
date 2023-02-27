import random
import threading
import time

from carla import Location, VehicleControl, command

import erdos
from erdos import ReadStream, Timestamp, WriteStream

import pylot.simulation.utils
import pylot.utils
from pylot.control.messages import ControlMessage


class CarlaOperator(erdos.Operator):
    """Initializes and controls a CARLA simulation.

    This operator connects to the simulation, sets the required weather in the
    simulation world, initializes the required number of actors, and the
    vehicle that the rest of the pipeline drives.

    Args:
        control_stream: Stream on which the operator receives control messages
            to apply to the ego vehicle.
        pipeline_finish_notify_stream: In pseudo-async mode, notifies the
            operator that the pipeline has finished executing.
        flags: A handle to the global flags instance to retrieve the
            configuration.

    Attributes:
        _client: A connection to the simulator.
        _world: A handle to the world running inside the simulation.
        _vehicles: A list of identifiers of the vehicles inside the simulation.
    """
    def __init__(self, control_stream: ReadStream,
                 pipeline_finish_notify_stream: ReadStream,
                 vehicle_id_stream: WriteStream, flags):
        if flags.random_seed:
            random.seed(flags.random_seed)
        # Register callback on control stream.
        control_stream.add_callback(self.on_control_msg)
        if flags.simulator_mode == "pseudo-asynchronous":
            erdos.add_watermark_callback([pipeline_finish_notify_stream], [],
                                         self.on_pipeline_finish)

        self.vehicle_id_stream = vehicle_id_stream

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

    @staticmethod
    def connect(control_stream: ReadStream,
                pipeline_finish_notify_stream: ReadStream):
        vehicle_id_stream = erdos.WriteStream()
        return [vehicle_id_stream]

    @erdos.profile_method()
    def on_control_msg(self, msg: ControlMessage):
        """ Invoked when a ControlMessage is received.

        Args:
            msg: A control.messages.ControlMessage message.
        """
        self._logger.debug('@{}: received control message {}'.format(
            msg.timestamp, msg))
        if self._flags.control not in ['simulator_auto_pilot', 'manual']:
            self._apply_control_msg(msg)
        # If auto pilot or manual mode is enabled then we do not apply the
        # control, but we still want to tick in this method to ensure that
        # all operators finished work before the world ticks.
        if not self._flags.simulator_mode == 'pseudo-asynchronous':
            self._tick_simulator()
        if self._flags.simulator_mode == 'pseudo-asynchronous':
            # Tick the world after the operator received a control command.
            # This usually indicates that all the operators have completed
            # processing the previous timestamp (with the exception of logging
            # operators that are not part of the main loop).
            self._apply_control_msg(msg)

    def on_pipeline_finish(self, timestamp: Timestamp):
        """Only invoked in pseudo-asynchronous mode."""
        self._logger.debug("@{}: Received pipeline finish.".format(timestamp))
        # Tick simulator forward until the next expected control message.
        game_time = timestamp.coordinates[0]
        if self._flags.simulator_control_frequency > -1:
            control_frequency = self._flags.simulator_control_frequency
        else:
            control_frequency = self._flags.simulator_fps
        next_control_msg_time = game_time + round(1000 / control_frequency)
        self._tick_simulator_until(next_control_msg_time)

    def run(self):
        self.vehicle_id_stream.send(
            erdos.Message(erdos.Timestamp(coordinates=[0]),
                          self._ego_vehicle.id))
        self.vehicle_id_stream.send(
            erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))
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
        self._tick_simulator()

    def _initialize_world(self):
        """Sets up the world town, and activates the desired weather."""
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
        self._logger.debug("ticking simulator until {}".format(goal_time))
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
        self.__update_spectactor_pose()

    def __update_spectactor_pose(self):
        # Set the world simulation view with respect to the vehicle.
        v_pose = self._ego_vehicle.get_transform()
        v_pose.location -= 10 * Location(v_pose.get_forward_vector())
        v_pose.location.z = 5
        self._spectator.set_transform(v_pose)
