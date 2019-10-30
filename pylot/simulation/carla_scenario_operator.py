import sys
import time

from erdos.op import Op
from erdos.utils import setup_logging, setup_csv_logging
from erdos.timestamp import Timestamp
from erdos.message import Message, WatermarkMessage

import pylot.utils
import pylot.simulation.utils
from pylot.simulation.carla_utils import get_world, set_synchronous_mode,\
        extract_data_in_pylot_format
import pylot.simulation.messages

import carla


class CarlaScenarioOperator(Op):
    def __init__(self, name, role_name, flags, log_file_name=None):
        super(CarlaScenarioOperator, self).__init__(name)
        self._flags = flags
        self._logger = setup_logging(self.name, log_file_name)

        # Connect to the simulator and retrieve the running world.
        self._client, self._world = get_world(self._flags.carla_host,
                                              self._flags.carla_port,
                                              self._flags.carla_timeout)
        if self._client is None or self._world is None:
            raise ValueError("There was an issue connecting to the simulator.")

        # Turn on the synchronous mode so we can control the simulation.
        set_synchronous_mode(self._world, self._flags.carla_fps)

        # Connect to the ego-vehicle spawned by the scenario runner.
        self._ego_vehicle = None
        while self._ego_vehicle is None:
            self._logger.info("Waiting for the scenario to be ready ...")
            time.sleep(1)
            self._ego_vehicle = CarlaScenarioOperator.retrieve_actor(
                self._world, 'vehicle.*', role_name)
            self._world.tick()
        self._logger.info("The ego vehicle with the identifier {} "
                          "was retrieved from the simulation.".format(
                              self._ego_vehicle.id))

    @staticmethod
    def setup_streams(input_streams):
        input_streams.filter(pylot.utils.is_control_stream).add_callback(
            CarlaScenarioOperator.on_control_msg)
        output_streams = [
            pylot.utils.create_can_bus_stream(),
            pylot.utils.create_ground_pedestrians_stream(),
            pylot.utils.create_vehicle_id_stream()
        ]
        return output_streams

    @staticmethod
    def retrieve_actor(world, bp_regex, role_name):
        """ Retrieves the actor from the world with the given blueprint and the
         role_name.

         Args:
             world: The instance of the simulator to retrieve the actors from.
             bp_regex: The blueprint of the actor to be retrieved from
                the simulator.
             role_name: The name of the actor to be retrieved.

         Returns:
             The actor retrieved from the given world with the role_name,
             if exists. Otherwise, returns None.
         """
        possible_actors = world.get_actors().filter(bp_regex)
        for actor in possible_actors:
            if actor.attributes['role_name'] == role_name:
                return actor
        return None

    def on_control_msg(self, msg):
        self._world.tick()

    def __publish_hero_vehicle_data(self, vehicle, timestamp, watermark_msg):
        vec_transform = pylot.simulation.utils.to_pylot_transform(
            vehicle.get_transform())
        forward_speed = pylot.simulation.utils.get_speed(
            vehicle.get_velocity())
        can_bus = pylot.simulation.utils.CanBus(vec_transform, forward_speed)
        self.get_output_stream('can_bus').send(Message(can_bus, timestamp))
        self.get_output_stream('can_bus').send(watermark_msg)

        # Set the world simulation view with respect to the vehicle.
        v_pose = vehicle.get_transform()
        v_pose.location -= 10 * carla.Location(v_pose.get_forward_vector())
        v_pose.location.z = 5
        self._world.get_spectator().set_transform(v_pose)

    def __publish_ground_pedestrians_data(self, timestamp, watermark_msg):
        # Get all the actors in the simulation.
        actor_list = self._world.get_actors()

        _, pedestrians, _, _, _ = extract_data_in_pylot_format(actor_list)

        pedestrians_msg = pylot.simulation.messages.GroundPedestriansMessage(
            pedestrians, timestamp)
        self.get_output_stream('pedestrians').send(pedestrians_msg)
        self.get_output_stream('pedestrians').send(watermark_msg)

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
        self.__publish_hero_vehicle_data(self._ego_vehicle, timestamp,
                                         watermark_msg)
        self.__publish_ground_pedestrians_data(timestamp, watermark_msg)

    def execute(self):
        # Send the vehicle id on the stream.
        timestamp = Timestamp(coordinates=[sys.maxint])
        vehicle_id_msg = Message(self._ego_vehicle.id, timestamp)
        self.get_output_stream('vehicle_id_stream').send(vehicle_id_msg)
        self.get_output_stream('vehicle_id_stream').send(
            WatermarkMessage(timestamp))

        # The dataflow graph needs to give enough time for downstream
        # operators to register callbacks, and register their assets before
        # ticking the simulation.
        time.sleep(20)
        self._world.tick()
        time.sleep(5)
        self._world.on_tick(self.publish_world_data)
        self._world.tick()
        self.spin()
