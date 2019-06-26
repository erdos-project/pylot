import sys
import threading
import time

# ERDOS specific imports.
from erdos.op import Op
from erdos.timestamp import Timestamp
from erdos.utils import setup_logging, setup_csv_logging
from erdos.message import Message, WatermarkMessage

import pylot.utils
from pylot.simulation.carla_utils import get_world, get_ground_data
import pylot.simulation.messages
from pylot.simulation.utils import to_erdos_transform
import pylot.simulation.utils


class CarlaReplayOperator(Op):
    def __init__(self,
                 name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        super(CarlaReplayOperator, self).__init__(name)
        self._flags = flags
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)

        # Connect to CARLA and retrieve the world running.
        self._client, self._world = get_world(self._flags.carla_host,
                                              self._flags.carla_port,
                                              self._flags.carla_timeout)
        if self._client is None or self._world is None:
            raise ValueError('There was an issue connecting to the simulator.')

        # Lock to ensure that the callbacks do not execute simultaneously.
        self._lock = threading.Lock()

    @staticmethod
    def setup_streams(input_streams):
        ground_agent_streams = [
            pylot.utils.create_can_bus_stream(),
            pylot.utils.create_ground_traffic_lights_stream(),
            pylot.utils.create_ground_vehicles_stream(),
            pylot.utils.create_ground_pedestrians_stream(),
            pylot.utils.create_ground_speed_limit_signs_stream(),
            pylot.utils.create_ground_stop_signs_stream()]
        return ground_agent_streams + [pylot.utils.create_vehicle_id_stream()]

    def on_world_tick(self, msg):
        """ Callback function that gets called when the world is ticked.
        This function sends a WatermarkMessage to the downstream operators as
        a signal that they need to release data to the rest of the pipeline.

        Args:
            msg: Data recieved from the simulation at a tick.
        """
        with self._lock:
            game_time = int(msg.elapsed_seconds * 1000)
            timestamp = Timestamp(coordinates=[game_time])
            watermark_msg = WatermarkMessage(timestamp)
            self.__publish_hero_vehicle_data(timestamp, watermark_msg)
            self.__publish_ground_actors_data(timestamp, watermark_msg)
            # XXX(ionel): Hack! Sleep a bit to not overlead the subscribers.
            time.sleep(0.2)

    def __publish_hero_vehicle_data(self, timestamp, watermark_msg):
        vec_transform = to_erdos_transform(
            self._driving_vehicle.get_transform())
        forward_speed = pylot.simulation.utils.get_speed(
            self._driving_vehicle.get_velocity())
        can_bus = pylot.simulation.utils.CanBus(vec_transform, forward_speed)
        self.get_output_stream('can_bus').send(
            Message(can_bus, timestamp))
        self.get_output_stream('can_bus').send(watermark_msg)

    def __publish_ground_actors_data(self, timestamp, watermark_msg):
        # Get all the actors in the simulation.
        actor_list = self._world.get_actors()

        (vehicles,
         pedestrians,
         traffic_lights,
         speed_limits,
         traffic_stops) = get_ground_data(actor_list)

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


    def execute(self):
        # Replayer time factor is only available in > 0.9.5.
        # self._client.set_replayer_time_factor(0.1)
        print(self._client.replay_file(self._flags.carla_replay_file,
                                       self._flags.carla_replay_start_time,
                                       self._flags.carla_replay_duration,
                                       self._flags.carla_replay_id))
        # Sleep a bit to allow the server to start the replay.
        time.sleep(1)
        self._driving_vehicle = self._world.get_actors().find(
            self._flags.carla_replay_id)
        if self._driving_vehicle is None:
            raise ValueError("There was an issue finding the vehicle.")
        # TODO(ionel): We do not currently have a top message.
        timestamp = Timestamp(coordinates=[sys.maxint])
        vehicle_id_msg = Message(self._driving_vehicle.id, timestamp)
        self.get_output_stream('vehicle_id_stream').send(vehicle_id_msg)
        self.get_output_stream('vehicle_id_stream').send(
            WatermarkMessage(timestamp))
        self._world.on_tick(self.on_world_tick)
        self.spin()
