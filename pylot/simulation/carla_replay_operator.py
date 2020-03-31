import erdos
import sys
import threading
import time

import pylot.simulation.utils
import pylot.utils
from pylot.perception.messages import ObstaclesMessage, SpeedSignsMessage, \
    StopSignsMessage, TrafficLightsMessage
from pylot.simulation.utils import extract_data_in_pylot_format, get_world


class CarlaReplayOperator(erdos.Operator):
    """ Replays a prior simulation from logs.

    The operator reads data from a log file, and publishes it on the ground
    streams.

    Attributes:
        _client: A connection to the simulator.
        _world: A handle to the world running inside the simulation.
    """

    def __init__(self, pose_stream, ground_traffic_lights_stream,
                 ground_obstacles_stream, ground_speed_limit_signs_stream,
                 ground_stop_signs_stream, vehicle_id_stream, flags):
        self._pose_stream = pose_stream
        self._ground_traffic_lights_stream = ground_traffic_lights_stream
        self._ground_obstacles_stream = ground_obstacles_stream
        self._ground_speed_limit_signs_stream = ground_speed_limit_signs_stream
        self._ground_stop_signs_stream = ground_stop_signs_stream
        self._vehicle_id_stream = vehicle_id_stream
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._client = None
        self._world = None
        # Lock to ensure that the callbacks do not execute simultaneously.
        self._lock = threading.Lock()

    @staticmethod
    def connect():
        pose_stream = erdos.WriteStream()
        ground_traffic_lights_stream = erdos.WriteStream()
        ground_obstacles_stream = erdos.WriteStream()
        ground_speed_limit_signs_stream = erdos.WriteStream()
        ground_stop_signs_stream = erdos.WriteStream()
        vehicle_id_stream = erdos.WriteStream()
        return [
            pose_stream, ground_traffic_lights_stream, ground_obstacles_stream,
            ground_speed_limit_signs_stream, ground_stop_signs_stream,
            vehicle_id_stream
        ]

    def on_world_tick(self, msg):
        """ Callback function that gets called when the world is ticked.
        This function sends a WatermarkMessage to the downstream operators as
        a signal that they need to release data to the rest of the pipeline.

        Args:
            msg: Data recieved from the simulation at a tick.
        """
        with self._lock:
            game_time = int(msg.elapsed_seconds * 1000)
            timestamp = erdos.Timestamp(coordinates=[game_time])
            watermark_msg = erdos.WatermarkMessage(timestamp)
            self.__publish_hero_vehicle_data(timestamp, watermark_msg)
            self.__publish_ground_actors_data(timestamp, watermark_msg)
            # XXX(ionel): Hack! Sleep a bit to not overlead the subscribers.
            time.sleep(0.2)

    def __publish_hero_vehicle_data(self, timestamp, watermark_msg):
        vec_transform = pylot.utils.Transform.from_carla_transform(
            self._driving_vehicle.get_transform())
        velocity_vector = pylot.utils.Vector3D.from_carla_vector(
            self._driving_vehicle.get_velocity())
        forward_speed = velocity_vector.magnitude()
        pose = pylot.utils.Pose(vec_transform, forward_speed, velocity_vector)
        self._pose_stream.send(erdos.Message(timestamp, pose))
        self._pose_stream.send(watermark_msg)

    def __publish_ground_actors_data(self, timestamp, watermark_msg):
        # Get all the actors in the simulation.
        actor_list = self._world.get_actors()

        (vehicles, people, traffic_lights, speed_limits,
         traffic_stops) = extract_data_in_pylot_format(actor_list)

        obstacles_msg = ObstaclesMessage(timestamp, vehicles + people)
        self._ground_obstacles_stream.send(obstacles_msg)
        self._ground_obstacles_stream.send(watermark_msg)
        traffic_lights_msg = TrafficLightsMessage(timestamp, traffic_lights)
        self._ground_traffic_lights_stream.send(traffic_lights_msg)
        self._ground_traffic_lights_stream.send(watermark_msg)
        speed_limit_signs_msg = SpeedSignsMessage(timestamp, speed_limits)
        self._ground_speed_limit_signs_stream.send(speed_limit_signs_msg)
        self._ground_speed_limit_signs_stream.send(watermark_msg)
        stop_signs_msg = StopSignsMessage(timestamp, traffic_stops)
        self._ground_stop_signs_stream.send(stop_signs_msg)
        self._ground_stop_signs_stream.send(watermark_msg)

    def run(self):
        # Connect to CARLA and retrieve the world running.
        self._client, self._world = get_world(self._flags.carla_host,
                                              self._flags.carla_port,
                                              self._flags.carla_timeout)
        if self._client is None or self._world is None:
            raise ValueError('There was an issue connecting to the simulator.')

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
        timestamp = erdos.Timestamp(coordinates=[sys.maxsize])
        vehicle_id_msg = erdos.Message(timestamp, self._driving_vehicle.id)
        self._vehicle_id_stream.send(vehicle_id_msg)
        self._vehicle_id_stream.send(erdos.WatermarkMessage(timestamp))
        self._world.on_tick(self.on_world_tick)
