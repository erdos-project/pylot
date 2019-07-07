import numpy as np
import time
import ray

from carla.client import CarlaClient
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings

from erdos.message import Message, WatermarkMessage
from erdos.op import Op
from erdos.timestamp import Timestamp
from erdos.utils import setup_csv_logging, setup_logging, time_epoch_ms

from pylot.perception.detection.utils import TrafficLightColor
from pylot.perception.messages import SegmentedFrameMessage
import pylot.utils
import pylot.simulation.messages
from pylot.simulation.utils import depth_to_array, labels_to_array, to_bgra_array
import pylot.simulation.utils


class CarlaLegacyOperator(Op):
    """ CarlaLegacyOperator initializes and controls the simulator.

    This operator connects to the simulator, spawns actors, gets and publishes
    ground info, and sends vehicle commands. The operator works with
    Carla 0.8.4.
    """
    def __init__(self,
                 name,
                 flags,
                 auto_pilot,
                 camera_setups=[],
                 lidar_setups=[],
                 log_file_name=None,
                 csv_file_name=None):
        super(CarlaLegacyOperator, self).__init__(name)
        self._flags = flags
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._auto_pilot = auto_pilot
        if self._flags.carla_high_quality:
            quality = 'Epic'
        else:
            quality = 'Low'
        self._settings = CarlaSettings()
        self._settings.set(
            SynchronousMode=self._flags.carla_synchronous_mode,
            SendNonPlayerAgentsInfo=True,
            NumberOfVehicles=self._flags.carla_num_vehicles,
            NumberOfPedestrians=self._flags.carla_num_pedestrians,
            WeatherId=self._flags.carla_weather,
            QualityLevel=quality)
        self._settings.randomize_seeds()
        self._transforms = {}
        # Add cameras to the simulation.
        for cs in camera_setups:
            self.__add_camera(cs)
            self._transforms[cs.name] = cs.get_transform()
        # Add lidars to the simulation.
        for ls in lidar_setups:
            self.__add_lidar(ls)
            self._transforms[ls.name] = ls.get_transform()
        self.agent_id_map = {}
        self.pedestrian_count = 0

        # Initialize the control state.
        self.control = {
            'steer': 0.0,
            'throttle': 0.0,
            'brake': 0.0,
            'hand_brake': False,
            'reverse': False
        }
        # Register custom serializers for Messages and WatermarkMessages
        ray.register_custom_serializer(Message, use_pickle=True)
        ray.register_custom_serializer(WatermarkMessage, use_pickle=True)

    @staticmethod
    def setup_streams(input_streams, camera_setups, lidar_setups):
        input_streams.add_callback(CarlaLegacyOperator.on_control_msg)
        camera_streams = [pylot.utils.create_camera_stream(cs)
                          for cs in camera_setups]
        lidar_streams = [pylot.utils.create_lidar_stream(ls)
                         for ls in lidar_setups]
        return [
            pylot.utils.create_can_bus_stream(),
            pylot.utils.create_ground_traffic_lights_stream(),
            pylot.utils.create_ground_vehicles_stream(),
            pylot.utils.create_ground_pedestrians_stream(),
            pylot.utils.create_ground_speed_limit_signs_stream(),
            pylot.utils.create_ground_stop_signs_stream()
        ] + camera_streams + lidar_streams

    def __add_camera(self, camera_setup):
        """Adds a camera and a corresponding output stream.

        Args:
            camera_setup: A camera setup object.
        """
        # Transform from Carla 0.9.x postprocessing strings to Carla 0.8.4.
        if camera_setup.camera_type == 'sensor.camera.rgb':
            postprocessing = 'SceneFinal'
        elif camera_setup.camera_type == 'sensor.camera.depth':
            postprocessing = 'Depth'
        elif camera_setup.camera_type == 'sensor.camera.semantic_segmentation':
            postprocessing = 'SemanticSegmentation'
        transform = camera_setup.get_transform()
        camera = Camera(
            name=camera_setup.name,
            PostProcessing=postprocessing,
            FOV=camera_setup.fov,
            ImageSizeX=camera_setup.width,
            ImageSizeY=camera_setup.height,
            PositionX=transform.location.x,
            PositionY=transform.location.y,
            PositionZ=transform.location.z,
            RotationPitch=transform.rotation.pitch,
            RotationRoll=transform.rotation.roll,
            RotationYaw=transform.rotation.yaw)

        self._settings.add_sensor(camera)

    def __add_lidar(self, lidar_setup):
        """Adds a LIDAR sensor and a corresponding output stream.

        Args:
            lidar_setup: A LidarSetup object..
        """
        transform = lidar_setup.get_transform()
        lidar = Lidar(
            lidar_setup.name,
            Channels=lidar_setup.channels,
            Range=lidar_setup.range,
            PointsPerSecond=lidar_setup.points_per_second,
            RotationFrequency=lidar_setup.rotation_frequency,
            UpperFovLimit=lidar_setup.upper_fov,
            LowerFovLimit=lidar_setup.lower_fov,
            PositionX=transform.location.x,
            PositionY=transform.location.y,
            PositionZ=transform.location.z,
            RotationPitch=transform.rotation.pitch,
            RotationYaw=transform.rotation.yaw,
            RotationRoll=transform.rotation.roll)

        self._settings.add_sensor(lidar)

    def publish_world_data(self):
        read_start_time = time.time()
        measurements, sensor_data = self.client.read_data()
        measure_time = time.time()

        self._logger.info(
            'Got readings for game time {} and platform time {}'.format(
                measurements.game_timestamp, measurements.platform_timestamp))

        timestamp = Timestamp(
            coordinates=[measurements.game_timestamp])
        watermark = WatermarkMessage(timestamp)

        # Send player data on data streams.
        self.__send_player_data(
            measurements.player_measurements, timestamp, watermark)
        # Extract agent data from measurements.
        agents = self.__get_ground_agents(measurements)
        # Send agent data on data streams.
        self.__send_ground_agent_data(agents, timestamp, watermark)
        # Send sensor data on data stream.
        self.__send_sensor_data(sensor_data, timestamp, watermark)
        # Get Autopilot control data.
        if self._auto_pilot:
            self.control = measurements.player_measurements.autopilot_control
        end_time = time.time()
        measurement_runtime = (measure_time - read_start_time) * 1000
        total_runtime = (end_time - read_start_time) * 1000
        self._logger.info('Carla measurement time {}; total time {}'.format(
            measurement_runtime, total_runtime))
        self._csv_logger.info('{},{},{},{}'.format(
            time_epoch_ms(), self.name, measurement_runtime, total_runtime))

    def __send_player_data(self, player_measurements, timestamp, watermark):
        """ Sends hero vehicle information.

        It populates a CanBus tuple.
        """
        location = pylot.simulation.utils.Location(
            carla_loc=player_measurements.transform.location)
        rotation = pylot.simulation.utils.Rotation(
            player_measurements.transform.rotation.pitch,
            player_measurements.transform.rotation.yaw,
            player_measurements.transform.rotation.roll)
        orientation = pylot.simulation.utils.Orientation(
            player_measurements.transform.orientation.x,
            player_measurements.transform.orientation.y,
            player_measurements.transform.orientation.z)
        vehicle_transform = pylot.simulation.utils.Transform(
            location, rotation, orientation=orientation)
        forward_speed = player_measurements.forward_speed * 3.6
        can_bus = pylot.simulation.utils.CanBus(
            vehicle_transform, forward_speed)
        self.get_output_stream('can_bus').send(Message(can_bus, timestamp))
        self.get_output_stream('can_bus').send(watermark)

    def __get_ground_agents(self, measurements):
        vehicles = []
        pedestrians = []
        traffic_lights = []
        speed_limit_signs = []
        for agent in measurements.non_player_agents:
            if agent.HasField('vehicle'):
                transform = pylot.simulation.utils.to_pylot_transform(
                    agent.vehicle.transform)
                bb = pylot.simulation.utils.BoundingBox(
                    agent.vehicle.bounding_box)
                forward_speed = agent.vehicle.forward_speed
                vehicle = pylot.simulation.utils.Vehicle(
                    transform, bb, forward_speed)
                vehicles.append(vehicle)
            elif agent.HasField('pedestrian'):
                if not self.agent_id_map.get(agent.id):
                    self.pedestrian_count += 1
                    self.agent_id_map[agent.id] = self.pedestrian_count
                pedestrian_index = self.agent_id_map[agent.id]
                transform = pylot.simulation.utils.to_pylot_transform(
                    agent.pedestrian.transform)
                bb = pylot.simulation.utils.BoundingBox(
                    agent.pedestrian.bounding_box)
                forward_speed = agent.pedestrian.forward_speed
                pedestrian = pylot.simulation.utils.Pedestrian(
                    pedestrian_index, transform, bb, forward_speed)
                pedestrians.append(pedestrian)
            elif agent.HasField('traffic_light'):
                transform = pylot.simulation.utils.to_pylot_transform(
                    agent.traffic_light.transform)
                if agent.traffic_light.state == 2:
                    erdos_tl_state = TrafficLightColor.RED
                elif agent.traffic_light.state == 1:
                    erdos_tl_state = TrafficLightColor.YELLOW
                elif agent.traffic_light.state == 0:
                    erdos_tl_state = TrafficLightColor.GREEN
                else:
                    erdos_tl_state = TrafficLightColor.OFF
                traffic_light = pylot.simulation.utils.TrafficLight(
                    transform, erdos_tl_state)
                traffic_lights.append(traffic_light)
            elif agent.HasField('speed_limit_sign'):
                transform = pylot.simulation.utils.to_pylot_transform(
                    agent.speed_limit_sign.transform)
                speed_sign = pylot.simulation.utils.SpeedLimitSign(
                    transform, agent.speed_limit_sign.speed_limit)
                speed_limit_signs.append(speed_sign)

        return vehicles, pedestrians, traffic_lights, speed_limit_signs

    def __send_ground_agent_data(self, agents, timestamp, watermark):
        vehicles, pedestrians, traffic_lights, speed_limit_signs = agents
        vehicles_msg = pylot.simulation.messages.GroundVehiclesMessage(
            vehicles, timestamp)
        self.get_output_stream('vehicles').send(vehicles_msg)
        self.get_output_stream('vehicles').send(watermark)
        pedestrians_msg = pylot.simulation.messages.GroundPedestriansMessage(
            pedestrians, timestamp)
        self.get_output_stream('pedestrians').send(pedestrians_msg)
        self.get_output_stream('pedestrians').send(watermark)
        traffic_lights_msg = pylot.simulation.messages.GroundTrafficLightsMessage(
            traffic_lights, timestamp)
        self.get_output_stream('traffic_lights').send(traffic_lights_msg)
        self.get_output_stream('traffic_lights').send(watermark)
        speed_limits_msg = pylot.simulation.messages.GroundSpeedSignsMessage(
            speed_limit_signs, timestamp)
        self.get_output_stream('speed_limit_signs').send(speed_limits_msg)
        self.get_output_stream('speed_limit_signs').send(watermark)
        # We do not have any stop signs.
        stop_signs_msg = pylot.simulation.messages.GroundStopSignsMessage(
            [], timestamp)
        self.get_output_stream('stop_signs').send(stop_signs_msg)
        self.get_output_stream('stop_signs').send(watermark)

    def __send_sensor_data(self, sensor_data, timestamp, watermark):
        for name, measurement in sensor_data.items():
            data_stream = self.get_output_stream(name)
            if data_stream.get_label('camera_type') == 'sensor.camera.rgb':
                # Transform the Carla RGB images to BGR.
                data_stream.send(
                    pylot.simulation.messages.FrameMessage(
                        pylot.utils.bgra_to_bgr(to_bgra_array(measurement)), timestamp))
            elif data_stream.get_label('camera_type') == 'sensor.camera.semantic_segmentation':
                frame = labels_to_array(measurement)
                data_stream.send(SegmentedFrameMessage(frame, 0, timestamp))
            elif data_stream.get_label('camera_type') == 'sensor.camera.depth':
                # NOTE: depth_to_array flips the image.
                data_stream.send(
                    pylot.simulation.messages.DepthFrameMessage(
                        depth_to_array(measurement),
                        self._transforms[name],
                        measurement.fov,
                        timestamp))
            elif data_stream.get_label('sensor_type') == 'sensor.lidar.ray_cast':
                pc_msg = pylot.simulation.messages.PointCloudMessage(
                    measurement.data, self._transforms[name], timestamp)
                data_stream.send(pc_msg)
            else:
                data_stream.send(Message(measurement, timestamp))
            data_stream.send(watermark)

    def _tick_simulator(self):
        if (not self._flags.carla_synchronous_mode or
            self._flags.carla_step_frequency == -1):
            # Run as fast as possible.
            self.publish_world_data()
            return
        time_until_tick = self._tick_at - time.time()
        if time_until_tick > 0:
            time.sleep(time_until_tick)
        else:
            self._logger.error(
                'Cannot tick Carla at frequency {}'.format(
                    self._flags.carla_step_frequency))
        self._tick_at += 1.0 / self._flags.carla_step_frequency
        self.publish_world_data()

    def on_control_msg(self, msg):
        """ Invoked when a ControlMessage is received.

        Args:
            msg: A control.messages.ControlMessage message.
        """
        if not self._auto_pilot:
            # Update the control dict state.
            self.control['steer'] = msg.steer
            self.control['throttle'] = msg.throttle
            self.control['brake'] = msg.brake
            self.control['hand_brake'] = msg.hand_brake
            self.control['reverse'] = msg.reverse
            self.client.send_control(**self.control)
        else:
            self.client.send_control(self.control)
        self._tick_simulator()

    def execute(self):
        # Connect to the simulator.
        self.client = CarlaClient(self._flags.carla_host,
                                  self._flags.carla_port,
                                  timeout=self._flags.carla_timeout)
        self.client.connect()
        scene = self.client.load_settings(self._settings)
        # Choose one player start at random.
        number_of_player_starts = len(scene.player_start_spots)
        player_start = self._flags.carla_start_player_num
        if self._flags.carla_random_player_start:
            player_start = np.random.randint(
                0, max(0, number_of_player_starts - 1))

        self.client.start_episode(player_start)
        self._tick_at = time.time()
        self._tick_simulator()
        self.spin()
