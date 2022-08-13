import threading

import erdos
from erdos import ReadStream, Timestamp, WriteStream

from pylot.simulation.utils import get_vehicle_handle, get_world, \
    set_simulation_mode
import pylot.utils
from pylot.perception.detection.traffic_light import TrafficLight
from pylot.perception.messages import TrafficLightsMessage


class CarlaTrafficLightsDriverOperator(erdos.Operator):
    """Publishes the location and state of all traffic lights.
    
    This operator attaches to the vehicle using the vehicle ID provided by the
    ``vehicle_id_stream``, registers callback functions to retrieve the
    state of the traffic lights, and publishes them to downstream operators.
    
    Args:
        vehicle_id_stream: Stream on which the operator receives the ID of the
            ego vehicle. The ID is used to get a simulator handle to the
            vehicle.
        traffic_lights_stream: Stream on which the operator sends the status of
            all traffic lights.
        frequency: Rate at which the pose is published, in Hertz.
        flags: Object used to access absl flags.
    """
    def __init__(self, vehicle_id_stream: ReadStream,
                 traffic_lights_stream: WriteStream, frequency: float, flags):
        # Save the streams.
        self._vehicle_id_stream = vehicle_id_stream
        self._traffic_lights_stream = traffic_lights_stream

        # Save the flags and initialize logging.
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)

        # Save the setup, the vehicle and the sensor.
        self._frequency = frequency
        self._traffic_light_actors = []
        self._gnss = None  # Use GNSS sensor to tick at a regular interval.
        self._lock = threading.Lock()

    @staticmethod
    def connect(vehicle_id_stream: ReadStream):
        traffic_lights_stream = WriteStream()
        return [traffic_lights_stream]

    def send_traffic_lights(self, timestamp: Timestamp):
        """Sends pose information followed by a watermark."""
        traffic_lights = list(
            map(TrafficLight.from_simulator_actor, self._traffic_light_actors))
        self._traffic_lights_stream.send(
            TrafficLightsMessage(timestamp, traffic_lights))
        self._traffic_lights_stream.send(erdos.WatermarkMessage(timestamp))

    def process_gnss(self, gnss_msg):
        """Callback attached to a GNSS sensor to ensure that traffic lights are
        sent at the provided frequency."""
        game_time = int(gnss_msg.timestamp * 1000)
        timestamp = erdos.Timestamp(coordinates=[game_time])
        with erdos.profile(self.config.name + '.process_gnss',
                           self,
                           event_data={'timestamp': str(timestamp)}):
            with self._lock:
                self.send_traffic_lights(timestamp)

    def run(self):
        # Read the vehicle ID from the vehicle ID stream.
        vehicle_id = self._vehicle_id_stream.read().data
        self._logger.debug("{} received vehicle id: {}".format(
            self.config.name, vehicle_id))

        # Connect to the world.
        _, world = get_world(self._flags.simulator_host,
                             self._flags.simulator_port,
                             self._flags.simulator_timeout)
        set_simulation_mode(world, self._flags)

        # Get traffic light actors
        self._traffic_light_actors = world.get_actors().filter(
            'traffic.traffic_light*')

        # Retrieve the vehicle and install the GNSS sensor.
        vehicle = get_vehicle_handle(world, vehicle_id)
        gnss_blueprint = world.get_blueprint_library().find(
            'sensor.other.gnss')

        if self._frequency == -1:
            gnss_blueprint.set_attribute('sensor_tick', '0.0')
        else:
            gnss_blueprint.set_attribute('sensor_tick',
                                         str(1.0 / self._frequency))
        transform = pylot.utils.Transform(
            pylot.utils.Location(),
            pylot.utils.Rotation()).as_simulator_transform()
        self._logger.debug("Spawning a GNSS sensor")
        self._gnss = world.spawn_actor(gnss_blueprint,
                                       transform,
                                       attach_to=vehicle)

        # Register the callback on the GNSS sensor.
        self._gnss.listen(self.process_gnss)
