import threading

import carla

import erdos

import pylot.utils
from pylot.perception.detection.traffic_light import TrafficLight
from pylot.perception.messages import TrafficLightsMessage
from pylot.drivers.carla_base_gnss_driver_operator import (
    CarlaBaseGNSSDriverOperator)


class CarlaTrafficLightsDriverOperator(CarlaBaseGNSSDriverOperator):
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
    def __init__(self, vehicle_id_stream: erdos.ReadStream,
                 traffic_lights_stream: erdos.WriteStream, frequency: float,
                 flags):
        transform = pylot.utils.Transform(pylot.utils.Location(),
                                          pylot.utils.Rotation())
        gnss_setup = pylot.drivers.sensor_setup.GNSSSetup(
            self.config.name, transform)
        super().__init__(vehicle_id_stream, traffic_lights_stream, gnss_setup,
                         frequency, flags)
        self._traffic_light_actors = None

    def process_gnss(self, timestamp: erdos.Timestamp,
                     gnss_msg: carla.GnssMeasurement):
        """Sends traffic light information followed by a watermark."""
        self._logger.debug(
            '@{}: sending traffic light information'.format(timestamp))
        traffic_lights = list(
            map(TrafficLight.from_simulator_actor, self._traffic_light_actors))
        self._output_stream.send(
            TrafficLightsMessage(timestamp, traffic_lights))
        self._output_stream.send(erdos.WatermarkMessage(timestamp))

    def run(self):
        super().run()
        # Get traffic light actors
        self._traffic_light_actors = self._world.get_actors().filter(
            'traffic.traffic_light*')
