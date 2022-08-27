import carla

import erdos

from pylot.perception.detection.stop_sign import StopSign
from pylot.perception.messages import StopSignsMessage
import pylot.utils
from pylot.drivers.carla_base_gnss_driver_operator import (
    CarlaBaseGNSSDriverOperator)


class CarlaStopSignsDriverOperator(CarlaBaseGNSSDriverOperator):
    """Publishes the locations and values of all stop signs retrieved from the
    simulator at the provided frequency.
    
    Args:
        vehicle_id_stream: Stream on which the operator receives the ID of the
            ego vehicle. The ID is used to get a simulator handle to the
            vehicle.
        stop_signs_stream: Stream on which the operator sends the speed
            limit signs.
        frequency: Rate at which the pose is published, in Hertz.
        flags: Object used to access absl flags.
    """
    def __init__(self, vehicle_id_stream: erdos.ReadStream,
                 stop_signs_stream: erdos.WriteStream, frequency: float,
                 flags):
        transform = pylot.utils.Transform(pylot.utils.Location(),
                                          pylot.utils.Rotation())
        gnss_setup = pylot.drivers.sensor_setup.GNSSSetup(
            self.config.name, transform)
        super().__init__(vehicle_id_stream, stop_signs_stream, gnss_setup,
                         frequency, flags)
        self._stop_sign_actors = None

    def process_gnss(self, timestamp: erdos.Timestamp,
                     gnss_msg: carla.GnssMeasurement):
        """"""
        stop_signs = list(
            map(StopSign.from_simulator_actor, self._stop_sign_actors))
        self._output_stream.send(StopSignsMessage(timestamp, stop_signs))
        self._output_stream.send(erdos.WatermarkMessage(timestamp))

    def run(self):
        super().run()
        # Get speed limit actors
        self._stop_sign_actors = self._world.get_actors().filter(
            'traffic.stop')
