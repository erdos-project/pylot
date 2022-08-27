import carla

import erdos

from pylot.perception.detection.speed_limit_sign import SpeedLimitSign
from pylot.perception.messages import ObstaclesMessage, SpeedSignsMessage
import pylot.utils
from pylot.drivers.carla_base_gnss_driver_operator import (
    CarlaBaseGNSSDriverOperator)


class CarlaSpeedLimitSignsDriverOperator(CarlaBaseGNSSDriverOperator):
    """Publishes the locations and values of all speed limit signs retrieved
    from the simulator at the provided frequency.
    
    Args:
        vehicle_id_stream: Stream on which the operator receives the ID of the
            ego vehicle. The ID is used to get a simulator handle to the
            vehicle.
        speed_limit_signs_stream: Stream on which the operator sends the speed
            limit signs.
        frequency: Rate at which the pose is published, in Hertz.
        flags: Object used to access absl flags.
    """
    def __init__(self, vehicle_id_stream: erdos.ReadStream,
                 speed_limit_signs_stream: erdos.WriteStream, frequency: float,
                 flags):
        transform = pylot.utils.Transform(pylot.utils.Location(),
                                          pylot.utils.Rotation())
        gnss_setup = pylot.drivers.sensor_setup.GNSSSetup(
            self.config.name, transform)
        super().__init__(vehicle_id_stream, speed_limit_signs_stream,
                         gnss_setup, frequency, flags)
        self._speed_limit_actors = None

    def process_gnss(self, timestamp: erdos.Timestamp,
                     gnss_msg: carla.GnssMeasurement):
        """"""
        actor_list = self._world.get_actors()

        speed_limit_actors = actor_list.filter('traffic.speed_limit*')
        speed_limits = list(
            map(SpeedLimitSign.from_simulator_actor, speed_limit_actors))

        self._output_stream.send(SpeedSignsMessage(timestamp, speed_limits))
        self._output_stream.send(erdos.WatermarkMessage(timestamp))

    def run(self):
        super().run()
        # Get speed limit actors
        self._speed_limit_actors = self._world.get_actors().filter(
            'traffic.speed_limit*')
