import carla

import erdos

from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.messages import ObstaclesMessage
import pylot.utils
from pylot.drivers.carla_base_gnss_driver_operator import (
    CarlaBaseGNSSDriverOperator)


class CarlaObstaclesDriverOperator(CarlaBaseGNSSDriverOperator):
    """Publishes the bounding boxes of all vehicles and people retrieved from
    the simulator at the provided frequency.
    
    Args:
        vehicle_id_stream: Stream on which the operator receives the ID of the
            ego vehicle. The ID is used to get a simulator handle to the
            vehicle.
        obstacles_stream: Stream on which the operator sends the obstacles.
        frequency: Rate at which the pose is published, in Hertz.
        flags: Object used to access absl flags.
    """
    def __init__(self, vehicle_id_stream: erdos.ReadStream,
                 obstacles_stream: erdos.WriteStream, frequency: float, flags):
        transform = pylot.utils.Transform(pylot.utils.Location(),
                                          pylot.utils.Rotation())
        gnss_setup = pylot.drivers.sensor_setup.GNSSSetup(
            self.config.name, transform)
        super().__init__(vehicle_id_stream, obstacles_stream, gnss_setup,
                         frequency, flags)

    def process_gnss(self, timestamp: erdos.Timestamp,
                     gnss_msg: carla.GnssMeasurement):
        """"""
        actor_list = self._world.get_actors()

        vec_actors = actor_list.filter('vehicle.*')
        vehicles = list(map(Obstacle.from_simulator_actor, vec_actors))

        person_actors = actor_list.filter('walker.pedestrian.*')
        people = list(map(Obstacle.from_simulator_actor, person_actors))

        self._output_stream.send(ObstaclesMessage(timestamp,
                                                  vehicles + people))
        self._output_stream.send(erdos.WatermarkMessage(timestamp))
