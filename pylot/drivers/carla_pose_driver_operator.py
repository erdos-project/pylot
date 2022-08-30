import carla

import erdos

import pylot.utils
from pylot.drivers.carla_base_gnss_driver_operator import (
    CarlaBaseGNSSDriverOperator)


class CarlaPoseDriverOperator(CarlaBaseGNSSDriverOperator):
    """Publishes the pose (location, orientation, and velocity) at the provided
    frequency.

    This operator attaches to the vehicle using the vehicle ID provided by the
    ``vehicle_id_stream``, registers callback functions to retrieve the
    pose at the provided frequency, and publishes it to downstream operators.

    Args:
        vehicle_id_stream: Stream on which the operator receives the ID of the
            ego vehicle. The ID is used to get a simulator handle to the
            vehicle.
        pose_stream: Stream on which the operator sends the vehicle's pose.
        frequency: Rate at which the pose is published, in Hertz.
        flags: Object used to access absl flags.
    """
    def __init__(self, vehicle_id_stream: erdos.ReadStream,
                 pose_stream: erdos.WriteStream, frequency: float, flags):
        transform = pylot.utils.Transform(pylot.utils.Location(),
                                          pylot.utils.Rotation())
        gnss_setup = pylot.drivers.sensor_setup.GNSSSetup(
            self.config.name, transform)
        super().__init__(vehicle_id_stream, pose_stream, gnss_setup, frequency,
                         flags)

    def process_gnss(self, timestamp: erdos.Timestamp,
                     gnss_msg: carla.GnssMeasurement):
        """Sends pose information followed by a watermark."""
        self._logger.debug('@{}: sending pose'.format(timestamp))
        vec_transform = pylot.utils.Transform.from_simulator_transform(
            self._vehicle.get_transform())
        velocity_vector = pylot.utils.Vector3D.from_simulator_vector(
            self._vehicle.get_velocity())
        forward_speed = velocity_vector.magnitude()
        pose = pylot.utils.Pose(vec_transform, forward_speed, velocity_vector,
                                timestamp.coordinates[0])
        self._output_stream.send(erdos.Message(timestamp, pose))
        self._output_stream.send(erdos.WatermarkMessage(timestamp))
