"""This module implements an operator acts like a IMU driver when
using the simulator.

The operator attaches an IMU sensor to the ego vehicle, receives
IMU measurements from the simulator, and sends them on its output stream.
"""

import threading

import erdos

from pylot.localization.messages import IMUMessage
from pylot.simulation.utils import get_vehicle_handle, get_world, \
    set_simulation_mode
from pylot.utils import Transform, Vector3D


class CarlaIMUDriverOperator(erdos.Operator):
    """Publishes IMU mesurements (transform, acceleration, gyro and
    compass) from IMU (inertial measurement unit) sensor.

    This operator attaches to a vehicle at the required position with respect
    to the vehicle, registers callback functions to retrieve the IMU
    measurements and publishes it to downstream operators.

    Args:
        ego_vehicle_id_stream (:py:class:`erdos.ReadStream`): Stream on
            which the operator receives the id of the ego vehicle. It uses this
            id to get a simulator handle to the vehicle.
        imu_stream (:py:class:`erdos.WriteStream`): Stream on which the
            operator sends IMU info.
        imu_setup (:py:class:`pylot.drivers.sensor_setup.IMUSetup`):
            Setup of the IMU sensor.
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, ego_vehicle_id_stream: erdos.ReadStream,
                 imu_stream: erdos.WriteStream, imu_setup, flags):
        self._vehicle_id_stream = ego_vehicle_id_stream
        self._imu_stream = imu_stream
        # The operator does not pass watermarks by defaults.
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._imu_setup = imu_setup
        # The hero vehicle actor object we obtain from the simulator.
        self._vehicle = None
        # The IMU sensor actor object we obtain from the simulator.
        self._imu = None
        # Lock to ensure that the callbacks do not execute simultaneously.
        self._lock = threading.Lock()

    @staticmethod
    def connect(ego_vehicle_id_stream: erdos.ReadStream):
        imu_stream = erdos.WriteStream()
        return [imu_stream]

    def process_imu(self, imu_msg):
        """Invoked when an IMU measurement is received from the simulator.

        Sends IMU measurements to downstream operators.
        """
        game_time = int(imu_msg.timestamp * 1000)
        timestamp = erdos.Timestamp(coordinates=[game_time])
        watermark_msg = erdos.WatermarkMessage(timestamp)
        with erdos.profile(self.config.name + '.process_imu',
                           self,
                           event_data={'timestamp': str(timestamp)}):
            with self._lock:
                msg = IMUMessage(
                    timestamp,
                    Transform.from_simulator_transform(imu_msg.transform),
                    Vector3D.from_simulator_vector(imu_msg.accelerometer),
                    Vector3D.from_simulator_vector(imu_msg.gyroscope),
                    imu_msg.compass)
                self._imu_stream.send(msg)
                # Note: The operator is set not to automatically propagate
                # watermarks received on input streams. Thus, we can issue
                # watermarks only after the simulator callback is invoked.
                self._imu_stream.send(watermark_msg)

    def run(self):
        # Read the vehicle id from the vehicle id stream
        vehicle_id_msg = self._vehicle_id_stream.read()
        vehicle_id = vehicle_id_msg.data
        self._logger.debug(
            "The IMUDriverOperator received the vehicle id: {}".format(
                vehicle_id))

        # Connect to the world. We connect here instead of in the constructor
        # to ensure we're connected to the latest world.
        _, world = get_world(self._flags.simulator_host,
                             self._flags.simulator_port,
                             self._flags.simulator_timeout)
        set_simulation_mode(world, self._flags)

        self._vehicle = get_vehicle_handle(world, vehicle_id)

        # Install the IMU.
        imu_blueprint = world.get_blueprint_library().find('sensor.other.imu')

        # Set noise attributes.
        imu_blueprint.set_attribute('noise_accel_stddev_x',
                                    str(self._flags.accel_noise_stddev_x))
        imu_blueprint.set_attribute('noise_accel_stddev_y',
                                    str(self._flags.accel_noise_stddev_y))
        imu_blueprint.set_attribute('noise_accel_stddev_z',
                                    str(self._flags.accel_noise_stddev_z))
        imu_blueprint.set_attribute('noise_gyro_stddev_x',
                                    str(self._flags.gyro_noise_stddev_x))
        imu_blueprint.set_attribute('noise_gyro_stddev_y',
                                    str(self._flags.gyro_noise_stddev_y))
        imu_blueprint.set_attribute('noise_gyro_stddev_z',
                                    str(self._flags.gyro_noise_stddev_z))

        if self._flags.simulator_imu_frequency == -1:
            imu_blueprint.set_attribute('sensor_tick', '0.0')
        else:
            imu_blueprint.set_attribute(
                'sensor_tick', str(1.0 / self._flags.simulator_imu_frequency))

        transform = self._imu_setup.get_transform().as_simulator_transform()

        self._logger.debug("Spawning an IMU: {}".format(self._imu_setup))

        self._imu = world.spawn_actor(imu_blueprint,
                                      transform,
                                      attach_to=self._vehicle)

        # Register the callback on the IMU.
        self._imu.listen(self.process_imu)
