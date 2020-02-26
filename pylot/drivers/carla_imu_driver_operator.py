import erdos
import threading
import time

from pylot.perception.messages import IMUMessage
from pylot.simulation.utils import get_world, set_synchronous_mode
from pylot.utils import Transform, Vector3D


class CarlaIMUDriverOperator(erdos.Operator):
    """Publishes carla.IMUMeasurements (transform, acceleration, gyro and
    compass) from IMU (inertial measurement unit) sensor.

    This operator attaches to a vehicle at the required position with respect
    to the vehicle, registers callback functions to retrieve the IMU
    measurements and publishes it to downstream operators.

    Args:
        ground_vehicle_id_stream (:py:class:`erdos.ReadStream`): Stream on
            which the operator receives the id of the ego vehicle. It uses this
            id to get a Carla handle to the vehicle.
        imu_stream (:py:class:`erdos.WriteStream`): Stream on which the
            operator sends IMU info.
        name (:obj:`str`): The name of the operator.
        imu_setup (:py:class:`pylot.drivers.sensor_setup.IMUSetup`):
            Setup of the IMU sensor.
        flags (absl.flags): Object to be used to access absl flags.
        log_file_name (:obj:`str`, optional): Name of file where log messages
            are written to. If None, then messages are written to stdout.
    """
    def __init__(self,
                 ground_vehicle_id_stream,
                 imu_stream,
                 name,
                 imu_setup,
                 flags,
                 log_file_name=None):
        self._vehicle_id_stream = ground_vehicle_id_stream
        self._imu_stream = imu_stream
        # The operator does not pass watermarks by defaults.
        self._name = name
        self._flags = flags
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._imu_setup = imu_setup
        # The hero vehicle actor object we obtain from Carla.
        self._vehicle = None
        # The IMU sensor actor object we obtain from Carla.
        self._imu = None
        # Lock to ensure that the callbacks do not execute simultaneously.
        self._lock = threading.Lock()

    @staticmethod
    def connect(ground_vehicle_id_stream):
        imu_stream = erdos.WriteStream()
        return [imu_stream]

    def process_imu(self, imu_msg):
        """Invoked when an IMU message is received from the simulator.

        Sends IMU measurements to downstream operators.

        Args:
            imu_msg: carla.IMUMeasurement
        """
        with self._lock:
            game_time = int(imu_msg.timestamp * 1000)
            timestamp = erdos.Timestamp(coordinates=[game_time])
            watermark_msg = erdos.WatermarkMessage(timestamp)

            msg = IMUMessage(timestamp,
                             Transform.from_carla_transform(imu_msg.transform),
                             Vector3D.from_carla_vector(imu_msg.accelerometer),
                             Vector3D.from_carla_vector(imu_msg.gyroscope),
                             imu_msg.compass)
            self._imu_stream.send(msg)
            # Note: The operator is set not to automatically propagate
            # watermark messages received on input streams. Thus, we can
            # issue watermarks only after the Carla callback is invoked.
            self._imu_stream.send(watermark_msg)

    def run(self):
        # Read the vehicle id from the vehicle id stream
        vehicle_id_msg = self._vehicle_id_stream.read()
        vehicle_id = vehicle_id_msg.data
        self._logger.debug(
            "The CarlaIMUDriverOperator received the vehicle id: {}".format(
                vehicle_id))

        # Connect to the world. We connect here instead of in the constructor
        # to ensure we're connected to the latest world.
        _, world = get_world(self._flags.carla_host, self._flags.carla_port,
                             self._flags.carla_timeout)
        if world is None:
            raise ValueError("There was an issue connecting to the simulator.")
        if self._flags.carla_synchronous_mode:
            set_synchronous_mode(world, self._flags.carla_fps)

        num_tries = 0
        while self._vehicle is None and num_tries < 30:
            self._vehicle = world.get_actors().find(vehicle_id)
            self._logger.debug(
                "Could not find vehicle. Try {}".format(num_tries))
            time.sleep(1)
            num_tries += 1
        if self._vehicle is None:
            raise ValueError("There was an issue finding the vehicle.")

        # Install the IMU.
        imu_blueprint = world.get_blueprint_library().find('sensor.other.imu')

        transform = self._imu_setup.get_transform().as_carla_transform()

        self._logger.debug("Spawning an IMU: {}".format(self._imu_setup))

        self._imu = world.spawn_actor(imu_blueprint,
                                      transform,
                                      attach_to=self._vehicle)

        # Register the callback on the IMU.
        self._imu.listen(self.process_imu)
        # TODO: We might have to loop here to keep hold of the thread so that
        # Carla callbacks are still invoked.
        # while True:
        #     time.sleep(0.01)
