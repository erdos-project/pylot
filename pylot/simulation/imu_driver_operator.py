import threading
import time

# ERDOS specific imports.
from erdos.op import Op
from erdos.utils import setup_logging
from erdos.message import WatermarkMessage
from erdos.timestamp import Timestamp

import pylot.utils
from pylot.simulation.carla_utils import get_world, set_synchronous_mode
from pylot.simulation.messages import IMUMessage
from pylot.simulation.utils import Vector3D, Transform


class IMUDriverOperator(Op):
    """ Publishes carla.IMUMeasurements (transform, acceleration, gyro and compass)
    from IMU (inertial measurement unit) sensor.

    This operator attaches to a vehicle at the required position with respect to
    the vehicle, registers callback functions to retrieve the IMU measurements and
    publishes it to downstream operators.

    Attributes:
        _imu_setup: An IMUSetup tuple.
        _imu: Handle to the IMU inside the simulation.
        _vehicle: Handle to the simulated hero vehicle.
    """
    def __init__(self,
                 name,
                 imu_setup,
                 flags,
                 log_file_name=None):
        """ Initializes the camera inside the simulation with the given
        parameters.

        Args:
            name: The unique name of the operator.
            imu_setup: A IMUSetup tuple.
            flags: A handle to the global flags instance to retrieve the
                configuration.
            log_file_name: The file to log the required information to.
        """
        super(IMUDriverOperator, self).__init__(
            name, no_watermark_passthrough=True)
        # The operator does not pass watermarks by defaults.
        self._flags = flags
        self._logger = setup_logging(self.name, log_file_name)
        self._imu_setup = imu_setup
        # The hero vehicle actor object we obtain from Carla.
        self._vehicle = None
        # The IMU sensor actor object we obtain from Carla.
        self._imu = None
        # Lock to ensure that the callbacks do not execute simultaneously.
        self._lock = threading.Lock()

    @staticmethod
    def setup_streams(input_streams, imu_setup):
        """ Set up callback functions on the input streams and return the
        output stream that publishes the IMU measurements.

        Args:
            input_streams: The streams that this operator is connected to.
            imu_setup: An IMUSetup tuple.
        """
        input_streams.filter(pylot.utils.is_ground_vehicle_id_stream)\
                     .add_callback(IMUDriverOperator.on_vehicle_id)
        return [pylot.utils.create_imu_stream(imu_setup)]

    def process_imu(self, imu_msg):
        """ Invoked when an IMU message is received from the simulator.
        Sends IMU measurements to downstream operators.

        Args:
            imu_msg: carla.IMUMeasurement
        """
        with self._lock:
            game_time = int(imu_msg.timestamp * 1000)
            timestamp = Timestamp(coordinates=[game_time])
            watermark_msg = WatermarkMessage(timestamp)

            msg = IMUMessage(
                Transform(carla_transform=imu_msg.transform),
                Vector3D(carla_vector=imu_msg.accelerometer),
                Vector3D(carla_vector=imu_msg.gyroscope),
                imu_msg.compass,
                timestamp
            )

            self.get_output_stream(self._imu_setup.name).send(msg)
            # Note: The operator is set not to automatically propagate
            # watermark messages received on input streams. Thus, we can
            # issue watermarks only after the Carla callback is invoked.
            self.get_output_stream(self._imu_setup.name).send(watermark_msg)

    def on_vehicle_id(self, msg):
        """ This function receives the identifier for the vehicle, retrieves
        the handler for the vehicle from the simulation and attaches the
        IMU to it.

        Args:
            msg: The identifier for the vehicle to attach the camera to.
        """
        vehicle_id = msg.data
        self._logger.info(
            "The IMUDriverOperator received the vehicle id: {}".format(
                vehicle_id))

        # Connect to the world. We connect here instead of in the constructor
        # to ensure we're connected to the latest world.
        _, world = get_world(self._flags.carla_host,
                             self._flags.carla_port,
                             self._flags.carla_timeout)
        if world is None:
            raise ValueError("There was an issue connecting to the simulator.")
        if self._flags.carla_synchronous_mode:
            set_synchronous_mode(world, self._flags.carla_fps)

        num_tries = 0
        while self._vehicle is None and num_tries < 30:
            self._vehicle = world.get_actors().find(vehicle_id)
            self._logger.info(
                "Could not find vehicle. Try {}".format(num_tries))
            time.sleep(1)
            num_tries += 1
        if self._vehicle is None:
            raise ValueError("There was an issue finding the vehicle.")

        # Install the IMU.
        imu_blueprint = world.get_blueprint_library().find(
            self._imu_setup.imu_type)

        transform = self._imu_setup.get_transform().as_carla_transform()

        self._logger.info("Spawning an IMU: {}".format(self._imu_setup))

        self._imu = world.spawn_actor(imu_blueprint,
                                      transform,
                                      attach_to=self._vehicle)

        # Register the callback on the camera.
        self._imu.listen(self.process_imu)
