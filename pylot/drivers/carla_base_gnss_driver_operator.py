"""This module implements an operator which executes a callback at a the
provided frequency.

The operator attaches a GNSS sensor to the ego vehicle, receives GNSS
measurements from the simulator, and invokes the user-defined callback.
"""
import threading
from abc import abstractmethod

import carla

import erdos

from pylot.drivers.sensor_setup import GNSSSetup
from pylot.simulation.utils import get_vehicle_handle, get_world, \
    set_simulation_mode


class CarlaBaseGNSSDriverOperator(erdos.Operator):
    """Invokes ``process_gnss`` at the provided frequency.

    This operator attaches to a vehicle at the required position with respect
    to the vehicle, registers callback functions to execute a ``process_gnss``
    at the provided frequency.

    Attributes:
        _vehicle_id_stream (ReadStream): Stream on which the operator receives
            the id of the ego vehicle. The ID is used to get a simulator handle
            to the vehicle.
        _output_stream (WriteStream): Stream on which the operator sends
            messages.
        _flags: Object to be used to access absl flags.
        _logger (logging.Logger): Used to log information.
        _gnss_setup (GNSSSetup): Setup of the GNSS sensor.
        _frequency (float): Rate at which the callback is invoked.
        _vehicle (Optional[carla.Vehicle]): Handle to the CARLA vehicle.
            Initialized once the vehicle ID is received.
        _world (Optional[carla.World]): Handle to the CARLA world. Initialized
            once the vehicle ID is received.
        _gnss (Optional[carla.Actor]): Handle to the CARLA GNSS sensor.
            Initialized once the vehicle ID is received.
        _log (threading.Lock): used to ensure that only 1 GNSS reading is
            processed at a time.


    Args:
        vehicle_id_stream: Stream on which the operator receives the id of the
            ego vehicle. The ID is used to get a simulator handle to the
            vehicle.
        output_stream: Stream on which the operator sends messages.
        gnss_setup: Setup of the GNSS sensor.
        frequency: Rate at which the pose is published, in Hertz. Set to -1 to
            invoke on every simulator tick.
        flags: Object used to access absl flags.
    """
    def __init__(self, vehicle_id_stream: erdos.ReadStream,
                 output_stream: erdos.WriteStream, gnss_setup: GNSSSetup,
                 frequency: float, flags):
        # Save the streams.
        self._vehicle_id_stream = vehicle_id_stream
        self._output_stream = output_stream

        # Save the flags and initialize logging.
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)

        # Save the setup, the vehicle, the world, and the sensor.
        self._gnss_setup = gnss_setup
        self._frequency = frequency
        self._vehicle = None
        self._world = None
        self._gnss = None
        self._lock = threading.Lock()

    @staticmethod
    def connect(ground_vehicle_id_stream):
        gnss_stream = erdos.WriteStream()
        return [gnss_stream]

    @abstractmethod
    def process_gnss(self, timestamp: erdos.Timestamp,
                     gnss_msg: carla.GnssMeasurement):
        """Invoked when a GNSS measurement is received from the simulator.

        Note:
            Only 1 invocation of this callback will run at a time.
        """
        raise NotImplementedError

    def on_gnss(self, gnss_msg: carla.GnssMeasurement):
        """Invoked when a GNSS measurement is received from the simulator.
        """
        game_time = int(gnss_msg.timestamp * 1000)
        timestamp = erdos.Timestamp(coordinates=[game_time])
        with erdos.profile(self.config.name + '.process_gnss',
                           self,
                           event_data={'timestamp': str(timestamp)}):
            with self._lock:
                self.process_gnss(timestamp, gnss_msg)

    def run(self):
        # Read the vehicle ID from the vehicle ID stream.
        vehicle_id = self._vehicle_id_stream.read().data
        self._logger.debug("received the vehicle id: {}".format(vehicle_id))

        # Connect to the world.
        _, self._world = get_world(self._flags.simulator_host,
                                   self._flags.simulator_port,
                                   self._flags.simulator_timeout)
        set_simulation_mode(self._world, self._flags)

        # Retrieve the vehicle and install the GNSS sensor.
        self._vehicle = get_vehicle_handle(self._world, vehicle_id)
        gnss_blueprint = self._world.get_blueprint_library().find(
            'sensor.other.gnss')

        # Set the noise and bias parameters.
        gnss_blueprint.set_attribute('noise_alt_stddev',
                                     str(self._flags.gnss_noise_stddev_alt))
        gnss_blueprint.set_attribute('noise_lat_stddev',
                                     str(self._flags.gnss_noise_stddev_lat))
        gnss_blueprint.set_attribute('noise_lon_stddev',
                                     str(self._flags.gnss_noise_stddev_lon))
        gnss_blueprint.set_attribute('noise_alt_bias',
                                     str(self._flags.gnss_bias_alt))
        gnss_blueprint.set_attribute('noise_lat_bias',
                                     str(self._flags.gnss_bias_lat))
        gnss_blueprint.set_attribute('noise_lon_bias',
                                     str(self._flags.gnss_bias_lon))

        if self._frequency == -1:
            gnss_blueprint.set_attribute('sensor_tick', '0.0')
        else:
            gnss_blueprint.set_attribute('sensor_tick',
                                         str(1.0 / self._frequency))
        transform = self._gnss_setup.get_transform().as_simulator_transform()
        self._logger.debug("Spawning a GNSS sensor: {}".format(
            self._gnss_setup))
        self._gnss = self._world.spawn_actor(gnss_blueprint,
                                             transform,
                                             attach_to=self._vehicle)

        # Register the callback on the GNSS sensor.
        self._gnss.listen(self.on_gnss)
