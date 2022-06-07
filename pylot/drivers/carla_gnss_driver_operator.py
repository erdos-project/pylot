"""This module implements an operator acts like a GNSS driver when
using the simulator.

The operator attaches a GNSS sensor to the ego vehicle, receives GNSS
 measurements from the simulator, and sends them on its output stream.
"""

import threading

from carla import GnssMeasurement

import erdos
from erdos import ReadStream, WriteStream
from erdos.operator import OneInOneOut

from pylot.drivers.sensor_setup import GNSSSetup
from pylot.localization.messages import GNSSMessageTuple
from pylot.simulation.utils import get_vehicle_handle, get_world, \
    set_simulation_mode
from pylot.utils import Transform


class CarlaGNSSDriverOperator(OneInOneOut):
    """Publishes GNSSMessages (transform, altitude, latitude, longitude) from
    the GNSS sensor.

    This operator attaches to a vehicle at the required position with respect
    to the vehicle, registers callback functions to retrieve the GNSS
    measurements and publishes it to downstream operators.

    Args:
        gnss_setup (:py:class:`pylot.drivers.sensor_setup.GNSSSetup`):
            Setup of the GNSS sensor.
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, gnss_setup: GNSSSetup, flags):
        # Save the flags and initialize logging.
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)

        # Save the setup, the vehicle and the sensor.
        self._gnss_setup = gnss_setup
        self._vehicle = None
        self._gnss = None
        self._lock = threading.Lock()

    def process_gnss(self, gnss_msg: GnssMeasurement,
                     write_stream: WriteStream):
        """Invoked when a GNSS measurement is received from the simulator.

        Sends GNSS measurements to downstream operators.
        """
        game_time = int(gnss_msg.timestamp * 1000)
        timestamp = erdos.Timestamp(coordinates=[game_time])
        watermark_msg = erdos.WatermarkMessage(timestamp)
        with erdos.profile(self.config.name + '.process_gnss',
                           self,
                           event_data={'timestamp': str(timestamp)}):
            with self._lock:
                gnss_data = GNSSMessageTuple(
                    Transform.from_simulator_transform(gnss_msg.transform),
                    gnss_msg.altitude, gnss_msg.latitude, gnss_msg.longitude)
                write_stream.send(erdos.Message(timestamp, gnss_data))
                write_stream.send(watermark_msg)

    def run(self, read_stream: ReadStream,
            write_stream: WriteStream):
        # Read the vehicle ID from the vehicle ID stream.
        vehicle_id_msg = read_stream.read()
        vehicle_id = vehicle_id_msg.data
        self._logger.debug(
            "The GNSSDriverOperator received the vehicle id: {}".format(
                vehicle_id))

        # Connect to the world.
        _, world = get_world(self._flags.simulator_host,
                             self._flags.simulator_port,
                             self._flags.simulator_timeout)
        set_simulation_mode(world, self._flags)

        # Retrieve the vehicle and install the GNSS sensor.
        self._vehicle = get_vehicle_handle(world, vehicle_id)
        gnss_blueprint = world.get_blueprint_library().find(
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

        if self._flags.simulator_gnss_frequency == -1:
            gnss_blueprint.set_attribute('sensor_tick', '0.0')
        else:
            gnss_blueprint.set_attribute(
                'sensor_tick', str(1.0 / self._flags.simulator_gnss_frequency))
        transform = self._gnss_setup.get_transform().as_simulator_transform()
        self._logger.debug("Spawning a GNSS sensor: {}".format(
            self._gnss_setup))
        self._gnss = world.spawn_actor(gnss_blueprint,
                                       transform,
                                       attach_to=self._vehicle)

        # Register the callback on the GNSS sensor.
        def _process_gnss(gnss_msg):
            self.process_gnss(gnss_msg, write_stream)

        self._gnss.listen(_process_gnss)
