"""This module implements an operator acts like a GNSS driver when
using the simulator.

The operator attaches a GNSS sensor to the ego vehicle, receives GNSS
 measurements from the simulator, and sends them on its output stream.
"""

import threading

import erdos

from pylot.localization.messages import GNSSMessage
from pylot.simulation.utils import get_vehicle_handle, get_world, \
    set_simulation_mode
from pylot.utils import Transform


class CarlaGNSSDriverOperator(erdos.Operator):
    """Publishes GNSSMessages (transform, altitude, latitude, longitude) from
    the GNSS sensor.

    This operator attaches to a vehicle at the required position with respect
    to the vehicle, registers callback functions to retrieve the GNSS
    measurements and publishes it to downstream operators.

    Args:
        ground_vehicle_id_stream (:py:class:`erdos.ReadStream`): Stream on
            which the operator receives the id of the ego vehicle. It uses this
            id to get a simulator handle to the vehicle.
        gnss_stream (:py:class:`erdos.WriteStream`): Stream on which the
            operator sends GNSS info.
        gnss_setup (:py:class:`pylot.drivers.sensor_setup.GNSSSetup`):
            Setup of the GNSS sensor.
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, ground_vehicle_id_stream: erdos.ReadStream,
                 gnss_stream: erdos.WriteStream, gnss_setup, flags):
        # Save the streams.
        self._vehicle_id_stream = ground_vehicle_id_stream
        self._gnss_stream = gnss_stream

        # Save the flags and initialize logging.
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)

        # Save the setup, the vehicle and the sensor.
        self._gnss_setup = gnss_setup
        self._vehicle = None
        self._gnss = None
        self._lock = threading.Lock()

    @staticmethod
    def connect(ground_vehicle_id_stream):
        gnss_stream = erdos.WriteStream()
        return [gnss_stream]

    def process_gnss(self, gnss_msg):
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
                msg = GNSSMessage(
                    timestamp,
                    Transform.from_simulator_transform(gnss_msg.transform),
                    gnss_msg.altitude, gnss_msg.latitude, gnss_msg.longitude)
                self._gnss_stream.send(msg)
                self._gnss_stream.send(watermark_msg)

    def run(self):
        # Read the vehicle ID from the vehicle ID stream.
        vehicle_id = self._vehicle_id_stream.read().data
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
        self._gnss.listen(self.process_gnss)
