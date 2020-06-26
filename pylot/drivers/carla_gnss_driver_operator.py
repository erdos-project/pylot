import erdos
import threading

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
            id to get a Carla handle to the vehicle.
        gnss_stream (:py:class:`erdos.WriteStream`): Stream on which the
            operator sends GNSS info.
        gnss_setup (:py:class:`pylot.drivers.sensor_setup.GNSSSetup`):
            Setup of the GNSS sensor.
        flags (absl.flags): Object to be used to access absl flags.
    """

    def __init__(self, ground_vehicle_id_stream, gnss_stream, gnss_setup,
                 flags):
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
        """ Invoked when a GNSS message is received from the simulator.

        Sends GNSS measurements to downstream operators.

        Args:
            gnss_msg: carla.GnssMeasurement
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
                    Transform.from_carla_transform(gnss_msg.transform),
                    gnss_msg.altitude, gnss_msg.latitude, gnss_msg.longitude)
                print(msg)
                self._gnss_stream.send(msg)
                self._gnss_stream.send(watermark_msg)

    def run(self):
        # Read the vehicle ID from the vehicle ID stream.
        vehicle_id = self._vehicle_id_stream.read().data
        self._logger.debug(
            "The CarlaGNSSDriverOperator received the vehicle id: {}".format(
                vehicle_id))

        # Connect to the world.
        _, world = get_world(self._flags.carla_host, self._flags.carla_port,
                             self._flags.carla_timeout)
        if world is None:
            raise ValueError("There was an issue connecting to the simulator.")
        set_simulation_mode(world, self._flags)

        # Retrieve the vehicle and install the GNSS sensor.
        self._vehicle = get_vehicle_handle(world, vehicle_id)
        gnss_blueprint = world.get_blueprint_library().find(
            'sensor.other.gnss')
        if self._flags.carla_gnss_frequency == -1:
            gnss_blueprint.set_attribute('sensor_tick', '0.0')
        else:
            gnss_blueprint.set_attribute(
                'sensor_tick', str(1.0 / self._flags.carla_gnss_frequency))
        transform = self._gnss_setup.get_transform().as_carla_transform()
        self._logger.debug("Spawning a GNSS sensor: {}".format(
            self._gnss_setup))
        self._gnss = world.spawn_actor(gnss_blueprint,
                                       transform,
                                       attach_to=self._vehicle)

        # Register the callback on the GNSS sensor.
        self._gnss.listen(self.process_gnss)
