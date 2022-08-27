"""This module implements an operator acts like a GNSS driver when
using the simulator.

The operator attaches a GNSS sensor to the ego vehicle, receives GNSS
 measurements from the simulator, and sends them on its output stream.
"""
import carla

import erdos

from pylot.drivers.carla_base_gnss_driver_operator import (
    CarlaBaseGNSSDriverOperator)
from pylot.drivers.sensor_setup import GNSSSetup
from pylot.localization.messages import GNSSMessage
from pylot.simulation.utils import get_vehicle_handle, get_world, \
    set_simulation_mode
from pylot.utils import Transform


class CarlaGNSSDriverOperator(CarlaBaseGNSSDriverOperator):
    """Publishes GNSSMessages (transform, altitude, latitude, longitude) from
    the GNSS sensor.

    This operator attaches to a vehicle at the required position with respect
    to the vehicle, registers callback functions to retrieve the GNSS
    measurements and publishes it to downstream operators.

    Args:
        vehicle_id_stream: Stream on which the operator receives the id of the
            ego vehicle. The ID is used to get a simulator handle to the
            vehicle.
        gnss_stream: Stream on which the operator sends GNSS info.
        gnss_setup: Setup of the GNSS sensor.
        flags: Object used to access absl flags.
    """
    def __init__(self, ground_vehicle_id_stream: erdos.ReadStream,
                 gnss_stream: erdos.WriteStream, gnss_setup: GNSSSetup, flags):

        frequency = flags.simulator_gnss_frequency
        super().__init__(ground_vehicle_id_stream, gnss_stream, gnss_setup,
                         frequency, flags)

    def process_gnss(self, timestamp: erdos.Timestamp,
                     gnss_msg: carla.GnssMeasurement):
        """Invoked when a GNSS measurement is received from the simulator.

        Sends GNSS measurements to downstream operators.
        """
        watermark_msg = erdos.WatermarkMessage(timestamp)
        msg = GNSSMessage(
            timestamp, Transform.from_simulator_transform(gnss_msg.transform),
            gnss_msg.altitude, gnss_msg.latitude, gnss_msg.longitude)
        self._output_stream.send(msg)
        self._output_stream.send(watermark_msg)
