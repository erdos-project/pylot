import erdos

from pylot.simulation.utils import get_world, set_simulation_mode


class CarlaOpenDriveDriverOperator(erdos.Operator):
    """Sends a string containing the ground-truth map of the world in OpenDRIVE
    format, followed by a top watermark."""
    def __init__(self, open_drive_stream: erdos.WriteStream, flags):
        self._open_drive_stream = open_drive_stream

        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)

    @staticmethod
    def connect():
        open_drive_stream = erdos.WriteStream()
        return [open_drive_stream]

    def run(self):
        _, world = get_world(self._flags.simulator_host,
                             self._flags.simulator_port,
                             self._flags.simulator_timeout)
        set_simulation_mode(world, self._flags)

        self._logger.debug('Sending the map in OpenDRIVE format')
        self._open_drive_stream.send(
            erdos.Message(erdos.Timestamp(coordinates=[0]),
                          world.get_map().to_opendrive()))
        self._open_drive_stream.send(
            erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))
