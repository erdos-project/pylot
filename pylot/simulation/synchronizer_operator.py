import erdos
from erdos import ReadStream, Timestamp, WriteStream

from pylot.control.messages import ControlMessage
from pylot.simulation.utils import get_vehicle_handle, get_world, \
    set_simulation_mode


class SynchronizerOperator(erdos.Operator):
    """Sends control messages when it receives a watermark on a stream.

    The operator can be used to ensure that simulator does not tick before the
    slowest stream in a data-flow completes processing a timestmap.

    Warning:
       The operator should only be used with the simulator auto pilot enabled.

    Args:
        wait_stream (:py:class:`erdos.ReadStream`): The stream on which to wait
            for watermark messages.
        control_stream (:py:class:`erdos.WriteStream`): Stream on which control
            messages are published.
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, ground_vehicle_id_stream: ReadStream,
                 wait_stream: ReadStream, control_stream: WriteStream, flags):
        erdos.add_watermark_callback([wait_stream], [control_stream],
                                     self.on_watermark)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._vehicle_id_stream = ground_vehicle_id_stream
        self._vehicle = None
        self._flags = flags

    @staticmethod
    def connect(ground_vehicle_id_stream: ReadStream, wait_stream: ReadStream):
        control_stream = erdos.WriteStream()
        return [control_stream]

    def run(self):
        # Read the vehicle ID from the vehicle ID stream.
        vehicle_id_msg = self._vehicle_id_stream.read()
        vehicle_id = vehicle_id_msg.data

        # Connect to the world. We connect here instead of in the constructor
        # to ensure we're connected to the latest world.
        _, world = get_world(self._flags.simulator_host,
                             self._flags.simulator_port,
                             self._flags.simulator_timeout)
        set_simulation_mode(world, self._flags)

        self._vehicle = get_vehicle_handle(world, vehicle_id)

    def on_watermark(self, timestamp: Timestamp, control_stream: WriteStream):
        """Invoked when the input stream has received a watermark.

        The method sends a control message.

        Args:
            timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of
                the watermark.
        """
        self._logger.debug('@{}: received watermark'.format(timestamp))
        # The control message is ignored by the bridge operator because
        # data gathering is conducted using auto pilot.
        # Send the control that the vehicle is currently applying.
        vehicle_control = self._vehicle.get_control()
        control_msg = ControlMessage(vehicle_control.steer,
                                     vehicle_control.throttle,
                                     vehicle_control.brake,
                                     vehicle_control.hand_brake,
                                     vehicle_control.reverse, timestamp)
        control_stream.send(control_msg)
        control_stream.send(erdos.WatermarkMessage(timestamp))
