import erdos

from pylot.control.messages import ControlMessage


class SynchronizerOperator(erdos.Operator):
    """Sends control messages when it receives a watermark on a stream.

    The operator can be used to ensure that simulator does not tick before the
    slowest stream in a data-flow completes processing a timestmap.

    Warning:
       The operator should only be used with the CARLA auto pilot enabled.

    Args:
        wait_stream (:py:class:`erdos.ReadStream`): The stream on which to wait
            for watermark messages.
        control_stream (:py:class:`erdos.WriteStream`): Stream on which control
            messages are published.
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, wait_stream, control_stream, flags):
        erdos.add_watermark_callback([wait_stream], [control_stream],
                                     self.on_watermark)
        self._flags = flags

    @staticmethod
    def connect(wait_stream):
        control_stream = erdos.WriteStream()
        return [control_stream]

    def on_watermark(self, timestamp, control_stream):
        """Invoked when the input stream has received a watermark.

        The method sends a control message.

        Args:
            timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of
                the watermark.
        """
        # The control message is ignored by the bridge operator because
        # data gathering is conducted using auto pilot. Send default control
        # message.
        control_msg = ControlMessage(0, 0, 0, False, False, timestamp)
        control_stream.send(control_msg)
