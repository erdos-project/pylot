import erdust

from pylot.control.messages import ControlMessage


class SynchronizerOperator(erdust.Operator):
    def __init__(self, wait_stream, control_stream, flags):
        erdust.add_watermark_callback([wait_stream], [control_stream],
                                      self.on_watermark)
        self._flags = flags

    @staticmethod
    def connect(wait_stream):
        control_stream = erdust.WriteStream()
        return [control_stream]

    def on_watermark(self, timestamp, control_stream):
        # The control message is ignored by the bridge operator because
        # data gathering is conducted using auto pilot. Send default control
        # message.
        control_msg = ControlMessage(0, 0, 0, False, False, timestamp)
        control_stream.send(control_msg)
