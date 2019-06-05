from erdos.message import Message


class ControlMessage(Message):
    """ This class represents a message to be used to send control commands.

    Attributes:
        steer: Steer angle between [-1.0, 1.0].
        throttle: Throttle command between [0.0, 1.0].
        brake: Brake command between [0.0, 1.0].
        hand_brake: Boolean controlling hand-brake engagement.
        reverse: Boolean controlling reverse gear engagement.
    """

    def __init__(self, steer, throttle, brake, hand_brake, reverse, timestamp,
                 stream_name='default'):
        super(ControlMessage, self).__init__(None, timestamp, stream_name)
        self.steer = steer
        self.throttle = throttle
        self.brake = brake
        self.hand_brake = hand_brake
        self.reverse = reverse

    def __str__(self):
        return 'timestamp: {}, steer: {}, throttle: {}, brake: {}, hand_brake: {}, reverse: {}'.format(
            self.timestamp, self.steer, self.throttle, self.brake, self.hand_brake, self.reverse)
