"""Implements messages related to vehicle control."""

import erdos


class ControlMessage(erdos.Message):
    """This class represents a message to be used to send control commands.

    Args:
        steer (:obj:`float`): Steer angle between [-1.0, 1.0].
        throttle (:obj:`float`): Throttle command between [0.0, 1.0].
        brake (:obj:`float`): Brake command between [0.0, 1.0].
        hand_brake (bool): Boolean controlling hand-brake engagement.
        reverse (bool): Boolean controlling reverse gear engagement.
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of
            the message.
    """
    def __init__(self, steer, throttle, brake, hand_brake, reverse, timestamp):
        super(ControlMessage, self).__init__(timestamp, None)
        assert steer >= -1 and steer <= 1, 'Steer angle must be in [-1, 1]'
        self.steer = steer
        assert throttle >= 0 and throttle <= 1, 'Throttle must be in [0, 1]'
        self.throttle = throttle
        assert brake >= 0 and brake <= 1, 'Brake must be in [0, 1]'
        self.brake = brake
        self.hand_brake = hand_brake
        self.reverse = reverse

    def __str__(self):
        return ('ControlMessage(timestamp: {}, steer: {}, throttle: {}, '
                'brake: {}, hand_brake: {}, reverse: {})'.format(
                    self.timestamp, self.steer, self.throttle, self.brake,
                    self.hand_brake, self.reverse))
