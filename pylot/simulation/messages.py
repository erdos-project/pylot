""" This module implements Messages sent out by the simulator-based drivers. """

import erdos
import carla

from pylot.utils import Vector3D


class CollisionMessage(erdos.Message):
    """ Message class to be used to send collision events.

    Args:
        collided_actor (:py:class:`carla.Actor`): The actor with which the
            ego-vehicle collided.
        impulse (:py:class:`pylot.utils.Vector3D`): The impulse as a result of
            the collision.
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of the
            message.

    Attributes:
        collided_actor (:py:class:`str`): The type of the actor with which the
            ego-vehicle collided.
        impulse (:py:class:`pyot.utils.Vector3D`): The impulse as a result of
            the collision.
        intensity (:py:class:`float`): The intensity of the collision.
    """

    def __init__(self, collided_actor, impulse, timestamp):
        super(CollisionMessage, self).__init__(timestamp, None)

        # Ensure the correct types of the arguments.
        if not isinstance(collided_actor, carla.Actor):
            raise ValueError(
                "The collided_actor should be of type carla.Actor")

        if not isinstance(impulse, Vector3D):
            raise ValueError(
                "The impulse should be of type pylot.utils.Vector3D")

        # Set the required attributes.
        self.collided_actor = collided_actor.type_id
        self.impulse = impulse
        self.intensity = self.impulse.magnitude()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'CollisionMessage(timestamp: {}, collided_actor: {}, " \
                "impulse: {}, intensity: {}'.format(self.timestamp,
                                                    self.collided_actor,
                                                    self.impulse,
                                                    self.intensity)
