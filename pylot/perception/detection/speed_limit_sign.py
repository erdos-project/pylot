import pylot.utils


class SpeedLimitSign(object):
    """Class used to store info about speed limit signs.

    Args:
        transform (:py:class:`~pylot.utils.Transform`): Transform of the
            speed limit sign.
        limit (:obj:`int`): The speed limit of the sign.

    Attributes:
        transform (:py:class:`~pylot.utils.Transform`): Transform of the
            speed limit sign.
        limit (:obj:`int`): The speed limit of the sign.
    """
    def __init__(self, transform, limit):
        self.transform = transform
        self.limit = limit

    @classmethod
    def from_carla_actor(cls, actor):
        """Creates a speed limit sign from a CARLA actor.

        Args:
            actor (carla.TrafficSign): A carla speed limit sign actor.

        Returns:
            :py:class:`.SpeedLimitSign`: A speed limit sign.
        """
        import carla
        if not isinstance(actor, carla.TrafficSign):
            raise ValueError('actor should be of type carla.TrafficSign')
        transform = pylot.utils.Transform.from_carla_transform(
            actor.get_transform())
        speed_limit = int(actor.type_id.split('.')[-1])
        return cls(transform, speed_limit)
