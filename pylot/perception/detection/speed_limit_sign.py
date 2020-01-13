import pylot.utils


class SpeedLimitSign(object):
    """ Class used to store info about speed limit signs.

    Attributes:
        transform: The transform of the speed limit sign.
        limit: The speed limit of the sign.
    """
    def __init__(self, transform, limit):
        self.transform = transform
        self.limit = limit

    @classmethod
    def from_carla_actor(cls, actor):
        import carla
        if not isinstance(actor, carla.TrafficSign):
            raise ValueError('actor should be of type carla.TrafficSign')
        transform = pylot.utils.Transform.from_carla_transform(
            actor.get_transform())
        speed_limit = int(actor.type_id.split('.')[-1])
        return cls(transform, speed_limit)
