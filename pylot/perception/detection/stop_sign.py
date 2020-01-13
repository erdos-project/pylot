from pylot.perception.detection.utils import BoundingBox3D
import pylot.utils


class StopSign(object):
    """ Class used to store info about stop signs.

    Attributes:
        transform: The transform of the stop sign.
        bounding_box: A BoundingBox3D of the stop sign.
    """
    def __init__(self, transform, bounding_box):
        self.transform = transform
        self.bounding_box = bounding_box

    @classmethod
    def from_carla_actor(cls, actor):
        import carla
        if not isinstance(actor, carla.TrafficSign):
            raise ValueError('actor should be of type carla.TrafficSign')
        transform = pylot.utils.Transform.from_carla_transform(
            actor.get_transform())
        world_trigger_volume = actor.get_transform().transform(
            actor.trigger_volume.location)
        bbox = BoundingBox3D(world_trigger_volume, actor.trigger_volume.extent)
        return cls(transform, bbox)
