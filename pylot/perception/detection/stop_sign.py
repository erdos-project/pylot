import pylot.utils
from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.detection.utils import BoundingBox2D, BoundingBox3D


class StopSign(Obstacle):
    """Class used to store info about stop signs.

    Args:
        confidence (:obj:`float`): The confidence of the detection.
        bounding_box (:py:class:`.BoundingBox2D`): The bounding box of the
            stop sign in camera view.
        id (:obj:`int`): Id associated with the sign.
        transform (:py:class:`~pylot.utils.Transform`): Transform of the
            stop sign.
        bounding_box_3d (:py:class:`~pylot.utisl.BoundingBox3D`): 3D bounding
            box of the stop sign.

    Attributes:
        confidence (:obj:`float`): The confidence of the detection.
        bounding_box (:py:class:`.BoundingBox2D`): The bounding box of the
            stop sign in camera view.
        id (:obj:`int`): Id associated with the sign.
        transform (:py:class:`~pylot.utils.Transform`): Transform of the
            stop sign.
        bounding_box_3d (:py:class:`~pylot.utisl.BoundingBox3D`): 3D bounding
            box of the stop sign.
    """
    def __init__(self,
                 confidence: float,
                 bounding_box: BoundingBox2D = None,
                 id: int = -1,
                 transform: pylot.utils.Transform = None,
                 bounding_box_3d: BoundingBox3D = None):
        super(StopSign, self).__init__(bounding_box, confidence,
                                       'stop marking', id, transform)
        self.bounding_box_3d = bounding_box_3d

    @classmethod
    def from_simulator_actor(cls, actor):
        """Creates a stop sign from a simulator actor.

        Args:
            actor: A simulator stop sign actor.

        Returns:
            :py:class:`.StopSign`: A stop sign.
        """
        from carla import TrafficSign
        if not isinstance(actor, TrafficSign):
            raise ValueError('actor should be of type TrafficSign')
        transform = pylot.utils.Transform.from_simulator_transform(
            actor.get_transform())
        world_trigger_volume = actor.get_transform().transform(
            actor.trigger_volume.location)
        extent = pylot.utils.Vector3D.from_simulator_vector(
            actor.trigger_volume.extent)
        trigger_loc = pylot.utils.Location(world_trigger_volume.x,
                                           world_trigger_volume.y,
                                           world_trigger_volume.z)
        trigger_transform = pylot.utils.Transform(trigger_loc,
                                                  pylot.utils.Rotation())
        bbox = BoundingBox3D(trigger_transform, extent)
        return cls(1.0, id=actor.id, transform=transform, bounding_box_3d=bbox)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'StopSign(label: {}, confidence: {}, id: {}, transform: {}, '\
            'bbox: {})'.format(self.label, self.confidence, self.id,
                               self.transform, self.bounding_box)
