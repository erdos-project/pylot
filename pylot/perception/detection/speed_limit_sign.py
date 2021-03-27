from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.detection.utils import BoundingBox2D
from pylot.utils import Transform


class SpeedLimitSign(Obstacle):
    """Class that stores info about a detected speed limit signs.

    Args:
        speed_limit (:obj:`int`): The speed limit (in km/h).
        confidence (:obj:`float`): The confidence of the detection.
        bounding_box (:py:class:`.BoundingBox2D`): The bounding box of the
            speed limit sign in camera view.
        id (:obj:`int`): Id associated with the sign.
        transform (:py:class:`~pylot.utils.transform`): Transform of the sign
            in the world.

    Attributes:
        speed_limit (:obj:`int`): The speed limit (in km/h).
        confidence (:obj:`float`): The confidence of the detection.
        bounding_box (:py:class:`.BoundingBox2D`): The bounding box of the
            speed limit sign in camera view.
        id (:obj:`int`): Id associated with the sign.
        transform (:py:class:`~pylot.utils.transform`): Transform of the sign
            in the world.
    """
    def __init__(self,
                 speed_limit: float,
                 confidence: float,
                 bounding_box: BoundingBox2D = None,
                 id: int = -1,
                 transform: Transform = None):
        super(SpeedLimitSign, self).__init__(bounding_box, confidence,
                                             'speed limit', id, transform)
        self.speed_limit = speed_limit

    @classmethod
    def from_simulator_actor(cls, actor):
        """Creates a detected speed limit sign from a simulator actor.

        Args:
            actor: A simulator speed limit sign actor.

        Returns:
            :py:class:`.SpeedLimitSign`: A detected speed limit sign.
        """
        from carla import TrafficSign
        if not isinstance(actor, TrafficSign):
            raise ValueError('actor should be of type TrafficSign')
        transform = Transform.from_simulator_transform(actor.get_transform())
        speed_limit = int(actor.type_id.split('.')[-1])
        return cls(speed_limit, 1.0, id=actor.id, transform=transform)

    def get_in_log_format(self):
        min_point = self.bounding_box_2D.get_min_point()
        max_point = self.bounding_box_2D.get_max_point()
        return (self.label + ' ' + str(self.speed_limit),
                ((min_point.x, min_point.y), (max_point.x, max_point.y)))

    def draw_on_frame(self, frame, bbox_color_map, ego_transform=None):
        assert self.bounding_box_2D, \
            'Speed limit sign does not have a 2D bounding box'
        text = '{} {} {:.1f}'.format(self.speed_limit, self.label,
                                     self.confidence)
        super(SpeedLimitSign, self).draw_on_frame(frame, bbox_color_map,
                                                  ego_transform, text)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'SpeedLimitSign(label: {}, limit: {}, '\
            'confidence: {}, id: {}, transform: {}, bbox: {})'.format(
                self.label, self.speed_limit, self.confidence, self.id,
                self.transform, self.bounding_box)
