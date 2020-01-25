from pylot.perception.detection.utils import DetectedObstacle
import pylot.utils


class DetectedSpeedLimit(DetectedObstacle):
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
        limit (:obj:`int`): The speed limit (in km/h).
    """
    def __init__(self,
                 speed_limit,
                 confidence,
                 bounding_box=None,
                 id=-1,
                 transform=None):
        super(DetectedSpeedLimit, self).__init__(bounding_box, confidence,
                                                 'speed limit', id, transform)
        self.limit = speed_limit

    @classmethod
    def from_carla_actor(cls, actor):
        """Creates a detected speed limit sign from a CARLA actor.

        Args:
            actor (carla.TrafficSign): A carla speed limit sign actor.

        Returns:
            :py:class:`.DetectedSpeedLimit`: A detected speed limit sign.
        """
        import carla
        if not isinstance(actor, carla.TrafficSign):
            raise ValueError('actor should be of type carla.TrafficSign')
        transform = pylot.utils.Transform.from_carla_transform(
            actor.get_transform())
        speed_limit = int(actor.type_id.split('.')[-1])
        return cls(speed_limit, 1.0, id=actor.id, transform=transform)

    def get_bbox_label(self):
        return (self.label + ' ' + str(self.limit),
                (self.bounding_box.get_min_point(),
                 self.bounding_box.get_max_point()))

    def visualize_on_img(self, image_np, bbox_color_map):
        text = '{} {} {:.1f}'.format(self.limit, self.label, self.confidence)
        super(DetectedSpeedLimit,
              self).visualize_on_img(image_np, bbox_color_map, text)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'DetectedSpeedLimit(label: {}, limit: {}, '\
            'confidence: {}, id: {}, transform: {}, bbox: {})'.format(
                self.label, self.limit, self.confidence, self.id,
                self.transform, self.bounding_box)
