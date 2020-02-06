from pylot.perception.detection.utils import BoundingBox3D


class ObstaclePrediction(object):
    """Class storing info about an obstacle prediction.

    Args:
        label (:obj:`str`): The label of the obstacle.
        id (:obj:`int`): The id of the obstacle.
        transform (:py:class:`~pylot.utils.Transform`): The current transform
            of the obstacle.
        bounding_box (:py:class:`~pylot.perception.detection.utils.BoundingBox3D`):
            The current bounding box of the obstacle.
        probability (:obj: `float`): The probability of the prediction.
        trajectory (list(:py:class:`~pylot.utils.Transform`)): The trajectory
            prediction.

    Attributes:
        label (:obj:`str`): The label of the obstacle.
        id (:obj:`int`): The id of the obstacle.
        transform (:py:class:`~pylot.utils.Transform`): The current transform
            of the obstacle.
        bounding_box (:py:class:`~pylot.perception.detection.utils.BoundingBox3D`):
            The current bounding box of the obstacle.
        probability (:obj: `float`): The probability of the prediction.
        trajectory (list(:py:class:`~pylot.utils.Transform`)): The trajectory
            prediction.
    """
    def __init__(self, label, id, transform, bounding_box, probability,
                 trajectory):
        self.label = label
        self.id = id
        self.transform = transform
        self.bounding_box = bounding_box
        if not isinstance(bounding_box, BoundingBox3D):
            raise ValueError('bounding box should be of type BoundingBox3D')
        self.probability = probability
        self.trajectory = trajectory

    def __str__(self):
        return ('Prediction for obstacle {}, label {}, probability {}, '
                'trajectory {}'.format(self.id, self.label, self.probability,
                                       self.trajectory))
