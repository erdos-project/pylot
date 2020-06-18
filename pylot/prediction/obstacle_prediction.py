from pylot.perception.detection.utils import BoundingBox2D, BoundingBox3D

VEHICLE_LABELS = {'car', 'bicycle', 'motorcycle', 'bus', 'truck', 'vehicle'}


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
        if not (isinstance(bounding_box, BoundingBox3D)
                or isinstance(bounding_box, BoundingBox2D)):
            raise ValueError('bounding box should be of type '
                             'BoundingBox2D or BoundingBox3D')
        self.bounding_box = bounding_box
        self.probability = probability
        self.trajectory = trajectory

    def is_person(self):
        return self.label == 'person'

    def is_speed_limit(self):
        return self.label in [
            'speed limit 30', 'speed limit 60', 'speed limit 90'
        ]

    def is_stop_sign(self):
        return self.label == 'stop sign' or self.label == 'stop marking'

    def is_traffic_light(self):
        return self.label in [
            'red traffic light', 'yellow traffic light', 'green traffic light',
            'off traffic light'
        ]

    def is_vehicle(self):
        # Might want to include train.
        return self.label in VEHICLE_LABELS

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return ('Prediction for obstacle {}, label {}, probability {}, '
                'trajectory {}'.format(self.id, self.label, self.probability,
                                       self.trajectory))
