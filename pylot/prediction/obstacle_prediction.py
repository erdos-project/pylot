class ObstaclePrediction(object):
    """Class storing info about an obstacle prediction.

    Args:
        label (:obj:`str`): The label of the obstacle.
        id (:obj:`int`): The id of the obstacle.
        probability (:obj: `float`): The probability of the prediction.
        trajectory (list(:py:class:`~pylot.utils.Transform`)): The trajectory
            prediction.

    Attributes:
        label (:obj:`str`): The label of the obstacle.
        id (:obj:`int`): The id of the obstacle.
        probability (:obj: `float`): The probability of the prediction.
        trajectory (list(:py:class:`~pylot.utils.Transform`)): The trajectory
            prediction.
    """
    def __init__(self, label, id, probability, trajectory):
        self.label = label
        self.id = id
        self.probability = probability
        self.trajectory = trajectory

    def __str__(self):
        return ('Prediction for obstacle {}, label {}, probability {}, '
                'trajectory {}'.format(self.id, self.label, self.probability,
                                       self.trajectory))
