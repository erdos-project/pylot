class ObstaclePrediction(object):
    """Class storing info about an obstacle prediction.

    Attributes:
        label: The label of the obstacle.
        id: The id of the obstacle.
        probability: The probability of the prediction.
        trajectory: The prediction as a list of pylot.utils.Transforms.
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
