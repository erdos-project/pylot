import erdust


class ObjPrediction(object):
    def __init__(self, obj_class, obj_id, probability, trajectory):
        self.obj_class = obj_class
        self.id = obj_id
        self.probability = probability
        # List of (x, y, yaw, timestamp).
        self.trajectory = trajectory

    def __str__(self):
        return 'Object {}, class {}, probability {}, trajectory {}'.format(
            self.id, self.obj_class, self.probability, self.trajectory)


class PredictionMessage(erdust.Message):
    def __init__(self, timestamp, predictions):
        super(PredictionMessage, self).__init__(timestamp, None)
        self.predictions = predictions

    def __str__(self):
        return 'PredictionMessage(timestamp: {}, predictions: {})'.format(
            self.timestamp, self.predictions)
