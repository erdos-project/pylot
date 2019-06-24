from erdos.message import Message


class ObjPrediction(object):
    def __init__(self, obj_id, probability, trajectory):
        self.id = obj_id
        self.probability = probability
        # List of (x, y, yaw, timestamp).
        self.trajectory = trajectory

    def __str__(self):
        return 'Object {}, probability {}, trajectory {}'.format(
            self.id, self.probability, self.trajectory)


class PredictionMessage(Message):
    def __init__(self, timestamp, predictions):
        super(PredictionMessage, self).__init__(None, timestamp, 'default')
        self.predictions = predictions

    def __str__(self):
        return 'PredictionMessage(timestamp: {}, predictions: {})'.format(
            self.timestamp, self.predictions)
