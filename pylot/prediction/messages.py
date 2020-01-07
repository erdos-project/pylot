import erdos


class PredictionMessage(erdos.Message):
    """Message class to be used to send obstacle predictions."""
    def __init__(self, timestamp, predictions):
        """ Constructs a PredictionMessage.

        Args:
            timestamp: The timestamp of the message.
            predictions: A list of pylot.prediction.utils.ObstaclePredictions.
        """
        super(PredictionMessage, self).__init__(timestamp, None)
        self.predictions = predictions

    def __str__(self):
        return 'PredictionMessage(timestamp: {}, predictions: {})'.format(
            self.timestamp, self.predictions)
