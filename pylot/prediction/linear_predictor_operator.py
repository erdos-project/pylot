"""Implements an operator that fits a linear model to predict trajectories."""

import erdos
import numpy as np

from pylot.prediction.messages import PredictionMessage
from pylot.prediction.obstacle_prediction import ObstaclePrediction
from pylot.utils import Location, Rotation, Transform


class LinearPredictorOperator(erdos.Operator):
    """Operator that implements a linear predictor.

    It takes (x,y) locations of agents in past, and fits a linear model to
    these locations.

    Args:
        tracking_stream (:py:class:`erdos.ReadStream`): The stream on which
            :py:class:`~pylot.perception.messages.ObstacleTrajectoriesMessage`
            are received.
        linear_prediction_stream (:py:class:`erdos.WriteStream`): Stream on
            which the operator sends
            :py:class:`~pylot.prediction.messages.PredictionMessage` messages.
        name (:obj:`str`): The name of the operator.
        flags (absl.flags): Object to be used to access absl flags.
        log_file_name (:obj:`str`, optional): Name of file where log messages
            are written to. If None, then messages are written to stdout.
    """
    def __init__(self,
                 tracking_stream,
                 linear_prediction_stream,
                 name,
                 flags,
                 log_file_name=None):
        tracking_stream.add_callback(self.generate_predicted_trajectories,
                                     [linear_prediction_stream])
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._flags = flags

    @staticmethod
    def connect(tracking_stream):
        """Connects the operator to other streams.

        Args:
            tracking_stream (:py:class:`erdos.ReadStream`): The stream on which
                :py:class:`~pylot.perception.messages.ObstacleTrajectoriesMessage`
                are received.

        Returns:
            :py:class:`erdos.WriteStream`: Stream on which the operator sends
            :py:class:`~pylot.prediction.messages.PredictionMessage` messages.
        """
        linear_prediction_stream = erdos.WriteStream()
        return [linear_prediction_stream]

    def generate_predicted_trajectories(self, msg, linear_prediction_stream):
        self._logger.debug('@{}: received trajectories message'.format(
            msg.timestamp))
        obstacle_predictions_list = []

        for obstacle in msg.obstacle_trajectories:
            # Time step matrices used in regression.
            num_steps = min(self._flags.prediction_num_past_steps,
                            len(obstacle.trajectory))
            ts = np.zeros((num_steps, 2))
            future_ts = np.zeros((self._flags.prediction_num_future_steps, 2))
            for t in range(num_steps):
                ts[t][0] = -t
                ts[t][1] = 1
            for i in range(self._flags.prediction_num_future_steps):
                future_ts[i][0] = i + 1
                future_ts[i][1] = 1

            xy = np.zeros((num_steps, 2))
            for t in range(num_steps):
                # t-th most recent step
                transform = obstacle.trajectory[-(t + 1)]
                xy[t][0] = transform.location.x
                xy[t][1] = transform.location.y
            linear_model_params = np.linalg.lstsq(ts, xy)[0]
            # Predict future steps and convert to list of locations.
            predict_array = np.matmul(future_ts, linear_model_params)
            predictions = []
            for t in range(self._flags.prediction_num_future_steps):
                predictions.append(
                    Transform(location=Location(x=predict_array[t][0],
                                                y=predict_array[t][1]),
                              rotation=Rotation()))
            # Get the current transform of the obstacle, which is the last
            # trajectory value.
            cur_transform = obstacle.trajectory[-1]
            obstacle_predictions_list.append(
                ObstaclePrediction(
                    obstacle.label,
                    obstacle.id,
                    cur_transform,
                    obstacle.bounding_box,
                    1.0,  # probability
                    predictions))
        linear_prediction_stream.send(
            PredictionMessage(msg.timestamp, obstacle_predictions_list))
