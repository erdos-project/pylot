"""Implements an operator that fits a linear model to predict trajectories."""

import erdos
from erdos import Message, ReadStream, WriteStream

import numpy as np

from pylot.prediction.messages import PredictionMessage
from pylot.prediction.obstacle_prediction import ObstaclePrediction
from pylot.utils import Location, Transform


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
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, tracking_stream: ReadStream,
                 time_to_decision_stream: ReadStream,
                 linear_prediction_stream: WriteStream, flags):
        tracking_stream.add_callback(self.generate_predicted_trajectories,
                                     [linear_prediction_stream])
        time_to_decision_stream.add_callback(self.on_time_to_decision_update)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags

    @staticmethod
    def connect(tracking_stream: ReadStream,
                time_to_decision_stream: ReadStream):
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

    def destroy(self):
        self._logger.warn('destroying {}'.format(self.config.name))

    def on_time_to_decision_update(self, msg):
        self._logger.debug('@{}: {} received ttd update {}'.format(
            msg.timestamp, self.config.name, msg))

    @erdos.profile_method()
    def generate_predicted_trajectories(self, msg: Message,
                                        linear_prediction_stream: WriteStream):
        self._logger.debug('@{}: received trajectories message'.format(
            msg.timestamp))
        obstacle_predictions_list = []

        nearby_obstacle_trajectories, nearby_obstacles_ego_transforms = \
            msg.get_nearby_obstacles_info(self._flags.prediction_radius)
        num_predictions = len(nearby_obstacle_trajectories)

        self._logger.info(
            '@{}: Getting linear predictions for {} obstacles'.format(
                msg.timestamp, num_predictions))

        for idx in range(len(nearby_obstacle_trajectories)):
            obstacle_trajectory = nearby_obstacle_trajectories[idx]
            # Time step matrices used in regression.
            num_steps = min(self._flags.prediction_num_past_steps,
                            len(obstacle_trajectory.trajectory))
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
                transform = obstacle_trajectory.trajectory[-(t + 1)]
                xy[t][0] = transform.location.x
                xy[t][1] = transform.location.y
            linear_model_params = np.linalg.lstsq(ts, xy, rcond=None)[0]
            # Predict future steps and convert to list of locations.
            predict_array = np.matmul(future_ts, linear_model_params)
            predictions = []
            for t in range(self._flags.prediction_num_future_steps):
                # Linear prediction does not predict vehicle orientation, so we
                # use our estimated orientation of the vehicle at its latest
                # location.
                predictions.append(
                    Transform(location=Location(x=predict_array[t][0],
                                                y=predict_array[t][1]),
                              rotation=nearby_obstacles_ego_transforms[idx].
                              rotation))
            obstacle_predictions_list.append(
                ObstaclePrediction(obstacle_trajectory,
                                   obstacle_trajectory.obstacle.transform, 1.0,
                                   predictions))
        linear_prediction_stream.send(
            PredictionMessage(msg.timestamp, obstacle_predictions_list))
