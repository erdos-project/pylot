"""Implements an operator that fits a linear model to predict trajectories."""

from typing import Any

import erdos
from erdos.operator import TwoInOneOut
from erdos.context import TwoInOneOutContext

import numpy as np

from pylot.perception.messages import ObstacleTrajectoriesMessageTuple
from pylot.prediction.obstacle_prediction import ObstaclePrediction
from pylot.utils import Location, Transform


class LinearPredictorOperator(TwoInOneOut):
    """Operator that implements a linear predictor.

    It takes (x,y) locations of agents in past, and fits a linear model to
    these locations.

    Args:
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, flags):
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags

    def on_left_data(self, context: TwoInOneOutContext,
                     data: ObstacleTrajectoriesMessageTuple):
        self._logger.debug('@{}: received trajectories message'.format(
            context.timestamp))
        obstacle_predictions_list = []

        nearby_obstacle_trajectories, nearby_obstacles_ego_transforms = \
            data.get_nearby_obstacles_info(self._flags.prediction_radius)
        num_predictions = len(nearby_obstacle_trajectories)

        self._logger.info(
            '@{}: Getting linear predictions for {} obstacles'.format(
                context.timestamp, num_predictions))

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
        context.write_stream.send(
            erdos.Message(context.timestamp, obstacle_predictions_list))

    def on_right_data(self, context: TwoInOneOutContext, data: Any):
        self._logger.debug('@{}: {} received ttd update {}'.format(
            context.timestamp, self.config.name, data))

    def destroy(self):
        self._logger.warn('destroying {}'.format(self.config.name))
