"""Implements an operator that eveluates prediction output."""

from collections import deque
import erdos

from pylot.perception.tracking.obstacle_trajectory import ObstacleTrajectory
from pylot.prediction.messages import PredictionMessage
from pylot.prediction.obstacle_prediction import ObstaclePrediction
from pylot.utils import time_epoch_ms


class PredictionEvalOperator(erdos.Operator):
    """Operator that calculates accuracy metrics for predicted trajectories.

    Args:
        can_bus_stream (:py:class:`erdos.ReadStream`): The stream on which can
            bus info is received.
        tracking_stream (:py:class:`erdos.ReadStream`): The stream on which
            perfect
            :py:class:`~pylot.perception.messages.ObstacleTrajectoriesMessage`
            are received.
        prediction_stream (:py:class:`erdos.ReadStream`): Stream on
            :py:class:`~pylot.prediction.messages.PredictionMessage` are
            received from the prediction operator.
        name (:obj:`str`): The name of the operator.
        flags (absl.flags): Object to be used to access absl flags.
        log_file_name (:obj:`str`, optional): Name of file where log messages
            are written to. If None, then messages are written to stdout.
        csv_file_name (:obj:`str`, optional): Name of file where stats logs are
            written to. If None, then messages are written to stdout.
    """
    def __init__(self,
                 can_bus_stream,
                 tracking_stream,
                 prediction_stream,
                 name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        can_bus_stream.add_callback(self._on_can_bus_update)
        tracking_stream.add_callback(self._on_tracking_update)
        prediction_stream.add_callback(self._on_prediction_update)
        erdos.add_watermark_callback(
            [can_bus_stream, tracking_stream, prediction_stream], [],
            self.on_watermark)
        self._name = name
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self._name, log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            self._name + '-csv', csv_file_name)
        # Message buffers.
        self._prediction_msgs = deque()
        self._tracking_msgs = deque()
        self._can_bus_msgs = deque()
        # Accumulated list of predictions, from oldest to newest.
        self._predictions = deque(
            maxlen=self._flags.prediction_num_future_steps)

    @staticmethod
    def connect(can_bus_stream, tracking_stream, prediction_stream):
        return []

    def _on_prediction_update(self, msg):
        self._prediction_msgs.append(msg)

    def _on_tracking_update(self, msg):
        self._tracking_msgs.append(msg)

    def _on_can_bus_update(self, msg):
        self._can_bus_msgs.append(msg)

    def on_watermark(self, timestamp):
        """Invoked when all input streams have received a watermark.

        Args:
            timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of
                the watermark.
        """
        tracking_msg = self._tracking_msgs.popleft()
        prediction_msg = self._prediction_msgs.popleft()
        vehicle_transform = self._can_bus_msgs.popleft().data.transform

        # Start calculating metrics when we've taken sufficiently many steps.
        if len(self._predictions) == self._flags.prediction_num_future_steps:
            # Convert the tracking message to a dictionary with trajectories
            # in world coordinates, for speedup when calculating metrics.
            ground_trajectories_dict = {}
            for obstacle in tracking_msg.obstacle_trajectories:
                cur_trajectory = []
                for past_transform in obstacle.trajectory:
                    world_coord = vehicle_transform * past_transform
                    cur_trajectory.append(world_coord)

                ground_trajectories_dict[obstacle.id] = \
                    ObstacleTrajectory(obstacle.label,
                                       obstacle.id,
                                       obstacle.bounding_box,
                                       cur_trajectory)
            # Evaluate the prediction corresponding to the current set of
            # ground truth past trajectories.
            self._calculate_metrics(ground_trajectories_dict,
                                    self._predictions[0].predictions)

        # Convert the prediction to world coordinates and append it to the
        # queue.
        obstacle_predictions_list = []
        for obstacle in prediction_msg.predictions:
            cur_trajectory = []
            for past_transform in obstacle.trajectory:
                world_coord = vehicle_transform * past_transform
                cur_trajectory.append(world_coord)
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
                    cur_trajectory))
        self._predictions.append(
            PredictionMessage(timestamp, obstacle_predictions_list))

    def _calculate_metrics(self, ground_trajectories, predictions):
        """ Calculates and logs MSD (mean squared distance), ADE (average
            displacement error), and FDE (final displacement error).

            Args:
                ground_trajectories: A dict of perfect past trajectories.
                predictions: A list of obstacle predictions.
        """
        # Vehicle metrics.
        vehicle_cnt = 0
        vehicle_msd = 0.0
        vehicle_ade = 0.0
        vehicle_fde = 0.0

        # Person metrics.
        person_cnt = 0
        person_msd = 0.0
        person_ade = 0.0
        person_fde = 0.0

        for obstacle in predictions:
            predicted_trajectory = obstacle.trajectory
            ground_trajectory = ground_trajectories[obstacle.id].trajectory
            if obstacle.label == 'vehicle':
                vehicle_cnt += 1
            elif obstacle.label == 'person':
                person_cnt += 1
            else:
                raise ValueError('Unexpected obstacle label {}'.format(
                    obstacle.label))
            l2_distance = 0.0
            l1_distance = 0.0
            for idx in range(1, len(predicted_trajectory) + 1):
                # Calculate MSD
                l2_distance += predicted_trajectory[-idx].location.distance(
                    ground_trajectory[-idx].location)
                # Calculate ADE
                l1_distance += predicted_trajectory[-idx].location.l1_distance(
                    ground_trajectory[-idx].location)
            l2_distance /= len(predicted_trajectory)
            l1_distance /= len(predicted_trajectory)
            fde = predicted_trajectory[-1].location.l1_distance(
                ground_trajectory[-1].location)
            if obstacle.label == 'vehicle':
                vehicle_msd += l2_distance
                vehicle_ade += l1_distance
                vehicle_fde += fde
            elif obstacle.label == 'person':
                person_msd += l2_distance
                person_ade += l1_distance
                person_fde += fde
            else:
                raise ValueError('Unexpected obstacle label {}'.format(
                    obstacle.label))

        vehicle_msd /= vehicle_cnt
        vehicle_ade /= vehicle_cnt
        vehicle_fde /= vehicle_cnt
        person_msd /= person_cnt
        person_ade /= person_cnt
        person_fde /= person_cnt
        # Log metrics.
        self._logger.info('Vehicle MSD is: {}'.format(vehicle_msd))
        self._logger.info('Vehicle ADE is: {}'.format(vehicle_ade))
        self._logger.info('Vehicle FDE is: {}'.format(vehicle_fde))
        self._logger.info('Person MSD is: {}'.format(person_msd))
        self._logger.info('Person ADE is: {}'.format(person_ade))
        self._logger.info('Person FDE is: {}'.format(person_fde))

        self._csv_logger.info('{},{},{},{}'.format(time_epoch_ms(), self._name,
                                                   'vehicle-MSD', vehicle_msd))
        self._csv_logger.info('{},{},{},{}'.format(time_epoch_ms(), self._name,
                                                   'vehicle-ADE', vehicle_ade))
        self._csv_logger.info('{},{},{},{}'.format(time_epoch_ms(), self._name,
                                                   'vehicle-FDE', vehicle_fde))
        self._csv_logger.info('{},{},{},{}'.format(time_epoch_ms(), self._name,
                                                   'person-MSD', person_msd))
        self._csv_logger.info('{},{},{},{}'.format(time_epoch_ms(), self._name,
                                                   'person-ADE', person_ade))
        self._csv_logger.info('{},{},{},{}'.format(time_epoch_ms(), self._name,
                                                   'person-FDE', person_fde))
