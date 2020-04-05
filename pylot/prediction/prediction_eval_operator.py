"""Implements an operator that eveluates prediction output."""

from collections import deque
import erdos

from pylot.perception.detection.utils import VEHICLE_LABELS
from pylot.perception.tracking.obstacle_trajectory import ObstacleTrajectory
from pylot.prediction.messages import PredictionMessage
from pylot.prediction.obstacle_prediction import ObstaclePrediction
from pylot.utils import Vector2D, time_epoch_ms


class PredictionEvalOperator(erdos.Operator):
    """Operator that calculates accuracy metrics for predicted trajectories.

    Args:
        pose_stream (:py:class:`erdos.ReadStream`): The stream on which pose
            info is received.
        tracking_stream (:py:class:`erdos.ReadStream`): The stream on which
            perfect
            :py:class:`~pylot.perception.messages.ObstacleTrajectoriesMessage`
            are received.
        prediction_stream (:py:class:`erdos.ReadStream`): Stream on
            :py:class:`~pylot.prediction.messages.PredictionMessage` are
            received from the prediction operator.
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, pose_stream, tracking_stream, prediction_stream, flags):
        pose_stream.add_callback(self._on_pose_update)
        tracking_stream.add_callback(self._on_tracking_update)
        prediction_stream.add_callback(self._on_prediction_update)
        erdos.add_watermark_callback(
            [pose_stream, tracking_stream, prediction_stream], [],
            self.on_watermark)
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            self.config.name + '-csv', self.config.csv_log_file_name)
        # Message buffers.
        self._prediction_msgs = deque()
        self._tracking_msgs = deque()
        self._pose_msgs = deque()
        # Accumulated list of predictions, from oldest to newest.
        self._predictions = deque(
            maxlen=self._flags.prediction_num_future_steps)

    @staticmethod
    def connect(pose_stream, tracking_stream, prediction_stream):
        return []

    def _on_prediction_update(self, msg):
        self._prediction_msgs.append(msg)

    def _on_tracking_update(self, msg):
        self._tracking_msgs.append(msg)

    def _on_pose_update(self, msg):
        self._pose_msgs.append(msg)

    def on_watermark(self, timestamp):
        """Invoked when all input streams have received a watermark.

        Args:
            timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of
                the watermark.
        """
        tracking_msg = self._tracking_msgs.popleft()
        prediction_msg = self._prediction_msgs.popleft()
        vehicle_transform = self._pose_msgs.popleft().data.transform

        # TODO: The evaluator assumes that the obstacle tracker assigns the
        # same ids to the obstacles as they have in the CARLA simulation.

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
            self._calculate_metrics(timestamp, ground_trajectories_dict,
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

    def _calculate_metrics(self, timestamp, ground_trajectories, predictions):
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
            # We remove altitude from the accuracy calculation because the
            # prediction operators do not currently predict altitude.
            predicted_trajectory = [
                Vector2D(transform.location.x, transform.location.y)
                for transform in obstacle.trajectory
            ]
            ground_trajectory = [
                Vector2D(transform.location.x, transform.location.y)
                for transform in ground_trajectories[obstacle.id].trajectory
            ]
            if obstacle.label in VEHICLE_LABELS:
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
                l2_distance += predicted_trajectory[-idx].l2_distance(
                    ground_trajectory[-idx])
                # Calculate ADE
                l1_distance += predicted_trajectory[-idx].l1_distance(
                    ground_trajectory[-idx])
            l2_distance /= len(predicted_trajectory)
            l1_distance /= len(predicted_trajectory)
            fde = predicted_trajectory[-1].l1_distance(ground_trajectory[-1])
            if obstacle.label in VEHICLE_LABELS:
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

        sim_time = timestamp.coordinates[0]
        # Log metrics.
        if person_cnt > 0:
            person_msd /= person_cnt
            person_ade /= person_cnt
            person_fde /= person_cnt
            self._logger.info('Person MSD is: {:.2f}'.format(person_msd))
            self._logger.info('Person ADE is: {:.2f}'.format(person_ade))
            self._logger.info('Person FDE is: {:.2f}'.format(person_fde))
            self._csv_logger.info('{},{},{},{},{:.2f}'.format(
                time_epoch_ms(), sim_time, self.config.name, 'person-MSD',
                person_msd))
            self._csv_logger.info('{},{},{},{},{:.2f}'.format(
                time_epoch_ms(), sim_time, self.config.name, 'person-ADE',
                person_ade))
            self._csv_logger.info('{},{},{},{},{:.2f}'.format(
                time_epoch_ms(), sim_time, self.config.name, 'person-FDE',
                person_fde))
        if vehicle_cnt > 0:
            vehicle_msd /= vehicle_cnt
            vehicle_ade /= vehicle_cnt
            vehicle_fde /= vehicle_cnt
            self._logger.info('Vehicle MSD is: {:.2f}'.format(vehicle_msd))
            self._logger.info('Vehicle ADE is: {:.2f}'.format(vehicle_ade))
            self._logger.info('Vehicle FDE is: {:.2f}'.format(vehicle_fde))
            self._csv_logger.info('{},{},{},{},{:.2f}'.format(
                time_epoch_ms(), sim_time, self.config.name, 'vehicle-MSD',
                vehicle_msd))
            self._csv_logger.info('{},{},{},{},{:.2f}'.format(
                time_epoch_ms(), sim_time, self.config.name, 'vehicle-ADE',
                vehicle_ade))
            self._csv_logger.info('{},{},{},{},{:.2f}'.format(
                time_epoch_ms(), sim_time, self.config.name, 'vehicle-FDE',
                vehicle_fde))
