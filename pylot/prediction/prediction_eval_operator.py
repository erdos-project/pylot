"""Implements an operator that eveluates prediction output."""

from collections import deque

import erdos
from erdos import Message, ReadStream, Timestamp, WriteStream

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
    def __init__(self, pose_stream: ReadStream, tracking_stream: ReadStream,
                 prediction_stream: ReadStream,
                 finished_indicator_stream: WriteStream, flags):
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
    def connect(pose_stream: ReadStream, tracking_stream: ReadStream,
                prediction_stream: ReadStream):
        finished_indicator_stream = erdos.WriteStream()
        return [finished_indicator_stream]

    def destroy(self):
        self._logger.warn('destroying {}'.format(self.config.name))

    def _on_prediction_update(self, msg: Message):
        self._prediction_msgs.append(msg)

    def _on_tracking_update(self, msg: Message):
        self._tracking_msgs.append(msg)

    def _on_pose_update(self, msg: Message):
        self._pose_msgs.append(msg)

    def on_watermark(self, timestamp: Timestamp):
        """Invoked when all input streams have received a watermark.

        Args:
            timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of
                the watermark.
        """
        if timestamp.is_top:
            return
        tracking_msg = self._tracking_msgs.popleft()
        prediction_msg = self._prediction_msgs.popleft()
        vehicle_transform = self._pose_msgs.popleft().data.transform

        # TODO: The evaluator assumes that the obstacle tracker assigns the
        # same ids to the obstacles as they have in the simulation.

        # Start calculating metrics when we've taken sufficiently many steps.
        if len(self._predictions) == self._flags.prediction_num_future_steps:
            # Convert the tracking message to a dictionary with trajectories
            # in world coordinates, for speedup when calculating metrics.
            ground_trajectories_dict = {}
            for obstacle_trajectory in tracking_msg.obstacle_trajectories:
                obstacle_trajectory.to_world_coordinates(vehicle_transform)
                ground_trajectories_dict[obstacle_trajectory.id] = \
                    obstacle_trajectory
            # Evaluate the prediction corresponding to the current set of
            # ground truth past trajectories.
            self._calculate_metrics(timestamp, ground_trajectories_dict,
                                    self._predictions[0].predictions)

        # Convert the prediction to world coordinates and append it to the
        # queue.
        for obstacle_prediction in prediction_msg.predictions:
            obstacle_prediction.to_world_coordinates(vehicle_transform)
        self._predictions.append(prediction_msg)

    def _calculate_metrics(self, timestamp: Timestamp, ground_trajectories,
                           predictions):
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

        for obstacle_prediction in predictions:
            # We remove altitude from the accuracy calculation because the
            # prediction operators do not currently predict altitude.
            predicted_trajectory = [
                Vector2D(transform.location.x, transform.location.y)
                for transform in obstacle_prediction.predicted_trajectory
            ]
            ground_trajectory = [
                Vector2D(transform.location.x, transform.location.y)
                for transform in ground_trajectories[
                    obstacle_prediction.id].trajectory
            ]
            if obstacle_prediction.is_vehicle():
                vehicle_cnt += 1
            elif obstacle_prediction.is_person():
                person_cnt += 1
            else:
                raise ValueError('Unexpected obstacle label {}'.format(
                    obstacle_prediction.label))
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
            if obstacle_prediction.is_vehicle():
                vehicle_msd += l2_distance
                vehicle_ade += l1_distance
                vehicle_fde += fde
            elif obstacle_prediction.is_person():
                person_msd += l2_distance
                person_ade += l1_distance
                person_fde += fde
            else:
                raise ValueError('Unexpected obstacle label {}'.format(
                    obstacle_prediction.label))

        # Log metrics.
        sim_time = timestamp.coordinates[0]
        actor_cnt = person_cnt + vehicle_cnt
        if actor_cnt > 0:
            msd = (person_msd + vehicle_msd) / actor_cnt
            ade = (person_ade + vehicle_ade) / actor_cnt
            fde = (person_fde + vehicle_fde) / actor_cnt
            self._csv_logger.info('{},{},prediction,MSD,{:.4f}'.format(
                time_epoch_ms(), sim_time, msd))
            self._csv_logger.info('{},{},prediction,ADE,{:.4f}'.format(
                time_epoch_ms(), sim_time, ade))
            self._csv_logger.info('{},{},prediction,FDE,{:.4f}'.format(
                time_epoch_ms(), sim_time, fde))
        if person_cnt > 0:
            person_msd /= person_cnt
            person_ade /= person_cnt
            person_fde /= person_cnt
            self._logger.info('Person MSD is: {:.4f}'.format(person_msd))
            self._logger.info('Person ADE is: {:.4f}'.format(person_ade))
            self._logger.info('Person FDE is: {:.4f}'.format(person_fde))
            self._csv_logger.info('{},{},prediction,person-MSD,{:.4f}'.format(
                time_epoch_ms(), sim_time, person_msd))
            self._csv_logger.info('{},{},prediction,person-ADE,{:.4f}'.format(
                time_epoch_ms(), sim_time, person_ade))
            self._csv_logger.info('{},{},prediction,person-FDE,{:.4f}'.format(
                time_epoch_ms(), sim_time, person_fde))
        if vehicle_cnt > 0:
            vehicle_msd /= vehicle_cnt
            vehicle_ade /= vehicle_cnt
            vehicle_fde /= vehicle_cnt
            self._logger.info('Vehicle MSD is: {:.4f}'.format(vehicle_msd))
            self._logger.info('Vehicle ADE is: {:.4f}'.format(vehicle_ade))
            self._logger.info('Vehicle FDE is: {:.4f}'.format(vehicle_fde))
            self._csv_logger.info('{},{},prediction,vehicle-MSD,{:.4f}'.format(
                time_epoch_ms(), sim_time, vehicle_msd))
            self._csv_logger.info('{},{},prediction,vehicle-ADE,{:.4f}'.format(
                time_epoch_ms(), sim_time, vehicle_ade))
            self._csv_logger.info('{},{},prediction,vehicle-FDE,{:.4f}'.format(
                time_epoch_ms(), sim_time, vehicle_fde))
