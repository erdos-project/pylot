from collections import deque
import erdos

from pylot.perception.messages import ObstacleTrajectory
from pylot.prediction.messages import PredictionMessage
from pylot.prediction.utils import ObstaclePrediction
from pylot.utils import time_epoch_ms


class PredictionEvalOperator(erdos.Operator):
    """ Operator that calculates metrics for the quality of
        predicted trajectories."""
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
            self.on_notification)
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

    def on_notification(self, timestamp):
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
                                       cur_trajectory)
            # Evaluate the prediction corresponding to the current set of
            # ground truth past trajectories.
            self.calculate_metrics(ground_trajectories_dict,
                                   self._predictions[0].predictions)

        # Convert the prediction to world coordinates and append it to the
        # queue.
        obstacle_predictions_list = []
        for obstacle in prediction_msg.predictions:
            cur_trajectory = []
            for past_transform in obstacle.trajectory:
                world_coord = vehicle_transform * past_transform
                cur_trajectory.append(world_coord)
            obstacle_predictions_list.append(
                ObstaclePrediction(
                    obstacle.label,
                    obstacle.id,
                    1.0,  # probability
                    cur_trajectory))
        self._predictions.append(
            PredictionMessage(timestamp, obstacle_predictions_list))

    def calculate_metrics(self, ground_trajectories, predictions):
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

        # Pedestrian metrics.
        pedestrian_cnt = 0
        pedestrian_msd = 0.0
        pedestrian_ade = 0.0
        pedestrian_fde = 0.0

        for obstacle in predictions:
            predicted_trajectory = obstacle.trajectory
            ground_trajectory = ground_trajectories[obstacle.id].trajectory
            if obstacle.label == 'vehicle':
                vehicle_cnt += 1
            elif obstacle.label == 'pedestrian':
                pedestrian_cnt += 1
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
            elif obstacle.label == 'pedestrian':
                pedestrian_msd += l2_distance
                pedestrian_ade += l1_distance
                pedestrian_fde += fde
            else:
                raise ValueError('Unexpected obstacle label {}'.format(
                    obstacle.label))

        vehicle_msd /= vehicle_cnt
        vehicle_ade /= vehicle_cnt
        vehicle_fde /= vehicle_cnt
        pedestrian_msd /= pedestrian_cnt
        pedestrian_ade /= pedestrian_cnt
        pedestrian_fde /= pedestrian_cnt
        # Log metrics.
        self._logger.info('Vehicle MSD is: {}'.format(vehicle_msd))
        self._logger.info('Vehicle ADE is: {}'.format(vehicle_ade))
        self._logger.info('Vehicle FDE is: {}'.format(vehicle_fde))
        self._logger.info('Pedestrian MSD is: {}'.format(pedestrian_msd))
        self._logger.info('Pedestrian ADE is: {}'.format(pedestrian_ade))
        self._logger.info('Pedestrian FDE is: {}'.format(pedestrian_fde))

        self._csv_logger.info('{},{},{},{}'.format(time_epoch_ms(), self._name,
                                                   'vehicle-MSD', vehicle_msd))
        self._csv_logger.info('{},{},{},{}'.format(time_epoch_ms(), self._name,
                                                   'vehicle-ADE', vehicle_ade))
        self._csv_logger.info('{},{},{},{}'.format(time_epoch_ms(), self._name,
                                                   'vehicle-FDE', vehicle_fde))
        self._csv_logger.info('{},{},{},{}'.format(time_epoch_ms(), self._name,
                                                   'pedestrian-MSD',
                                                   pedestrian_msd))
        self._csv_logger.info('{},{},{},{}'.format(time_epoch_ms(), self._name,
                                                   'pedestrian-ADE',
                                                   pedestrian_ade))
        self._csv_logger.info('{},{},{},{}'.format(time_epoch_ms(), self._name,
                                                   'pedestrian-FDE',
                                                   pedestrian_fde))
