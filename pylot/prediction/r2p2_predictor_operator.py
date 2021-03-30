import time
from collections import deque

import erdos

import numpy as np

import pylot.prediction.utils
from pylot.prediction.messages import PredictionMessage
from pylot.prediction.obstacle_prediction import ObstaclePrediction
from pylot.utils import Location, Transform, time_epoch_ms

import torch

try:
    from pylot.prediction.prediction.r2p2.r2p2_model import R2P2
except ImportError:
    raise Exception('Error importing R2P2.')


class R2P2PredictorOperator(erdos.Operator):
    """Wrapper operator for R2P2 ego-vehicle prediction module.

    Args:
        point_cloud_stream (:py:class:`erdos.ReadStream`, optional): Stream on
            which point cloud messages are received.
        tracking_stream (:py:class:`erdos.ReadStream`):
            Stream on which
            :py:class:`~pylot.perception.messages.ObstacleTrajectoriesMessage`
            are received.
        prediction_stream (:py:class:`erdos.ReadStream`): Stream on which
            :py:class:`~pylot.prediction.messages.PredictionMessage`
            messages are published.
        lidar_setup (:py:class:`pylot.drivers.sensor_setup.LidarSetup`): Setup
            of the lidar. This setup is used to get the maximum range of the
            lidar.
    """
    def __init__(self, point_cloud_stream: erdos.ReadStream,
                 tracking_stream: erdos.ReadStream,
                 time_to_decision_stream: erdos.ReadStream,
                 prediction_stream: erdos.WriteStream, flags, lidar_setup):
        print("WARNING: R2P2 predicts only vehicle trajectories")
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            self.config.name + '-csv', self.config.csv_log_file_name)
        self._flags = flags

        self._device = torch.device('cuda')
        self._r2p2_model = R2P2().to(self._device)
        state_dict = torch.load(flags.r2p2_model_path)
        self._r2p2_model.load_state_dict(state_dict)

        point_cloud_stream.add_callback(self.on_point_cloud_update)
        tracking_stream.add_callback(self.on_trajectory_update)
        time_to_decision_stream.add_callback(self.on_time_to_decision_update)
        erdos.add_watermark_callback([point_cloud_stream, tracking_stream],
                                     [prediction_stream], self.on_watermark)

        self._lidar_setup = lidar_setup

        self._point_cloud_msgs = deque()
        self._tracking_msgs = deque()

    @staticmethod
    def connect(point_cloud_stream: erdos.ReadStream,
                tracking_stream: erdos.ReadStream,
                time_to_decision_stream: erdos.ReadStream):
        prediction_stream = erdos.WriteStream()
        return [prediction_stream]

    def destroy(self):
        self._logger.warn('destroying {}'.format(self.config.name))

    @erdos.profile_method()
    def on_watermark(self, timestamp: erdos.Timestamp,
                     prediction_stream: erdos.WriteStream):
        self._logger.debug('@{}: received watermark'.format(timestamp))
        if timestamp.is_top:
            return
        point_cloud_msg = self._point_cloud_msgs.popleft()
        tracking_msg = self._tracking_msgs.popleft()

        start_time = time.time()
        nearby_trajectories, nearby_vehicle_ego_transforms, \
            nearby_trajectories_tensor, binned_lidars_tensor = \
            self._preprocess_input(tracking_msg, point_cloud_msg)

        num_predictions = len(nearby_trajectories)
        self._logger.info(
            '@{}: Getting R2P2 predictions for {} vehicles'.format(
                timestamp, num_predictions))

        if num_predictions == 0:
            prediction_stream.send(PredictionMessage(timestamp, []))
            return

        # Run the forward pass.
        z = torch.tensor(
            np.random.normal(size=(num_predictions,
                                   self._flags.prediction_num_future_steps,
                                   2))).to(torch.float32).to(self._device)
        model_start_time = time.time()
        prediction_array, _ = self._r2p2_model.forward(
            z, nearby_trajectories_tensor, binned_lidars_tensor)
        model_runtime = (time.time() - model_start_time) * 1000
        self._csv_logger.debug("{},{},{},{:.4f}".format(
            time_epoch_ms(), timestamp.coordinates[0],
            'r2p2-modelonly-runtime', model_runtime))
        prediction_array = prediction_array.cpu().detach().numpy()

        obstacle_predictions_list = self._postprocess_predictions(
            prediction_array, nearby_trajectories,
            nearby_vehicle_ego_transforms)
        runtime = (time.time() - start_time) * 1000
        self._csv_logger.debug("{},{},{},{:.4f}".format(
            time_epoch_ms(), timestamp.coordinates[0], 'r2p2-runtime',
            runtime))
        prediction_stream.send(
            PredictionMessage(timestamp, obstacle_predictions_list))

    def _preprocess_input(self, tracking_msg, point_cloud_msg):

        nearby_vehicle_trajectories, nearby_vehicle_ego_transforms = \
            tracking_msg.get_nearby_obstacles_info(
                self._flags.prediction_radius,
                lambda t: t.obstacle.is_vehicle())
        point_cloud = point_cloud_msg.point_cloud.points
        num_nearby_vehicles = len(nearby_vehicle_trajectories)
        if num_nearby_vehicles == 0:
            return [], [], [], []

        # Pad and rotate the trajectory of each nearby vehicle to its
        # coordinate frame. Also, remove the z-coordinate of the trajectory.
        nearby_trajectories_tensor = []  # Pytorch tensor for network input.

        for i in range(num_nearby_vehicles):
            cur_trajectory = nearby_vehicle_trajectories[
                i].get_last_n_transforms(self._flags.prediction_num_past_steps)
            cur_trajectory = np.stack(
                [[point.location.x, point.location.y, point.location.z]
                 for point in cur_trajectory])

            rotated_trajectory = nearby_vehicle_ego_transforms[
                i].inverse_transform_points(cur_trajectory)[:, :2]

            nearby_trajectories_tensor.append(rotated_trajectory)

        nearby_trajectories_tensor = np.stack(nearby_trajectories_tensor)
        nearby_trajectories_tensor = torch.tensor(
            nearby_trajectories_tensor).to(torch.float32).to(self._device)

        # For each vehicle, transform the lidar point cloud to that vehicle's
        # coordinate frame for purposes of prediction.
        binned_lidars = []
        for i in range(num_nearby_vehicles):
            rotated_point_cloud = nearby_vehicle_ego_transforms[
                i].inverse_transform_points(point_cloud)
            binned_lidars.append(
                pylot.prediction.utils.get_occupancy_grid(
                    rotated_point_cloud,
                    self._lidar_setup.transform.location.z,
                    int(self._lidar_setup.get_range_in_meters())))
        binned_lidars = np.concatenate(binned_lidars)
        binned_lidars_tensor = torch.tensor(binned_lidars).to(
            torch.float32).to(self._device)

        return nearby_vehicle_trajectories, nearby_vehicle_ego_transforms, \
            nearby_trajectories_tensor, binned_lidars_tensor

    def _postprocess_predictions(self, prediction_array, vehicle_trajectories,
                                 vehicle_ego_transforms):
        # The prediction_array consists of predictions with respect to each
        # vehicle. Transform each predicted trajectory to be in relation to the
        # ego-vehicle, then convert into an ObstaclePrediction.
        obstacle_predictions_list = []
        num_predictions = len(vehicle_trajectories)

        for idx in range(num_predictions):
            cur_prediction = prediction_array[idx]

            obstacle_transform = vehicle_trajectories[idx].obstacle.transform
            predictions = []
            # Because R2P2 only predicts (x,y) coordinates, we assume the
            # vehicle stays at the same height as its last location.
            for t in range(self._flags.prediction_num_future_steps):
                cur_point = vehicle_ego_transforms[idx].transform_points(
                    np.array([[
                        cur_prediction[t][0], cur_prediction[t][1],
                        vehicle_ego_transforms[idx].location.z
                    ]]))[0]
                # R2P2 does not predict vehicle orientation, so we use our
                # estimated orientation of the vehicle at its latest location.
                predictions.append(
                    Transform(location=Location(cur_point[0], cur_point[1],
                                                cur_point[2]),
                              rotation=vehicle_ego_transforms[idx].rotation))

            # Probability; currently a filler value because we are taking
            # just one sample from distribution
            obstacle_predictions_list.append(
                ObstaclePrediction(vehicle_trajectories[idx],
                                   obstacle_transform, 1.0, predictions))
        return obstacle_predictions_list

    def on_point_cloud_update(self, msg: erdos.Message):
        self._logger.debug('@{}: received point cloud message'.format(
            msg.timestamp))
        self._point_cloud_msgs.append(msg)

    def on_trajectory_update(self, msg: erdos.Message):
        self._logger.debug('@{}: received trajectories message'.format(
            msg.timestamp))
        self._tracking_msgs.append(msg)

    def on_time_to_decision_update(self, msg: erdos.Message):
        self._logger.debug('@{}: {} received ttd update {}'.format(
            msg.timestamp, self.config.name, msg))
