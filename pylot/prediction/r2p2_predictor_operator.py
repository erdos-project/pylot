import erdos

from collections import deque
import numpy as np
import os
import torch

try:
    from pylot.prediction.prediction.r2p2.r2p2_model import R2P2
except ImportError:
    raise Exception('Error importing R2P2.')

import pylot.prediction.flags
from pylot.prediction.messages import PredictionMessage
from pylot.prediction.obstacle_prediction import ObstaclePrediction
from pylot.utils import Location, Rotation, Transform


class R2P2PredictorOperator(erdos.Operator):
    """Wrapper operator for R2P2 ego-vehicle prediction module."""

    def __init__(self,
                 can_bus_stream,
                 point_cloud_stream,
                 tracking_stream,
                 vehicle_id_stream,
                 prediction_stream,
                 flags):
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags

        self._device = torch.device('cuda')
        self._r2p2_model = R2P2().to(self._device)
        state_dict = torch.load(flags.r2p2_model_path)
        self._r2p2_model.load_state_dict(state_dict)

        can_bus_stream.add_callback(self.on_can_bus_update)
        point_cloud_stream.add_callback(self.on_point_cloud_update)
        tracking_stream.add_callback(self.on_trajectory_update)
        erdos.add_watermark_callback(
            [can_bus_stream, point_cloud_stream, tracking_stream],
            [prediction_stream], self.on_watermark)

        self._vehicle_id_stream = vehicle_id_stream

        self._can_bus_msgs = deque()
        self._point_cloud_msgs = deque()
        self._tracking_msgs = deque()

        
    @staticmethod
    def connect(can_bus_stream, point_cloud_stream, tracking_stream, vehicle_id_stream):
        prediction_stream = erdos.WriteStream()
        return [prediction_stream]


    def run(self):
        # Read the ego vehicle id from the vehicle id stream.
        vehicle_id_msg = self._vehicle_id_stream.read()
        self.vehicle_id = vehicle_id_msg.data
        self._logger.debug(
            "The R2P2PredictorOperator received the vehicle id: {}".format(
                self.vehicle_id))


    def on_watermark(self, timestamp, prediction_stream):
        self._logger.debug('@{}: received watermark'.format(timestamp))

        can_bus_msg = self._can_bus_msgs.popleft()
        point_cloud_msg = self._point_cloud_msgs.popleft()
        tracking_msg = self._tracking_msgs.popleft()

        binned_lidar = self.get_occupancy_grid(point_cloud_msg.point_cloud.points)

        self._vehicle_transform = can_bus_msg.data.transform
        
        # Get the ego-vehicle track and bounding box.
        ego_vehicle_trajectory = None
        for vehicle in tracking_msg.obstacle_trajectories:
            if vehicle.id == self.vehicle_id:
                ego_vehicle_trajectory = vehicle.trajectory
                ego_vehicle_bbox = vehicle.bounding_box
                break
        assert ego_vehicle_trajectory is not None, "ego-vehicle trajectory not found"
        ego_vehicle_trajectory = np.stack(
            [[point.location.x, point.location.y] for
              point in ego_vehicle_trajectory])

        # If we haven't seen enough past locations yet, pad the start
        # of the trajectory with copies of the earliest location.
        num_past_locations = ego_vehicle_trajectory.shape[0]
        if num_past_locations < self._flags.prediction_num_past_steps:
            initial_copies = np.repeat([np.array(ego_vehicle_trajectory[0])],
                                       self._flags.prediction_num_past_steps - num_past_locations,
                                       axis=0)
            ego_vehicle_trajectory = np.vstack(
                (initial_copies, ego_vehicle_trajectory))
        ego_vehicle_trajectory = np.expand_dims(ego_vehicle_trajectory, axis=0)

        z = torch.tensor(np.random.normal(
            size=(1, self._flags.prediction_num_future_steps, 2))).to(
            torch.float32).to(self._device)
        ego_vehicle_trajectory = torch.tensor(ego_vehicle_trajectory).to(torch.float32).to(self._device)
        binned_lidar = torch.tensor(binned_lidar).to(torch.float32).to(self._device)

        prediction_array, _ = self._r2p2_model.forward(z,
                                  ego_vehicle_trajectory,
                                  binned_lidar)
        prediction_array = prediction_array.cpu().detach().numpy()[0]

        # Convert prediction array to a list of Location objects.
        predictions = []
        for t in range(self._flags.prediction_num_future_steps):
            predictions.append(
                Transform(location=Location(x=prediction_array[t][0],
                                            y=prediction_array[t][1]),
                          rotation=Rotation()))

        obstacle_predictions_list = []
        obstacle_predictions_list.append(
            ObstaclePrediction('vehicle',
                               self.vehicle_id,
                               self._vehicle_transform,
                               ego_vehicle_bbox,
                               1.0, # Probability; currently a filler value because we are taking just one sample from distribution
                               predictions))
        prediction_stream.send(
            PredictionMessage(timestamp, obstacle_predictions_list))

    def get_occupancy_grid(self, point_cloud):
        """Get occupancy grids at two different heights."""

        z_threshold = -3.0 # Might need to be adjusted.

        above_mask = point_cloud[:, 2] > z_threshold
        # print ("Points above threshold:", sum(above_mask))
        # print ("Points below threshold:", sum(1 - above_mask))

        def get_occupancy_from_masked_lidar(mask):
            masked_lidar = point_cloud[mask]
            meters_max = 50
            pixels_per_meter = 2
            xbins = np.linspace(-meters_max, meters_max, meters_max * 2 * pixels_per_meter + 1) 
            ybins = xbins
            grid = np.histogramdd(masked_lidar[..., :2], bins=(xbins, ybins))[0]
            grid[grid > 0.] = 1
            return grid

        feats = ()
        # Above z_threshold.
        feats += (get_occupancy_from_masked_lidar(above_mask),)
        # Below z_threshold.
        feats += (get_occupancy_from_masked_lidar((1 - above_mask).astype(np.bool)),)

        stacked_feats = np.stack(feats, axis=-1)
        return np.expand_dims(stacked_feats, axis=0)

    def on_can_bus_update(self, msg):
        self._logger.debug('@{}: received can bus message'.format(
            msg.timestamp))
        self._can_bus_msgs.append(msg)

    def on_point_cloud_update(self, msg):
        self._logger.debug('@{}: received point cloud message'.format(
            msg.timestamp))
        self._point_cloud_msgs.append(msg)

    def on_trajectory_update(self, msg):
        self._logger.debug('@{}: received trajectories message'.format(
            msg.timestamp))
        self._tracking_msgs.append(msg)
