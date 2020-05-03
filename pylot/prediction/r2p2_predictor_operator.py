import erdos

from collections import deque
import math
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
from pylot.utils import Location, Rotation, Transform, Vector2D


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
            of the lidar. This setup is used to get the maximum range of the lidar.
        """

    def __init__(self,
                 point_cloud_stream,
                 tracking_stream,
                 prediction_stream,
                 flags,
                 lidar_setup):
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags

        self._device = torch.device('cuda')
        self._r2p2_model = R2P2().to(self._device)
        state_dict = torch.load(flags.r2p2_model_path)
        self._r2p2_model.load_state_dict(state_dict)

        point_cloud_stream.add_callback(self.on_point_cloud_update)
        tracking_stream.add_callback(self.on_trajectory_update)
        erdos.add_watermark_callback(
            [point_cloud_stream, tracking_stream],
            [prediction_stream], self.on_watermark)

        self._lidar_setup = lidar_setup

        self._point_cloud_msgs = deque()
        self._tracking_msgs = deque()

        
    @staticmethod
    def connect(point_cloud_stream, tracking_stream):
        prediction_stream = erdos.WriteStream()
        return [prediction_stream]


    def on_watermark(self, timestamp, prediction_stream):
        self._logger.debug('@{}: received watermark'.format(timestamp))

        point_cloud_msg = self._point_cloud_msgs.popleft()
        tracking_msg = self._tracking_msgs.popleft()

        all_vehicles = [trajectory for trajectory in tracking_msg.obstacle_trajectories if trajectory.label == 'vehicle']
        closest_vehicles, ego_vehicle = self._get_closest_vehicles(all_vehicles)

        closest_vehicles_ego_transforms = self._get_closest_vehicles_ego_transforms(closest_vehicles, ego_vehicle)

        num_predictions = len(closest_vehicles)

        # For each vehicle, transform lidar to that vehicle's coordinate frame
        # for purposes of prediction.
        point_cloud = point_cloud_msg.point_cloud.points
        binned_lidars = []

        for i in range(num_predictions):
            rotated_point_cloud = closest_vehicles_ego_transforms[i].inverse_transform_points(point_cloud)
            binned_lidars.append(self._get_occupancy_grid(rotated_point_cloud))

        binned_lidars = np.concatenate(binned_lidars)

        # Rotate and pad the closest trajectories.
        closest_trajectories = []
        for i in range(num_predictions):
            cur_trajectory = np.stack([[point.location.x,
                                        point.location.y,
                                        point.location.z] \
                for point in closest_vehicles[i].trajectory])

            # Remove z-coordinate from trajectory.
            closest_trajectories.append(
                closest_vehicles_ego_transforms[i].inverse_transform_points(cur_trajectory)[:,:2])
        closest_trajectories = np.stack([self.pad_trajectory(t) for t in closest_trajectories])

        # Run the forward pass.
        z = torch.tensor(np.random.normal(
            size=(num_predictions, self._flags.prediction_num_future_steps, 2))).to(
            torch.float32).to(self._device)
        closest_trajectories = torch.tensor(closest_trajectories).to(torch.float32).to(self._device)
        binned_lidars = torch.tensor(binned_lidars).to(torch.float32).to(self._device)
        prediction_array, _ = self._r2p2_model.forward(z,
                                  closest_trajectories,
                                  binned_lidars)
        prediction_array = prediction_array.cpu().detach().numpy()

        # Transform each predicted trajectory to be in relation to the
        # ego-vehicle, then convert into an ObstaclePrediction. Because R2P2
        # performs top-down prediction, we assume the vehicle stays at the same
        # height as its last location.
        obstacle_predictions_list = []

        for idx in range(num_predictions):
            cur_prediction = prediction_array[idx]

            last_location = closest_vehicles_ego_transforms[idx].location
            predictions = []
            for t in range(self._flags.prediction_num_future_steps):
                cur_point = closest_vehicles_ego_transforms[idx].transform_points(
                                 np.array([[cur_prediction[t][0],
                                            cur_prediction[t][1],
                                            last_location.z]]))[0]
                predictions.append(Transform(
                    location=Location(cur_point[0], cur_point[1], cur_point[2]),
                    rotation=Rotation()))
 
            obstacle_predictions_list.append(
                ObstaclePrediction('vehicle',
                                   closest_vehicles[idx].id,
                                   closest_vehicles_ego_transforms[idx],
                                   closest_vehicles[idx].bounding_box,
                                   1.0, # Probability; currently a filler value because we are taking just one sample from distribution
                                   predictions))
        prediction_stream.send(
            PredictionMessage(timestamp, obstacle_predictions_list))


    def pad_trajectory(self, trajectory):
        # Take the appropriate number of past steps as specified by flags. 
        # If we have not seen enough past locations of the vehicle, pad the
        # trajectory with the appropriate number of copies of the original
        # locations.
        num_past_locations = trajectory.shape[0]
        if num_past_locations < self._flags.prediction_num_past_steps:
            initial_copies = np.repeat([np.array(trajectory[0])],
                                       self._flags.prediction_num_past_steps - num_past_locations,
                                       axis=0)
            trajectory = np.vstack(
                (initial_copies, trajectory))
        elif num_past_locations > self._flags.prediction_num_past_steps:
            trajectory = trajectory[-self._flags.prediction_num_past_steps:]
        assert trajectory.shape[0] == self._flags.prediction_num_past_steps
        return trajectory


    def _get_closest_vehicles(self, all_vehicles):
        # Number of vehicles to make predictions for.
        distances = [v.trajectory[-1].get_vector_magnitude_angle(Location())[1]
                        for v in all_vehicles]
        sorted_vehicles = [v for v, d in sorted(
            zip(all_vehicles, distances), key=lambda pair: pair[1])
            if d <= self._flags.prediction_radius]
        self._logger.info('Getting predictions for {} vehicles'.format(
            len(sorted_vehicles)))

        if self._flags.prediction_ego_agent:
            return sorted_vehicles, sorted_vehicles[0]
        else: # Exclude the ego vehicle.
            return sorted_vehicles[1:], sorted_vehicles[0]


    def _get_closest_vehicles_ego_transforms(self, closest_vehicles, ego_vehicle):
        closest_vehicles_ego_locations = np.stack([v.trajectory[-1] for v in closest_vehicles])
        closest_vehicles_ego_transforms = []

        # Add appropriate rotations to closest_vehicles_ego_transforms, which
        # we estimate using the direction determined by the last two distinct
        # locations
        ego_vehicle_angle = self._get_vehicle_orientation(ego_vehicle)
        for i in range(len(closest_vehicles)):
            cur_vehicle_angle = self._get_vehicle_orientation(closest_vehicles[i])
            new_yaw = cur_vehicle_angle - ego_vehicle_angle
            if new_yaw > 180:
                new_yaw -= 360
            elif new_yaw < -180:
                new_yaw += 360
            #print ("New yaw for vehicle {}: {}".format(i, new_yaw))
            closest_vehicles_ego_transforms.append(
                Transform(location=closest_vehicles_ego_locations[i].location, 
                          rotation=Rotation(yaw=new_yaw)
            ))
        return closest_vehicles_ego_transforms


    def _get_vehicle_orientation(self, vehicle):
        # Gets the angle from the positive x-axis for the given vehicle.
        other_idx = len(vehicle.trajectory) - 2
        yaw = 0.0
        while other_idx >= 0:
            vec, magnitude = vehicle.trajectory[-1].location.get_vector_and_magnitude(vehicle.trajectory[other_idx].location)
            if magnitude > 0.01:
                yaw = vec.get_angle(Vector2D(0,0))
                break
            else:
                other_idx -= 1
        return math.degrees(yaw)


    def _get_occupancy_grid(self, point_cloud):
        """Get occupancy grids at two different heights."""

        # Threshold used when generating the PRECOG
        # (https://arxiv.org/pdf/1905.01296.pdf) dataset
        z_threshold = -self._lidar_setup.transform.location.z - 2.0
        
        # Transform the point cloud to unreal coordinates.
        transformed_point_cloud = np.zeros_like(point_cloud)
        transformed_point_cloud[:,0] = point_cloud[:,2]
        transformed_point_cloud[:,1] = point_cloud[:,0]
        transformed_point_cloud[:,2] = -point_cloud[:,1]
        point_cloud = transformed_point_cloud

        above_mask = point_cloud[:, 2] > z_threshold

        def get_occupancy_from_masked_lidar(mask):
            masked_lidar = point_cloud[mask]
            meters_max = int(self._lidar_setup.get_range_in_meters())

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

    def on_point_cloud_update(self, msg):
        self._logger.debug('@{}: received point cloud message'.format(
            msg.timestamp))
        self._point_cloud_msgs.append(msg)

    def on_trajectory_update(self, msg):
        self._logger.debug('@{}: received trajectories message'.format(
            msg.timestamp))
        self._tracking_msgs.append(msg)
