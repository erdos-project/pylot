import json
import os
from collections import deque

import PIL.Image as Image

from carla import BoundingBox, Color, TrafficLightState

import cv2

import erdos

import numpy as np

import pylot.simulation.utils
import pylot.utils

TL_STATE_TO_PIXEL_COLOR = {
    TrafficLightState.Red: [255, 1, 1],
    TrafficLightState.Yellow: [2, 255, 2],
    TrafficLightState.Green: [3, 3, 255],
}

# Draw bounding boxes and record them within TL_LOGGING_RADIUS meters from car.
TL_LOGGING_RADIUS = 40
TL_BBOX_LIFETIME_BUFFER = 0.1


class ChauffeurLoggerOperator(erdos.Operator):
    """ Logs data in Chauffeur format. """
    def __init__(self, vehicle_id_stream: erdos.ReadStream,
                 pose_stream: erdos.ReadStream,
                 obstacle_tracking_stream: erdos.ReadStream,
                 top_down_camera_stream: erdos.ReadStream,
                 top_down_segmentation_stream: erdos.ReadStream, flags,
                 top_down_camera_setup):
        """ Initializes the operator with the given parameters.

        Args:
            flags: A handle to the global flags instance to retrieve the
                configuration.
            top_down_camera_setup: The setup of the top down camera.
        """
        vehicle_id_stream.add_callback(self.on_ground_vehicle_id_update)
        pose_stream.add_callback(self.on_pose_update)
        obstacle_tracking_stream.add_callback(self.on_tracking_update)
        top_down_camera_stream.add_callback(self.on_top_down_camera_update)
        top_down_segmentation_stream.add_callback(
            self.on_top_down_segmentation_update)
        self._flags = flags
        self._buffer_length = 10
        self._ground_vehicle_id = None
        self._waypoints = None
        # Holds history of global transforms at each timestep.
        self._global_transforms = deque(maxlen=self._buffer_length)
        self._current_transform = None
        self._previous_transform = None
        self._top_down_camera_setup = top_down_camera_setup

    @staticmethod
    def connect(vehicle_id_stream: erdos.ReadStream,
                pose_stream: erdos.ReadStream,
                obstacle_tracking_stream: erdos.ReadStream,
                top_down_camera_stream: erdos.ReadStream,
                top_down_segmentation_stream: erdos.ReadStream):
        return []

    def run(self):
        # Run method is invoked after all operators finished initializing.
        # Thus, we're sure the world is up-to-date here.
        _, self._world = pylot.simulation.utils.get_world(
            self._flags.simulator_host, self._flags.simulator_port,
            self._flags.simulator_timeout)

    def on_tracking_update(self, msg: erdos.Message):
        assert len(msg.timestamp.coordinates) == 1
        past_poses = np.zeros((self._top_down_camera_setup.height,
                               self._top_down_camera_setup.width, 3),
                              dtype=np.uint8)
        future_poses = np.zeros((self._top_down_camera_setup.height,
                                 self._top_down_camera_setup.width, 3),
                                dtype=np.uint8)

        # Intrinsic and extrinsic matrix of the top down segmentation camera.
        extrinsic_matrix = self._top_down_camera_setup.get_extrinsic_matrix()
        intrinsic_matrix = self._top_down_camera_setup.get_intrinsic_matrix()

        rotation = pylot.utils.Rotation()
        for obstacle_trajectory in msg.obstacle_trajectories:
            # Convert to screen points.
            screen_points = [
                transform.location.to_camera_view(extrinsic_matrix,
                                                  intrinsic_matrix)
                for transform in obstacle_trajectory.trajectory
            ]

            # Keep track of ground vehicle waypoints
            if obstacle_trajectory.id == self._ground_vehicle_id:
                self._waypoints = obstacle_trajectory.trajectory

            # Draw trajectory points on segmented image.
            for point in screen_points:
                if (0 <= point.x <= self._flags.camera_image_width) and \
                   (0 <= point.y <= self._flags.camera_image_height):
                    r = 3
                    if obstacle_trajectory.id == self._ground_vehicle_id:
                        r = 10
                    cv2.circle(past_poses, (int(point.x), int(point.y)), r,
                               (100, 100, 100), -1)

        # Transform to previous and back to current frame
        self._waypoints = [
            self._current_transform.transform_locations(
                self._previous_transform.inverse_transform_locations(
                    [wp.location]))[0] for wp in self._waypoints
        ]

        # Center first point at 0, 0
        center_transform = pylot.utils.Transform(self._waypoints[0], rotation)
        self._waypoints = [
            center_transform.inverse_transform_locations([wp])[0]
            for wp in self._waypoints
        ]

        # Convert to screen points
        screen_waypoints = [
            wp.to_camera_view(extrinsic_matrix, intrinsic_matrix)
            for wp in self._waypoints
        ]

        # Draw screen points
        for point in screen_waypoints:
            cv2.circle(future_poses, (int(point.x), int(point.y)), 10,
                       (100, 100, 100), -1)

        # Log future screen points
        future_poses_img = Image.fromarray(future_poses)
        future_poses_img = future_poses_img.convert('RGB')
        file_name = os.path.join(
            self._flags.data_path,
            'future_poses-{}.png'.format(msg.timestamp.coordinates[0] -
                                         len(self._waypoints) * 100))
        future_poses_img.save(file_name)

        # Log future poses
        waypoints = [str(wp) for wp in self._waypoints]
        file_name = os.path.join(
            self._flags.data_path,
            'waypoints-{}.json'.format(msg.timestamp.coordinates[0] -
                                       len(self._waypoints) * 100))
        with open(file_name, 'w') as outfile:
            json.dump(waypoints, outfile)

        # Log past screen points
        past_poses_img = Image.fromarray(past_poses)
        past_poses_img = past_poses_img.convert('RGB')
        file_name = os.path.join(
            self._flags.data_path,
            'past_poses-{}.png'.format(msg.timestamp.coordinates[0]))
        past_poses_img.save(file_name)

    def on_top_down_segmentation_update(self, msg: erdos.Message):
        assert len(msg.timestamp.coordinates) == 1
        # Save the segmented channels
        msg.frame.save_per_class_masks(self._flags.data_path, msg.timestamp)
        msg.frame.save(msg.timestamp.coordinates[0], self._flags.data_path,
                       'top_down_segmentation')

    def on_ground_vehicle_id_update(self, msg: erdos.Message):
        self._ground_vehicle_id = msg.data

    def on_pose_update(self, msg: erdos.Message):
        assert len(msg.timestamp.coordinates) == 1
        # Make sure transforms deque is full
        self._current_transform = msg.data.transform
        while len(self._global_transforms) != self._buffer_length:
            self._global_transforms.append(msg.data.transform)
        self._previous_transform = self._global_transforms.popleft()
        self._global_transforms.append(msg.data.transform)

        # Log heading
        file_name = os.path.join(
            self._flags.data_path,
            'heading-{}.json'.format(msg.timestamp.coordinates[0]))
        with open(file_name, 'w') as outfile:
            json.dump(str(self._current_transform.rotation.yaw), outfile)

        # Log speed
        file_name = os.path.join(
            self._flags.data_path,
            'speed-{}.json'.format(msg.timestamp.coordinates[0]))
        with open(file_name, 'w') as outfile:
            json.dump(str(msg.data.forward_speed), outfile)

    def on_top_down_camera_update(self, msg: erdos.Message):
        assert len(msg.timestamp.coordinates) == 1
        # Draw traffic light bboxes within TL_LOGGING_RADIUS meters from car
        tl_actors = self._world.get_actors().filter('traffic.traffic_light*')
        for tl_actor in tl_actors:
            dist = self._current_transform.location.distance(
                tl_actor.get_transform().location)
            if dist <= TL_LOGGING_RADIUS:
                self._draw_trigger_volume(self._world, tl_actor)

        # Record traffic light masks
        img = np.uint8(msg.frame.as_rgb_numpy_array())
        tl_mask = self._get_traffic_light_channel_from_top_down_rgb(img)
        tl_img = Image.fromarray(tl_mask)
        tl_img = tl_img.convert('RGB')
        file_name = os.path.join(
            self._flags.data_path,
            'traffic_lights-{}.png'.format(msg.timestamp.coordinates[0]))
        tl_img.save(file_name)

    def _draw_trigger_volume(self, world, tl_actor):
        transform = tl_actor.get_transform()
        tv = transform.transform(tl_actor.trigger_volume.location)
        bbox = BoundingBox(tv, tl_actor.trigger_volume.extent)
        tl_state = tl_actor.get_state()
        if tl_state in TL_STATE_TO_PIXEL_COLOR:
            r, g, b = TL_STATE_TO_PIXEL_COLOR[tl_state]
            bbox_color = Color(r, g, b)
        else:
            bbox_color = Color(0, 0, 0)
        bbox_life_time = \
            (1 / self._flags.simulator_fps + TL_BBOX_LIFETIME_BUFFER)
        world.debug.draw_box(bbox,
                             transform.rotation,
                             thickness=0.5,
                             color=bbox_color,
                             life_time=bbox_life_time)

    def _get_traffic_light_channel_from_top_down_rgb(self,
                                                     img,
                                                     tl_bbox_colors=[[
                                                         200, 0, 0
                                                     ], [13, 0,
                                                         196], [5, 200, 0]]):
        """
        Returns a mask of the traffic light extent bounding boxes seen from a
        top-down view. The bounding boxes in the mask are colored differently
        depending on the state of each traffic light.

        *Note: Not sure why the colors do not match up with the original colors
        the boxes are drawn with. The default colors are estimates learned by
        examining the output of the top down camera.

        Args:
            img: Top-down RGB frame with traffic light extent bboxes drawn.
            tl_bbox_colors: The colors of the traffic light extent bboxes.
        """
        if tl_bbox_colors is None:
            tl_bbox_colors = TL_STATE_TO_PIXEL_COLOR.values()
        h, w = img.shape[:2]
        tl_mask = np.zeros((h + 2, w + 2), np.uint8)
        # Grayscale values for different traffic light states
        vals = [33, 66, 99]
        for i, bbox_color in enumerate(tl_bbox_colors):
            tl_mask_for_bbox_color = np.zeros((h + 2, w + 2), np.uint8)
            # Using a tolerance of 20 to locate correct boxes.
            mask = np.all(abs(img - bbox_color) < 20, axis=2).astype(np.uint8)
            # Flood fill from (0, 0) corner.
            cv2.floodFill(mask, tl_mask_for_bbox_color, (0, 0), 1)
            # Invert image so mask highlights lights.
            tl_mask_for_bbox_color = 1 - tl_mask_for_bbox_color
            tl_mask += tl_mask_for_bbox_color * vals[i]
        # Remove extra rows and cols added for floodFill.
        return tl_mask[1:-1, 1:-1]
