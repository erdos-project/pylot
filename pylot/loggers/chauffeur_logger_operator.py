from collections import deque

import carla
import cv2
import json
import PIL.Image as Image
import numpy as np

# ERDOS specific imports.
from erdos.op import Op
from erdos.utils import setup_logging

# Pylot specific imports.
from pylot.perception.segmentation.utils import transform_to_cityscapes_palette, LABEL_2_PIXEL
from pylot.planning.utils import get_distance
import pylot.utils
import pylot.simulation.carla_utils

TL_STATE_TO_PIXEL_COLOR = {
    carla.TrafficLightState.Red: [255, 1, 1],
    carla.TrafficLightState.Yellow: [2, 255, 2],
    carla.TrafficLightState.Green: [3, 3, 255],
}

# Draw bounding boxes and record them within TL_LOGGING_RADIUS meters from car.
TL_LOGGING_RADIUS = 40
TL_BBOX_LIFETIME_BUFFER = 0.1


class ChauffeurLoggerOp(Op):
    """ Logs data in Chauffeur format. """

    def __init__(self, name, flags, top_down_camera_setup, log_file_name=None):
        """ Initializes the operator with the given parameters.

        Args:
            name: The name of the operator.
            flags: A handle to the global flags instance to retrieve the
                configuration.
            top_down_camera_setup: The setup of the top down camera.
            log_file_name: The file to log the required information to.
        """
        super(ChauffeurLoggerOp, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self._flags = flags
        self._buffer_length = 10

        self._ground_vehicle_id = None
        self._waypoints = None
        # Holds history of global transforms at each timestep.
        self._global_transforms = deque(maxlen=self._buffer_length)
        self._current_transform = None
        self._previous_transform = None

        # Queues of incoming data.
        self._track_count = 0
        self._frame_count = 0

        self._top_down_camera_setup = top_down_camera_setup
        # Get world to access traffic lights.
        _, self._world = pylot.simulation.carla_utils.get_world(
            self._flags.carla_host,
            self._flags.carla_port,
            self._flags.carla_timeout)
        if self._world is None:
            raise ValueError('There was an issue connecting to the simulator.')

    @staticmethod
    def setup_streams(input_streams, top_down_stream_name):
        input_streams.filter(pylot.utils.is_tracking_stream).add_callback(
            ChauffeurLoggerOp.on_tracking_update)
        input_streams.filter(
            pylot.utils.is_segmented_camera_stream).filter_name(
                top_down_stream_name).add_callback(
                    ChauffeurLoggerOp.on_top_down_segmentation_update)
        input_streams.filter(
            pylot.utils.is_ground_vehicle_id_stream).add_callback(
                ChauffeurLoggerOp.on_ground_vehicle_id_update)
        input_streams.filter(pylot.utils.is_can_bus_stream).add_callback(
            ChauffeurLoggerOp.on_can_bus_update)
        input_streams.filter(
            pylot.utils.is_top_down_camera_stream).add_callback(
                ChauffeurLoggerOp.on_top_down_camera_update)
        return []

    def on_tracking_update(self, msg):
        past_poses = np.zeros((self._top_down_camera_setup.height,
                               self._top_down_camera_setup.width,
                               3),
                              dtype=np.uint8)
        future_poses = np.zeros((self._top_down_camera_setup.height,
                                 self._top_down_camera_setup.width,
                                 3),
                                dtype=np.uint8)

        # Intrinsic matrix of the top down segmentation camera.
        intrinsic_matrix = pylot.simulation.utils.create_intrinsic_matrix(
            self._top_down_camera_setup.width,
            self._top_down_camera_setup.height,
            fov=self._top_down_camera_setup.fov)

        rotation = pylot.simulation.utils.Rotation(0, 0, 0)
        for obj in msg.obj_trajectories:
            # Convert to screen points.
            screen_points = pylot.simulation.utils.locations_3d_to_view(
                obj.trajectory,
                self._top_down_camera_setup.transform.matrix,
                intrinsic_matrix)

            # Keep track of ground vehicle waypoints
            if obj.obj_id == self._ground_vehicle_id:
                self._waypoints = obj.trajectory

            # Draw trajectory points on segmented image.
            for point in screen_points:
                if (0 <= point.x <= self._flags.carla_camera_image_width) and \
                   (0 <= point.y <= self._flags.carla_camera_image_height):
                    r = 3
                    if obj.obj_id == self._ground_vehicle_id:
                        r = 10
                    cv2.circle(past_poses,
                               (int(point.x), int(point.y)),
                               r, (100, 100, 100), -1)

        # Transform to previous and back to current frame
        new_transform = self._current_transform * self._previous_transform.inverse_transform()
        self._waypoints = [
            (new_transform * pylot.simulation.utils.Transform(wp, rotation)).location
            for wp in self._waypoints
        ]

        # Center first point at 0, 0
        center_transform = pylot.simulation.utils.Transform(
            self._waypoints[0],
            rotation
        ).inverse_transform()
        self._waypoints = [
            (center_transform * pylot.simulation.utils.Transform(wp, rotation)).location
            for wp in self._waypoints
        ]

        # Convert to screen points
        screen_waypoints = pylot.simulation.utils.locations_3d_to_view(
                self._waypoints,
                self._top_down_camera_setup.transform.matrix,
                intrinsic_matrix)

        # Draw screen points
        for point in screen_waypoints:
            cv2.circle(future_poses,
                       (int(point.x), int(point.y)),
                       10, (100, 100, 100), -1)

        # Log future screen points
        future_poses_img = Image.fromarray(future_poses)
        future_poses_img = future_poses_img.convert('RGB')
        future_poses_img.save('{}{}-{}.png'.format(
            self._flags.data_path, "future_poses",
            msg.timestamp.coordinates[0] - len(self._waypoints) * 100))

        # Log future poses
        waypoints = [str(wp) for wp in self._waypoints]
        file_name = '{}waypoints-{}.json'.format(
            self._flags.data_path,
            msg.timestamp.coordinates[0] - len(self._waypoints) * 100)
        with open(file_name, 'w') as outfile:
            json.dump(waypoints, outfile)

        # Log past screen points
        past_poses_img = Image.fromarray(past_poses)
        past_poses_img = past_poses_img.convert('RGB')
        past_poses_img.save('{}{}-{}.png'.format(
            self._flags.data_path,
            "past_poses", msg.timestamp.coordinates[0]))

    def on_top_down_segmentation_update(self, msg):
        top_down = np.uint8(transform_to_cityscapes_palette(msg.frame))

        # Save the segmented channels
        for k, v in LABEL_2_PIXEL.items():
            mask = np.all(top_down == v, axis=-1)
            tmp = np.zeros(top_down.shape[:2])
            tmp[mask] = 1
            name = '{}{}-{}.png'.format(
                self._flags.data_path, k, msg.timestamp.coordinates[0])
            img = Image.fromarray(tmp)
            img = img.convert('RGB')
            img.save(name)

        top_down_img = Image.fromarray(top_down)
        top_down_img.save('{}{}-{}.png'.format(
            self._flags.data_path,
            "top_down_segmentation",
            msg.timestamp.coordinates[0]))

    def on_ground_vehicle_id_update(self, msg):
        self._ground_vehicle_id = msg.data

    def on_can_bus_update(self, msg):
        # Make sure transforms deque is full
        self._current_transform = msg.data.transform
        while len(self._global_transforms) != self._buffer_length:
            self._global_transforms.append(msg.data.transform)
        self._previous_transform = self._global_transforms.popleft()
        self._global_transforms.append(msg.data.transform)

        # Log heading
        file_name = '{}heading-{}.json'.format(self._flags.data_path,
                                               msg.timestamp.coordinates[0])
        with open(file_name, 'w') as outfile:
            json.dump(str(self._current_transform.rotation.yaw), outfile)

        # Log speed
        file_name = '{}speed-{}.json'.format(self._flags.data_path,
                                             msg.timestamp.coordinates[0])
        with open(file_name, 'w') as outfile:
            json.dump(str(msg.data.forward_speed), outfile)

    def on_top_down_camera_update(self, msg):
        # Draw traffic light bboxes within TL_LOGGING_RADIUS meters from car
        tl_actors = self._world.get_actors().filter('traffic.traffic_light*')
        for tl_actor in tl_actors:
            x = self._current_transform.location
            y = tl_actor.get_transform().location
            dist = get_distance(x, y)
            if dist <= TL_LOGGING_RADIUS:
                self._draw_trigger_volume(self._world, tl_actor)

        # Record traffic light masks
        img = np.uint8(msg.frame)
        tl_mask = self._get_traffic_light_channel_from_top_down_rgb(img)
        tl_img = Image.fromarray(tl_mask)
        tl_img = tl_img.convert('RGB')
        tl_img.save('{}{}-{}.png'.format(
            self._flags.data_path,
            "traffic_lights",
            msg.timestamp.coordinates[0]))

    def _draw_trigger_volume(self, world, tl_actor):
        transform = tl_actor.get_transform()
        tv = transform.transform(tl_actor.trigger_volume.location)
        bbox = carla.BoundingBox(tv, tl_actor.trigger_volume.extent)
        tl_state = tl_actor.get_state()
        if tl_state in TL_STATE_TO_PIXEL_COLOR:
            r, g, b = TL_STATE_TO_PIXEL_COLOR[tl_state]
            bbox_color = carla.Color(r, g, b)
        else:
            bbox_color = carla.Color(0, 0, 0)
        bbox_life_time = 1 / self._flags.carla_step_frequency + TL_BBOX_LIFETIME_BUFFER
        world.debug.draw_box(bbox,
                             transform.rotation,
                             thickness=0.5,
                             color=bbox_color,
                             life_time=bbox_life_time)

    def _get_traffic_light_channel_from_top_down_rgb(
            self,
            img,
            tl_bbox_colors=[[200, 0, 0], [13, 0, 196], [5, 200, 0]]):
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
        tl_mask = np.zeros((h+2, w+2), np.uint8)
        # Grayscale values for different traffic light states
        vals = [33, 66, 99]
        for i, bbox_color in enumerate(tl_bbox_colors):
            tl_mask_for_bbox_color = np.zeros((h+2, w+2), np.uint8)
            # Using a tolerance of 20 to locate correct boxes.
            mask = np.all(abs(img - bbox_color) < 20, axis=2).astype(np.uint8)
            # Flood fill from (0, 0) corner.
            cv2.floodFill(mask, tl_mask_for_bbox_color, (0, 0), 1)
            # Invert image so mask highlights lights.
            tl_mask_for_bbox_color = 1 - tl_mask_for_bbox_color
            tl_mask += tl_mask_for_bbox_color * vals[i]
        # Remove extra rows and cols added for floodFill.
        return tl_mask[1:-1, 1:-1]
