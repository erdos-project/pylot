"""This module implements an operator for visualizing the state of
the different pipeline components (e.g., detections, tracked obstacles,
planning waypoints)."""

from collections import deque
from functools import partial

import erdos

import numpy as np

import pygame
from pygame.locals import K_n

import pylot.utils
from pylot.drivers.sensor_setup import RGBCameraSetup
from pylot.perception.camera_frame import CameraFrame
from pylot.planning.world import World

DEFAULT_VIS_TIME = 30000.0


class VisualizerOperator(erdos.Operator):
    """ The `VisualizerOperator` allows developers to see the current state
    of the entire pipeline by visualizing it on a pygame instance.

    This receives input data from almost the entire pipeline and renders the
    results of the operator currently chosen by the developer on the screen.
    """
    def __init__(self, pose_stream, rgb_camera_stream, tl_camera_stream,
                 prediction_camera_stream, depth_camera_stream,
                 point_cloud_stream, segmentation_stream, imu_stream,
                 obstacles_stream, traffic_lights_stream,
                 tracked_obstacles_stream, lane_detection_stream,
                 prediction_stream, waypoints_stream, control_stream,
                 display_control_stream, pygame_display, flags):
        visualize_streams = []
        self._pose_msgs = deque()
        pose_stream.add_callback(
            partial(self.save, msg_type="Pose", queue=self._pose_msgs))
        visualize_streams.append(pose_stream)

        self._bgr_msgs = deque()
        rgb_camera_stream.add_callback(
            partial(self.save, msg_type="RGB", queue=self._bgr_msgs))
        visualize_streams.append(rgb_camera_stream)

        self._imu_msgs = deque()
        imu_stream.add_callback(
            partial(self.save, msg_type="IMU", queue=self._imu_msgs))
        visualize_streams.append(imu_stream)

        self._obstacle_msgs = deque()
        obstacles_stream.add_callback(
            partial(self.save, msg_type="Obstacle", queue=self._obstacle_msgs))
        visualize_streams.append(obstacles_stream)

        self._tracked_obstacle_msgs = deque()
        tracked_obstacles_stream.add_callback(
            partial(self.save,
                    msg_type="TrackedObstacle",
                    queue=self._tracked_obstacle_msgs))
        visualize_streams.append(tracked_obstacles_stream)

        self._tl_camera_msgs = deque()
        tl_camera_stream.add_callback(
            partial(self.save, msg_type="TLCamera",
                    queue=self._tl_camera_msgs))
        visualize_streams.append(tl_camera_stream)

        self._traffic_light_msgs = deque()
        traffic_lights_stream.add_callback(
            partial(self.save,
                    msg_type="TrafficLight",
                    queue=self._traffic_light_msgs))
        visualize_streams.append(traffic_lights_stream)

        self._waypoint_msgs = deque()
        waypoints_stream.add_callback(
            partial(self.save, msg_type="Waypoint", queue=self._waypoint_msgs))
        visualize_streams.append(waypoints_stream)

        self._prediction_camera_msgs = deque()
        prediction_camera_stream.add_callback(
            partial(self.save,
                    msg_type="PredictionCamera",
                    queue=self._prediction_camera_msgs))
        visualize_streams.append(prediction_camera_stream)

        self._prediction_msgs = deque()
        prediction_stream.add_callback(
            partial(self.save,
                    msg_type="Prediction",
                    queue=self._prediction_msgs))
        visualize_streams.append(prediction_stream)

        self._point_cloud_msgs = deque()
        point_cloud_stream.add_callback(
            partial(self.save,
                    msg_type="PointCloud",
                    queue=self._point_cloud_msgs))
        visualize_streams.append(point_cloud_stream)

        self._lane_detection_msgs = deque()
        lane_detection_stream.add_callback(
            partial(self.save,
                    msg_type="Lanes",
                    queue=self._lane_detection_msgs))
        visualize_streams.append(lane_detection_stream)

        self._depth_msgs = deque()
        depth_camera_stream.add_callback(
            partial(self.save, msg_type="Depth", queue=self._depth_msgs))
        visualize_streams.append(depth_camera_stream)

        self._segmentation_msgs = deque()
        segmentation_stream.add_callback(
            partial(self.save,
                    msg_type="Segmentation",
                    queue=self._segmentation_msgs))
        visualize_streams.append(segmentation_stream)

        self._control_msgs = deque()
        control_stream.add_callback(
            partial(self.save, msg_type="Control", queue=self._control_msgs))
        visualize_streams.append(control_stream)

        # Register a watermark callback on all the streams to be visualized.
        erdos.add_watermark_callback(visualize_streams, [], self.on_watermark)

        # Add a callback on a control stream to figure out what to display.
        display_control_stream.add_callback(self.change_display)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self.display = pygame_display

        # Set the font.
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self.font = pygame.font.Font(mono, 14)

        # Array of keys to figure out which message to display.
        self.current_display = 0
        self.display_array = []
        self.window_titles = []
        if flags.visualize_rgb_camera:
            self.display_array.append("RGB")
            self.window_titles.append("RGB Camera")
        if flags.visualize_detected_obstacles:
            self.display_array.append("Obstacle")
            self.window_titles.append("Detected obstacles")
        if flags.visualize_tracked_obstacles:
            self.display_array.append("TrackedObstacle")
            self.window_titles.append("Obstacle tracking")
        if flags.visualize_detected_traffic_lights:
            self.display_array.append("TLCamera")
            self.window_titles.append("Detected traffic lights")
        if flags.visualize_waypoints:
            self.display_array.append("Waypoint")
            self.window_titles.append("Planning")
        if flags.visualize_prediction:
            self.display_array.append("PredictionCamera")
            self.window_titles.append("Prediction")
        if flags.visualize_lidar:
            self.display_array.append("PointCloud")
            self.window_titles.append("LiDAR")
        if flags.visualize_detected_lanes:
            self.display_array.append("Lanes")
            self.window_titles.append("Detected lanes")
        if flags.visualize_depth_camera:
            self.display_array.append("Depth")
            self.window_titles.append("Depth Camera")
        if flags.visualize_segmentation:
            self.display_array.append("Segmentation")
            self.window_titles.append("Segmentation")
        if flags.visualize_world:
            self._planning_world = World(flags, self._logger)
            top_down_transform = pylot.utils.get_top_down_transform(
                pylot.utils.Transform(pylot.utils.Location(),
                                      pylot.utils.Rotation()),
                flags.top_down_camera_altitude)
            self._bird_eye_camera_setup = RGBCameraSetup(
                'bird_eye_camera', flags.camera_image_width,
                flags.camera_image_height, top_down_transform, 90)
            self.display_array.append("PlanningWorld")
            self.window_titles.append("Planning world")
        else:
            self._planning_world = None
        assert len(self.display_array) == len(self.window_titles), \
            "The display and titles differ."

        # Save the flags.
        self._flags = flags

    @staticmethod
    def connect(pose_stream, rgb_camera_stream, tl_camera_stream,
                prediction_camera_stream, depth_stream, point_cloud_stream,
                segmentation_stream, imu_stream, obstacles_stream,
                traffic_lights_stream, tracked_obstacles_stream,
                lane_detection_stream, prediction_stream, waypoints_stream,
                control_stream, display_control_stream):
        return []

    def destroy(self):
        self._logger.warn('destroying {}'.format(self.config.name))

    def save(self, msg, msg_type, queue):
        self._logger.debug("@{}: Received {} message.".format(
            msg.timestamp, msg_type))
        queue.append(msg)

    def change_display(self, display_message):
        if display_message.data == K_n:
            self.current_display = (self.current_display + 1) % len(
                self.display_array)
            self._logger.debug("@{}: Visualizer changed to {}".format(
                display_message.timestamp, self.current_display))

    def get_message(self, queue, timestamp, name):
        msg = None
        if queue:
            while len(queue) > 0:
                retrieved_msg = queue.popleft()
                if retrieved_msg.timestamp == timestamp:
                    msg = retrieved_msg
                    break
            if not msg:
                self._logger.warning(
                    "@{}: message for {} was not found".format(
                        timestamp, name))
        return msg

    def render_text(self, pose, control, timestamp):
        # Generate the text to be shown on the box.
        info_text = [
            "Display  : {}".format(self.window_titles[self.current_display]),
            "Timestamp: {}".format(timestamp.coordinates[0]),
        ]

        # Add information from the pose.
        if pose:
            info_text += [
                "Location : {:.1f}, {:.1f}, {:.1f}".format(
                    *tuple(pose.transform.location.as_numpy_array())),
                "Rotation : {:.1f}, {:.1f}, {:.1f}".format(
                    *tuple(pose.transform.rotation.as_numpy_array())),
                "Speed    : {:.2f} m/s".format(pose.forward_speed),
            ]

        # Add information from the control message
        if control:
            info_text += [
                "Throttle : {:.2f}".format(control.throttle),
                "Steer    : {:.2f}".format(control.steer),
                "Brake    : {:.2f}".format(control.brake),
                "Reverse  : {:.2f}".format(control.reverse),
            ]

        # Display the information box.
        info_surface = pygame.Surface(
            (220, self._flags.camera_image_height // 3))
        info_surface.set_alpha(100)
        self.display.blit(info_surface, (0, 0))

        # Render the text.
        v_offset = 10
        for line in info_text:
            if v_offset + 18 > self._flags.camera_image_height:
                break
            surface = self.font.render(line, True, (255, 255, 255))
            self.display.blit(surface, (8, v_offset))
            v_offset += 18
        pygame.display.flip()

    def on_watermark(self, timestamp):
        self._logger.debug("@{}: received watermark.".format(timestamp))
        if timestamp.is_top:
            return
        pose_msg = self.get_message(self._pose_msgs, timestamp, "Pose")
        bgr_msg = self.get_message(self._bgr_msgs, timestamp, "BGR")
        tl_camera_msg = self.get_message(self._tl_camera_msgs, timestamp,
                                         "TLCamera")
        depth_msg = self.get_message(self._depth_msgs, timestamp, "Depth")
        point_cloud_msg = self.get_message(self._point_cloud_msgs, timestamp,
                                           "PointCloud")
        segmentation_msg = self.get_message(self._segmentation_msgs, timestamp,
                                            "Segmentation")
        imu_msg = self.get_message(self._imu_msgs, timestamp, "IMU")
        obstacle_msg = self.get_message(self._obstacle_msgs, timestamp,
                                        "Obstacle")
        traffic_light_msg = self.get_message(self._traffic_light_msgs,
                                             timestamp, "TrafficLight")
        tracked_obstacle_msg = self.get_message(self._tracked_obstacle_msgs,
                                                timestamp, "TrackedObstacle")
        lane_detection_msg = self.get_message(self._lane_detection_msgs,
                                              timestamp, "Lanes")
        prediction_camera_msg = self.get_message(self._prediction_camera_msgs,
                                                 timestamp, "PredictionCamera")
        prediction_msg = self.get_message(self._prediction_msgs, timestamp,
                                          "Prediction")
        waypoint_msg = self.get_message(self._waypoint_msgs, timestamp,
                                        "Waypoint")
        control_msg = self.get_message(self._control_msgs, timestamp,
                                       "Control")
        if pose_msg:
            ego_transform = pose_msg.data.transform
        else:
            ego_transform = None

        # Add the visualizations on world.
        if self._flags.visualize_pose:
            self._visualize_pose(ego_transform)
        if self._flags.visualize_imu:
            self._visualize_imu(imu_msg)

        sensor_to_display = self.display_array[self.current_display]
        if sensor_to_display == "RGB" and bgr_msg:
            bgr_msg.frame.visualize(self.display, timestamp=timestamp)
        elif sensor_to_display == "Obstacle" and bgr_msg and obstacle_msg:
            bgr_msg.frame.annotate_with_bounding_boxes(timestamp,
                                                       obstacle_msg.obstacles,
                                                       ego_transform)
            bgr_msg.frame.visualize(self.display, timestamp=timestamp)
        elif (sensor_to_display == "TLCamera" and tl_camera_msg
              and traffic_light_msg):
            tl_camera_msg.frame.annotate_with_bounding_boxes(
                timestamp, traffic_light_msg.obstacles)
            tl_camera_msg.frame.visualize(self.display, timestamp=timestamp)
        elif (sensor_to_display == "TrackedObstacle" and bgr_msg
              and tracked_obstacle_msg):
            bgr_msg.frame.annotate_with_bounding_boxes(
                timestamp, tracked_obstacle_msg.obstacle_trajectories,
                ego_transform)
            bgr_msg.frame.visualize(self.display)
        elif sensor_to_display == "Waypoint" and (bgr_msg and pose_msg
                                                  and waypoint_msg):
            bgr_frame = bgr_msg.frame
            if self._flags.draw_waypoints_on_camera_frames:
                bgr_frame.camera_setup.set_transform(
                    pose_msg.data.transform * bgr_frame.camera_setup.transform)
                waypoint_msg.waypoints.draw_on_frame(bgr_frame)
            if self._flags.draw_waypoints_on_world:
                waypoint_msg.waypoints.draw_on_world(self._world)
            bgr_frame.visualize(self.display, timestamp=timestamp)
        elif (sensor_to_display == "PredictionCamera" and prediction_camera_msg
              and prediction_msg):
            frame = prediction_camera_msg.frame
            frame.transform_to_cityscapes()
            for obstacle_prediction in prediction_msg.predictions:
                obstacle_prediction.draw_trajectory_on_frame(frame)
            frame.visualize(self.display, timestamp=timestamp)
        elif sensor_to_display == "PointCloud" and point_cloud_msg:
            point_cloud_msg.point_cloud.visualize(
                self.display, self._flags.camera_image_width,
                self._flags.camera_image_height)
        elif (sensor_to_display == "Lanes" and bgr_msg and lane_detection_msg):
            for lane in lane_detection_msg.data:
                lane.draw_on_frame(bgr_msg.frame)
            bgr_msg.frame.visualize(self.display, timestamp)
        elif sensor_to_display == "Depth" and depth_msg:
            depth_msg.frame.visualize(self.display, timestamp=timestamp)
        elif sensor_to_display == "Segmentation" and segmentation_msg:
            segmentation_msg.frame.visualize(self.display, timestamp=timestamp)
        elif sensor_to_display == "PlanningWorld":
            if prediction_camera_msg is None:
                # Top-down prediction is not available. Show planning
                # world on a black image.
                black_img = np.zeros((self._bird_eye_camera_setup.height,
                                      self._bird_eye_camera_setup.width, 3),
                                     dtype=np.dtype("uint8"))
                frame = CameraFrame(black_img, 'RGB',
                                    self._bird_eye_camera_setup)
            else:
                frame = prediction_camera_msg.frame
                frame.transform_to_cityscapes()
            if lane_detection_msg:
                lanes = lane_detection_msg.data
            else:
                lanes = None
            self._planning_world.update(timestamp,
                                        pose_msg.data,
                                        prediction_msg.predictions,
                                        traffic_light_msg.obstacles,
                                        None,
                                        lanes=lanes)
            self._planning_world.update_waypoints(None, waypoint_msg.waypoints)
            self._planning_world.draw_on_frame(frame)
            frame.visualize(self.display, timestamp=timestamp)

        self.render_text(pose_msg.data, control_msg, timestamp)

    def run(self):
        # Run method is invoked after all operators finished initializing.
        # Thus, we're sure the world is up-to-date here.
        if (self._flags.visualize_pose or self._flags.visualize_imu
                or (self._flags.visualize_waypoints
                    and self._flags.draw_waypoints_on_world)):
            from pylot.simulation.utils import get_world
            _, self._world = get_world(self._flags.simulator_host,
                                       self._flags.simulator_port,
                                       self._flags.simulator_timeout)

    def _visualize_pose(self, ego_transform):
        # Draw position. We add 0.5 to z to ensure that the point is above
        # the road surface.
        loc = (ego_transform.location +
               pylot.utils.Location(0, 0, 0.5)).as_simulator_location()
        self._world.debug.draw_point(loc, size=0.2, life_time=DEFAULT_VIS_TIME)

    def _visualize_imu(self, msg):
        transform = msg.transform
        # Acceleration measured in ego frame, not global
        # z acceleration not useful for visualization so set to 0
        rotation_transform = pylot.utils.Transform(
            location=pylot.utils.Location(0, 0, 0),
            rotation=transform.rotation)
        rotated_acceleration = rotation_transform.transform_locations(
            [pylot.utils.Location(msg.acceleration.x, msg.acceleration.y,
                                  0)])[0]

        # Construct arrow.
        begin_acc = transform.location + pylot.utils.Location(z=0.5)
        end_acc = begin_acc + pylot.utils.Location(rotated_acceleration.x,
                                                   rotated_acceleration.y, 0)

        # draw arrow
        self._logger.debug("Acc: {}".format(rotated_acceleration))
        self._world.debug.draw_arrow(begin_acc.as_simulator_location(),
                                     end_acc.as_simulator_location(),
                                     arrow_size=0.1,
                                     life_time=0.1)
