from collections import deque
from functools import partial

import erdos

import pygame
from pygame.locals import K_n


class VisualizerOperator(erdos.Operator):
    """ The `VisualizerOperator` allows developers to see the current state
    of the entire pipeline by visualizing it on a pygame instance.

    This receives input data from almost the entire pipeline and renders the
    results of the operator currently chosen by the developer on the screen.
    """
    def __init__(self, pose_stream, rgb_camera_stream, depth_camera_stream,
                 segmentation_stream, obstacles_stream,
                 tracked_obstacles_stream, waypoints_stream, control_stream,
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

        self._waypoint_msgs = deque()
        waypoints_stream.add_callback(
            partial(self.save, msg_type="Waypoint", queue=self._waypoint_msgs))
        visualize_streams.append(waypoints_stream)

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
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self.font = pygame.font.Font(mono, 14)

        # Array of keys to figure out which message to display.
        self.current_display = 0
        self.display_array = [
            "RGB", "Depth", "Segmentation", "Obstacle", "TrackedObstacle",
            "Waypoint"
        ]
        self.window_titles = [
            "RGB Camera", "Depth Camera", "Segmentation", "Detection",
            "Obstacle tracking", "Planning"
        ]
        assert len(self.display_array) == len(self.window_titles), \
            "The display and titles differ."

        # Save the flags.
        self._flags = flags

    @staticmethod
    def connect(pose_stream, rgb_camera_stream, depth_stream,
                segmentation_stream, obstacles_stream,
                tracked_obstacles_stream, waypoints_stream, control_stream,
                display_control_stream):
        return []

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
                raise ValueError(
                    "The message for {} with timestamp {} was not found".
                    format(name, timestamp))
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
            (220, self._flags.carla_camera_image_height // 3))
        info_surface.set_alpha(100)
        self.display.blit(info_surface, (0, 0))

        # Render the text.
        v_offset = 10
        for line in info_text:
            if v_offset + 18 > self._flags.carla_camera_image_height:
                break
            surface = self.font.render(line, True, (255, 255, 255))
            self.display.blit(surface, (8, v_offset))
            v_offset += 18
        pygame.display.flip()

    def on_watermark(self, timestamp):
        self._logger.debug("@{}: received watermark.".format(timestamp))
        pose_msg = self.get_message(self._pose_msgs, timestamp, "Pose")
        bgr_msg = self.get_message(self._bgr_msgs, timestamp, "BGR")
        depth_msg = self.get_message(self._depth_msgs, timestamp, "Depth")
        segmentation_msg = self.get_message(self._segmentation_msgs, timestamp,
                                            "Segmentation")
        obstacle_msg = self.get_message(self._obstacle_msgs, timestamp,
                                        "Obstacle")
        tracked_obstacle_msg = self.get_message(self._tracked_obstacle_msgs,
                                                timestamp, "TrackedObstacle")
        waypoint_msg = self.get_message(self._waypoint_msgs, timestamp,
                                        "Waypoint")
        control_msg = self.get_message(self._control_msgs, timestamp,
                                       "Control")

        sensor_to_display = self.display_array[self.current_display]
        if sensor_to_display == "RGB" and bgr_msg:
            bgr_msg.frame.visualize(self.display, timestamp=timestamp)
        elif sensor_to_display == "Depth" and depth_msg:
            depth_msg.frame.visualize(self.display, timestamp=timestamp)
        elif sensor_to_display == "Segmentation" and segmentation_msg:
            segmentation_msg.frame.visualize(self.display, timestamp=timestamp)
        elif sensor_to_display == "Obstacle" and (bgr_msg and obstacle_msg
                                                  and pose_msg):
            bgr_msg.frame.annotate_with_bounding_boxes(timestamp,
                                                       obstacle_msg.obstacles,
                                                       pose_msg.data.transform)
            bgr_msg.frame.visualize(self.display, timestamp=timestamp)
        elif sensor_to_display == "TrackedObstacle" and (
                bgr_msg and tracked_obstacle_msg and pose_msg):
            bgr_msg.frame.annotate_with_bounding_boxes(
                timestamp, tracked_obstacle_msg.obstacle_trajectories,
                pose_msg.data.transform)
            bgr_msg.frame.visualize(self.display)
        elif sensor_to_display == "Waypoint" and (bgr_msg and pose_msg
                                                  and waypoint_msg):
            bgr_frame = bgr_msg.frame
            bgr_frame.camera_setup.set_transform(
                pose_msg.data.transform * bgr_frame.camera_setup.transform)
            waypoint_msg.waypoints.draw_on_frame(bgr_frame)
            bgr_frame.visualize(self.display, timestamp=timestamp)

        self.render_text(pose_msg.data, control_msg, timestamp)
