import erdos
import pygame
from pygame.locals import K_n
from collections import deque
from functools import partial


class VisualizerOperator(erdos.Operator):
    """ The `VisualizerOperator` allows developers to see the current state
    of the entire pipeline by visualizing it on a pygame instance.

    This receives input data from almost the entire pipeline and renders the
    results of the operator currently chosen by the developer on the screen.
    """

    def __init__(self, pose_stream, rgb_camera_stream, depth_camera_stream,
                 obstacles_stream, control_stream, pygame_display, flags):
        # Queue of saved images.
        self._pose_msgs = None
        self._bgr_msgs = None
        self._depth_msgs = None
        self._obstacle_msgs = None

        visualize_streams = []
        if pose_stream:
            self._pose_msgs = deque()
            pose_stream.add_callback(
                partial(self.save, msg_type="Pose", queue=self._pose_msgs))
            visualize_streams.append(pose_stream)

        if rgb_camera_stream:
            self._bgr_msgs = deque()
            rgb_camera_stream.add_callback(
                partial(self.save, msg_type="RGB", queue=self._bgr_msgs))
            visualize_streams.append(rgb_camera_stream)

        if depth_camera_stream:
            self._depth_msgs = deque()
            depth_camera_stream.add_callback(
                partial(self.save, msg_type="Depth", queue=self._depth_msgs))
            visualize_streams.append(depth_camera_stream)

        if obstacles_stream:
            self._obstacle_msgs = deque()
            obstacles_stream.add_callback(
                partial(self.save,
                        msg_type="Obstacle",
                        queue=self._obstacle_msgs))
            visualize_streams.append(obstacles_stream)

        # Register a watermark callback on all the streams to be visualized.
        erdos.add_watermark_callback(visualize_streams, [], self.on_watermark)

        # Add a callback on a control stream to figure out what to display.
        control_stream.add_callback(self.change_display)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self.display = pygame_display

        # Array of keys to figure out which message to display.
        self.current_display = 0
        self.display_array = ["RGB", "Depth", "Obstacle"]

    @staticmethod
    def connect(pose_stream, rgb_camera_stream, depth_stream, obstacles_stream,
                control_stream):
        return []

    def save(self, msg, msg_type, queue):
        self._logger.debug("@{}: Received {} message.".format(
            msg.timestamp, msg_type))
        queue.append(msg)

    def change_display(self, display_message):
        if display_message.data == K_n:
            self.current_display = (self.current_display + 1) % len(
                self.display_array)
            print("RECEIVED MESSAGE: {}".format(display_message))

    def get_message(self, queue, timestamp, name):
        msg = None
        if queue:
            msg = queue.popleft()
            assert msg.timestamp == timestamp, \
                    "The {} timestamp was incorrect".format(name)
        return msg

    def on_watermark(self, timestamp):
        self._logger.debug("@{}: received watermark.".format(timestamp))
        pose_msg = self.get_message(self._pose_msgs, timestamp, "Pose")
        bgr_msg = self.get_message(self._bgr_msgs, timestamp, "BGR")
        depth_msg = self.get_message(self._depth_msgs, timestamp, "Depth")
        obstacle_msg = self.get_message(self._obstacle_msgs, timestamp,
                                        "Obstacle")

        sensor_to_display = self.display_array[self.current_display]
        if sensor_to_display == "RGB" and bgr_msg:
            bgr_msg.frame.visualize('blah',
                                    timestamp=timestamp,
                                    pygame_display=self.display)
        elif sensor_to_display == "Depth" and depth_msg:
            depth_msg.frame.visualize('blah',
                                      timestamp=timestamp,
                                      pygame_display=self.display)
        elif sensor_to_display == "Obstacle" and (bgr_msg and obstacle_msg
                                                  and pose_msg):
            bgr_msg.frame.annotate_with_bounding_boxes(timestamp,
                                                       obstacle_msg.obstacles,
                                                       pose_msg.data.transform)
            bgr_msg.frame.visualize('blah',
                                    timestamp=timestamp,
                                    pygame_display=self.display)
