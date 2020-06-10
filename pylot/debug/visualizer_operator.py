import erdos
import pygame

from pygame.locals import K_n


class VisualizerOperator(erdos.Operator):
    """ The `VisualizerOperator` allows developers to see the current state
    of the entire pipeline by visualizing it on a pygame instance.

    This receives input data from almost the entire pipeline and renders the
    results of the operator currently chosen by the developer on the screen.
    """

    def __init__(self,
                 rgb_camera_stream,
                 depth_camera_stream,
                 control_stream,
                 pygame_display,
                 flags):
        if rgb_camera_stream:
            rgb_camera_stream.add_callback(lambda msg: self.display_frame(
                msg, "RGB"))

        if depth_camera_stream:
            depth_camera_stream.add_callback(lambda msg: self.display_frame(
                msg, "Depth"))

        # Add a callback on a control stream to figure out what to display.
        control_stream.add_callback(self.change_display)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self.display = pygame_display

        # Array of keys to figure out which message to display.
        self.current_display = 0
        self.display_array = ["RGB", "Depth"]

    @staticmethod
    def connect(rgb_camera_stream, depth_stream, control_stream):
        return []

    def display_frame(self, msg, sensor_type):
        self._logger.debug("@{}: Received {} camera message.".format(
            msg.timestamp, sensor_type))
        if sensor_type == self.display_array[self.current_display]:
            msg.frame.visualize('blah',
                                timestamp=msg.timestamp,
                                pygame_display=self.display)

    def change_display(self, display_message):
        if display_message.data == K_n:
            self.current_display = (self.current_display + 1) % len(
                self.display_array)
            print("RECEIVED MESSAGE: {}".format(display_message))
