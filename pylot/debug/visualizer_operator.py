import erdos
import pygame


class VisualizerOperator(erdos.Operator):
    """ The `VisualizerOperator` allows developers to see the current state
    of the entire pipeline by visualizing it on a pygame instance.

    This receives input data from almost the entire pipeline and renders the
    results of the operator currently chosen by the developer on the screen.
    """
    
    def __init__(self, rgb_camera_stream, flags):
        rgb_camera_stream.add_callback(self.display_frame)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)

        # Initialize the pygame display.
        pygame.init()
        self.display = pygame.display.set_mode(
            (flags.carla_camera_image_width, flags.carla_camera_image_height))

    @staticmethod
    def connect(rgb_camera_stream):
        return []

    def display_frame(self, msg):
        self._logger.debug("@{}: Received RGB camera message.".format(
            msg.timestamp))
        msg.frame.visualize('blah', timestamp=msg.timestamp,
                            pygame_display=self.display)
