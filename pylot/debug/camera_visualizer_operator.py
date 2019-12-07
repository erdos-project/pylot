import erdust
import cv2

from pylot.utils import add_timestamp


class CameraVisualizerOperator(erdust.Operator):
    """ Subscribes to camera streams, and visualizes frames."""

    def __init__(self, camera_stream, name):
        self.name = name
        camera_stream.add_callback(self.display_frame)

    @staticmethod
    def connect(camera_stream):
        return []

    def display_frame(self, msg):
        if msg.encoding == 'segmented':
            msg.frame.visualize(self.name, msg.timestamp)
        else:
            add_timestamp(msg.timestamp, msg.frame)
            cv2.imshow(self.name, msg.frame)
            cv2.waitKey(1)
