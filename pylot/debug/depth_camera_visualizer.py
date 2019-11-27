import cv2
import erdust

from pylot.utils import add_timestamp


class DepthCameraVisualizer(erdust.Operator):
    """ Subscribes to depth camera streams, and visualizes frames."""
    def __init__(self, depth_camera_stream, name):
        depth_camera_stream.add_callback(self.display_frame)

    @staticmethod
    def connect(depth_camera_stream):
        return []

    def display_frame(self, msg):
        add_timestamp(msg.timestamp, msg.frame)
        cv2.imshow(self.name, msg.frame)
        cv2.waitKey(1)
