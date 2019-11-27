import erdust
import cv2

from pylot.utils import add_timestamp


class VideoOperator(erdust.Operator):
    """ Subscribes to camera streams, and visualizes frames."""

    def __init__(self, camera_stream, name):
        self.name = name
        camera_stream.add_callback(self.display_frame)

    @staticmethod
    def connect(camera_stream):
        return []

    def display_frame(self, msg):
        assert msg.encoding == 'BGR', 'Expects BGR frames'
        add_timestamp(msg.timestamp, msg.frame)
        cv2.imshow(self.name, msg.frame)
        cv2.waitKey(1)
