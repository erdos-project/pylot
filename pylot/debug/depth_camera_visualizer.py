import cv2

from erdos.op import Op

from erdos.utils import setup_logging
from pylot.utils import add_timestamp, is_depth_camera_stream


class DepthCameraVisualizer(Op):
    """ Subscribes to depth camera streams, and visualizes frames."""
    def __init__(self, name, flags, log_file_name=None):
        super(DepthCameraVisualizer, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self._flags = flags

    @staticmethod
    def setup_streams(input_streams, filter_name=None):
        if filter_name:
            input_streams = input_streams.filter_name(filter_name)
        else:
            input_streams = input_streams.filter(is_depth_camera_stream)
    
        input_streams.add_callback(DepthCameraVisualizer.display_frame)
        return []

    def display_frame(self, msg):
        add_timestamp(msg.timestamp, msg.frame)
        cv2.imshow(self.name, msg.frame)
        cv2.waitKey(1)

    def execute(self):
        self.spin()
