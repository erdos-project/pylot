import cv2

from erdos.op import Op
from erdos.utils import setup_logging

from pylot.utils import add_timestamp, is_camera_stream


class VideoOperator(Op):
    def __init__(self, name, flags, log_file_name=None):
        super(VideoOperator, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self._flags = flags

    @staticmethod
    def setup_streams(input_streams, filter_name=None):
        if filter_name:
            input_streams = input_streams.filter_name(filter_name)
        else:
            input_streams = input_streams.filter(is_camera_stream)
        input_streams.add_callback(VideoOperator.display_frame)
        return []

    def display_frame(self, msg):
        assert msg.encoding == 'BGR', 'Expects BGR frames'
        add_timestamp(msg.timestamp, msg.frame)
        cv2.imshow(self.name, msg.frame)
        cv2.waitKey(1)

    def execute(self):
        self.spin()
