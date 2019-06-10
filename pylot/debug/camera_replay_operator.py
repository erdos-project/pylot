import numpy as np
import cv2
from cv_bridge import CvBridge

from erdos.data_stream import DataStream
from erdos.message import Message
from erdos.op import Op
from erdos.timestamp import Timestamp
from erdos.utils import frequency, setup_logging

from sensor_msgs.msg import Image


class CameraReplayOperator(Op):
    def __init__(self, name, log_file_name=None):
        super(CameraReplayOperator, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self._cnt = 1
        self._image = None

    @staticmethod
    def setup_streams(input_streams, op_name):
        return [DataStream(data_type=Image,
                           name='{}_output'.format(op_name),
                           labels={'camera': 'true'})]

    @frequency(0.1)
    def publish_frame(self):
        """Publish mock camera frames."""
        cv_image = self.read_image(self._cnt)
        self._image = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        self._image.header.seq = self._cnt
        timestamp = Timestamp(coordinates=[self._cnt])
        output_msg = Message(self._image, timestamp)
        self.get_output_stream('{}_output'.format(self.name)).send(output_msg)
        self.get_output_stream('{}_output'.format(self.name)).send(
            WatermarkMessage(timestamp))
        self._cnt += 1

    def create_image(self):
        np_image = np.zeros((1024, 768, 3), dtype=np.uint8)
        r, g, b = cv2.split(np_image)
        cv_image = cv2.merge([b, g, r])
        return cv_image

    def read_image(self, cnt):
        img = cv2.imread('images/carla{}.jpg'.format(cnt))
#        img = cv2.imread('images/car-cops/{:08d}.jpg'.format(cnt))
        return img

    def execute(self):
        """Operator entry method.
        The method implements the operator logic.
        """
        self.bridge = CvBridge()
        #cv_image = self.create_image()
        self.publish_frame()
        self.spin()
