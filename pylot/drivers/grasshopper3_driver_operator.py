"""This module implements an operator that forwards messages from the
 ROS topic created by the ROS Grasshopper camera driver.

Similar operators can be implemented for other types of cameras.
"""

import cv2

import cv_bridge

import erdos

import numpy as np

from pylot.perception.camera_frame import CameraFrame
from pylot.perception.messages import FrameMessage

import rospy

from sensor_msgs.msg import Image

CAMERA_FPS = 30


class Grasshopper3DriverOperator(erdos.Operator):
    """Subscribes to a ROS topic on which camera images are published.

    Args:
        camera_stream (:py:class:`erdos.WriteStream`): Stream on which the
            operator sends camera frames.
        camera_setup (:py:class:`pylot.drivers.sensor_setup.RGBCameraSetup`):
            Setup of the camera.
        topic_name (:obj:`str`): The name of the ROS topic on which to listen
            for camera frames.
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, camera_stream, camera_setup, topic_name, flags):
        self._camera_stream = camera_stream
        self._camera_setup = camera_setup
        self._topic_name = topic_name
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._bridge = cv_bridge.CvBridge()
        self._modulo_to_send = CAMERA_FPS // self._flags.sensor_frequency
        self._counter = 0
        self._msg_cnt = 0

    @staticmethod
    def connect():
        return [erdos.WriteStream()]

    @erdos.profile_method()
    def on_camera_frame(self, data):
        self._counter += 1
        if self._counter % self._modulo_to_send != 0:
            return
        cv2_image = self._bridge.imgmsg_to_cv2(data, "bgr8")
        resized_image = cv2.resize(
            cv2.flip(cv2_image, -1),
            (self._flags.camera_image_width, self._flags.camera_image_height))
        numpy_array = np.asarray(resized_image)
        timestamp = erdos.Timestamp(coordinates=[self._msg_cnt])
        camera_frame = CameraFrame(numpy_array, 'BGR', self._camera_setup)
        self._camera_stream.send(FrameMessage(timestamp, camera_frame))
        watermark_msg = erdos.WatermarkMessage(timestamp)
        self._camera_stream.send(watermark_msg)
        self._logger.debug('@{}: sent message'.format(timestamp))
        self._msg_cnt += 1

    def run(self):
        rospy.init_node(self.config.name, anonymous=True, disable_signals=True)
        rospy.Subscriber(self._topic_name, Image, self.on_camera_frame)
        rospy.spin()
