import cv_bridge
import cv2
import erdos
import numpy as np
import rospy
from sensor_msgs.msg import Image

from pylot.perception.camera_frame import CameraFrame
from pylot.perception.messages import FrameMessage

CAMERA_FPS = 30


class Grasshopper3DriverOperator(erdos.Operator):
    """Subscribes to a ROS topic on which camera images are published.

    Args:
        camera_stream (:py:class:`erdos.WriteStream`): Stream on which the
            operator sends camera frames.
        name (:obj:`str`): The name of the operator.
        camera_setup (:py:class:`pylot.simulation.sensor_setup.RGBCameraSetup`):
            Setup of the camera.
        topic_name (:obj:`str`): The name of the ROS topic on which to listen
            for camera frames.
        flags (absl.flags): Object to be used to access absl flags.
        log_file_name (:obj:`str`, optional): Name of file where log messages
            are written to. If None, then messages are written to stdout.
        csv_file_name (:obj:`str`, optional): Name of file where stats logs are
            written to. If None, then messages are written to stdout.
    """
    def __init__(self,
                 camera_stream,
                 name,
                 camera_setup,
                 topic_name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        self._camera_stream = camera_stream
        self._name = name
        self._camera_setup = camera_setup
        self._topic_name = topic_name
        self._flags = flags
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            name + '-csv', csv_file_name)
        self._bridge = cv_bridge.CvBridge()
        self._modulo_to_send = CAMERA_FPS // self._flags.sensor_frequency
        self._counter = 0
        self._msg_cnt = 0

    @staticmethod
    def connect():
        return [erdos.WriteStream()]

    def on_camera_frame(self, data):
        self._counter += 1
        if self._counter % self._modulo_to_send != 0:
            return
        cv2_image = self._bridge.imgmsg_to_cv2(data, "bgr8")
        resized_image = cv2.resize(cv2.flip(cv2_image, 0), (512, 512))
        numpy_array = np.asarray(resized_image)
        timestamp = erdos.Timestamp(coordinates=[self._msg_cnt])
        camera_frame = CameraFrame(numpy_array, 'BGR', self._camera_setup)
        self._camera_stream.send(FrameMessage(timestamp, camera_frame))
        watermark_msg = erdos.WatermarkMessage(timestamp)
        self._camera_stream.send(watermark_msg)
        self._logger.debug('@{}: sent message'.format(timestamp))
        self._msg_cnt += 1

    def run(self):
        rospy.init_node(self._name, anonymous=True, disable_signals=True)
        rospy.Subscriber(self._topic_name, Image, self.on_camera_frame)
        rospy.spin()
