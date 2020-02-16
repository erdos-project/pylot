import erdos
import rospy
from std_msgs.msg import Image

from pylot.perception.camera_frame import CameraFrame
from pylot.perception.messages import FrameMessage


class Grasshopper3DriverOperator(erdos.Operator):
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

    @staticmethod
    def connect():
        return [erdos.WriteStream()]

    def on_camera_frame(self, data):
        print('Received data {} encoding {}'.format(data.header.seq,
                                                    data.encoding))
        timestamp = erdos.Timestamp(coordinates=[data.header.seq])
        camera_frame = CameraFrame(data.data, 'RGB', self._camera_setup)
        self._camera_stream.send(FrameMessage(timestamp, camera_frame))
        watermark_msg = erdos.WatermarkMessage(timestamp)
        self._camera_stream.send(watermark_msg)

    def run(self):
        rospy.init_node(self._name, anonymous=True)
        rospy.Subscriber(self._topic_name, Image, self.on_camera_frame)
        rospy.spin()
