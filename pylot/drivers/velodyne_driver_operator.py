import erdos
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

from pylot.perception.messages import PointCloudMessage
import pylot.perception.point_cloud
from pylot.utils import Location

LIDAR_FREQUENCY = 10


class VelodyneDriverOperator(erdos.Operator):
    """Subscribes to a ROS topic on which point clouds are published.

    Args:
        point_cloud_stream (:py:class:`erdos.WriteStream`): Stream on which the
            operator sends point clouds.
        name (:obj:`str`): The name of the operator.
        lidar_setup (:py:class:`pylot.drivers.sensor_setup.LidarSetup`):
            Setup of the Lidar.
        topic_name (:obj:`str`): The name of the ROS topic on which to listen
            for point cloud messages.
        flags (absl.flags): Object to be used to access absl flags.
        log_file_name (:obj:`str`, optional): Name of file where log messages
            are written to. If None, then messages are written to stdout.
        csv_file_name (:obj:`str`, optional): Name of file where stats logs are
            written to. If None, then messages are written to stdout.
    """
    def __init__(self,
                 point_cloud_stream,
                 name,
                 lidar_setup,
                 topic_name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        self._point_cloud_stream = point_cloud_stream
        self._name = name
        self._lidar_setup = lidar_setup
        self._topic_name = topic_name
        self._flags = flags
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            name + '-csv', csv_file_name)
        self._modulo_to_send = LIDAR_FREQUENCY // self._flags.sensor_frequency
        self._counter = 0
        self._msg_cnt = 0

    @staticmethod
    def connect():
        return [erdos.WriteStream()]

    def on_point_cloud(self, data):
        self._counter += 1
        if self._counter % self._modulo_to_send != 0:
            return
        timestamp = erdos.Timestamp(coordinates=[self._msg_cnt])
        points = []
        for data in pc2.read_points(data,
                                    field_names=('x', 'y', 'z'),
                                    skip_nans=True):
            points.append(Location(data[0], data[1], data[2]))
        point_cloud = pylot.perception.point_cloud.PointCloud(
            points, self._lidar_setup.transform)
        msg = PointCloudMessage(timestamp, point_cloud)
        self._point_cloud_stream.send(msg)
        watermark_msg = erdos.WatermarkMessage(timestamp)
        self._point_cloud_stream.send(watermark_msg)
        self._logger.debug('@{}: sent message'.format(timestamp))
        self._msg_cnt += 1

    def run(self):
        rospy.init_node(self._name, anonymous=True, disable_signals=True)
        rospy.Subscriber(self._topic_name, PointCloud2, self.on_point_cloud)
        rospy.spin()
