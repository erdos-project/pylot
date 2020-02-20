import erdos
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

from pylot.perception.messages import PointCloudMessage
import pylot.perception.point_cloud
from pylot.utils import Location


class VelodyneDriverOperator(erdos.Operator):
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
        self._topic_name = topic_name
        self._flags = flags
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            name + '-csv', csv_file_name)

    @staticmethod
    def connect():
        return [erdos.WriteStream()]

    def on_point_cloud(self, data):
        print('Received {}'.format(data.header.seq))
        # data.header.stamp.secs
        # data.header.stamp.nsec
        transform = None
        timestamp = erdos.Timestamp(coordinates=[data.header.seq])
        points = []
        for data in pc2.read_points(data,
                                    field_names=('x', 'y', 'z'),
                                    skip_nans=True):
            points.append(Location(data[0], data[1], data[2]))
        point_cloud = pylot.perception.point_cloud.PointCloud(
            points, transform)
        msg = PointCloudMessage(timestamp, point_cloud)
        self._point_cloud_stream.send(msg)
        watermark_msg = erdos.WatermarkMessage(timestamp)
        self._point_cloud_stream.send(watermark_msg)

    def run(self):
        rospy.init_node(self._name, anonymous=True, disable_signals=True)
        rospy.Subscriber(self._topic_name, PointCloud2, self.on_point_cloud)
        rospy.spin()