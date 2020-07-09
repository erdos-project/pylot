"""This module implements an operator that forwards messages from the
 ROS topic created by the Velodyne LiDAR driver.

Similar operators can be implemented for other types of LiDARs.
"""

import erdos

import numpy as np

import pylot.perception.point_cloud
from pylot.perception.messages import PointCloudMessage

import rospy

import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

LIDAR_FREQUENCY = 10


class VelodyneDriverOperator(erdos.Operator):
    """Subscribes to a ROS topic on which point clouds are published.

    Args:
        point_cloud_stream (:py:class:`erdos.WriteStream`): Stream on which the
            operator sends point clouds.
        lidar_setup (:py:class:`pylot.drivers.sensor_setup.LidarSetup`):
            Setup of the Lidar.
        topic_name (:obj:`str`): The name of the ROS topic on which to listen
            for point cloud messages.
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, point_cloud_stream, lidar_setup, topic_name, flags):
        self._point_cloud_stream = point_cloud_stream
        self._lidar_setup = lidar_setup
        self._topic_name = topic_name
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._modulo_to_send = LIDAR_FREQUENCY // self._flags.sensor_frequency
        self._counter = 0
        self._msg_cnt = 0

    @staticmethod
    def connect():
        return [erdos.WriteStream()]

    @erdos.profile_method()
    def on_point_cloud(self, data):
        self._counter += 1
        if self._counter % self._modulo_to_send != 0:
            return
        timestamp = erdos.Timestamp(coordinates=[self._msg_cnt])
        points = []
        for data in pc2.read_points(data,
                                    field_names=('x', 'y', 'z'),
                                    skip_nans=True):
            points.append([data[0], data[1], data[2]])
        points = np.array(points)
        point_cloud = pylot.perception.point_cloud.PointCloud(
            points, self._lidar_setup)
        msg = PointCloudMessage(timestamp, point_cloud)
        self._point_cloud_stream.send(msg)
        watermark_msg = erdos.WatermarkMessage(timestamp)
        self._point_cloud_stream.send(watermark_msg)
        self._logger.debug('@{}: sent message'.format(timestamp))
        self._msg_cnt += 1

    def run(self):
        rospy.init_node(self.config.name, anonymous=True, disable_signals=True)
        rospy.Subscriber(self._topic_name, PointCloud2, self.on_point_cloud)
        rospy.spin()
