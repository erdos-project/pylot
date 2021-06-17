import sys
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField

class ROSLIDARPublisher:

    def __init__(self, topic:str):
        # publishes to the given topic 
        self.point_cloud_pub = rospy.Publisher(topic, PointCloud2, queue_size=10)
        
    def publish(self, points):
        # converts point cloud points array to a sensor_msgs/PointCloud2 datatype
        points = points.astype(np.float32)
        points[:,[0,2]] = points[:,[2,0]]
        points[:,[1,2]] = points[:,[2,1]]
        points[:,0] = -points[:,0]
        points[:,2] = -points[:,2]
        points_byte_array = points.tobytes()
        row_step = len(points_byte_array)
        point_step = len(points[0].tobytes())
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1)]
        point_cloud_msg = PointCloud2(height=1,
                                      width=len(points),
                                      is_dense=True,
                                      is_bigendian=False,
                                      fields=fields,
                                      point_step=point_step,
                                      row_step=row_step,
                                      data=points_byte_array)
        self.point_cloud_pub.publish(point_cloud_msg)
