import sys
import rospy
import numpy as np
from sensor_msgs.msg import Image

class ROSCameraPublisher:

    def __init__(self, topic:str):
        # publishes to the given topic 
        self.image_pub = rospy.Publisher(topic, Image, queue_size=10)
        
    def publish(self, img_arr):
        # converts the 3d np arrary img_arr to a sensor_msgs/Image datatype
        # inspired by https://github.com/eric-wieser/ros_numpy/blob/master/src/ros_numpy/image.py
        img_msg = Image(encoding='rgb8')
        if type(img_arr[0][0][0]) != np.int8:
            img_arr = img_arr.astype(np.int8)
        img_msg.height, img_msg.width, channels = img_arr.shape
        img_msg.data = img_arr.tobytes()
        img_msg.step = img_msg.width * img_msg.height 
        img_msg.is_bigendian = (
            img_arr.dtype.byteorder == '>' or 
            img_arr.dtype.byteorder == '=' and sys.byteorder == 'big'
        )
        self.image_pub.publish(img_msg)
