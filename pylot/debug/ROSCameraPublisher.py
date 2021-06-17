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
        img_msg = Image(encoding='rgb8')
        img_msg.height, img_msg.width, channels = img_arr.shape
        img_msg.data = img_arr.tobytes()
        img_msg.step = img_msg.width * img_msg.height 
        img_msg.is_bigendian = (
            img_arr.dtype.byteorder == '>' or 
            img_arr.dtype.byteorder == '=' and sys.byteorder == 'big'
        )
        self.image_pub.publish(img_msg)
