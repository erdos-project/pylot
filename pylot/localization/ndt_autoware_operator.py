import erdos
import numpy as np
import rospy

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
from tf.transformations import euler_from_quaternion

from pylot.utils import CanBus, Location, Rotation, Transform

NDT_FREQUENCY = 10


class NDTAutowareOperator(erdos.Operator):
    def __init__(self, can_bus_stream, flags, topic_name='/ndt_pose'):
        self._can_bus_stream = can_bus_stream
        self._flags = flags
        self._topic_name = topic_name
        self._logger = erdos.utils.setup_logging(self.name, log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            self.name + '-csv', csv_file_name)
        self._forward_speed = 0
        self._modulo_to_send = NDT_FREQUENCY // self._flags.sensor_frequency
        self._counter = 0
        self._msg_cnt = 0

    @staticmethod
    def connect():
        return [erdos.WriteStream()]

    def on_pose_update(self, data):
        self._counter += 1
        if self._counter % self._modulo_to_send != 0:
            return
        loc = Location(data.pose.position.x, data.pose.position.y,
                       data.pose.position.z)
        quaternion = [
            data.pose.orientation.x, data.pose.orientation.y,
            data.pose.orientation.z, data.pose.orientation.w
        ]
        roll, pitch, yaw = euler_from_quaternion(quaternion)
        rotation = Rotation(np.degrees(pitch), np.degrees(yaw),
                            np.degrees(roll))
        timestamp = erdos.Timestamp(coordinates=[self._msg_cnt])
        can_bus = CanBus(Transform(loc, rotation), self._forward_speed)
        self._logger.debug('NDT localization {}'.format(can_bus))
        self._can_bus_stream.send(erdos.Message(timestamp, can_bus))
        self._can_bus_stream.send(erdos.WatermarkMessage(timestamp))
        self._msg_cnt += 1

    def on_velocity_update(self, data):
        self._forward_speed = data.data

    def run(self):
        rospy.init_node(self.name, anonymous=True, disable_signals=True)
        rospy.Subscriber(self._topic_name, PoseStamped, self.on_pose_update)
        rospy.Subscriber('/estimated_vel_mps', Float32,
                         self.on_velocity_update)
        rospy.spin()
