import erdos
import rospy
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion

from pylot.utils import CanBus, Location, Rotation, Transform


class NDTAutowareOperator(erdos.Operator):
    def __init__(self,
                 can_bus_stream,
                 name,
                 flags,
                 topic_name='/ndt_pose',
                 log_file_name=None,
                 csv_file_name=None):
        self._can_bus_stream = can_bus_stream
        self._name = name
        self._flags = flags
        self._topic_name = topic_name
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            name + '-csv', csv_file_name)

    @staticmethod
    def connect():
        return [erdos.WriteStream()]

    def on_pose_update(self, data):
        loc = Location(data.pose.point.x, data.pose.point.y, data.pose.point.z)
        quaternion = [
            data.pose.orientation.x, data.pose.orientation.y,
            data.pose.orientation.z, data.pose.orientation.w
        ]
        roll, pitch, yaw = euler_from_quaternion(quaternion)
        rotation = Rotation(pitch, yaw, roll)
        # TODO: Set forward_speed.
        forward_speed = 0
        timestamp = erdos.Timestamp(coordinates=[data.header.seq])
        can_bus = CanBus(Transform(loc, rotation), forward_speed)
        print('Location {}; Rotation {}'.format(loc, rotation))
        self._can_bus_stream.send(erdos.Message(timestamp, can_bus))
        self._can_bus_stream.send(erdos.WatermarkMessage(timestamp))

    def run(self):
        rospy.init_node(self._name, anonymous=True, disable_signals=True)
        rospy.Subscriber(self._topic_name, PoseStamped, self.on_pose_update)
        rospy.spin()
