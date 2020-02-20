import erdos
import rospy
from std_msgs.msg import Empty
from dbw_mkz_msgs.msg import ThrottleCmd, BrakeCmd, SteeringCmd

ROS_NAMESPACE = "/vehicle/"
ENABLE_TOPIC = ROS_NAMESPACE + "enable"
DISABLE_TOPIC = ROS_NAMESPACE + "disable"
THROTTLE_TOPIC = ROS_NAMESPACE + "throttle_cmd"
BRAKE_TOPIC = ROS_NAMESPACE + "brake_cmd"
STEERING_TOPIC = ROS_NAMESPACE + "steering_cmd"


class DriveByWireOperator(erdos.Operator):
    """ Operator that converts ControlMessages that Pylot sends to Carla to
    messages for the Lincoln MKZ ADAS. """

    def __init__(self,
                 control_stream,
                 name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        control_stream.add_callback(self.on_control_stream_update)
        self._name = name
        self._flags = flags
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            name + '-csv', csv_file_name)

        # ROS Publishers to publish the commands to ADAS.
        self.enable_pub, self.disable_pub = None, None
        self.throttle_pub, self.brake_pub = None, None
        self.steering_pub = None

    @staticmethod
    def connect(control_stream):
        return []

    def on_control_stream_update(self, msg):
        # Send all the commands from a single ControlMessage one after the other.
        steer_message = SteeringCmd(enable=True,
                                    ignore=False,
                                    count=msg.timestamp,
                                    cmd_type=SteeringCmd.CMD_ANGLE,
                                    steering_wheel_angle_cmd=msg.steer *
                                    SteeringCmd.ANGLE_MAX,
                                    steering_wheel_angle_velocity=0.0)
        self.steering_pub.publish(steer_message)

        throttle_message = ThrottleCmd(enable=True,
                                       ignore=False,
                                       count=msg.timestamp,
                                       pedal_cmd_type=ThrottleCmd.CMD_PERCENT,
                                       pedal_cmd=msg.throttle)
        self.throttle_pub.publish(throttle_message)

        brake_message = BrakeCmd(enable=True,
                                 ignore=False,
                                 count=msg.timestamp,
                                 pedal_cmd_type=BrakeCmd.CMD_PERCENT,
                                 pedal_cmd=msg.brake)
        self.throttle_pub.publish(brake_message)

    def run(self):
        # Initialize all the publishers.
        self.enable_pub = rospy.Publisher(ENABLE_TOPIC, Empty, queue_size=10)
        self.disable_pub = rospy.Publisher(DISABLE_TOPIC, Empty, queue_size=10)
        self.throttle_pub = rospy.Publisher(THROTTLE_TOPIC,
                                            ThrottleCmd,
                                            queue_size=10)
        self.brake_pub = rospy.Publisher(BRAKE_TOPIC, BrakeCmd, queue_size=10)
        self.steering_pub = rospy.Publisher(STEERING_TOPIC,
                                            SteeringCmd,
                                            queue_size=10)

        # Initialize the Node.
        rospy.init_node(self._name, anonymous=True, disable_signals=True)
        rospy.spin()
