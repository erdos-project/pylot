from dbw_mkz_msgs.msg import BrakeCmd, SteeringCmd, ThrottleCmd

import erdos

from pylot.control.messages import ControlMessage

import rospy

from std_msgs.msg import Empty

ROS_NAMESPACE = "/vehicle/"
ROS_FREQUENCY = 100  # hz
ENABLE_TOPIC = ROS_NAMESPACE + "enable"
DISABLE_TOPIC = ROS_NAMESPACE + "disable"
THROTTLE_TOPIC = ROS_NAMESPACE + "throttle_cmd"
BRAKE_TOPIC = ROS_NAMESPACE + "brake_cmd"
STEERING_TOPIC = ROS_NAMESPACE + "steering_cmd"

STEERING_ANGLE_MAX = 8.2


class DriveByWireOperator(erdos.Operator):
    """Converts ControlMessages to messages for the Lincoln MKZ ADAS.

    Args:
        control_stream (:py:class:`erdos.ReadStream`): Stream on which the
            operator receives control commands.
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, control_stream, flags):
        self._control_stream = control_stream
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)

        # ROS Publishers to publish the commands to ADAS.
        self.enable_pub, self.disable_pub = None, None
        self.throttle_pub, self.brake_pub = None, None
        self.steering_pub = None

    @staticmethod
    def connect(control_stream):
        return []

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
        rospy.init_node(self.config.name, anonymous=True, disable_signals=True)

        # Enable the ADAS.
        # self.enable_pub.publish(Empty())

        # Pull from the control stream and publish messages continuously.
        r = rospy.Rate(ROS_FREQUENCY)
        last_control_message = ControlMessage(
            steer=0,
            throttle=0,
            brake=0,
            hand_brake=False,
            reverse=False,
            timestamp=erdos.Timestamp(coordinates=[0]))
        while not rospy.is_shutdown():
            control_message = self._control_stream.try_read()
            if control_message is None or isinstance(control_message,
                                                     erdos.WatermarkMessage):
                control_message = last_control_message
            else:
                last_control_message = control_message

            # Send all the commands from a single ControlMessage one after
            # the other.
            steer_angle = control_message.steer * STEERING_ANGLE_MAX
            self._logger.debug("The steering angle is {}".format(steer_angle))
            steer_message = SteeringCmd(enable=True,
                                        clear=True,
                                        ignore=False,
                                        quiet=False,
                                        count=0,
                                        steering_wheel_angle_cmd=steer_angle,
                                        steering_wheel_angle_velocity=0.0)
            self.steering_pub.publish(steer_message)

            throttle_message = ThrottleCmd(
                enable=True,
                ignore=False,
                count=0,
                pedal_cmd_type=ThrottleCmd.CMD_PERCENT,
                pedal_cmd=control_message.throttle)
            self.throttle_pub.publish(throttle_message)

            brake_message = BrakeCmd(enable=True,
                                     ignore=False,
                                     count=0,
                                     pedal_cmd_type=BrakeCmd.CMD_PERCENT,
                                     pedal_cmd=control_message.brake)
            self.brake_pub.publish(brake_message)

            # Run at frequency
            r.sleep()

        # Disable the ADAS.
        # self.disable_pub.publish(Empty())
