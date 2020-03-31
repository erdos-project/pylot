"""Implements an agent operator that uses info from other operators."""

from collections import deque
import erdos
import math
from pid_controller.pid import PID

# Pylot imports
import pylot.control.utils
from pylot.control.messages import ControlMessage
from pylot.utils import time_epoch_ms


class PIDAgentOperator(erdos.Operator):
    """Agent operator that uses PID to follow a list of waystops.

    The agent waits for the pose and waypoint streams to receive a watermark
    message for timestamp t, and then it computes and sends a control command.

    Args:
        pose_stream (:py:class:`erdos.ReadStream`): Stream on which pose
            info is received.
        waypoints_stream (:py:class:`erdos.ReadStream`): Stream on which
            :py:class:`~pylot.planning.messages.WaypointMessage` messages are
            received. The agent receives waypoints from the planning operator,
            and must follow these waypoints.
        control_stream (:py:class:`erdos.WriteStream`): Stream on which the
            operator sends :py:class:`~pylot.control.messages.ControlMessage`
            messages.
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, pose_stream, waypoints_stream, control_stream, flags):
        pose_stream.add_callback(self.on_pose_update)
        waypoints_stream.add_callback(self.on_waypoints_update)
        erdos.add_watermark_callback([pose_stream, waypoints_stream],
                                     [control_stream], self.on_watermark)
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._pid = PID(p=self._flags.pid_p,
                        i=self._flags.pid_i,
                        d=self._flags.pid_d)
        # Queues in which received messages are stored.
        self._waypoint_msgs = deque()
        self._pose_msgs = deque()

    @staticmethod
    def connect(pose_stream, waypoints_stream):
        control_stream = erdos.WriteStream()
        return [control_stream]

    @erdos.profile_method()
    def on_watermark(self, timestamp, control_stream):
        """Computes and sends the control command on the control stream.

        Invoked when all input streams have received a watermark.

        Args:
            timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of
                the watermark.
        """
        self._logger.debug('@{}: received watermark'.format(timestamp))
        pose_msg = self._pose_msgs.popleft()
        vehicle_transform = pose_msg.data.transform
        # Vehicle sped in m/s
        current_speed = pose_msg.data.forward_speed
        waypoint_msg = self._waypoint_msgs.popleft()
        if current_speed < 0:
            self._logger.warning(
                'Current speed is negative: {}'.format(current_speed))
            current_speed = 0

        if len(waypoint_msg.waypoints) > 0:
            # The operator picks the wp_num_steer-th waypoint to compute the
            # angle it has to steer by when taking a turn.
            # Use 10th waypoint for steering.
            _, wp_angle_steer = \
                pylot.planning.utils.compute_waypoint_vector_and_angle(
                    vehicle_transform, waypoint_msg.waypoints,
                    self._flags.pid_steer_wp)
            # Use 5th waypoint for speed.
            _, wp_angle_speed = \
                pylot.planning.utils.compute_waypoint_vector_and_angle(
                    vehicle_transform, waypoint_msg.waypoints,
                    self._flags.pid_speed_wp)
            target_speed = waypoint_msg.target_speeds[min(
                len(waypoint_msg.target_speeds) - 1, self._flags.pid_speed_wp)]
            throttle, brake = pylot.control.utils.compute_throttle_and_brake(
                self._pid, current_speed, target_speed, self._flags)
            steer = pylot.control.utils.radians_to_steer(
                wp_angle_steer, self._flags.steer_gain)
        else:
            self._logger.warning('Braking! No more waypoints to follow.')
            throttle, brake = 0.0, 0.5
            steer = 0.0
        self._logger.debug(
            '@{}: speed {}, location {}, steer {}, throttle {}, brake {}'.
            format(timestamp, current_speed, vehicle_transform, steer,
                   throttle, brake))
        control_stream.send(
            ControlMessage(steer, throttle, brake, False, False, timestamp))

    def on_waypoints_update(self, msg):
        self._logger.debug('@{}: waypoints update'.format(msg.timestamp))
        self._waypoint_msgs.append(msg)

    def on_pose_update(self, msg):
        self._logger.debug('@{}: pose update'.format(msg.timestamp))
        self._pose_msgs.append(msg)
