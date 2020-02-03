"""Implements an agent operator that uses info from other operators."""

from collections import deque
import erdos
import math
from pid_controller.pid import PID
import time

# Pylot imports
from pylot.control.messages import ControlMessage
import pylot.control.utils
from pylot.utils import time_epoch_ms

DEFAULT_STEER_WAYPOINT = 5  # Use the 5th waypoint for computing speed
DEFAULT_SPEED_WAYPOINT = 9  # Use the 10th waypoint for computing speed


class PIDAgentOperator(erdos.Operator):
    """Agent operator that uses PID to follow a list of waystops.

    The agent waits for the can bus and waypoint streams to receive a watermark
    message for timestamp t, and then it computes and sends a control command.

    Args:
        can_bus_stream (:py:class:`erdos.ReadStream`): Stream on which can bus
            info is received.
        waypoints_stream (:py:class:`erdos.ReadStream`): Stream on which
            :py:class:`~pylot.planning.messages.WaypointMessage` messages are
            received. The agent receives waypoints from the planning operator,
            and must follow these waypoints.
        control_stream (:py:class:`erdos.WriteStream`): Stream on which the
            operator sends :py:class:`~pylot.control.messages.ControlMessage`
            messages.
        name (:obj:`str`): The name of the operator.
        flags (absl.flags): Object to be used to access absl flags.
        log_file_name (:obj:`str`, optional): Name of file where log messages
            are written to. If None, then messages are written to stdout.
        csv_file_name (:obj:`str`, optional): Name of file where stats logs are
            written to. If None, then messages are written to stdout.
    """
    def __init__(self,
                 can_bus_stream,
                 waypoints_stream,
                 control_stream,
                 name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        can_bus_stream.add_callback(self.on_can_bus_update)
        waypoints_stream.add_callback(self.on_waypoints_update)
        erdos.add_watermark_callback([can_bus_stream, waypoints_stream],
                                     [control_stream], self.on_watermark)
        self._name = name
        self._flags = flags
        self._log_file_name = log_file_name
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            name + '-csv', csv_file_name)
        self._pid = PID(p=self._flags.pid_p,
                        i=self._flags.pid_i,
                        d=self._flags.pid_d)
        # Queues in which received messages are stored.
        self._waypoint_msgs = deque()
        self._can_bus_msgs = deque()

    @staticmethod
    def connect(can_bus_stream, waypoints_stream):
        control_stream = erdos.WriteStream()
        return [control_stream]

    def on_watermark(self, timestamp, control_stream):
        """Computes and sends the control command on the control stream.

        Invoked when all input streams have received a watermark.

        Args:
            timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of
                the watermark.
        """
        self._logger.debug('@{}: received watermark'.format(timestamp))
        start_time = time.time()
        can_bus_msg = self._can_bus_msgs.popleft()
        vehicle_transform = can_bus_msg.data.transform
        # Vehicle sped in m/s
        current_speed = can_bus_msg.data.forward_speed
        waypoint_msg = self._waypoint_msgs.popleft()
        if current_speed < 0:
            self._logger.warning(
                'Current speed is negative: {}'.format(current_speed))
            current_speed = 0

        # The operator picks the wp_num_steer-th waypoint to compute the angle
        # it has to steer by when taking a turn.
        # Use 10th waypoint for steering.
        _, wp_angle_steer = \
            pylot.planning.utils.compute_waypoint_vector_and_angle(
                vehicle_transform, waypoint_msg.waypoints,
                DEFAULT_STEER_WAYPOINT)
        # Use 5th waypoint for speed.
        _, wp_angle_speed = \
            pylot.planning.utils.compute_waypoint_vector_and_angle(
                vehicle_transform, waypoint_msg.waypoints,
                DEFAULT_SPEED_WAYPOINT)
        # Don't go to fast around corners
        target_speed_adjusted = waypoint_msg.target_speeds[min(
            len(waypoint_msg.target_speeds) - 1, DEFAULT_SPEED_WAYPOINT)]
        if math.fabs(wp_angle_speed) < 0.1:
            target_speed_adjusted /= 2

        throttle, brake = pylot.control.utils.compute_throttle_and_brake(
            self._pid, current_speed, target_speed_adjusted, self._flags)
        steer = pylot.control.utils.radians_to_steer(wp_angle_steer,
                                                     self._flags.steer_gain)

        self._logger.debug('@{}: speed {} and location {}'.format(
            timestamp, current_speed, vehicle_transform))
        # Get runtime in ms.
        runtime = (time.time() - start_time) * 1000
        self._csv_logger.info('{},{},"{}",{}'.format(time_epoch_ms(),
                                                     self._name, timestamp,
                                                     runtime))

        control_stream.send(
            ControlMessage(steer, throttle, brake, False, False, timestamp))

    def on_waypoints_update(self, msg):
        self._logger.debug('@{}: waypoints update'.format(msg.timestamp))
        self._waypoint_msgs.append(msg)

    def on_can_bus_update(self, msg):
        self._logger.debug('@{}: can bus update'.format(msg.timestamp))
        self._can_bus_msgs.append(msg)
