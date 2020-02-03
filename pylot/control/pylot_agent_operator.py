"""Implements an agent operator that uses info from other operators."""

from collections import deque
import erdos
import math
from pid_controller.pid import PID
import time

# Pylot imports
from pylot.control.messages import ControlMessage
import pylot.control.utils
from pylot.map.hd_map import HDMap
from pylot.simulation.utils import get_map
from pylot.utils import time_epoch_ms


class PylotAgentOperator(erdos.Operator):
    """Agent operator that stops whenever it encounters an obstacle.

    The agent stops whenever it encounters an obstacle, and waits for the
    obstacle to move. It does not implement an obstacle avoidance policy.

    The agent waits for all input streams to receive a watermark message for
    timestamp t, and then it computes and sends a control command.

    Args:
        can_bus_stream (:py:class:`erdos.ReadStream`): Stream on which can bus
            info is received.
        waypoints_stream (:py:class:`erdos.ReadStream`): Stream on which
            :py:class:`~pylot.planning.messages.WaypointMessage` messages are
            received. The agent receives waypoints from the planning operator,
            and must follow these waypoints.
        traffic_lights_stream (:py:class:`erdos.ReadStream`): Stream on which
            detected traffic lights are received.
        obstacles_stream (:py:class:`erdos.ReadStream`): Stream on which
            detected obstacles are received.
        open_drive_stream (:py:class:`erdos.ReadStream`): Stream on which open
            drive string representations are received. The operator can
            construct HDMaps out of the open drive strings.
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
                 traffic_lights_stream,
                 obstacles_stream,
                 open_drive_stream,
                 control_stream,
                 name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        can_bus_stream.add_callback(self.on_can_bus_update)
        waypoints_stream.add_callback(self.on_waypoints_update)
        traffic_lights_stream.add_callback(self.on_traffic_lights_update)
        obstacles_stream.add_callback(self.on_obstacles_update)
        open_drive_stream.add_callback(self.on_open_drive_map)
        erdos.add_watermark_callback([
            can_bus_stream, waypoints_stream, traffic_lights_stream,
            obstacles_stream
        ], [control_stream], self.on_watermark)
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
        self._traffic_lights_msgs = deque()
        self._obstacles_msgs = deque()

    @staticmethod
    def connect(can_bus_stream, waypoints_stream, traffic_lights_stream,
                obstacles_stream, open_drive_stream):
        control_stream = erdos.WriteStream()
        return [control_stream]

    def run(self):
        # Run method is invoked after all operators finished initializing,
        # including the CARLA operator, which reloads the world. Thus, if
        # we get the map here we're sure it is up-to-date.
        if not hasattr(self._flags, 'track'):
            # The agent is not used in the Carla challenge. It has access to
            # the simulator, and to the town map.
            self._map = HDMap(
                get_map(self._flags.carla_host, self._flags.carla_port,
                        self._flags.carla_timeout), self._log_file_name)
            self._logger.debug('Agent running using map')
        else:
            self._map = None

    def on_watermark(self, timestamp, control_stream):
        """Invoked when all input streams have received a watermark.

        The method computes and sends the control command on the control
        stream.

        Args:
            timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of
                the watermark.
        """
        self._logger.debug('@{}: received watermark'.format(timestamp))
        start_time = time.time()
        can_bus_msg = self._can_bus_msgs.popleft()
        vehicle_transform = can_bus_msg.data.transform
        # Vehicle sped in m/s
        vehicle_speed = can_bus_msg.data.forward_speed
        waypoint_msg = self._waypoint_msgs.popleft()
        # The operator picks the wp_num_steer-th waypoint to compute the angle
        # it has to steer by when taking a turn.
        # Use 10th waypoint for steering.
        wp_vector, wp_angle = \
            pylot.planning.utils.compute_waypoint_vector_and_angle(
                vehicle_transform, waypoint_msg.waypoints, 4)
        # Use 5th waypoint for speed.
        _, wp_angle_speed = \
            pylot.planning.utils.compute_waypoint_vector_and_angle(
                vehicle_transform, waypoint_msg.waypoints, 9)
        tl_msg = self._traffic_lights_msgs.popleft()
        obstacles_msg = self._obstacles_msgs.popleft()

        self._logger.debug('@{}: speed {} and location {}'.format(
            timestamp, vehicle_speed, vehicle_transform))

        speed_factor, _ = pylot.planning.utils.stop_for_agents(
            vehicle_transform.location, wp_angle, wp_vector,
            obstacles_msg.obstacles, tl_msg.obstacles, self._flags,
            self._logger, self._map, timestamp)

        control_msg = self.get_control_message(wp_angle, wp_angle_speed,
                                               speed_factor, vehicle_speed,
                                               self._flags.target_speed,
                                               timestamp)

        # Get runtime in ms.
        runtime = (time.time() - start_time) * 1000
        self._csv_logger.info('{},{},"{}",{}'.format(time_epoch_ms(),
                                                     self._name, timestamp,
                                                     runtime))

        control_stream.send(control_msg)

    def on_waypoints_update(self, msg):
        self._logger.debug('@{}: waypoints update'.format(msg.timestamp))
        self._waypoint_msgs.append(msg)

    def on_can_bus_update(self, msg):
        self._logger.debug('@{}: can bus update'.format(msg.timestamp))
        self._can_bus_msgs.append(msg)

    def on_traffic_lights_update(self, msg):
        self._logger.debug('@{}: traffic lights update'.format(msg.timestamp))
        self._traffic_lights_msgs.append(msg)

    def on_obstacles_update(self, msg):
        self._logger.debug('@{}: obstacles update'.format(msg.timestamp))
        self._obstacles_msgs.append(msg)

    def on_open_drive_map(self, msg):
        self._logger.debug('@{}: open drive update'.format(msg.timestamp))
        try:
            import carla
        except ImportError:
            raise Exception('Error importing carla.')
        self._map = HDMap(carla.Map('challenge', msg.data),
                          self._log_file_name)

    def get_control_message(self, wp_angle, wp_angle_speed, speed_factor,
                            current_speed, target_speed, timestamp):
        if current_speed < 0:
            self._logger.warning(
                'Current speed is negative: {}'.format(current_speed))
            current_speed = 0
        steer = pylot.control.utils.radians_to_steer(wp_angle,
                                                     self._flags.steer_gain)
        # Don't go to fast around corners
        if math.fabs(wp_angle_speed) < 0.1:
            target_speed_adjusted = target_speed * speed_factor / 2
        else:
            target_speed_adjusted = target_speed * speed_factor

        throttle, brake = pylot.control.utils.compute_throttle_and_brake(
            self._pid, current_speed, target_speed_adjusted, self._flags)

        return ControlMessage(steer, throttle, brake, False, False, timestamp)
