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
        if not hasattr(self._flags, 'track'):
            # The agent is not used in the Carla challenge. It has access to
            # the simulator, and to the town map.
            self._map = HDMap(
                get_map(self._flags.carla_host, self._flags.carla_port,
                        self._flags.carla_timeout), log_file_name)
            self._logger.debug('Agent running using map')
        else:
            self._map = None
        self._pid = PID(p=self._flags.pid_p,
                        i=self._flags.pid_i,
                        d=self._flags.pid_d)
        # Queues in which received messages are stored.
        self._waypoint_msgs = deque()
        self._can_bus_msgs = deque()
        self._traffic_lights_msgs = deque()
        self._obstacles_msgs = deque()
        self._vehicle_labels = {'car', 'bicycle', 'motorcycle', 'bus', 'truck'}

    @staticmethod
    def connect(can_bus_stream, waypoints_stream, traffic_lights_stream,
                obstacles_stream, open_drive_stream):
        control_stream = erdos.WriteStream()
        return [control_stream]

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
        waypoint_msg = self._waypoint_msgs.popleft()
        tl_msg = self._traffic_lights_msgs.popleft()
        obstacles_msg = self._obstacles_msgs.popleft()
        vehicle_transform = can_bus_msg.data.transform
        # Vehicle sped in m/s
        vehicle_speed = can_bus_msg.data.forward_speed
        wp_angle = waypoint_msg.wp_angle
        wp_vector = waypoint_msg.wp_vector
        wp_angle_speed = waypoint_msg.wp_angle_speed
        target_speed = waypoint_msg.target_speed

        people = []
        vehicles = []
        for obstacle in obstacles_msg.obstacles:
            if obstacle.label == 'person':
                people.append(obstacle)
            elif obstacle.label in self._vehicle_labels:
                vehicles.append(obstacle)

        self._logger.debug('@{}: speed {} and location {}'.format(
            timestamp, vehicle_speed, vehicle_transform))
        self._logger.debug('@{}: people {}'.format(timestamp, people))
        self._logger.debug('@{}: vehicles {}'.format(timestamp, vehicles))

        speed_factor, _ = self.__stop_for_agents(vehicle_transform.location,
                                                 wp_angle, wp_vector, vehicles,
                                                 people, tl_msg.obstacles,
                                                 timestamp)

        control_msg = self.get_control_message(wp_angle, wp_angle_speed,
                                               speed_factor, vehicle_speed,
                                               target_speed, timestamp)

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

    def __stop_for_agents(self, ego_vehicle_location, wp_angle, wp_vector,
                          vehicles, people, traffic_lights, timestamp):
        speed_factor = 1
        speed_factor_tl = 1
        speed_factor_p = 1
        speed_factor_v = 1

        for vehicle in vehicles:
            if (not self._map or self._map.are_on_same_lane(
                    ego_vehicle_location, vehicle.transform.location)):
                self._logger.debug(
                    '@{}: ego {} and vehicle {} are on the same lane'.format(
                        timestamp, ego_vehicle_location,
                        vehicle.transform.location))
                new_speed_factor_v = pylot.control.utils.stop_vehicle(
                    ego_vehicle_location, vehicle.transform.location,
                    wp_vector, speed_factor_v, self._flags)
                if new_speed_factor_v < speed_factor_v:
                    speed_factor_v = new_speed_factor_v
                    self._logger.debug(
                        '@{}: vehicle {} reduced speed factor to {}'.format(
                            timestamp, vehicle.transform.location,
                            speed_factor_v))

        for person in people:
            if (not self._map or self._map.are_on_same_lane(
                    ego_vehicle_location, person.transform.location)):
                self._logger.debug(
                    '@{}: ego {} and person {} are on the same lane'.format(
                        timestamp, ego_vehicle_location,
                        person.transform.location))
                new_speed_factor_p = pylot.control.utils.stop_person(
                    ego_vehicle_location, person.transform.location, wp_vector,
                    speed_factor_p, self._flags)
                if new_speed_factor_p < speed_factor_p:
                    speed_factor_p = new_speed_factor_p
                    self._logger.debug(
                        '@{}: person {} reduced speed factor to {}'.format(
                            timestamp, person.transform.location,
                            speed_factor_p))

        for tl in traffic_lights:
            if (not self._map or self._map.must_obbey_traffic_light(
                    ego_vehicle_location, tl.transform.location)):
                self._logger.debug(
                    '@{}: ego is obbeying traffic light {}'.format(
                        timestamp, ego_vehicle_location,
                        tl.transform.location))
                new_speed_factor_tl = pylot.control.utils.stop_traffic_light(
                    ego_vehicle_location, tl.transform.location, tl.label,
                    wp_vector, wp_angle, speed_factor_tl, self._flags)
                if new_speed_factor_tl < speed_factor_tl:
                    speed_factor_tl = new_speed_factor_tl
                    self._logger.debug(
                        '@{}: traffic light {} reduced speed factor to {}'.
                        format(timestamp, tl.transform.location,
                               speed_factor_tl))

        speed_factor = min(speed_factor_tl, speed_factor_p, speed_factor_v)
        state = {
            'stop_person': speed_factor_p,
            'stop_vehicle': speed_factor_v,
            'stop_traffic_lights': speed_factor_tl
        }
        self._logger.debug('@{}: agent speed factors {}'.format(
            timestamp, state))
        return speed_factor, state

    def get_control_message(self, wp_angle, wp_angle_speed, speed_factor,
                            current_speed, target_speed, timestamp):
        assert current_speed >= 0, 'Current speed is negative'
        steer = pylot.control.utils.radians_to_steer(wp_angle,
                                                     self._flags.steer_gain)
        # TODO(ionel): DO NOT HARDCODE VALUES!
        # Don't go to fast around corners
        if math.fabs(wp_angle_speed) < 0.1:
            target_speed_adjusted = target_speed * speed_factor
        elif math.fabs(wp_angle_speed) < 0.5:
            target_speed_adjusted = 6 * speed_factor
        else:
            target_speed_adjusted = 3 * speed_factor

        throttle, brake = pylot.control.utils.compute_throttle_and_brake(
            self._pid, current_speed, target_speed_adjusted, self._flags)

        return ControlMessage(steer, throttle, brake, False, False, timestamp)
