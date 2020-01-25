from collections import deque
import erdos
import itertools
import numpy as np
from pid_controller.pid import PID

from pylot.control.messages import ControlMessage
from pylot.control.mpc.mpc import ModelPredictiveController
from pylot.control.mpc.utils import zero_to_2_pi, global_config, CubicSpline2D
import pylot.control.utils
from pylot.map.hd_map import HDMap
from pylot.simulation.utils import get_map, get_world


class MPCAgentOperator(erdos.Operator):
    def __init__(self,
                 can_bus_stream,
                 ground_obstacles_stream,
                 ground_traffic_lights_stream,
                 waypoints_stream,
                 control_stream,
                 name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        can_bus_stream.add_callback(self.on_can_bus_update)
        ground_obstacles_stream.add_callback(self.on_obstacles_update)
        ground_traffic_lights_stream.add_callback(
            self.on_traffic_lights_update)
        waypoints_stream.add_callback(self.on_waypoints_update)
        erdos.add_watermark_callback([
            can_bus_stream, ground_obstacles_stream,
            ground_traffic_lights_stream, waypoints_stream
        ], [control_stream], self.on_watermark)
        self._name = name
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            name + '-csv', csv_file_name)
        self._flags = flags
        self._config = global_config
        self._map = HDMap(
            get_map(self._flags.carla_host, self._flags.carla_port,
                    self._flags.carla_timeout), log_file_name)
        _, self._world = get_world(self._flags.carla_host,
                                   self._flags.carla_port,
                                   self._flags.carla_timeout)
        self._pid = PID(p=flags.pid_p, i=flags.pid_i, d=flags.pid_d)
        self._can_bus_msgs = deque()
        self._obstacles_msgs = deque()
        self._traffic_light_msgs = deque()
        self._waypoint_msgs = deque()

    @staticmethod
    def connect(can_bus_stream, ground_obstacles_stream,
                ground_traffic_lights_stream, waypoints_stream):
        control_stream = erdos.WriteStream()
        return [control_stream]

    def on_waypoints_update(self, msg):
        self._logger.debug('@{}: waypoints update'.format(msg.timestamp))
        self._waypoint_msgs.append(msg)

    def on_can_bus_update(self, msg):
        self._logger.debug('@{}: can bus update'.format(msg.timestamp))
        self._can_bus_msgs.append(msg)

    def on_obstacles_update(self, msg):
        self._logger.debug('@{}: obstacles update'.format(msg.timestamp))
        self._obstacles_msgs.append(msg)

    def on_traffic_lights_update(self, msg):
        self._logger.debug('@{}: traffic lights update'.format(msg.timestamp))
        self._traffic_light_msgs.append(msg)

    def on_watermark(self, timestamp, control_stream):
        self._logger.debug('@{}: received watermark'.format(timestamp))
        # Get hero vehicle info.
        can_bus_msg = self._can_bus_msgs.popleft()
        vehicle_transform = can_bus_msg.data.transform
        vehicle_speed = can_bus_msg.data.forward_speed
        # Get waypoints.
        waypoint_msg = self._waypoint_msgs.popleft()
        wp_angle = waypoint_msg.wp_angle
        wp_vector = waypoint_msg.wp_vector
        # Only take the first 50 waypoints (50 meters).
        waypoints = deque(itertools.islice(waypoint_msg.waypoints, 0, 50))
        # Get ground obstacles info.
        obstacles = self._obstacles_msgs.popleft().obstacles
        # Get ground traffic lights info.
        traffic_lights = self._traffic_light_msgs.popleft().traffic_lights

        speed_factor, state = pylot.control.utils.stop_for_agents(
            vehicle_transform.location, wp_angle, wp_vector, obstacles,
            traffic_lights, self._flags, self._logger, self._map, timestamp)

        control_msg = self.get_control_message(waypoints, vehicle_transform,
                                               vehicle_speed, speed_factor,
                                               timestamp)
        self._logger.debug("Throttle: {}".format(control_msg.throttle))
        self._logger.debug("Steer: {}".format(control_msg.steer))
        self._logger.debug("Brake: {}".format(control_msg.brake))
        self._logger.debug("State: {}".format(state))

        control_stream.send(control_msg)

    def setup_mpc(self, waypoints):
        path = np.array([[wp.location.x, wp.location.y] for wp in waypoints])

        # convert target waypoints into spline
        spline = CubicSpline2D(path[:, 0], path[:, 1], self._logger)
        ss = []
        vels = []
        xs = []
        ys = []
        yaws = []
        ks = []
        for s in spline.s[:-2]:
            x, y = spline.calc_position(s)
            yaw = np.abs(spline.calc_yaw(s))
            k = spline.calc_curvature(s)
            xs.append(x)
            ys.append(y)
            yaws.append(yaw)
            ks.append(k)
            ss.append(s)
            vels.append(18)

        self._config["reference"] = {
            't_list': [],  # Time [s]
            's_list': ss,  # Arc distance [m]
            'x_list': xs,  # Desired X coordinates [m]
            'y_list': ys,  # Desired Y coordinates [m]
            'k_list': ks,  # Curvatures [1/m]
            'vel_list': vels,  # Desired tangential velocities [m/s]
            'yaw_list': yaws,  # Yaws [rad]
        }

        # initialize mpc controller
        self.mpc = ModelPredictiveController(config=self._config)

    def get_control_message(self, waypoints, vehicle_transform, current_speed,
                            speed_factor, timestamp):
        self.setup_mpc(waypoints)
        self.mpc.vehicle.x = vehicle_transform.location.x
        self.mpc.vehicle.y = vehicle_transform.location.y
        self.mpc.vehicle.yaw = np.deg2rad(
            zero_to_2_pi(vehicle_transform.rotation.yaw))

        try:
            self.mpc.step()
        except Exception as e:
            self._logger.error('Failed to solve MPC.')
            self._logger.error(e)
            return ControlMessage(0, 0, 1, False, False, timestamp)

        # Compute pid controls.
        target_speed = self.mpc.solution.vel_list[0] * speed_factor
        target_yaw = self.mpc.solution.yaw_list[0]
        target_steer_rad = self.mpc.horizon_steer[0]  # in rad
        steer = pylot.control.utils.radians_to_steer(target_steer_rad,
                                                     self._flags.steer_gain)
        throttle, brake = pylot.control.utils.compute_throttle_and_brake(
            self._pid, current_speed, target_speed, self._flags)

        return ControlMessage(steer, throttle, brake, False, False, timestamp)
