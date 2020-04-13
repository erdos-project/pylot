from collections import deque
import erdos
import numpy as np
from pid_controller.pid import PID

import pylot.control.utils
from pylot.planning.utils import remove_completed_waypoints
from pylot.control.messages import ControlMessage
from pylot.control.mpc.mpc import ModelPredictiveController
from pylot.control.mpc.utils import CubicSpline2D, global_config, zero_to_2_pi


class MPCAgentOperator(erdos.Operator):
    def __init__(self, pose_stream, waypoints_stream, control_stream, flags):
        pose_stream.add_callback(self.on_pose_update)
        waypoints_stream.add_callback(self.on_waypoints_update)
        erdos.add_watermark_callback([pose_stream, waypoints_stream],
                                     [control_stream], self.on_watermark)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags
        self._config = global_config
        self._pid = PID(p=flags.pid_p, i=flags.pid_i, d=flags.pid_d)
        self._pose_msgs = deque()
        self._obstacles_msgs = deque()
        self._traffic_light_msgs = deque()
        self._waypoint_msgs = deque()
        self._mpc = None

    @staticmethod
    def connect(pose_stream, waypoints_stream):
        control_stream = erdos.WriteStream()
        return [control_stream]

    def on_waypoints_update(self, msg):
        self._logger.debug('@{}: waypoints update'.format(msg.timestamp))
        self._waypoint_msgs.append(msg)

    def on_pose_update(self, msg):
        self._logger.debug('@{}: pose update'.format(msg.timestamp))
        self._pose_msgs.append(msg)

    @erdos.profile_method()
    def on_watermark(self, timestamp, control_stream):
        self._logger.debug('@{}: received watermark'.format(timestamp))

        # Get hero vehicle info.
        pose_msg = self._pose_msgs.popleft()
        vehicle_transform = pose_msg.data.transform
        vehicle_speed = pose_msg.data.forward_speed

        # Get first 50 waypoints (50 meters) waypoints.
        waypoint_msg = self._waypoint_msgs.popleft()
        trimmed_waypoints = remove_completed_waypoints(
            waypoint_msg.waypoints, vehicle_transform.location)
        target_speeds = waypoint_msg.target_speeds

        # Compute and send control message
        control_msg = self.get_control_message(waypoints, target_speeds,
                                               vehicle_transform,
                                               vehicle_speed, timestamp)
        self._logger.debug("Throttle: {}".format(control_msg.throttle))
        self._logger.debug("Steer: {}".format(control_msg.steer))
        self._logger.debug("Brake: {}".format(control_msg.brake))
        control_stream.send(control_msg)

    def setup_mpc(self, waypoints, target_speeds):
        path = np.array([[wp.location.x, wp.location.y] for wp in waypoints])
        # convert target waypoints into spline
        spline = CubicSpline2D(path[:, 0], path[:, 1])
        ss = []
        vs = []
        xs = []
        ys = []
        yaws = []
        ks = []
        for i, s in enumerate(spline.s[:-1]):
            x, y = spline.calc_position(s)
            yaw = np.abs(spline.calc_yaw(s))
            k = spline.calc_curvature(s)
            xs.append(x)
            ys.append(y)
            yaws.append(yaw)
            ks.append(k)
            ss.append(s)
            vs.append(target_speeds[i])

        self._config["reference"] = {
            't_list': [],  # Time [s]
            's_list': ss,  # Arc distance [m]
            'x_list': xs,  # Desired X coordinates [m]
            'y_list': ys,  # Desired Y coordinates [m]
            'k_list': ks,  # Curvatures [1/m]
            'vel_list': vs,  # Desired tangential velocities [m/s]
            'yaw_list': yaws,  # Yaws [rad]
        }

        # initialize mpc controller
        self._mpc = ModelPredictiveController(config=self._config)

    def get_control_message(self, waypoints, target_speeds, vehicle_transform,
                            current_speed, timestamp):
        self.setup_mpc(waypoints, target_speeds)
        self._mpc.vehicle.x = vehicle_transform.location.x
        self._mpc.vehicle.y = vehicle_transform.location.y
        self._mpc.vehicle.yaw = np.deg2rad(
            zero_to_2_pi(vehicle_transform.rotation.yaw))

        try:
            self._mpc.step()
        except Exception as e:
            self._logger.error('Failed to solve MPC. Emergency braking.')
            self._logger.error(e)
            return ControlMessage(0, 0, 1, False, False, timestamp)

        # Compute pid controls.
        target_speed = self._mpc.solution.vel_list[-1]
        target_steer_rad = self._mpc.horizon_steer[0]  # in rad
        steer = pylot.control.utils.radians_to_steer(target_steer_rad,
                                                     self._flags.steer_gain)
        throttle, brake = pylot.control.utils.compute_throttle_and_brake(
            self._pid, current_speed, target_speed, self._flags)
        return ControlMessage(steer, throttle, brake, False, False, timestamp)
