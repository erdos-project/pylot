import erdust
import numpy as np
import math
from pid_controller.pid import PID

# Pipeline imports.
from pylot.control.messages import ControlMessage


class PIDControlOperator(erdust.Operator):
    """ This class receives the vehicle identifier and low level waypoints
    from the local planner and sends out control commands to the vehicle
    being driven inside the simulation.

    Args:
        longitudinal_control_args: A dictionary of arguments to be used to
            initialize the longitudinal controller. It needs to contain
            three floating point values with the keys K_P, K_D, K_I, where
                K_P -- Proportional Term
                K_D -- Differential Term
                K_I -- Integral Term
                K_dt -- Time differential in seconds.
        _pid: The longitudinal PID controller.
    """

    def __init__(self,
                 waypoints_stream,
                 can_bus_stream,
                 control_stream,
                 name,
                 longitudinal_control_args,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        """ Initializes the operator to send out control information given
        waypoints on its input streams.

        Args:
            name: The name of the operator.
            longitudinal_control_args: A dictionary of arguments to be used to
                initialize the longitudinal controller. It needs to contain
                three floating point values with the keys K_P, K_D, K_I, where
                    K_P -- Proportional Term
                    K_D -- Differential Term
                    K_I -- Integral Term
            flags: A handle to the global flags instance to retrieve the
                configuration.
            log_file_name: The file to log the required information to.
            csv_file_name: The CSV file to log info to.
        """
        waypoints_stream.add_callback(self.on_waypoint)
        can_bus_stream.add_callback(self.on_can_bus_update, [control_stream])
        self._name = name
        self._flags = flags
        self._logger = erdust.setup_logging(name, log_file_name)
        self._csv_logger = erdust.setup_csv_logging(
            name + '-csv', csv_file_name)
        self._longitudinal_control_args = longitudinal_control_args
        self._pid = PID(
            p=longitudinal_control_args['K_P'],
            i=longitudinal_control_args['K_I'],
            d=longitudinal_control_args['K_D'],
        )
        self._vehicle_transform = None
        self._last_waypoint_msg = None
        self._latest_speed = 0

    @staticmethod
    def connect(waypoints_stream, can_bus_stream):
        control_stream = erdust.WriteStream()
        return [control_stream]

    def _get_throttle_brake(self, target_speed):
        """ Computes the throttle/brake required to reach the target speed.
        It uses the longitudinal controller to derive the required information.

        Args:
            target_speed: The target speed to reach.

        Returns:
            Throttle and brake values.
        """
        self._pid.target = target_speed
        pid_gain = self._pid(feedback=self._latest_speed)
        throttle = min(max(self._flags.default_throttle - 1.3 * pid_gain, 0),
                       self._flags.throttle_max)
        if pid_gain > 0.5:
            brake = min(0.35 * pid_gain * self._flags.brake_strength, 1)
        else:
            brake = 0
        return throttle, brake

    def _get_steering(self, waypoint_transform):
        """ Get the steering angle of the vehicle to reach the required
        waypoint.

        Args:
            waypoint: `simulation.utils.Transform` to retrieve the
                waypoint information from.

        Returns:
            The steering control in the range [-1, 1]
        """
        assert self._vehicle_transform is not None

        # Compute the vector to the waypoint.
        # The vehicle cannot move in the z-axis, so set that to 0.
        wp_vector = np.array([
            waypoint_transform.location.x,
            waypoint_transform.location.y,
            0.0,
        ]) - np.array([
            self._vehicle_transform.location.x,
            self._vehicle_transform.location.y,
            0.0,
        ])

        # Compute the vector of the vehicle.
        v_vector = np.array([
            math.cos(math.radians(self._vehicle_transform.rotation.yaw)),
            math.sin(math.radians(self._vehicle_transform.rotation.yaw)),
            0.0,
        ])

        # Normalize the vectors.
        wp_vector /= np.linalg.norm(wp_vector)
        v_vector /= np.linalg.norm(v_vector)

        # Compute the angle of the vehicle using the dot product.
        angle = math.acos(np.dot(v_vector, wp_vector))

        # Compute the sign of the angle.
        if np.cross(v_vector, wp_vector)[2] < 0:
            angle *= -1

        steering = self._flags.steer_gain * angle
        if steering > 0:
            steering = min(steering, 1)
        else:
            steering = max(steering, -1)
        return steering

    def on_can_bus_update(self, msg, control_stream):
        self._latest_speed = msg.data.forward_speed
        self._vehicle_transform = msg.data.transform
        throttle = 0.0
        brake = 0
        steer = 0
        if self._last_waypoint_msg:
            throttle, brake = self._get_throttle_brake(
                self._last_waypoint_msg.target_speed)
            steer = self._get_steering(self._last_waypoint_msg.waypoints[0])
        control_stream.send(
            ControlMessage(steer,
                           throttle,
                           brake,
                           False,
                           False,
                           msg.timestamp))

    def on_waypoint(self, msg):
        self._last_waypoint_msg = msg
