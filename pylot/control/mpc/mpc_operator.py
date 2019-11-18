import threading
import carla
import collections
import time
import pylot.utils

import numpy as np

from pid_controller.pid import PID
from erdos.op import Op
from erdos.utils import setup_logging

from pylot.control.messages import ControlMessage
from pylot.simulation.mpc_input import MPCInput, retrieve_actor
from pylot.control.mpc.cubic_spline import CubicSpline2D
from pylot.control.mpc.mpc import ModelPredictiveController
from pylot.control.mpc.utils import global_config
from pylot.simulation.carla_utils import get_world, to_carla_location


class MPCOperator(Op):
    def __init__(self, name, flags, log_file_name=None):
        """ Initializes the operator with the given information.

        Args:
            name: The name to be used for the operator in the dataflow graph.
            goal: The final location used to plan until.
            flags: The command line flags passed to the driver.
            log_file_name: The file name to log the intermediate messages to.
        """
        super(MPCOperator, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self._flags = flags

        _, world = get_world(self._flags.carla_host, self._flags.carla_port,
                             self._flags.carla_timeout)
        if world is None:
            raise ValueError("There was an issue connecting to the simulator.")
        self._world = world
        self._map = world.get_map()

        # Input retrieved from the various input streams.
        self._lock = threading.Lock()
        self._can_bus_msgs = collections.deque()
        self._pedestrians = collections.deque()

        # PID Controller
        self._pid = PID(p=self._flags.pid_p,
                        i=self._flags.pid_i,
                        d=self._flags.pid_d)

        # MPC
        self.mpc_input = None
        self.mpc = None

    @staticmethod
    def setup_streams(input_streams):
        input_streams.filter(pylot.utils.is_can_bus_stream).add_callback(
            MPCOperator.on_can_bus_update)
        input_streams.filter(
            pylot.utils.is_ground_pedestrians_stream).add_callback(
            MPCOperator.on_pedestrians_update)
        input_streams.add_completion_callback(
            MPCOperator.on_notification)
        return [pylot.utils.create_control_stream()]

    def on_can_bus_update(self, msg):
        """ Receives the CAN Bus update and adds it to the queue of messages.

        Args:
            msg: The message received for the given timestamp.
        """
        with self._lock:
            self._can_bus_msgs.append(msg)

    def on_pedestrians_update(self, msg):
        """ Receives the pedestrian update and adds it to the queue of
        messages.

        Args:
            msg: The message received for the given timestamp.
        """
        with self._lock:
            self._pedestrians.append(msg)

    @staticmethod
    def synchronize_msg_buffers(timestamp, buffers):
        """ Synchronizes the given buffers for the given timestamp.

       Args:
           timestamp: The timestamp to push all the top of the buffers to.
           buffers: The buffers to synchronize.

       Returns:
           True, if the buffers were successfully synchronized. False,
           otherwise.
       """
        for buffer in buffers:
            while (len(buffer) > 0 and buffer[0].timestamp < timestamp):
                buffer.popleft()
            if len(buffer) == 0:
                return False
            assert buffer[0].timestamp == timestamp
        return True

    def __rad2steer(self, rad):
        """
        Converts radians to steer input.

        :return: float [-1.0, 1.0]
        """
        steer = self._flags.steer_gain * rad
        if steer > 0:
            steer = min(steer, 1)
        else:
            steer = max(steer, -1)
        return steer

    def __steer2rad(self, steer):
        """
        Converts radians to steer input. Assumes max steering angle is -45, 45 degrees

        :return: float [-1.0, 1.0]
        """
        rad = steer / self._flags.steer_gain
        if rad > 0:
            rad = min(rad, np.pi/2)
        else:
            rad = max(rad, -np.pi/2)
        return rad

    def __get_throttle_brake_without_factor(self, current_speed, target_speed):
        self._pid.target = target_speed
        pid_gain = self._pid(feedback=current_speed)
        throttle = min(max(self._flags.default_throttle - 1.7 * pid_gain, 0),
                       self._flags.throttle_max)
        if pid_gain > 0.5:
            brake = min(0.35 * pid_gain * self._flags.brake_strength, 1)
        else:
            brake = 0
        return throttle, brake

    def on_notification(self, msg):
        """ The callback function invoked upon receipt of a WatermarkMessage.

        The function uses the latest location of the vehicle and drives to the
        next waypoint, while doing either a stop or a swerve upon the
        detection of a pedestrian.

        Args:
            msg: The timestamp for which the WatermarkMessage is retrieved.
        """
        with self._lock:
            if not self.synchronize_msg_buffers(
                    msg.timestamp, [self._can_bus_msgs, self._pedestrians]):
                self._logger.info("Could not synchronize the message buffers "
                                  "for the timestamp {}".format(msg.timestamp))

            can_bus_msg = self._can_bus_msgs.popleft()
            pedestrian_msg = self._pedestrians.popleft()

        # Assert that the timestamp of all the messages are the same.
        assert (can_bus_msg.timestamp == pedestrian_msg.timestamp)

        # Figure out the location of the ego vehicle and compute the next waypoint.
        ego_location = to_carla_location(can_bus_msg.data.transform.location)

        # step the controller
        self.mpc.step()

        # update vehicle info
        self.mpc.vehicle.x = ego_location.x
        self.mpc.vehicle.y = ego_location.y

        target_x = self.mpc.solution.x_list[-1]
        target_y = self.mpc.solution.y_list[-1]
        target_speed = self.mpc.solution.vel_list[-1]
        target_steer_rad = self.mpc.horizon_steer[0]  # in rad
        steer = self.__rad2steer(target_steer_rad)  # [-1.0, 1.0]
        throttle, brake = self.__get_throttle_brake_without_factor(
            self.mpc_input.get_ego_speed(), target_speed)

        # draw next waypoints
        self._world.debug.draw_point(carla.Location(x=target_x, y=target_y, z=0.5),
                                     size=0.2,
                                     life_time=30000.0)

        # send controls
        self.get_output_stream('control_stream').send(
            ControlMessage(steer, throttle, brake, False, False,
                           msg.timestamp))

    def execute(self):
        # wait for ego vehicle to spawn
        ego_vehicle = None
        while ego_vehicle is None:
            self._logger.info("Waiting for hero...")
            time.sleep(1)
            ego_vehicle = retrieve_actor(self._world, 'vehicle.*', 'hero')

        # intialize mpc input module
        self.mpc_input = MPCInput(ego_vehicle=ego_vehicle)

        # convert target waypoints into spline
        path = np.array(self.mpc_input.get_ego_path())
        spline = CubicSpline2D(path[:, 0], path[:, 1])
        ss = []
        vels = []
        xs = []
        ys = []
        yaws = []
        ks = []
        for s in spline.s[:-1]:
            x, y = spline.calc_position(s)
            yaw = np.abs(spline.calc_yaw(s))
            k = spline.calc_curvature(s)
            xs.append(x)
            ys.append(y)
            yaws.append(yaw)
            ks.append(k)
            ss.append(s)
            vels.append(self.mpc_input.speed_limit)

        config = global_config
        config["reference"] = {
            't_list': [],  # Time [s]
            's_list': ss,  # Arc distance [m]
            'x_list': xs,  # Desired X coordinates [m]
            'y_list': ys,  # Desired Y coordinates [m]
            'k_list': ks,  # Curvatures [1/m]
            'vel_list': vels,  # Desired tangential velocities [m/s]
            'yaw_list': yaws,  # Yaws [rad]
        }
        # draw intended trajectory
        for p in path:
            self._world.debug.draw_point(carla.Location(x=p[0], y=p[1], z=0.5),
                                         size=0.2,
                                         life_time=30000.0, color=carla.Color(0, 0, 255))

        # initialize mpc controller
        self.mpc = ModelPredictiveController(config=config)
        self.spin()

