import threading
import collections
import os
import json

from pid_controller.pid import PID

from erdos.op import Op
from erdos.utils import setup_logging

import pylot.utils
from pylot.control.messages import ControlMessage
from pylot.simulation.utils import to_pylot_transform, Planner, retrieve_actor
from pylot.simulation.carla_utils import get_world, to_carla_location
from pylot.planning.utils import get_waypoint_vector_and_angle


class MPCPlanningOperator(Op):
    def __init__(self, name, goal, flags, log_file_name=None):
        """ Initializes the operator with the given information.

        Args:
            name: The name to be used for the operator in the dataflow graph.
            goal: The final location used to plan until.
            flags: The command line flags passed to the driver.
            log_file_name: The file name to log the intermediate messages to.
        """
        super(MPCPlanningOperator, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self._flags = flags
        self._goal = goal

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

    @staticmethod
    def setup_streams(input_streams):
        input_streams.filter(pylot.utils.is_can_bus_stream).add_callback(
            MPCPlanningOperator.on_can_bus_update)
        input_streams.filter(
            pylot.utils.is_ground_pedestrians_stream).add_callback(
            MPCPlanningOperator.on_pedestrians_update)
        input_streams.add_completion_callback(
            MPCPlanningOperator.on_notification)
        return [pylot.utils.create_control_stream()]

    def on_can_bus_update(self, msg):
        """ Receives the CAN Bus update and adds it to the queue of messages.

        Args:
            msg: The message received for the given timestamp.
        """
        self._logger.info(
            "Received a CAN Bus update for the timestamp {}".format(
                msg.timestamp))
        with self._lock:
            self._can_bus_msgs.append(msg)

    def on_pedestrians_update(self, msg):
        """ Receives the pedestrian update and adds it to the queue of
        messages.

        Args:
            msg: The message received for the given timestamp.
        """
        self._logger.info(
            "Received a pedestrian update for the timestamp {}".format(
                msg.timestamp))
        with self._lock:
            self._pedestrians.append(msg)

    def synchronize_msg_buffers(self, timestamp, buffers):
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

    def __get_steer(self, wp_angle):
        steer = self._flags.steer_gain * wp_angle
        if steer > 0:
            steer = min(steer, 1)
        else:
            steer = max(steer, -1)
        return steer

    def __get_throttle_brake_without_factor(self, current_speed, target_speed):
        self._pid.target = target_speed
        pid_gain = self._pid(feedback=current_speed)
        throttle = min(max(self._flags.default_throttle - 1.3 * pid_gain, 0),
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
        self._logger.info("Received a watermark for the timestamp {}".format(
            msg.timestamp))
        with self._lock:
            if not self.synchronize_msg_buffers(
                    msg.timestamp, [self._can_bus_msgs, self._pedestrians]):
                self._logger.info("Could not synchronize the message buffers "
                                  "for the timestamp {}".format(msg.timestamp))

            can_bus_msg = self._can_bus_msgs.popleft()
            pedestrian_msg = self._pedestrians.popleft()

        # Assert that the timestamp of all the messages are the same.
        assert (can_bus_msg.timestamp == pedestrian_msg.timestamp)

        self._logger.info("Can Bus Message: {}, Pedestrian Message: {}".format(
            can_bus_msg.timestamp, pedestrian_msg.timestamp))

        ### TEST ###
        data = {}
        data["time"] = self.planner.get_time()
        data["ego_accel"] = self.planner.get_ego_accel()
        data["ego_speed"] = self.planner.get_ego_speed()
        data["ego_location"] = self.planner.get_ego_location()
        data["ego_bbox"] = self.planner.get_ego_bbox()
        data["speed_limit"] = self.planner.get_speed_limit()
        data["bounds_and_marks"] = self.planner.get_road_bounds_and_lane_marks()
        data["dynamic_bboxes"] = self.planner.get_dynamic_bboxes()
        data["static"] = self.planner.get_static_bboxes()
        with open(os.path.join("./test/", "{}.json".format(data["time"])), "w") as outfile:
            outfile.write(json.dumps(data, outfile))
        ### TEST ###

        # Figure out the location of the ego vehicle and compute the next
        # waypoint.
        ego_location = to_carla_location(can_bus_msg.data.transform.location)
        if ego_location.distance(self._goal) <= 5:
            self.get_output_stream('control_stream').send(
                ControlMessage(0.0, 0.0, 1.0, False, False, msg.timestamp))
        else:
            #wp_speed = self._map.get_waypoint(ego_location).next(4.0)[0]
            #wp_speed_vector, wp_speed_angle = get_waypoint_vector_and_angle(
            #    to_pylot_transform(wp_speed.transform),
            #    can_bus_msg.data.transform)

            wp_steer = self._map.get_waypoint(ego_location).next(1.0)[0].next(1.0)[0].next(1.0)[0]
            self._world.debug.draw_point(wp_steer.transform.location,
                                         size=0.2,
                                         life_time=30000.0)
            wp_steer_vector, wp_steer_angle = get_waypoint_vector_and_angle(
                to_pylot_transform(wp_steer.transform),
                can_bus_msg.data.transform)

            current_speed = max(0, can_bus_msg.data.forward_speed)
            steer = self.__get_steer(wp_steer_angle)
            throttle, brake = self.__get_throttle_brake_without_factor(
                current_speed, 10)
            self.get_output_stream('control_stream').send(
                ControlMessage(steer, throttle, brake, False, False,
                               msg.timestamp))

    def execute(self):
        ego_vehicle = retrieve_actor(self._world, 'vehicle.*', 'hero')
        self.planner = Planner(ego_vehicle=ego_vehicle)
        self.spin()

