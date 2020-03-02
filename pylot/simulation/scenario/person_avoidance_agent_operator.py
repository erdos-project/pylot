from absl import flags
import collections
import erdos
import numpy as np
from pid_controller.pid import PID

import pylot.control.utils
from pylot.control.messages import ControlMessage
from pylot.simulation.utils import get_world

flags.DEFINE_enum('avoidance_behavior', 'stop', ['stop', 'swerve'],
                  'Planning behavior in case of emergencies (stop/swerve)')


class PersonAvoidanceAgentOperator(erdos.Operator):
    def __init__(self, can_bus_stream, obstacles_stream,
                 ground_obstacles_stream, control_stream, goal, flags):
        """ Initializes the operator with the given information.

        Args:
            goal: The destination pylot.utils.Location used to plan until.
            flags: The command line flags passed to the driver.
        """
        can_bus_stream.add_callback(self.on_can_bus_update)
        obstacles_stream.add_callback(self.on_obstacles_update)
        ground_obstacles_stream.add_callback(self.on_ground_obstacles_update)
        erdos.add_watermark_callback([can_bus_stream, obstacles_stream],
                                     [control_stream], self.on_watermark)

        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            self.config.name + '-csv', self.config.csv_log_file_name)
        self._flags = flags
        self._goal = goal

        # Input retrieved from the various input streams.
        self._can_bus_msgs = collections.deque()
        self._ground_obstacles_msgs = collections.deque()
        self._obstacle_msgs = collections.deque()

        # PID Controller
        self._pid = PID(p=self._flags.pid_p,
                        i=self._flags.pid_i,
                        d=self._flags.pid_d)

        # Planning constants.
        self.SPEED = self._flags.target_speed
        self.DETECTION_DISTANCE = 12
        self.GOAL_DISTANCE = self.SPEED
        self.SAMPLING_DISTANCE = self.SPEED / 3
        self._goal_reached = False

    @staticmethod
    def connect(can_bus_stream, obstacles_stream, ground_obstacles_stream):
        control_stream = erdos.WriteStream()
        return [control_stream]

    def run(self):
        # Run method is invoked after all operators finished initializing,
        # including the CARLA operator, which reloads the world. Thus, if
        # we get the world here we're sure it is up-to-date.
        _, world = get_world(self._flags.carla_host, self._flags.carla_port,
                             self._flags.carla_timeout)
        if world is None:
            raise ValueError("There was an issue connecting to the simulator.")
        self._world = world
        self._map = world.get_map()

    def on_obstacles_update(self, msg):
        """ Receives the message from the detector and adds it to the queue.

        Args:
            msg: The message received for the given timestamp.
        """
        self._logger.debug('@{}: received detector obstacles update'.format(
            msg.timestamp))
        self._obstacle_msgs.append(msg)

    def on_can_bus_update(self, msg):
        """ Receives the CAN Bus update and adds it to the queue of messages.

        Args:
            msg: The message received for the given timestamp.
        """
        self._logger.debug('@{}: received can bus message'.format(
            msg.timestamp))
        self._can_bus_msgs.append(msg)

    def on_ground_obstacles_update(self, msg):
        """ Receives the ground obstacles update and adds it to the queue of
        messages.

        Args:
            msg: The message received for the given timestamp.
        """
        self._logger.debug('@{}: received ground obstacles message'.format(
            msg.timestamp))
        self._ground_obstacles_msgs.append(msg)

    def on_watermark(self, timestamp, control_stream):
        """ The callback function invoked upon receipt of a WatermarkMessage.

        The function uses the latest location of the vehicle and drives to the
        next waypoint, while doing either a stop or a swerve upon the
        detection of a person.

        Args:
            timestamp: Timestamp for which the WatermarkMessage was received.
            control_stream: Output stream on which the callback can write to.
        """
        self._logger.debug('@{}: received watermark'.format(timestamp))
        can_bus_msg = self._can_bus_msgs.popleft()
        ground_obstacles_msg = self._ground_obstacles_msgs.popleft()
        obstacle_msg = self._obstacle_msgs.popleft()

        self._logger.debug(
            "The vehicle is travelling at a speed of {} m/s.".format(
                can_bus_msg.data.forward_speed))

        ego_transform = can_bus_msg.data.transform
        ego_location = ego_transform.location
        ego_wp = self._map.get_waypoint(ego_location.as_carla_location())

        people_obstacles = [
            obstacle for obstacle in ground_obstacles_msg.obstacles
            if obstacle.label == 'person'
        ]
        # Heuristic to tell us how far away do we detect the person.
        for person in people_obstacles:
            person_wp = self._map.get_waypoint(
                person.transform.location.as_carla_location(),
                project_to_road=False)
            if person_wp and person_wp.road_id == ego_wp.road_id:
                for obstacle in obstacle_msg.obstacles:
                    if obstacle.label == 'person':
                        self._csv_logger.info(
                            "{},{},detected a person {}m away".format(
                                self.config.name, self.SPEED,
                                person.distance(ego_transform)))
                        self._csv_logger.info(
                            "{},{},vehicle speed {} m/s.".format(
                                self.config.name, self.SPEED,
                                can_bus_msg.data.forward_speed))

        # Figure out the location of the ego vehicle and compute the next
        # waypoint.
        if self._goal_reached or ego_location.distance(
                self._goal) <= self.GOAL_DISTANCE:
            self._logger.info(
                "The distance was {} and we reached the goal.".format(
                    ego_location.distance(self._goal)))
            control_stream.send(
                ControlMessage(0.0, 0.0, 1.0, True, False, timestamp))
            self._goal_reached = True
        else:
            person_detected = False
            for person in people_obstacles:
                person_wp = self._map.get_waypoint(
                    person.transform.location.as_carla_location(),
                    project_to_road=False)
                if person_wp and ego_location.distance(
                        person.transform.location) <= self.DETECTION_DISTANCE:
                    person_detected = True
                    break

            if person_detected and self._flags.avoidance_behavior == 'stop':
                control_stream.send(
                    ControlMessage(0.0, 0.0, 1.0, True, False, timestamp))
                return

            # Get the waypoint that is SAMPLING_DISTANCE away.
            sample_distance = self.SAMPLING_DISTANCE if \
                ego_transform.forward_vector.x > 0 else \
                -1 * self.SAMPLING_DISTANCE
            steer_loc = ego_location + pylot.utils.Location(
                x=sample_distance, y=0, z=0)
            wp_steer = self._map.get_waypoint(steer_loc.as_carla_location())

            in_swerve = False
            fwd_vector = wp_steer.transform.get_forward_vector()
            waypoint_fwd = [fwd_vector.x, fwd_vector.y, fwd_vector.z]
            if person_detected:
                # If a pedestrian was detected, make sure we're driving on the
                # wrong direction.
                if np.dot(ego_transform.forward_vector.as_numpy_array(),
                          waypoint_fwd) > 0:
                    # We're not driving in the wrong direction, get left
                    # lane waypoint.
                    if wp_steer.get_left_lane():
                        wp_steer = wp_steer.get_left_lane()
                        in_swerve = True
                else:
                    # We're driving in the right direction, continue driving.
                    pass
            else:
                # The person was not detected, come back from the swerve.
                if np.dot(ego_transform.forward_vector.as_numpy_array(),
                          waypoint_fwd) < 0:
                    # We're driving in the wrong direction, get the left lane
                    # waypoint.
                    if wp_steer.get_left_lane():
                        wp_steer = wp_steer.get_left_lane()
                        in_swerve = True
                else:
                    # We're driving in the right direction, continue driving.
                    pass

            self._world.debug.draw_point(wp_steer.transform.location,
                                         size=0.2,
                                         life_time=30000.0)

            wp_steer_vector, _, wp_steer_angle = \
                ego_transform.get_vector_magnitude_angle(
                    pylot.utils.Location.from_carla_location(
                        wp_steer.transform.location))
            current_speed = max(0, can_bus_msg.data.forward_speed)
            steer = pylot.control.utils.radians_to_steer(
                wp_steer_angle, self._flags.steer_gain)
            # target_speed = self.SPEED if not in_swerve else self.SPEED / 5.0
            target_speed = self.SPEED
            throttle, brake = pylot.control.utils.compute_throttle_and_brake(
                self._pid, current_speed, target_speed, self._flags)

            control_stream.send(
                ControlMessage(steer, throttle, brake, False, False,
                               timestamp))
