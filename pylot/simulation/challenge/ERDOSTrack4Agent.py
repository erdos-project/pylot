import logging

from absl import flags

from carla import VehicleControl

import erdos

import numpy as np

import pylot.flags
import pylot.operator_creator
import pylot.perception.messages
import pylot.utils
from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.detection.speed_limit_sign import SpeedLimitSign
from pylot.perception.detection.stop_sign import StopSign
from pylot.perception.detection.traffic_light import TrafficLight, \
    TrafficLightColor

from srunner.challenge.autoagents.autonomous_agent import AutonomousAgent,\
    Track

FLAGS = flags.FLAGS


class ERDOSTrack4Agent(AutonomousAgent):
    """Agent class that interacts with the challenge scenario runner.

    Warning:
        The agent is designed to work on track 4 only.
    """
    def setup(self, path_to_conf_file):
        """Setup phase code.

        Invoked by the scenario runner.
        """
        # Disable Tensorflow logging.
        pylot.utils.set_tf_loglevel(logging.ERROR)
        flags.FLAGS([__file__, '--flagfile={}'.format(path_to_conf_file)])
        self._logger = erdos.utils.setup_logging('erdos_agent',
                                                 FLAGS.log_file_name)
        enable_logging()
        self.track = Track.SCENE_LAYOUT
        # Stores the waypoints we get from the challenge planner.
        self._waypoints = None
        (pose_stream, global_trajectory_stream, ground_obstacles_stream,
         traffic_lights_stream, lanes_stream, open_drive_stream,
         control_stream) = create_data_flow()
        self._pose_stream = pose_stream
        self._global_trajectory_stream = global_trajectory_stream
        self._ground_obstacles_stream = ground_obstacles_stream
        self._traffic_lights_stream = traffic_lights_stream
        self._open_drive_stream = open_drive_stream
        self._sent_open_drive = False
        self._control_stream = control_stream

        # These are used for the timestamp hack.
        self._past_timestamp = None
        self._past_control_message = None
        # Execute the data-flow.
        erdos.run_async()

    def destroy(self):
        """Code to clean-up the agent.

        Invoked by the scenario runner between different runs.
        """
        self._logger.info('ERDOSTrack4Agent destroy method invoked')

    def sensors(self):
        """Defines the sensor suite required by the agent."""
        gnss_sensors = [{
            'type': 'sensor.other.gnss',
            'reading_frequency': 20,
            'x': 0.7,
            'y': -0.4,
            'z': 1.60,
            'id': 'gnss'
        }]
        can_sensors = [{
            'type': 'sensor.can_bus',
            'reading_frequency': 20,
            'id': 'can_bus'
        }]
        scene_layout_sensors = [{
            'type': 'sensor.scene_layout',
            'id': 'scene_layout',
        }]
        ground_objects = [{
            'type': 'sensor.object_finder',
            'reading_frequency': 20,
            'id': 'ground_objects'
        }]
        return (can_sensors + gnss_sensors + scene_layout_sensors +
                ground_objects)

    def run_step(self, input_data, timestamp):
        game_time = int(timestamp * 1000)
        self._logger.debug("Current game time {}".format(game_time))
        erdos_timestamp = erdos.Timestamp(coordinates=[game_time])

        if not self._sent_open_drive:
            # We do not have access to the open drive map. Send top watermark.
            self._sent_open_drive = True
            self._open_drive_stream.send(
                erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))

        self.send_waypoints_msg(erdos_timestamp)

        for key, val in input_data.items():
            if key == 'ground_objects':
                self.send_ground_objects(val[1], erdos_timestamp)
            elif key == 'scene_layout':
                self.send_scene_layout(val[1], erdos_timestamp)
            elif key == 'can_bus':
                self.send_pose_msg(val[1], erdos_timestamp)
            elif key == 'gnss':
                self.send_gnss_data(val[1], erdos_timestamp)
            else:
                self._logger.warning("Sensor {} not used".format(key))

        # Wait until the control is set.
        while True:
            control_msg = self._control_stream.read()
            if not isinstance(control_msg, erdos.WatermarkMessage):
                output_control = VehicleControl()
                output_control.throttle = control_msg.throttle
                output_control.brake = control_msg.brake
                output_control.steer = control_msg.steer
                output_control.reverse = control_msg.reverse
                output_control.hand_brake = control_msg.hand_brake
                output_control.manual_gear_shift = False
                return output_control

    def send_ground_objects(self, data, timestamp):
        # Input parsing is based on the description at
        # https://github.com/carla-simulator/carla/blob/master/PythonAPI/carla/scene_layout.py
        traffic_lights_list = []
        speed_limit_signs_list = []
        stop_signs_list = []
        static_obstacles_list = []

        # Dictionary that contains id, position, road_id, and lane_id of
        # the hero vehicle (position contains the same data as the gnss_data,
        # except with lower altitude).
        hero_vehicle = data['hero_vehicle']
        # Currently, we don't do anything with the road_id and lane_id of the
        # hero vehicle. This could potentially be useful in conjunction with
        # the data in scene_layout.
        # self._logger.debug('Hero vehicle id: {}'.format(hero_vehicle['id']))

        vehicles_list = self.parse_vehicles(data['vehicles'],
                                            hero_vehicle['id'])
        people_list = self.parse_people(data['walkers'])
        traffic_lights_list = self.parse_traffic_lights(data['traffic_lights'])
        stop_signs_list = self.parse_stop_signs(data['stop_signs'])
        speed_limit_signs_list = self.parse_speed_limit_signs(
            data['speed_limits'])
        static_obstacles_list = self.parse_static_obstacles(
            data['static_obstacles'])

        # Send messages.
        self._ground_obstacles_stream.send(
            pylot.perception.messages.ObstaclesMessage(
                timestamp,
                vehicles_list + people_list + speed_limit_signs_list +
                stop_signs_list + static_obstacles_list))
        self._ground_obstacles_stream.send(erdos.WatermarkMessage(timestamp))
        self._traffic_lights_stream.send(
            pylot.perception.messages.TrafficLightsMessage(
                timestamp, traffic_lights_list))
        self._traffic_lights_stream.send(erdos.WatermarkMessage(timestamp))

    def parse_vehicles(self, vehicles, ego_vehicle_id):
        # vehicles is a dictionary that maps each vehicle's id to
        # a dictionary of information about that vehicle. Each such dictionary
        # contains four items: the vehicle's id, position, orientation, and
        # bounding_box (represented as four points in GPS coordinates).
        vehicles_list = []
        for veh_dict in vehicles.values():
            vehicle_id = veh_dict['id']
            location = pylot.utils.Location.from_gps(*veh_dict['position'])
            roll, pitch, yaw = veh_dict['orientation']
            rotation = pylot.utils.Rotation(pitch, yaw, roll)
            if vehicle_id == ego_vehicle_id:
                # Can compare against canbus output to check that
                # transformations are working.
                self._logger.debug(
                    'Ego vehicle location with ground_obstacles: {}'.format(
                        location))
            else:
                vehicles_list.append(
                    Obstacle(
                        None,  # We currently don't use bounding box
                        1.0,  # confidence
                        'vehicle',
                        vehicle_id,
                        pylot.utils.Transform(location, rotation)))
        return vehicles_list

    def parse_people(self, people):
        # Similar to vehicles, each entry of people is a dictionary that
        # contains four items, the person's id, position, orientation,
        # and bounding box.
        people_list = []
        for person_dict in people.values():
            person_id = person_dict['id']
            location = pylot.utils.Location.from_gps(*person_dict['position'])
            roll, pitch, yaw = person_dict['orientation']
            rotation = pylot.utils.Rotation(pitch, yaw, roll)
            people_list.append(
                Obstacle(
                    None,  # bounding box
                    1.0,  # confidence
                    'person',
                    person_id,
                    pylot.utils.Transform(location, rotation)))
        return people_list

    def parse_traffic_lights(self, traffic_lights):
        # Each entry of traffic lights is a dictionary that contains four
        # items, the id, state, position, and trigger volume of the traffic
        # light.
        # WARNING: Some of the methods in the TrafficLight class may not work
        # here  (e.g. methods that depend knowing the town we are in).
        traffic_lights_list = []
        traffic_light_labels = {
            0: TrafficLightColor.RED,
            1: TrafficLightColor.YELLOW,
            2: TrafficLightColor.GREEN
        }
        for traffic_light_dict in traffic_lights.values():
            traffic_light_id = traffic_light_dict['id']
            traffic_light_state = traffic_light_labels[
                traffic_light_dict['state']]
            # Trigger volume is currently unused.
            # traffic_light_trigger_volume = \
            #    traffic_light_dict['trigger_volume']
            location = pylot.utils.Location.from_gps(
                *traffic_light_dict['position'])
            traffic_lights_list.append(
                TrafficLight(
                    1.0,  # confidence
                    traffic_light_state,
                    traffic_light_id,
                    pylot.utils.Transform(
                        location, pylot.utils.Rotation())  # No rotation given
                ))
        return traffic_lights_list

    def parse_stop_signs(self, stop_signs):
        # Each stop sign has an id, position, and trigger volume.
        stop_signs_list = []
        for stop_sign_dict in stop_signs.values():
            stop_sign_id = stop_sign_dict['id']
            location = pylot.utils.Location.from_gps(
                *stop_sign_dict['position'])
            # Trigger volume is currently unused.
            # trigger_volume = stop_sign_dict['trigger_volume']
            stop_signs_list.append(
                StopSign(
                    1.0,  # confidence
                    None,  # bounding box
                    stop_sign_id,
                    pylot.utils.Transform(
                        location, pylot.utils.Rotation())  # No rotation given
                ))
        return stop_signs_list

    def parse_speed_limit_signs(self, speed_limits):
        # Each speed limit sign has an id, position, and speed.
        speed_limit_signs_list = []
        for speed_limit_dict in speed_limits.values():
            speed_limit_id = speed_limit_dict['id']
            location = pylot.utils.Location.from_gps(
                *speed_limit_dict['position'])
            speed_limit = speed_limit_dict['speed']
            speed_limit_signs_list.append(
                SpeedLimitSign(
                    speed_limit,
                    1.0,  # confidence
                    None,  # bounding box
                    speed_limit_id,
                    pylot.utils.Transform(location, pylot.utils.Rotation())))
        return speed_limit_signs_list

    def parse_static_obstacles(self, static_obstacles):
        # Each static obstacle has an id and position.
        static_obstacles_list = []
        for static_obstacle_dict in static_obstacles.values():
            static_obstacle_id = static_obstacle_dict['id']
            location = pylot.utils.Location.from_gps(
                *static_obstacle_dict['position'])
            static_obstacles_list.append(
                Obstacle(
                    None,  # bounding box
                    1.0,  # confidence
                    'static_obstacle',
                    static_obstacle_id,
                    pylot.utils.Transform(location, pylot.utils.Rotation())))
        return static_obstacles_list

    def send_scene_layout(self, data, timestamp):
        # data is a dictionary describing the scene layout. Each key is a
        # waypoint id; the corresponding value is a dictionary with the
        # following attributes:
        #   road_id: ID of the road the waypoint is on. Each road consists of a
        #       list of lanes, each of which has a list of waypoints.
        #   lane_id: ID of the lane the waypoint is on.
        #   position: Location of the waypoint in GPS coordinates.
        #   orientation: Orientation of the waypoint.
        #   left_margin_position: position shifted in the left edge of the
        #       lane, in GPS coordinates.
        #   right_margin_position: position shifted to the right edge of the
        #       lane,in GPS coordinates.
        #   next_waypoint_ids: list of ids of future waypoints in the same
        #       lane.
        #   left_lane_waypoint_id: Id of the waypoint in the lane to the left.
        #       May be -1 if no valid waypoint.
        #   right_lane_waypoint_id: Id of the waypoint in the lane to the
        #       right. May be -1 if no valid waypoint.

        # TODO: Parse this information into a useful format for planning.
        pass

    def send_gnss_data(self, data, timestamp):
        # GPS coordinates for the ego-vehicle.
        # This is not useful, because the ground_objects message already gives
        # us the ego-vehicle's GPS coordinates.
        pass

    def send_pose_msg(self, data, timestamp):
        # The can bus dict contains other fields as well, but we don't use
        # them yet.
        vehicle_transform = pylot.utils.Transform.from_simulator_transform(
            data['transform'])
        forward_speed = data['speed']
        yaw = vehicle_transform.rotation.yaw
        velocity_vector = pylot.utils.Vector3D(forward_speed * np.cos(yaw),
                                               forward_speed * np.sin(yaw), 0)
        self._pose_stream.send(
            erdos.Message(
                timestamp,
                pylot.utils.Pose(vehicle_transform, forward_speed,
                                 velocity_vector)))
        self._logger.debug('{} Ego vehicle location with Pose: {}'.format(
            timestamp, vehicle_transform))
        self._pose_stream.send(erdos.WatermarkMessage(timestamp))

    def send_waypoints_msg(self, timestamp):
        # Send once the global waypoints.
        if self._waypoints is None:
            # Gets global waypoints from the agent.
            self._waypoints = self._global_plan_world_coord
            data = [(pylot.utils.Transform.from_simulator_transform(transform),
                     road_option)
                    for (transform, road_option) in self._waypoints]
            self._global_trajectory_stream.send(erdos.Message(timestamp, data))
            self._global_trajectory_stream.send(
                erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))


def create_data_flow():
    """Creates a data-flow which process input data, and outputs commands.

    Returns:
        A tuple of streams. The first N - 1 streams are input streams on
        which the agent can send input data. The last stream is the control
        stream on which the agent receives control commands.
    """
    pose_stream = erdos.IngestStream()
    open_drive_stream = erdos.IngestStream()
    global_trajectory_stream = erdos.IngestStream()
    # Currently, we do not use the scene layout information.
    # scene_layout_stream = erdos.IngestStream()
    ground_obstacles_stream = erdos.IngestStream()
    traffic_lights_stream = erdos.IngestStream()
    lanes_stream = erdos.IngestStream()
    time_to_decision_loop_stream = erdos.LoopStream()

    # Add waypoint planner.
    waypoints_stream = pylot.component_creator.add_planning(
        None, pose_stream, ground_obstacles_stream, traffic_lights_stream,
        lanes_stream, open_drive_stream, global_trajectory_stream,
        time_to_decision_loop_stream)
    control_stream = pylot.operator_creator.add_pid_control(
        pose_stream, waypoints_stream)
    extract_control_stream = erdos.ExtractStream(control_stream)

    time_to_decision_stream = pylot.operator_creator.add_time_to_decision(
        pose_stream, ground_obstacles_stream)
    time_to_decision_loop_stream.set(time_to_decision_stream)

    return (pose_stream, global_trajectory_stream, ground_obstacles_stream,
            traffic_lights_stream, lanes_stream, open_drive_stream,
            extract_control_stream)


def enable_logging():
    """Overwrites logging config so that loggers can control verbosity.

    This method is required because the challenge evaluator overwrites
    verbosity, which causes Pylot log messages to be discarded.
    """
    import logging
    logging.root.setLevel(logging.NOTSET)
