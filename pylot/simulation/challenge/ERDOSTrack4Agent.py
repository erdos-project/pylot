from absl import flags
import carla
import erdos
import sys
import numpy as np

import pylot.flags
import pylot.operator_creator
import pylot.utils

from srunner.challenge.autoagents.autonomous_agent import AutonomousAgent,\
    Track

FLAGS = flags.FLAGS


class ERDOSTrack4Agent(AutonomousAgent):
    """Agent class that interacts with the CARLA challenge scenario runner.

    Warning:
        The agent is designed to work on track 4 only.
    """
    def __init_attributes(self, path_to_conf_file):
        flags.FLAGS([__file__, '--flagfile={}'.format(path_to_conf_file)])
        self._logger = erdos.utils.setup_logging('erdos_agent',
                                                 FLAGS.log_file_name)
        enable_logging()
        self.track = Track.SCENE_LAYOUT
        # Stores the waypoints we get from the challenge planner.
        self._waypoints = None
        # Stores the open drive string we get when we run in track 3.
        self._open_drive_data = None
        (can_bus_stream, global_trajectory_stream, open_drive_stream,
         control_stream) = erdos.run_async(create_data_flow)
        self._can_bus_stream = can_bus_stream
        self._global_trajectory_stream = global_trajectory_stream
        self._open_drive_stream = open_drive_stream
        self._control_stream = control_stream

    def setup(self, path_to_conf_file):
        """Setup phase code.

        Invoked by the scenario runner.
        """
        self.__init_attributes(path_to_conf_file)

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

        self.send_waypoints_msg(erdos_timestamp)

        for key, val in input_data.items():
            if key == 'ground_objects':
                self.send_ground_objects(val[1], erdos_timestamp)
            elif key == 'scene_layout':
                self.send_scene_layout(val[1], erdos_timestamp)
            elif key == 'can_bus':
                self.send_can_bus_msg(val[1], erdos_timestamp)
            elif key == 'gnss':
                self.send_gnss_data(val[1], erdos_timestamp)
            else:
                self._logger.warning("Sensor {} not used".format(key))

        # Wait until the control is set.
        while True:
            control_msg = self._control_stream.read()
            if not isinstance(control_msg, erdos.WatermarkMessage):
                output_control = carla.VehicleControl()
                output_control.throttle = control_msg.throttle
                output_control.brake = control_msg.brake
                output_control.steer = control_msg.steer
                output_control.reverse = control_msg.reverse
                output_control.hand_brake = control_msg.hand_brake
                output_control.manual_gear_shift = False
        return output_control

    def send_ground_objects(self, data, timestamp):
        # TODO: Finish implementation
        vehicles = data['vehicles']
        for veh_dict in vehicles:
            id = veh_dict['id']
            rotation = pylot.utils.Rotation(*veh_dict['orientation'])
        hero_vehicle = data['hero_vehicle']
        people = data['walkers']
        for person_dict in people:
            id = person_dict['id']
            rotation = pylot.utils.Rotation(*person_dict['orientation'])
        traffic_lights = data['traffic_lights']
        stop_signs = data['stop_signs']
        speed_limits = data['speed_limits']
        static_obstacles = data['static_obstacles']

    def send_scene_layout(self, data, timestamp):
        # TODO: Implement.
        pass

    def send_gnss_data(self, data, timestamp):
        # TODO: Implement.
        pass

    def send_can_bus_msg(self, data, timestamp):
        # The can bus dict contains other fields as well, but we don't use
        # them yet.
        vehicle_transform = pylot.utils.Transform.from_carla_transform(
            data['transform'])
        forward_speed = data['speed']
        yaw = vehicle_transform.rotation.yaw
        velocity_vector = pylot.utils.Vector3D(forward_speed * np.cos(yaw),
                                               forward_speed * np.sin(yaw),
                                               0)
        self._can_bus_stream.send(
            erdos.Message(timestamp,
                          pylot.utils.CanBus(vehicle_transform,
                                             forward_speed, velocity_vector)))
        self._can_bus_stream.send(erdos.WatermarkMessage(timestamp))

    def send_waypoints_msg(self, timestamp):
        # Send once the global waypoints.
        if self._waypoints is None:
            # Gets global waypoints from the agent.
            self._waypoints = self._global_plan_world_coord
            data = [(pylot.utils.Transform.from_carla_transform(transform),
                     road_option)
                    for (transform, road_option) in self._waypoints]
            self._global_trajectory_stream.send(erdos.Message(timestamp, data))
            self._global_trajectory_stream.send(
                erdos.WatermarkMessage(
                    erdos.Timestamp(coordinates=[sys.maxsize])))


def create_data_flow():
    """Creates a data-flow which process input data, and outputs commands.

    Returns:
        A tuple of streams. The first N - 1 streams are input streams on
        which the agent can send input data. The last stream is the control
        stream on which the agent receives control commands.
    """
    can_bus_stream = erdos.IngestStream()
    global_trajectory_stream = erdos.IngestStream()
    open_drive_stream = erdos.IngestStream()
    gnss_stream = erdos.IngestStream()
    scene_layout_stream = erdos.IngestStream()
    # We do not have access to the open drive map. Send top watermark.
    open_drive_stream.send(
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[sys.maxsize])))

    # TODO: Initialize the data-flow operators.
    return (can_bus_stream, global_trajectory_stream, open_drive_stream,
            control_stream)


def enable_logging():
    """Overwrites logging config so that loggers can control verbosity.

    This method is required because the challenge evaluator overwrites
    verbosity, which causes Pylot log messages to be discarded.
    """
    import logging
    logging.root.setLevel(logging.NOTSET)
