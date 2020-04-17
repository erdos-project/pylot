"""
Author: Edward Fang
Email: edward.fang@berkeley.edu
"""
from collections import deque

import numpy as np

import erdos


class PlanningOperator(erdos.Operator):
    """Base Planning Operator for Carla 0.9.x.

    Args:
        flags: Config flags.
        goal_location: Goal pylot.utils.Location for planner to route to.
    """
    def __init__(self,
                 pose_stream,
                 prediction_stream,
                 global_trajectory_stream,
                 open_drive_stream,
                 time_to_decision_stream,
                 waypoints_stream,
                 flags,
                 goal_location=None,
                 log_file_name=None,
                 csv_file_name=None):
        pose_stream.add_callback(self.on_pose_update)
        prediction_stream.add_callback(self.on_prediction_update)
        global_trajectory_stream.add_callback(self.on_global_trajectory)
        open_drive_stream.add_callback(self.on_opendrive_map)
        time_to_decision_stream.add_callback(self.on_time_to_decision)
        erdos.add_watermark_callback(
            [pose_stream, prediction_stream, time_to_decision_stream],
            [waypoints_stream], self.on_watermark)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags
        self._vehicle_transform = None
        self._map = None
        self._waypoints = None
        self._prev_waypoints = None
        self._goal_location = goal_location

        self._pose_msgs = deque()
        self._prediction_msgs = deque()

    @staticmethod
    def connect(pose_stream, prediction_stream, global_trajectory_stream,
                open_drive_stream, time_to_decision_stream):
        waypoints_stream = erdos.WriteStream()
        return [waypoints_stream]

    def run(self):
        # Run method is invoked after all operators finished initializing,
        # including the CARLA operator, which reloads the world. Thus, if
        # we get the map here we're sure it is up-to-date.
        if self._flags.track == -1:
            from pylot.map.hd_map import HDMap
            from pylot.simulation.utils import get_map
            self._map = HDMap(
                get_map(self._flags.carla_host, self._flags.carla_port,
                        self._flags.carla_timeout))
            self._logger.info('Planner running in stand-alone mode')

    def on_pose_update(self, msg):
        self._logger.debug('@{}: received pose message'.format(msg.timestamp))
        self._pose_msgs.append(msg)

    def on_prediction_update(self, msg):
        self._logger.debug('@{}: received prediction message'.format(
            msg.timestamp))
        self._prediction_msgs.append(msg)

    def on_global_trajectory(self, msg):
        """Invoked whenever a message is received on the trajectory stream.
        Args:
            msg (:py:class:`~erdos.message.Message`): Message that contains
                a list of waypoints to the goal location.
        """
        self._logger.debug('@{}: global trajectory has {} waypoints'.format(
            msg.timestamp, len(msg.data)))
        if len(msg.data) > 0:
            # The last waypoint is the goal location.
            self._goal_location = msg.data[-1][0].location
        else:
            # Trajectory does not contain any waypoints. We assume we have
            # arrived at destionation.
            self._goal_location = self._vehicle_transform.location
        assert self._goal_location, 'Planner does not have a goal'
        self._waypoints = deque()
        for waypoint_option in msg.data:
            self._waypoints.append(waypoint_option[0])
        self._prev_waypoints = self._waypoints

    def on_opendrive_map(self, msg):
        """Invoked whenever a message is received on the open drive stream.
        Args:
            msg (:py:class:`~erdos.message.Message`): Message that contains
                the open drive string.
        """
        self._logger.debug('@{}: received open drive message'.format(
            msg.timestamp))
        try:
            import carla
        except ImportError:
            raise Exception('Error importing carla.')
        self._logger.info('Initializing HDMap from open drive stream')
        from pylot.map.hd_map import HDMap
        self._map = HDMap(carla.Map('map', msg.data))

    def on_time_to_decision(self, msg):
        self._logger.debug('@{}: {} received ttd update {}'.format(
            msg.timestamp, self.config.name, msg))

    @erdos.profile_method()
    def on_watermark(self, timestamp, waypoints_stream):
        raise NotImplementedError
