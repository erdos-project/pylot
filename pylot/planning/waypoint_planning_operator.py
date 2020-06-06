"""Implements an operator that computes waypoints to a goal location."""

import itertools
from collections import deque

import erdos

import pylot.utils
from pylot.planning.messages import WaypointsMessage
from pylot.planning.utils import remove_completed_waypoints

DEFAULT_TARGET_WAYPOINT = 9  # Use the 10th waypoint for computing speed
RECOMPUTE_WAYPOINT_EVERY_N_WATERMARKS = 5


class WaypointPlanningOperator(erdos.Operator):
    """Computes waypoints the ego vehicle must follow.

    If the operator is running in CARLA challenge mode, then it receives all
    the waypoints from the scenario runner agent (on the global trajectory
    stream). Otherwise, it computes waypoints using the HD Map.

    The planner reduces speed/stops whenever it encounters an obstacle,
    and waits for the obstacle to move. It does not implement an obstacle
    avoidance policy.

    Args:
        pose_stream (:py:class:`erdos.ReadStream`): Stream on which pose
            info is received.
        open_drive_stream (:py:class:`erdos.ReadStream`): Stream on which open
            drive string representations are received. The operator can
            construct HDMaps out of the open drive strings.
        global_trajectory_stream (:py:class:`erdos.ReadStream`): Stream on
            which the scenario runner publishes waypoints.
        waypoints_stream (:py:class:`erdos.WriteStream`): Stream on which the
            operator sends waypoints the ego vehicle must follow.
        flags (absl.flags): Object to be used to access absl flags.
        goal_location (:py:class:`~pylot.utils.Location`): The goal location of
            the ego vehicle.
    """
    def __init__(self,
                 pose_stream,
                 open_drive_stream,
                 global_trajectory_stream,
                 obstacles_stream,
                 traffic_lights_stream,
                 waypoints_stream,
                 flags,
                 goal_location=None):
        pose_stream.add_callback(self.on_pose_update)
        open_drive_stream.add_callback(self.on_opendrive_map)
        global_trajectory_stream.add_callback(self.on_global_trajectory)
        obstacles_stream.add_callback(self.on_obstacles_update)
        traffic_lights_stream.add_callback(self.on_traffic_lights_update)
        erdos.add_watermark_callback(
            [pose_stream, obstacles_stream, traffic_lights_stream],
            [waypoints_stream], self.on_watermark)

        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags
        # We do not know yet the vehicle's location.
        self._vehicle_transform = None
        self._goal_location = goal_location
        self._map = None
        # Deque of waypoints the vehicle must follow. The waypoints are either
        # received on the global trajectory stream when running using the
        # scenario runner, or computed using the Carla global planner when
        # running in stand-alone mode. The waypoints are Pylot transforms.
        self._waypoints = deque()
        self._pose_msgs = deque()
        self._obstacles_msgs = deque()
        self._traffic_light_msgs = deque()

    @staticmethod
    def connect(pose_stream, open_drive_stream, global_trajectory_stream,
                obstacles_stream, traffic_lights_stream):
        waypoints_stream = erdos.WriteStream()
        return [waypoints_stream]

    def run(self):
        # Run method is invoked after all operators finished initializing,
        # including the CARLA operator, which reloads the world. Thus, if
        # we get the map here we're sure it is up-to-date.

        # We're not running in challenge mode if no track flag is present.
        # Thus, we can directly get the map from the simulator.
        if self._flags.track == -1:
            from pylot.map.hd_map import HDMap
            from pylot.simulation.utils import get_map
            self._map = HDMap(
                get_map(self._flags.carla_host, self._flags.carla_port,
                        self._flags.carla_timeout))
            self._logger.info('Planner running in stand-alone mode')
            # Recompute waypoints every RECOMPUTE_WAYPOINT_EVERY_N_WATERMARKS.
            self._recompute_waypoints = True
        else:
            # Do not recompute waypoints upon each run.
            self._recompute_waypoints = False
        self._watermark_cnt = 0

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
            # arrived at destination.
            self._goal_location = self._vehicle_transform.location
        assert self._goal_location, 'Planner does not have a goal'
        self._waypoints = deque()
        for waypoint_option in msg.data:
            self._waypoints.append(waypoint_option[0])

    def on_pose_update(self, msg):
        """Invoked whenever a message is received on the pose stream.

        Args:
            msg (:py:class:`~erdos.message.Message`): Message that contains
                info about the ego vehicle.
        """
        self._logger.debug('@{}: received pose message'.format(msg.timestamp))
        self._pose_msgs.append(msg)

    def on_obstacles_update(self, msg):
        self._logger.debug('@{}: obstacles update'.format(msg.timestamp))
        self._obstacles_msgs.append(msg)

    def on_traffic_lights_update(self, msg):
        self._logger.debug('@{}: traffic lights update'.format(msg.timestamp))
        self._traffic_light_msgs.append(msg)

    @erdos.profile_method()
    def on_watermark(self, timestamp, waypoints_stream):
        self._logger.debug('@{}: received watermark'.format(timestamp))
        self._watermark_cnt += 1
        self._vehicle_transform = self._pose_msgs.popleft().data.transform
        tl_msg = self._traffic_light_msgs.popleft()
        obstacles_msg = self._obstacles_msgs.popleft()

        if (self._recompute_waypoints and self._watermark_cnt %
                RECOMPUTE_WAYPOINT_EVERY_N_WATERMARKS == 0):
            self._waypoints = self._map.compute_waypoints(
                self._vehicle_transform.location, self._goal_location)
        self._waypoints, _ = remove_completed_waypoints(
            self._waypoints, None, self._vehicle_transform.location)
        if not self._waypoints or len(self._waypoints) == 0:
            # If waypoints are empty (e.g., reached destination), set waypoint
            # to current vehicle location.
            self._waypoints = deque([self._vehicle_transform])

        wp_vector, wp_angle = \
            pylot.planning.utils.compute_waypoint_vector_and_angle(
                self._vehicle_transform, self._waypoints,
                DEFAULT_TARGET_WAYPOINT)

        speed_factor, _ = pylot.planning.utils.stop_for_agents(
            self._vehicle_transform.location, wp_angle, wp_vector,
            obstacles_msg.obstacles, tl_msg.obstacles, self._flags,
            self._logger, self._map, timestamp)

        target_speed = speed_factor * self._flags.target_speed
        self._logger.debug('@{}: speed factor: {}, target speed: {}'.format(
            timestamp, speed_factor, target_speed))
        head_waypoints = deque(
            itertools.islice(self._waypoints, 0,
                             self._flags.num_waypoints_ahead))
        target_speeds = deque(
            [target_speed for _ in range(len(head_waypoints))])
        waypoints_stream.send(
            WaypointsMessage(timestamp, head_waypoints, target_speeds))
