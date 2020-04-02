"""
Author: Edward Fang
Email: edward.fang@berkeley.edu
"""
import numpy as np
import itertools
from collections import deque

import erdos

from pylot.planning.frenet_optimal_trajectory.frenet_optimal_trajectory_planner. \
    FrenetOptimalTrajectory.fot_wrapper \
    import compute_initial_conditions, get_fot_frenet_space
from pylot.planning.messages import WaypointsMessage
from pylot.utils import Location, Rotation, Transform

DEFAULT_DISTANCE_THRESHOLD = 30  # 30 meters radius around of ego
DEFAULT_NUM_WAYPOINTS = 100  # 100 waypoints to plan for


class FOTPlanningOperator(erdos.Operator):
    """ Frenet Optimal Trajectory (FOT) Planning operator for Carla 0.9.x.

    This planning operator uses a global route and listens for predictions
    to produce a frenet optimal trajectory plan. Details can be found in
    `~pylot.planning.frenet_optimal_trajectory.frenet_optimal_trajectory.py`.

     Args:
        flags(:absl.flags:): Object to be used to access absl flags
        goal_location(:pylot.utils.Location:): Goal location for route planning
    """
    def __init__(self,
                 pose_stream,
                 prediction_stream,
                 global_trajectory_stream,
                 open_drive_stream,
                 waypoints_stream,
                 flags,
                 goal_location=None,
                 log_file_name=None,
                 csv_file_name=None):
        pose_stream.add_callback(self.on_pose_update)
        prediction_stream.add_callback(self.on_prediction_update)
        global_trajectory_stream.add_callback(self.on_global_trajectory)
        open_drive_stream.add_callback(self.on_opendrive_map)
        erdos.add_watermark_callback([pose_stream, prediction_stream],
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
        self.s0 = 0

    @staticmethod
    def connect(pose_stream, prediction_stream, global_trajectory_stream,
                open_drive_stream):
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

    @erdos.profile_method()
    def on_watermark(self, timestamp, waypoints_stream):
        self._logger.debug('@{}: received watermark'.format(timestamp))

        # get ego info
        pose_msg = self._pose_msgs.popleft()
        vehicle_transform = pose_msg.data.transform
        self._vehicle_transform = vehicle_transform

        # get obstacles
        prediction_msg = self._prediction_msgs.popleft()
        obstacle_list = self._build_obstacle_list(vehicle_transform,
                                                  prediction_msg)
        # update waypoints
        if not self._waypoints:
            # running in CARLA
            if self._map is not None:
                self._waypoints = self._map.compute_waypoints(
                    vehicle_transform.location, self._goal_location)
            # haven't received waypoints from global trajectory stream
            else:
                self._logger.debug(
                    "@{}: Sending target speed 0, haven't"
                    "received global trajectory".format(timestamp))
                head_waypoints = deque([vehicle_transform])
                target_speeds = deque([0])
                waypoints_stream.send(
                    WaypointsMessage(timestamp, head_waypoints, target_speeds))
                return

        # compute optimal frenet trajectory
        path_x, path_y, speeds, params, success, s0 = \
            self._compute_optimal_frenet_trajectory(pose_msg, obstacle_list)

        if success:
            self._logger.debug("@{}: Frenet Path X: {}".format(
                timestamp, path_x.tolist()))
            self._logger.debug("@{}: Frenet Path Y: {}".format(
                timestamp, path_y.tolist()))
            self._logger.debug("@{}: Frenet Speeds: {}".format(
                timestamp, speeds.tolist()))

        # construct and send waypoint message
        waypoints_message = self._construct_waypoints(timestamp, path_x,
                                                      path_y, speeds, success)
        waypoints_stream.send(waypoints_message)

    def _compute_optimal_frenet_trajectory(self, pose_msg, obstacle_list):
        """
        Compute the optimal frenet trajectory, given current environment info.
        """
        # convert waypoints to frenet coordinates
        wx = []
        wy = []
        for wp in itertools.islice(self._waypoints, 0, DEFAULT_NUM_WAYPOINTS):
            wx.append(wp.location.x)
            wy.append(wp.location.y)
        wx = np.array(wx)
        wy = np.array(wy)

        # compute frenet optimal trajectory
        s0, c_speed, c_d, c_d_d, c_d_dd = \
            self._compute_initial_conditions(pose_msg, wx, wy)
        self.s0 = s0
        target_speed = (c_speed + self._flags.target_speed) / 2
        path_x, path_y, speeds, params, success = get_fot_frenet_space(
            s0, c_speed, c_d, c_d_d, c_d_dd, wx, wy, obstacle_list,
            target_speed)

        # log initial conditions for debugging
        initial_conditions = {
            "s0": s0,
            "c_speed": c_speed,
            "c_d": c_d,
            "c_d_d": c_d_d,
            "c_d_dd": c_d_dd,
            "wx": wx.tolist(),
            "wy": wy.tolist(),
            "obstacle_list": obstacle_list.tolist(),
            "x": pose_msg.data.transform.location.x,
            "y": pose_msg.data.transform.location.y,
            "vx": pose_msg.data.velocity_vector.x,
            "vy": pose_msg.data.velocity_vector.y,
        }
        timestamp = pose_msg.timestamp
        self._logger.debug("@{}: Initial conditions: {}".format(
            timestamp, initial_conditions))

        return path_x, path_y, speeds, params, success, s0

    def _construct_waypoints(self, timestamp, path_x, path_y, speeds, success):
        """
        Convert the optimal frenet path into a waypoints message.
        """
        path_transforms = []
        target_speeds = []
        if not success:
            self._logger.debug("@{}: Frenet Optimal Trajectory failed. "
                               "Sending emergency stop.".format(timestamp))
            for wp in itertools.islice(self._prev_waypoints, 0,
                                       DEFAULT_NUM_WAYPOINTS):
                path_transforms.append(wp)
                target_speeds.append(0)
        else:
            self._logger.debug(
                "@{}: Frenet Optimal Trajectory succeeded.".format(timestamp))
            for point in zip(path_x, path_y, speeds):
                if self._map is not None:
                    p_loc = self._map.get_closest_lane_waypoint(
                        Location(x=point[0], y=point[1], z=0)).location
                else:
                    p_loc = Location(x=point[0], y=point[1], z=0)
                path_transforms.append(
                    Transform(
                        location=Location(x=point[0], y=point[1], z=p_loc.z),
                        rotation=Rotation(),
                    ))
                target_speeds.append(point[2])

        waypoints = deque(path_transforms)
        self._prev_waypoints = waypoints
        return WaypointsMessage(timestamp, waypoints, target_speeds)

    def _compute_initial_conditions(self, pose_msg, wx, wy):
        """
        Convert the initial conditions of vehicle into frenet frame parameters.
        """
        x = pose_msg.data.transform.location.x
        y = pose_msg.data.transform.location.y
        vx = pose_msg.data.velocity_vector.x
        vy = pose_msg.data.velocity_vector.y
        return compute_initial_conditions(self.s0, x, y, vx, vy,
                                          pose_msg.data.forward_speed, wx, wy)

    @staticmethod
    def _build_obstacle_list(vehicle_transform, prediction_msg):
        """
        Construct an obstacle list of proximal objects given vehicle_transform.
        """
        obstacle_list = []
        # look over all predictions
        for prediction in prediction_msg.predictions:
            # use all prediction times as potential obstacles
            for transform in prediction.trajectory:
                global_obstacle = vehicle_transform * transform
                obstacle_origin = [
                    global_obstacle.location.x, global_obstacle.location.y
                ]
                dist_to_ego = np.linalg.norm([
                    vehicle_transform.location.x - obstacle_origin[0],
                    vehicle_transform.location.y - obstacle_origin[1]
                ])
                # TODO (@fangedward): Fix this hack
                # Prediction also sends a prediction for ego vehicle
                # This will always be the closest to the ego vehicle
                # Filter out until this is removed from prediction
                if dist_to_ego < 2:  # this allows max vel to be 20m/s
                    break
                elif dist_to_ego < DEFAULT_DISTANCE_THRESHOLD:
                    obstacle_list.append(obstacle_origin)

        if len(obstacle_list) == 0:
            return np.empty((0, 2))
        return np.array(obstacle_list)
