"""
Author: Edward Fang
Email: edward.fang@berkeley.edu
"""
import numpy as np
import shapely.geometry as geom
import itertools
from collections import deque

import erdos

from pylot.map.hd_map import HDMap
from pylot.planning.frenet_optimal_trajectory.frenet_optimal_trajectory import \
    generate_target_course, frenet_optimal_planning
from pylot.planning.messages import WaypointsMessage
from pylot.simulation.utils import get_map
from pylot.utils import Location, Transform, Rotation

DEFAULT_DISTANCE_THRESHOLD = 30  # 30 meters radius around of ego
DEFAULT_NUM_WAYPOINTS = 100  # 100 waypoints to plan for


class FOTPlanningOperator(erdos.Operator):
    """ Frenet Optimal Trajectory (FOT) Planning operator for Carla 0.9.x."""
    def __init__(self,
                 can_bus_stream,
                 prediction_stream,
                 waypoints_stream,
                 name,
                 flags,
                 goal_location,
                 log_file_name=None,
                 csv_file_name=None):
        """
        Initialize the FOT planner. Setup logger and map attributes.

        Args:
            name: Name of the operator.
            flags: Config flags.
            goal_location: Goal pylot.utils.Location for planner to route to.
        """
        can_bus_stream.add_callback(self.on_can_bus_update)
        prediction_stream.add_callback(self.on_prediction_update)
        erdos.add_watermark_callback([can_bus_stream, prediction_stream],
                                     [waypoints_stream], self.on_watermark)
        self._name = name
        self._log_file_name = log_file_name
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            name + '-csv', csv_file_name)
        self._flags = flags

        self._waypoints = None
        self._goal_location = goal_location

        self._can_bus_msgs = deque()
        self._prediction_msgs = deque()
        self.s0 = 0

    @staticmethod
    def connect(can_bus_stream, prediction_stream):
        waypoints_stream = erdos.WriteStream()
        return [waypoints_stream]

    def run(self):
        # Run method is invoked after all operators finished initializing,
        # including the CARLA operator, which reloads the world. Thus, if
        # we get the map here we're sure it is up-to-date.
        self._hd_map = HDMap(
            get_map(self._flags.carla_host, self._flags.carla_port,
                    self._flags.carla_timeout), self._log_file_name)

    def on_can_bus_update(self, msg):
        self._logger.debug('@{}: received can bus message'.format(
            msg.timestamp))
        self._can_bus_msgs.append(msg)

    def on_prediction_update(self, msg):
        self._logger.debug('@{}: received prediction message'.format(
            msg.timestamp))
        self._prediction_msgs.append(msg)

    def on_watermark(self, timestamp, waypoints_stream):
        self._logger.debug('@{}: received watermark'.format(timestamp))

        # get ego info
        can_bus_msg = self._can_bus_msgs.popleft()
        vehicle_transform = can_bus_msg.data.transform

        # get obstacles
        prediction_msg = self._prediction_msgs.popleft()
        obstacle_list = self._build_obstacle_list(vehicle_transform,
                                                  prediction_msg)
        # update waypoints
        if not self._waypoints:
            self._update_global_waypoints(vehicle_transform)

        # compute optimal frenet trajectory
        path, csp, s0, initial_conditions = \
            self._compute_optimal_frenet_trajectory(can_bus_msg, obstacle_list)

        # send debug info
        self._logger.debug("@{}: Initial conditions: {}"
                           .format(timestamp, initial_conditions))
        if path:
            self._logger.debug("@{}: Frenet Path X: {}".format(timestamp, path.x))
            self._logger.debug("@{}: Frenet Path Y: {}".format(timestamp, path.y))
            self._logger.debug("@{}: Frenet Path V: {}".format(timestamp, path.s_d))

        # construct and send waypoint message
        waypoints_message = self._construct_waypoints(timestamp, path, csp, s0)
        waypoints_stream.send(waypoints_message)

    def _compute_optimal_frenet_trajectory(self, can_bus_msg, obstacle_list):
        """
        Compute the optimal frenet trajectory, given current environment info.
        """
        # convert waypoints to frenet coordinates
        wx = []
        wy = []
        for wp in itertools.islice(self._waypoints, 0, DEFAULT_NUM_WAYPOINTS):
            wx.append(wp.location.x)
            wy.append(wp.location.y)
        tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)

        # compute frenet optimal trajectory
        s0, c_speed, c_d, c_d_d, c_d_dd = \
            self._compute_initial_conditions(can_bus_msg, wx, wy, csp)
        target_speed = (c_speed + self._flags.target_speed) / 2
        path = frenet_optimal_planning(
            csp, s0, c_speed, c_d, c_d_d, c_d_dd, obstacle_list, target_speed)

        initial_conditions = {
            "s0": s0,
            "c_speed": c_speed,
            "c_d": c_d,
            "c_d_d": c_d_d,
            "c_d_dd": c_d_dd,
            "wx": wx,
            "wy": wy,
            "obstacle_list": obstacle_list.tolist(),
            "x": can_bus_msg.data.transform.location.x,
            "y": can_bus_msg.data.transform.location.y,
            "vx": can_bus_msg.data.velocity_vector.x,
            "vy": can_bus_msg.data.velocity_vector.y,
        }
        return path, csp, s0, initial_conditions

    def _construct_waypoints(self, timestamp, path, csp, s0):
        """
        Convert the optimal frenet path into a waypoints message.
        """
        path_transforms = []
        target_speeds = []
        if path is None:
            self._logger.debug("@{}: Frenet Optimal Trajectory failed. "
                               "Sending emergency stop.".format(timestamp))
            for s in np.arange(s0, csp.s[-1], 1):
                x, y = csp.calc_position(s)
                p_loc = self._hd_map.get_closest_lane_waypoint(
                    Location(x=x, y=y, z=0)).location
                path_transforms.append(
                    Transform(
                        location=Location(x=x, y=y, z=p_loc.z),
                        rotation=Rotation(),
                    ))
                target_speeds.append(0)
        else:
            self._logger.debug("@{}: Frenet Optimal Trajectory succeeded."
                               .format(timestamp))
            for point in zip(path.x, path.y, path.s_d):
                p_loc = self._hd_map.get_closest_lane_waypoint(
                    Location(x=point[0], y=point[1], z=0)).location
                path_transforms.append(
                    Transform(
                        location=Location(x=point[0], y=point[1], z=p_loc.z),
                        rotation=Rotation(),
                    ))
                target_speeds.append(point[2])

        waypoints = deque(path_transforms)
        return WaypointsMessage(timestamp, waypoints, target_speeds)

    def _compute_initial_conditions(self, can_bus_msg, wx, wy, csp):
        """
        Convert the initial conditions of vehicle into frenet frame parameters.
        """
        x = can_bus_msg.data.transform.location.x
        y = can_bus_msg.data.transform.location.y
        vx = can_bus_msg.data.velocity_vector.x
        vy = can_bus_msg.data.velocity_vector.y

        # get distance from car to spline and projection
        line = geom.LineString(zip(wx, wy))
        point = geom.Point(x, y)
        distance = point.distance(line)
        point_on_line = line.interpolate(line.project(point))

        # compute tangent / normal spline vectors
        x0, y0 = csp.calc_position(self.s0)
        x1, y1 = csp.calc_position(self.s0 + 2)
        svec = np.array([x1-x0, y1-y0])
        svec = svec / np.linalg.norm(svec)  # unit vector tangent to spline
        tvec = np.array([svec[1], -svec[0]])  # unit vector orthog. to spline

        # compute tangent / normal car vectors
        fvec = np.array([vx, vy])
        fvec = fvec / np.linalg.norm(fvec)  # unit vector tangent to velocity
        bvec = np.array([point_on_line.x - x, point_on_line.y - y])
        bvec = bvec / np.linalg.norm(bvec)  # unit vector between car and spline

        # get initial conditions in frenet frame
        s0 = csp.find_s(x, y, self.s0)  # cur course position
        c_speed = can_bus_msg.data.forward_speed  # cur speed [m/s]
        c_d = np.sign(np.dot(tvec, bvec)) * distance  # cur lateral position [m]
        c_d_d = c_speed * np.dot(tvec, fvec)  # cur lateral speed [m\s]
        c_d_dd = 0.0  # cur lateral acceleration [m\s]
        # TODO (@fangedward) add IMU for lat. acc. when 0.9.7 is fixed

        self.s0 = s0
        return s0, c_speed, c_d, c_d_d, c_d_dd

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
                obstacle_origin = [global_obstacle.location.x,
                                   global_obstacle.location.y]
                dist_to_ego = np.linalg.norm([vehicle_transform.location.x -
                                              obstacle_origin[0],
                                              vehicle_transform.location.y -
                                              obstacle_origin[1]])
                # TODO (@fangedward): Fix this hack
                # Prediction also sends a prediction for ego vehicle
                # This will always be the closest to the ego vehicle
                # Filter out until this is removed from prediction
                if dist_to_ego < 2:  # this allows max vel to be 20m/s
                    break
                elif dist_to_ego < DEFAULT_DISTANCE_THRESHOLD:
                    obstacle_list.append(obstacle_origin)
        return np.array(obstacle_list)

    def _update_global_waypoints(self, vehicle_transform):
        """
        Update the global waypoint route.
        """
        self._waypoints = self._hd_map.compute_waypoints(
            vehicle_transform.location, self._goal_location)
