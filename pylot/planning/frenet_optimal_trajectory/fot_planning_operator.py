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

DEFAULT_DISTANCE_THRESHOLD = 50  # 20 meters ahead of ego
DEFAULT_NUM_WAYPOINTS = 50  # 50 waypoints to plan for
WAYPOINT_COMPLETION_THRESHOLD = 0.9


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
        # self.__remove_completed_waypoints(vehicle_transform)

        # convert waypoints to frenet coordinates
        wx = []
        wy = []
        for wp in itertools.islice(self._waypoints, 0, DEFAULT_NUM_WAYPOINTS):
            wx.append(wp.location.x)
            wy.append(wp.location.y)
        tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)

        s0, c_speed, c_d, c_d_d, c_d_dd = \
            self._compute_initial_conditions(can_bus_msg, wx, wy, csp)

        self._logger.info("s_0: {}".format(s0))
        self._logger.info("c_speed: {}".format(c_speed))
        self._logger.info("c_d: {}".format(c_d))
        self._logger.info("c_d_d: {}".format(c_d_d))

        # compute frenet optimal trajectory
        target_speed = min(c_speed + 15, self._flags.target_speed)
        path = frenet_optimal_planning(
            csp, s0, c_speed, c_d, c_d_d, c_d_dd, obstacle_list, target_speed)

        if path is None:
            self._logger.info("Path not found. Emergency stop.")
            self._logger.info("Wx: {}".format(wx))
            self._logger.info("Wy: {}".format(wy))
            self._logger.info("Ob: {}".format(obstacle_list.tolist()))
            waypoints_stream.send(WaypointsMessage(timestamp, self._waypoints,
                                                   [0] * len(self._waypoints)))
            return

        # construct waypoint message
        path_transforms = []
        target_speeds = []
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
        waypoints_stream.send(WaypointsMessage(timestamp, waypoints,
                                               target_speeds))
        self._logger.info("FOUND PATH!")
        self._logger.info("Ob: {}".format(obstacle_list.tolist()))
        self._logger.info("Wx: {}".format(wx))
        self._logger.info("Wy: {}".format(wy))

    def _compute_initial_conditions(self, can_bus_msg, wx, wy, csp):
        x = can_bus_msg.data.transform.location.x
        y = can_bus_msg.data.transform.location.y
        vx = can_bus_msg.data.velocity_vector.x
        vy = can_bus_msg.data.velocity_vector.y

        # get distance from car to spline and projection
        line = geom.LineString(zip(wx, wy))
        point = geom.Point(x, y)
        distance = point.distance(line)
        point_on_line = line.interpolate(line.project(point))

        # compute tanget / normal spline vectors
        svec = np.array([wx[1] - wx[0], wy[1] - wy[0]])
        svec = svec / np.linalg.norm(svec)  # unit vector tangent to spline
        tvec = np.array([-svec[1], svec[0]])  # unit vector orthog. to spline

        # compute tangent / normal car vectors
        fvec = np.array([vx, vy])
        fvec = fvec / np.linalg.norm(fvec)  # unit vector tangent to velocity
        bvec = np.array([point_on_line.x - x, point_on_line.y - y])
        bvec = bvec / np.linalg.norm(bvec)  # unit vector between car and spline

        self._logger.info("fvec: {}".format(fvec))
        self._logger.info("tvec: {}".format(tvec))
        self._logger.info("svec: {}".format(svec))
        self._logger.info("bvec: {}".format(bvec))
        self._logger.info("x, y: {}, {}".format(x, y))

        s0 = csp.find_s(x, y, self.s0)  # current course position
        c_speed = can_bus_msg.data.forward_speed  # current speed [m/s]
        c_d = np.sign(np.dot(tvec, bvec)) * distance  # current lateral position [m]
        c_d_d = c_speed * np.dot(tvec, fvec)  # current lateral speed [m\s]
        c_d_dd = 0.0  # current lateral acceleration [m\s]

        self.s0 = s0
        return s0, c_speed, c_d, c_d_d, c_d_dd

    @staticmethod
    def _build_obstacle_list(vehicle_transform, prediction_msg):
        """
        Construct an obstacle map given vehicle_transform.

        Args:
            vehicle_transform: pylot.utils.Transform of vehicle from can_bus
                stream

        Returns:
            an obstacle list of coordinates from the prediction stream.
            only obstacles within DEFAULT_DISTANCE_THRESHOLD of the
            ego vehicle are considered to save computation cost.
        """
        obstacle_list = []
        # look over all predictions
        for prediction in prediction_msg.predictions:
            # use all prediction times as potential obstacles
            for transform in prediction.trajectory:
                _, dist, _ = vehicle_transform.get_vector_magnitude_angle(transform.location)
                if dist < DEFAULT_DISTANCE_THRESHOLD:
                    # add the obstacle origin to the map
                    obstacle_origin = [transform.location.x,
                                       transform.location.y]
                    obstacle_list.append(obstacle_origin)
        return np.array(obstacle_list)

    def _update_global_waypoints(self, vehicle_transform):
        """
        Update the global waypoint route.

        Args:
            vehicle_transform: pylot.utils.Transform of vehicle from can_bus
                stream

        Returns:
            target location
        """
        self._waypoints = self._hd_map.compute_waypoints(
            vehicle_transform.location, self._goal_location)

    # def __remove_completed_waypoints(self, vehicle_transform):
    #     """Removes waypoints that the ego vehicle has already completed.
    #
    #     The method first finds the closest waypoint, removes all waypoints
    #     that are before the closest waypoint, and finally removes the closest
    #     waypoint if the ego vehicle is very close to it (i.e., close to
    #     completion).
    #     """
    #     min_dist = 10000000
    #     min_index = 0
    #     index = 0
    #     for waypoint in self._waypoints:
    #         # XXX(ionel): We only check the first 10 waypoints.
    #         if index > 30:
    #             break
    #         dist = waypoint.location.distance(vehicle_transform.location)
    #         if dist < min_dist:
    #             min_dist = dist
    #             min_index = index
    #
    #     # Remove waypoints that are before the closest waypoint. The ego
    #     # vehicle already completed them.
    #     while min_index > 0:
    #         self._logger.info("POP LEFT: {}".format(min_index))
    #         self._waypoints.popleft()
    #         min_index -= 1
    #
    #     # The closest waypoint is almost complete, remove it.
    #     if min_dist < WAYPOINT_COMPLETION_THRESHOLD:
    #         self._logger.info("POP LEFT: {}".format(min_index))
    #         self._waypoints.popleft()
