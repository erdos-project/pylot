"""
Author: Edward Fang
Email: edward.fang@berkeley.edu
"""
from collections import deque

import itertools
import numpy as np

import erdos

from pylot.map.hd_map import HDMap
from pylot.perception.detection.obstacle import BoundingBox3D
from pylot.planning.messages import WaypointsMessage
from pylot.planning.rrt_star.rrt_star_planning.RRTStar.rrt_star_wrapper import apply_rrt_star
from pylot.simulation.utils import get_map
from pylot.utils import Location, Rotation, Transform

STEP_SIZE = 0.5
MAX_ITERATIONS = 2000
DEFAULT_DISTANCE_THRESHOLD = 30  # 30 meters radius around of ego
DEFAULT_NUM_WAYPOINTS = 100  # 100 waypoints to plan for
DEFAULT_OBSTACLE_SIZE = 2  # 2 x 2 meter square
DEFAULT_TARGET_WAYPOINT = 20  # use the 30th waypoint as a target


class RRTStarPlanningOperator(erdos.Operator):
    """RRTStar Planning operator for Carla 0.9.x.

    Args:
        flags: Config flags.
        goal_location: Goal pylot.utils.Location for planner to route to.
    """
    def __init__(self,
                 can_bus_stream,
                 prediction_stream,
                 global_trajectory_stream,
                 open_drive_stream,
                 waypoints_stream,
                 flags,
                 goal_location=None,
                 log_file_name=None,
                 csv_file_name=None):
        can_bus_stream.add_callback(self.on_can_bus_update)
        prediction_stream.add_callback(self.on_prediction_update)
        global_trajectory_stream.add_callback(self.on_global_trajectory)
        open_drive_stream.add_callback(self.on_opendrive_map)
        erdos.add_watermark_callback([can_bus_stream, prediction_stream],
                                     [waypoints_stream], self.on_watermark)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags

        self._vehicle_transform = None
        self._map = None
        self._waypoints = None
        self._prev_waypoints = None
        self._goal_location = goal_location
        self._can_bus_msgs = deque()
        self._prediction_msgs = deque()

    @staticmethod
    def connect(can_bus_stream, prediction_stream, global_trajectory_stream,
                open_drive_stream):
        waypoints_stream = erdos.WriteStream()
        return [waypoints_stream]

    def run(self):
        # Run method is invoked after all operators finished initializing,
        # including the CARLA operator, which reloads the world. Thus, if
        # we get the map here we're sure it is up-to-date.
        if not hasattr(self._flags, 'track'):
            from pylot.map.hd_map import HDMap
            from pylot.simulation.utils import get_map
            self._map = HDMap(
                get_map(self._flags.carla_host, self._flags.carla_port,
                        self._flags.carla_timeout))
            self._logger.info('Planner running in stand-alone mode')

    def on_can_bus_update(self, msg):
        self._logger.debug('@{}: received can bus message'.format(
            msg.timestamp))
        self._can_bus_msgs.append(msg)

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
        can_bus_msg = self._can_bus_msgs.popleft()
        vehicle_transform = can_bus_msg.data.transform
        self._vehicle_transform = vehicle_transform

        # get obstacles
        prediction_msg = self._prediction_msgs.popleft()
        obstacle_list = self._build_obstacle_list(vehicle_transform,
                                                  prediction_msg)

        # update waypoints
        if not self._waypoints:
            self._waypoints = self._map.compute_waypoints(
                vehicle_transform.location, self._goal_location)
            # running in CARLA
            if self._map is not None:
                self._waypoints = self._map.compute_waypoints(
                    vehicle_transform.location, self._goal_location)
            # haven't received waypoints from global trajectory stream
            else:
                self._logger.debug("@{}: Sending target speed 0, haven't"
                                   "received global trajectory"
                                   .format(timestamp))
                head_waypoints = deque([vehicle_transform])
                target_speeds = deque([0])
                waypoints_stream.send(
                    WaypointsMessage(timestamp, head_waypoints, target_speeds))
                return

        # run rrt star
        success, (path_x, path_y) = \
            self._apply_rrt_star(obstacle_list, timestamp)

        speeds = [0]
        if success:
            speeds = [self._flags.target_speed] * len(path_x)
            self._logger.debug("@{}: RRT* Path X: {}".format(
                timestamp,
                path_x.tolist())
            )
            self._logger.debug("@{}: RRT* Path Y: {}".format(
                timestamp,
                path_y.tolist())
            )
            self._logger.debug("@{}: RRT* Speeds: {}".format(
                timestamp,
                [self._flags.target_speed] * len(path_x))
            )

        # construct and send waypoint message
        waypoint_message = self._construct_waypoints(
            timestamp, path_x, path_y, speeds, success
        )
        waypoints_stream.send(waypoint_message)

    def _get_closest_index(self, start):
        min_dist = np.infty
        mindex = 0
        for ind, wp in enumerate(self._waypoints):
            dist = np.linalg.norm([start[0] - wp.location.x,
                                   start[1] - wp.location.y])
            if dist <= min_dist:
                mindex = ind
                min_dist = dist
        return mindex

    def _apply_rrt_star(self, obstacles, timestamp):
        start = [
            self._vehicle_transform.location.x,
            self._vehicle_transform.location.y
        ]

        # find the closest point to current location
        mindex = self._get_closest_index(start)
        end_ind = min(mindex + DEFAULT_TARGET_WAYPOINT, len(self._waypoints) - 1)
        end = [
            self._waypoints[end_ind].location.x,
            self._waypoints[end_ind].location.y
        ]

        # log initial conditions for debugging
        initial_conditions = {
            "start": start,
            "end": end,
            "obstacles": obstacles.tolist(),
            "step_size": STEP_SIZE,
            "max_iterations": MAX_ITERATIONS,
        }
        self._logger.debug("@{}: Initial conditions: {}".format(
            timestamp, initial_conditions))
        return apply_rrt_star(start, end, STEP_SIZE, MAX_ITERATIONS, obstacles)

    def _construct_waypoints(self, timestamp, path_x, path_y, speeds, success):
        """
        Convert the rrt* path into a waypoints message.
        """
        path_transforms = []
        target_speeds = []
        if not success:
            self._logger.debug("@{}: RRT* failed. "
                               "Sending emergency stop.".format(timestamp))
            for wp in itertools.islice(self._prev_waypoints, 0,
                                       DEFAULT_NUM_WAYPOINTS):
                path_transforms.append(wp)
                target_speeds.append(0)
        else:
            self._logger.debug("@{}: RRT* succeeded."
                               .format(timestamp))
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
                    # use 3d bounding boxes if available, otherwise use default
                    if isinstance(prediction.bounding_box, BoundingBox3D):
                        start_location = \
                            prediction.bounding_box.transform.location - \
                            prediction.bounding_box.extent
                        end_location = \
                            prediction.bounding_box.transform.location + \
                            prediction.bounding_box.extent
                        start_transform = global_obstacle.transform_locations(
                            [start_location])
                        end_transform = global_obstacle.transform_locations(
                            [end_location])
                    else:
                        start_transform = [
                            Location(
                                obstacle_origin[0] - DEFAULT_OBSTACLE_SIZE,
                                obstacle_origin[1] - DEFAULT_OBSTACLE_SIZE,
                                0
                            )
                        ]
                        end_transform = [
                            Location(
                                obstacle_origin[0] + DEFAULT_OBSTACLE_SIZE,
                                obstacle_origin[1] + DEFAULT_OBSTACLE_SIZE,
                                0
                            )
                        ]
                    obstacle_list.append([min(start_transform[0].x,
                                              end_transform[0].x),
                                          min(start_transform[0].y,
                                              end_transform[0].y),
                                          max(start_transform[0].x,
                                              end_transform[0].x),
                                          max(start_transform[0].y,
                                              end_transform[0].y)])

        if len(obstacle_list) == 0:
            return np.empty((0, 4))

        return np.array(obstacle_list)
