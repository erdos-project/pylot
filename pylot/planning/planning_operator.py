"""
Author: Edward Fang
Email: edward.fang@berkeley.edu
"""
from collections import deque

import erdos

import numpy as np

from pylot.perception.detection.obstacle import BoundingBox3D
from pylot.planning.waypoints import Waypoints
from pylot.utils import Location, Rotation, Transform


class PlanningOperator(erdos.Operator):
    """Base Planning Operator.

    If the operator is running in CARLA challenge mode, then it receives all
    the waypoints from the scenario runner agent (on the global trajectory
    stream). Otherwise, it computes waypoints using the HD Map.

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
                 prediction_stream,
                 traffic_lights_stream,
                 global_trajectory_stream,
                 open_drive_stream,
                 time_to_decision_stream,
                 waypoints_stream,
                 flags,
                 goal_location=None):
        pose_stream.add_callback(self.on_pose_update)
        prediction_stream.add_callback(self.on_prediction_update)
        traffic_lights_stream.add_callback(self.on_traffic_lights_update)
        global_trajectory_stream.add_callback(self.on_global_trajectory)
        open_drive_stream.add_callback(self.on_opendrive_map)
        time_to_decision_stream.add_callback(self.on_time_to_decision)
        erdos.add_watermark_callback([
            pose_stream, prediction_stream, traffic_lights_stream,
            time_to_decision_stream
        ], [waypoints_stream], self.on_watermark)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags
        # We do not know yet the vehicle's location.
        self._ego_transform = None
        self._map = None
        # Waypoints the vehicle must follow. The waypoints are either
        # received on the global trajectory stream when running using the
        # scenario runner, or computed using the Carla global planner when
        # running in stand-alone mode.
        self._waypoints = None
        self._goal_location = goal_location

        self._pose_msgs = deque()
        self._prediction_msgs = deque()
        self._traffic_light_msgs = deque()
        self._ttd_msgs = deque()

    @staticmethod
    def connect(pose_stream, prediction_stream, traffic_lights_stream,
                global_trajectory_stream, open_drive_stream,
                time_to_decision_stream):
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
            # Recompute waypoints every RECOMPUTE_WAYPOINT_EVERY_N_WATERMARKS.
            self._recompute_waypoints = True
        else:
            # Do not recompute waypoints upon each run.
            self._recompute_waypoints = False
        self._watermark_cnt = 0

    def on_pose_update(self, msg):
        """Invoked whenever a message is received on the pose stream.

        Args:
            msg (:py:class:`~erdos.message.Message`): Message that contains
                info about the ego vehicle.
        """
        self._logger.debug('@{}: received pose message'.format(msg.timestamp))
        self._pose_msgs.append(msg)
        self._ego_transform = msg.data.transform

    @erdos.profile_method()
    def on_prediction_update(self, msg):
        self._logger.debug('@{}: received prediction message'.format(
            msg.timestamp))
        self._prediction_msgs.append(msg)

    def on_traffic_lights_update(self, msg):
        self._logger.debug('@{}: received traffic lights update'.format(
            msg.timestamp))
        self._traffic_light_msgs.append(msg)

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
            waypoints = deque()
            for waypoint_option in msg.data:
                waypoints.append(waypoint_option[0])
            self._waypoints = Waypoints(waypoints)
        else:
            # Trajectory does not contain any waypoints. We assume we have
            # arrived at destionation.
            self._goal_location = self._ego_transform.location
            self._waypoints = Waypoints(deque(), deque())
        assert self._goal_location, 'Planner does not have a goal'

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
    def on_time_to_decision(self, msg):
        self._logger.debug('@{}: {} received ttd update {}'.format(
            msg.timestamp, self.config.name, msg))
        self._ttd_msgs.append(msg)

    def build_obstacle_list(self, ego_transform, prediction_msg):
        """
        Construct an obstacle list of proximal objects given ego_transform.
        """
        obstacle_list = []
        for prediction in prediction_msg.predictions:
            # Use all prediction times as potential obstacles.
            previous_origin = None
            for transform in prediction.trajectory:
                global_obstacle = ego_transform * transform
                obstacle_origin = global_obstacle.location.as_numpy_array_2D()
                # distance filtering
                if (previous_origin is None
                        or np.linalg.norm(previous_origin - obstacle_origin) >
                        self._flags.obstacle_filtering_distance):
                    previous_origin = obstacle_origin
                    dist_to_ego = np.linalg.norm([
                        ego_transform.location.x - obstacle_origin[0],
                        ego_transform.location.y - obstacle_origin[1]
                    ])
                    if dist_to_ego < self._flags.distance_threshold:
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
                                    obstacle_origin[0] -
                                    self._flags.obstacle_radius,
                                    obstacle_origin[1] -
                                    self._flags.obstacle_radius, 0)
                            ]
                            end_transform = [
                                Location(
                                    obstacle_origin[0] +
                                    self._flags.obstacle_radius,
                                    obstacle_origin[1] +
                                    self._flags.obstacle_radius, 0)
                            ]
                        obstacle_list.append([
                            min(start_transform[0].x, end_transform[0].x),
                            min(start_transform[0].y, end_transform[0].y),
                            max(start_transform[0].x, end_transform[0].x),
                            max(start_transform[0].y, end_transform[0].y)
                        ])
        if len(obstacle_list) == 0:
            return np.empty((0, 4))

        return np.array(obstacle_list)

    def build_output_waypoints(self, path_x, path_y, speeds):
        wps = deque()
        target_speeds = deque()
        for point in zip(path_x, path_y, speeds):
            if self._map is not None:
                p_loc = self._map.get_closest_lane_waypoint(
                    Location(x=point[0], y=point[1], z=0)).location
            else:
                p_loc = Location(x=point[0], y=point[1], z=0)
            wps.append(
                Transform(
                    location=Location(x=point[0], y=point[1], z=p_loc.z),
                    rotation=Rotation(),
                ))
            target_speeds.append(point[2])
        return Waypoints(wps, target_speeds)

    @erdos.profile_method()
    def on_watermark(self, timestamp, waypoints_stream):
        raise NotImplementedError
