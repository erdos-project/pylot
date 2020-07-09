"""
Author: Edward Fang
Email: edward.fang@berkeley.edu
"""
from collections import deque

import erdos

from pylot.perception.messages import ObstaclesMessage
from pylot.perception.tracking.obstacle_trajectory import ObstacleTrajectory
from pylot.planning.messages import WaypointsMessage
from pylot.planning.utils import BehaviorPlannerState
from pylot.planning.world import World
from pylot.prediction.messages import PredictionMessage
from pylot.prediction.obstacle_prediction import ObstaclePrediction


class PlanningOperator(erdos.Operator):
    """Base Planning Operator.

    If the operator is running in CARLA challenge mode, then it receives all
    the waypoints from the scenario runner agent (on the global trajectory
    stream). Otherwise, it computes waypoints using the HD Map.

    Args:
        pose_stream (:py:class:`erdos.ReadStream`): Stream on which pose
            info is received.
        prediction_stream (:py:class:`erdos.ReadStream`): Stream on which
            trajectory predictions of dynamic obstacles is received.
        static_obstacles_stream (:py:class:`erdos.ReadStream`): Stream on
            which static obstacles (e.g., traffic lights) are received.
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
                 static_obstacles_stream,
                 lanes_stream,
                 global_trajectory_stream,
                 open_drive_stream,
                 time_to_decision_stream,
                 waypoints_stream,
                 flags,
                 goal_location=None):
        pose_stream.add_callback(self.on_pose_update)
        prediction_stream.add_callback(self.on_prediction_update)
        static_obstacles_stream.add_callback(self.on_static_obstacles_update)
        lanes_stream.add_callback(self.on_lanes_update)
        global_trajectory_stream.add_callback(self.on_global_trajectory)
        open_drive_stream.add_callback(self.on_opendrive_map)
        time_to_decision_stream.add_callback(self.on_time_to_decision)
        erdos.add_watermark_callback([
            pose_stream, prediction_stream, static_obstacles_stream,
            lanes_stream, time_to_decision_stream, global_trajectory_stream
        ], [waypoints_stream], self.on_watermark)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags
        # We do not know yet the vehicle's location.
        self._ego_transform = None
        self._map = None
        self._world = World(flags, self._logger)
        # Waypoints the vehicle must follow. The waypoints are either
        # received on the global trajectory stream when running using the
        # scenario runner, or computed using the Carla global planner when
        # running in stand-alone mode.
        self._world.update_waypoints(goal_location, None)
        if self._flags.planning_type == 'waypoint':
            # Use the FOT planner for overtaking.
            from pylot.planning.frenet_optimal_trajectory.fot_planner \
                import FOTPlanner
            self._planner = FOTPlanner(self._world, self._flags, self._logger)
        elif self._flags.planning_type == 'frenet_optimal_trajectory':
            from pylot.planning.frenet_optimal_trajectory.fot_planner \
                import FOTPlanner
            self._planner = FOTPlanner(self._world, self._flags, self._logger)
        elif self._flags.planning_type == 'hybrid_astar':
            from pylot.planning.hybrid_astar.hybrid_astar_planner \
                import HybridAStarPlanner
            self._planner = HybridAStarPlanner(self._world, self._flags,
                                               self._logger)
        elif self._flags.planning_type == 'rrt_star':
            from pylot.planning.rrt_star.rrt_star_planner import RRTStarPlanner
            self._planner = RRTStarPlanner(self._world, self._flags,
                                           self._logger)
        else:
            raise ValueError('Unexpected planning type: {}'.format(
                self._flags.planning_type))
        self._state = BehaviorPlannerState.FOLLOW_WAYPOINTS
        self._pose_msgs = deque()
        self._prediction_msgs = deque()
        self._static_obstacles_msgs = deque()
        self._lanes_msgs = deque()
        self._ttd_msgs = deque()

    @staticmethod
    def connect(pose_stream, prediction_stream, static_obstacles_stream,
                lanes_steam, global_trajectory_stream, open_drive_stream,
                time_to_decision_stream):
        waypoints_stream = erdos.WriteStream()
        return [waypoints_stream]

    def run(self):
        # Run method is invoked after all operators finished initializing,
        # including the CARLA operator, which reloads the world. Thus, if
        # we get the map here we're sure it is up-to-date.
        if self._flags.execution_mode == 'simulation':
            from pylot.map.hd_map import HDMap
            from pylot.simulation.utils import get_map
            self._map = HDMap(
                get_map(self._flags.carla_host, self._flags.carla_port,
                        self._flags.carla_timeout), self.config.log_file_name)
            self._logger.info('Planner running in stand-alone mode')

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

    def on_static_obstacles_update(self, msg):
        self._logger.debug('@{}: received static obstacles update'.format(
            msg.timestamp))
        self._static_obstacles_msgs.append(msg)

    def on_lanes_update(self, msg):
        self._logger.debug('@{}: received lanes update'.format(msg.timestamp))
        self._lanes_msgs.append(msg)

    def on_global_trajectory(self, msg):
        """Invoked whenever a message is received on the trajectory stream.

        Args:
            msg (:py:class:`~erdos.message.Message`): Message that contains
                a list of waypoints to the goal location.
        """
        self._logger.debug('@{}: global trajectory has {} waypoints'.format(
            msg.timestamp, len(msg.waypoints.waypoints)))
        if msg.agent_state:
            self._logger.debug('@{}: updating planner state to {}'.format(
                msg.timestamp, msg.agent_state))
            self._state = msg.agent_state
        # The last waypoint is the goal location.
        self._world.update_waypoints(msg.waypoints.waypoints[-1].location,
                                     msg.waypoints)

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

    @erdos.profile_method()
    def on_watermark(self, timestamp, waypoints_stream):
        self._logger.debug('@{}: received watermark'.format(timestamp))
        self.update_world(timestamp)
        if self._state == BehaviorPlannerState.OVERTAKE:
            # Ignore traffic lights and obstacle.
            output_wps = self._planner.run(timestamp)
        else:
            (speed_factor, _, _, speed_factor_tl,
             speed_factor_stop) = self._world.stop_for_agents(timestamp)
            if self._flags.planning_type == 'waypoint':
                target_speed = speed_factor * self._flags.target_speed
                self._logger.debug(
                    '@{}: speed factor: {}, target speed: {}'.format(
                        timestamp, speed_factor, target_speed))
                output_wps = self._world.follow_waypoints(target_speed)
            else:
                output_wps = self._planner.run(timestamp)
                speed_factor = min(speed_factor_stop, speed_factor_tl)
                self._logger.debug('@{}: speed factor: {}'.format(
                    timestamp, speed_factor))
                output_wps.apply_speed_factor(speed_factor)
        waypoints_stream.send(WaypointsMessage(timestamp, output_wps))
        waypoints_stream.send(erdos.WatermarkMessage(timestamp))

    def get_predictions(self, prediction_msg, ego_transform):
        predictions = []
        if isinstance(prediction_msg, ObstaclesMessage):
            # Transform the obstacle into a prediction.
            predictions = []
            for obstacle in prediction_msg.obstacles:
                obstacle_trajectory = ObstacleTrajectory(obstacle, [])
                prediction = ObstaclePrediction(
                    obstacle_trajectory, obstacle.transform, 1.0,
                    [ego_transform.inverse_transform() * obstacle.transform])
                predictions.append(prediction)
        elif isinstance(prediction_msg, PredictionMessage):
            predictions = prediction_msg.predictions
        else:
            raise ValueError('Unexpected obstacles msg type {}'.format(
                type(prediction_msg)))
        return predictions

    def update_world(self, timestamp):
        pose_msg = self._pose_msgs.popleft()
        ego_transform = pose_msg.data.transform
        prediction_msg = self._prediction_msgs.popleft()
        predictions = self.get_predictions(prediction_msg, ego_transform)
        static_obstacles_msg = self._static_obstacles_msgs.popleft()
        if len(self._lanes_msgs) > 0:
            lanes = self._lanes_msgs.popleft().data
        else:
            lanes = None

        # Update the representation of the world.
        self._world.update(timestamp,
                           pose_msg.data,
                           predictions,
                           static_obstacles_msg.obstacles,
                           hd_map=self._map,
                           lanes=lanes)
