from collections import defaultdict, deque
from typing import Union

import erdos
from erdos.operator import OneInOneOut
from erdos.context import OneInOneOutContext
from pylot.perception.depth_frame import DepthFrame

import pylot.utils
from pylot.perception.detection.utils import get_obstacle_locations
from pylot.perception.messages import ObstacleTrajectoriesMessageTuple, \
    ObstaclesMessageTuple
from pylot.perception.tracking.obstacle_trajectory import ObstacleTrajectory


class ObstacleLocationHistoryOperator(OneInOneOut):
    def __init__(self, flags, camera_setup):
        self._flags = flags
        self._camera_setup = camera_setup
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            self.config.name + '-csv', self.config.csv_log_file_name)
        # Queues in which received messages are stored.
        self._obstacles_msgs = deque()
        self._depth_msgs = deque()
        self._pose_msgs = deque()
        self._obstacle_history = defaultdict(deque)
        self._timestamp_history = deque()
        # Stores the id of obstacles that have values for a given timestamp.
        # This is used to GC the state from timestamp_history.
        self._timestamp_to_id = defaultdict(list)

    def on_data(self, context: OneInOneOutContext,
                data: Union[ObstaclesMessageTuple, DepthFrame,
                            pylot.utils.Pose]):
        if isinstance(data, ObstaclesMessageTuple):
            self.on_obstacles_update(context, data)
        elif isinstance(data, DepthFrame):
            self.on_depth_update(context, data)
        elif isinstance(data, pylot.utils.Pose):
            self.on_pose_update(context, data)

    def on_obstacles_update(self, context, data):
        self._logger.debug('@{}: obstacles update'.format(context.timestamp))
        self._obstacles_msgs.append(data)

    def on_depth_update(self, context, data):
        self._logger.debug('@{}: depth update'.format(context.timestamp))
        self._depth_msgs.append(data)

    def on_pose_update(self, context, data):
        self._logger.debug('@{}: pose update'.format(context.timestamp))
        self._pose_msgs.append(data)

    def on_watermark(self, context: OneInOneOutContext):
        """Invoked when all input streams have received a watermark.

        Args:
            timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of
                the watermark.
        """
        self._logger.debug('@{}: received watermark'.format(context.timestamp))
        if context.timestamp.is_top:
            context.write_stream.send(erdos.WatermarkMessage(
                context.timestamp))
            return
        obstacles_msg = self._obstacles_msgs.popleft()
        depth_msg = self._depth_msgs.popleft()
        vehicle_transform = self._pose_msgs.popleft().transform

        obstacles_with_location = get_obstacle_locations(
            obstacles_msg.obstacles, depth_msg, vehicle_transform,
            self._camera_setup, self._logger)

        ids_cur_timestamp = []
        obstacle_trajectories = []
        for obstacle in obstacles_with_location:
            # Ignore obstacles that are far away.
            if (vehicle_transform.location.distance(
                    obstacle.transform.location) >
                    self._flags.dynamic_obstacle_distance_threshold):
                continue
            ids_cur_timestamp.append(obstacle.id)
            self._obstacle_history[obstacle.id].append(obstacle)
            # Transform obstacle location from global world coordinates to
            # ego-centric coordinates.
            cur_obstacle_trajectory = []
            for obstacle in self._obstacle_history[obstacle.id]:
                new_location = \
                    vehicle_transform.inverse_transform_locations(
                        [obstacle.transform.location])[0]
                cur_obstacle_trajectory.append(
                    pylot.utils.Transform(new_location,
                                          pylot.utils.Rotation()))
            # The trajectory is relative to the current location.
            obstacle_trajectories.append(
                ObstacleTrajectory(obstacle, cur_obstacle_trajectory))

        context.write_stream.send(
            erdos.Message(
                context.timestamp,
                ObstacleTrajectoriesMessageTuple(obstacle_trajectories)))
        context.write_stream.send(erdos.WatermarkMessage(context.timestamp))

        self._log_obstacles(context.timestamp, obstacles_with_location)

        self._timestamp_history.append(context.timestamp)
        self._timestamp_to_id[context.timestamp] = ids_cur_timestamp
        if len(self._timestamp_history) >= self._flags.tracking_num_steps:
            gc_timestamp = self._timestamp_history.popleft()
            for obstacle_id in self._timestamp_to_id[gc_timestamp]:
                self._obstacle_history[obstacle_id].popleft()
                if len(self._obstacle_history[obstacle_id]) == 0:
                    del self._obstacle_history[obstacle_id]
            del self._timestamp_to_id[gc_timestamp]

    def destroy(self):
        self._logger.warn('destroying {}'.format(self.config.name))

    def _log_obstacles(self, timestamp, obstacles):
        for obstacle in obstacles:
            obstacle_location = obstacle.transform.location
            x = obstacle_location.x
            y = obstacle_location.y
            z = obstacle_location.z
            self._csv_logger.debug('{},{},obstacle,{},{}'.format(
                pylot.utils.time_epoch_ms(), timestamp.coordinates[0],
                "[{} {}]".format(obstacle.id, obstacle.label),
                "[{:.4f} {:.4f} {:.4f}]".format(x, y, z)))
