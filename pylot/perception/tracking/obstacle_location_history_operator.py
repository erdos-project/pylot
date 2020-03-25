from collections import deque, defaultdict
import erdos

from pylot.perception.detection.utils import get_obstacle_locations
from pylot.perception.messages import ObstacleTrajectoriesMessage
from pylot.perception.tracking.obstacle_trajectory import ObstacleTrajectory
import pylot.utils


class ObstacleLocationHistoryOperator(erdos.Operator):
    def __init__(self, obstacles_stream, depth_stream, can_bus_stream,
                 camera_stream, tracked_obstacles_stream, flags, camera_setup):
        obstacles_stream.add_callback(self.on_obstacles_update)
        depth_stream.add_callback(self.on_depth_update)
        can_bus_stream.add_callback(self.on_can_bus_update)
        camera_stream.add_callback(self.on_camera_update)
        erdos.add_watermark_callback(
            [obstacles_stream, depth_stream, can_bus_stream, camera_stream],
            [tracked_obstacles_stream], self.on_watermark)
        self._flags = flags
        self._camera_setup = camera_setup
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        # Queues in which received messages are stored.
        self._obstacles_msgs = deque()
        self._depth_msgs = deque()
        self._can_bus_msgs = deque()
        self._frame_msgs = deque()
        self._obstacle_history = defaultdict(deque)
        self._timestamp_history = deque()
        # Stores the id of obstacles that have values for a given timestamp.
        # This is used to GC the state from timestamp_history.
        self._timestamp_to_id = defaultdict(list)

    @staticmethod
    def connect(obstacles_stream, depth_stream, can_bus_stream, camera_stream):
        tracked_obstacles_stream = erdos.WriteStream()
        return [tracked_obstacles_stream]

    @erdos.profile_method()
    def on_watermark(self, timestamp, tracked_obstacles_stream):
        """Invoked when all input streams have received a watermark.

        Args:
            timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of
                the watermark.
        """
        self._logger.debug('@{}: received watermark'.format(timestamp))
        obstacles_msg = self._obstacles_msgs.popleft()
        depth_msg = self._depth_msgs.popleft()
        vehicle_transform = self._can_bus_msgs.popleft().data.transform
        frame_msg = self._frame_msgs.popleft()

        obstacles_with_location = get_obstacle_locations(
            obstacles_msg.obstacles, depth_msg, vehicle_transform,
            self._camera_setup, self._logger)

        ids_cur_timestamp = []
        obstacle_trajectories = []
        for obstacle in obstacles_with_location:
            ids_cur_timestamp.append(obstacle.id)
            # Transform obstacle location from global world coordinates to
            # ego-centric coordinates.
            location_ego_relative = \
                vehicle_transform.inverse_transform_locations(
                    [obstacle.transform.location])
            obstacle.transform = pylot.utils.Transform(
                location_ego_relative[0], pylot.utils.Rotation())
            self._obstacle_history[obstacle.id].append(obstacle)
            cur_obstacle_trajectory = [
                obstacle.transform
                for obstacle in self._obstacle_history[obstacle.id]
            ]
            obstacle_trajectories.append(
                ObstacleTrajectory(obstacle.label, obstacle.id,
                                   obstacle.bounding_box,
                                   cur_obstacle_trajectory))

        tracked_obstacles_stream.send(
            ObstacleTrajectoriesMessage(timestamp, obstacle_trajectories))

        self._timestamp_history.append(timestamp)
        self._timestamp_to_id[timestamp] = ids_cur_timestamp
        if len(self._timestamp_history) >= self._flags.tracking_num_steps:
            gc_timestamp = self._timestamp_history.popleft()
            for obstacle_id in self._timestamp_to_id[gc_timestamp]:
                self._obstacle_history[obstacle_id].popleft()
                if len(self._obstacle_history[obstacle_id]) == 0:
                    del self._obstacle_history[obstacle_id]
            del self._timestamp_to_id[gc_timestamp]

        if self._flags.visualize_obstacles_with_distance:
            frame_msg.frame.annotate_with_bounding_boxes(
                timestamp, obstacles_with_location, vehicle_transform)
            frame_msg.frame.visualize(self.config.name)

    def on_obstacles_update(self, msg):
        self._logger.debug('@{}: obstacles update'.format(msg.timestamp))
        self._obstacles_msgs.append(msg)

    def on_depth_update(self, msg):
        self._logger.debug('@{}: depth update'.format(msg.timestamp))
        self._depth_msgs.append(msg)

    def on_can_bus_update(self, msg):
        self._logger.debug('@{}: can bus update'.format(msg.timestamp))
        self._can_bus_msgs.append(msg)

    def on_camera_update(self, msg):
        self._logger.debug('@{}: camera update'.format(msg.timestamp))
        self._frame_msgs.append(msg)
