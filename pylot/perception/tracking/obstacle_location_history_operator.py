from collections import defaultdict, deque

import erdos

import pylot.utils
from pylot.perception.detection.utils import VEHICLE_LABELS, \
    get_obstacle_locations
from pylot.perception.messages import ObstacleTrajectoriesMessage
from pylot.perception.tracking.obstacle_trajectory import ObstacleTrajectory


class ObstacleLocationHistoryOperator(erdos.Operator):
    def __init__(self, obstacles_stream, depth_stream, pose_stream,
                 ground_obstacles_stream, vehicle_id_stream,
                 tracked_obstacles_stream, flags, camera_setup):
        self._vehicle_id_stream = vehicle_id_stream
        obstacles_stream.add_callback(self.on_obstacles_update)
        depth_stream.add_callback(self.on_depth_update)
        pose_stream.add_callback(self.on_pose_update)
        ground_obstacles_stream.add_callback(self.on_ground_obstacles_update)
        erdos.add_watermark_callback([
            obstacles_stream, depth_stream, pose_stream,
            ground_obstacles_stream
        ], [tracked_obstacles_stream], self.on_watermark)
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
        self._ground_obstacles_msgs = deque()
        self._obstacle_history = defaultdict(deque)
        self._timestamp_history = deque()
        # Stores the id of obstacles that have values for a given timestamp.
        # This is used to GC the state from timestamp_history.
        self._timestamp_to_id = defaultdict(list)

    @staticmethod
    def connect(obstacles_stream, depth_stream, pose_stream,
                ground_obstacles_stream, vehicle_id_stream):
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
        ground_obstacles_msg = self._ground_obstacles_msgs.popleft()
        assert obstacles_msg.timestamp.coordinates[
            0] == ground_obstacles_msg.timestamp.coordinates[
                0], 'Ground obstacles are not synchronized with obstacles'
        depth_msg = self._depth_msgs.popleft()
        vehicle_transform = self._pose_msgs.popleft().data.transform

        obstacles_with_location = get_obstacle_locations(
            obstacles_msg.obstacles, depth_msg, vehicle_transform,
            self._camera_setup, self._logger)

        self.assign_ground_locations(obstacles_with_location,
                                     ground_obstacles_msg.obstacles)
        ids_cur_timestamp = []
        obstacle_trajectories = []
        for obstacle in obstacles_with_location:
            # Ignore obstacles that are far away.
            if (vehicle_transform.location.distance(
                    obstacle.transform.location) >
                    self._flags.obstacle_distance_threshold):
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

        tracked_obstacles_stream.send(
            ObstacleTrajectoriesMessage(timestamp, obstacle_trajectories))
        tracked_obstacles_stream.send(erdos.WatermarkMessage(timestamp))

        self._log_obstacles(timestamp, obstacles_with_location)

        self._timestamp_history.append(timestamp)
        self._timestamp_to_id[timestamp] = ids_cur_timestamp
        if len(self._timestamp_history) >= self._flags.tracking_num_steps:
            gc_timestamp = self._timestamp_history.popleft()
            for obstacle_id in self._timestamp_to_id[gc_timestamp]:
                self._obstacle_history[obstacle_id].popleft()
                if len(self._obstacle_history[obstacle_id]) == 0:
                    del self._obstacle_history[obstacle_id]
            del self._timestamp_to_id[gc_timestamp]

    def assign_ground_locations(self, obstacles, ground_obstacles):
        for obstacle in obstacles:
            min_distance = 100
            closest_obstacle_index = None
            for index, ground_obstacle in enumerate(ground_obstacles):
                if ground_obstacle.id == self._vehicle_id:
                    continue
                # We're only matching pedestrians.
                if (obstacle.label == ground_obstacle.label
                        or (ground_obstacle.label == 'vehicle'
                            and obstacle.label in VEHICLE_LABELS)):
                    distance = obstacle.transform.location.distance(
                        ground_obstacle.transform.location)
                    if distance < min_distance and distance < 20:
                        min_distance = distance
                        closest_obstacle_index = index
            if closest_obstacle_index is not None:
                obstacle.transform = \
                    ground_obstacles[closest_obstacle_index].transform
                obstacle.id = ground_obstacles[closest_obstacle_index].id

    def on_obstacles_update(self, msg):
        self._logger.debug('@{}: obstacles update'.format(msg.timestamp))
        self._obstacles_msgs.append(msg)

    def on_ground_obstacles_update(self, msg):
        self._logger.debug('@{}: ground obstacles update'.format(
            msg.timestamp))
        self._ground_obstacles_msgs.append(msg)

    def on_depth_update(self, msg):
        self._logger.debug('@{}: depth update'.format(msg.timestamp))
        self._depth_msgs.append(msg)

    def on_pose_update(self, msg):
        self._logger.debug('@{}: pose update'.format(msg.timestamp))
        self._pose_msgs.append(msg)

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

    def run(self):
        # Read the vehicle id from the vehicle id stream
        vehicle_id_msg = self._vehicle_id_stream.read()
        self._vehicle_id = vehicle_id_msg.data
