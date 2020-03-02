from collections import defaultdict, deque
import erdos

import pylot.utils
from pylot.perception.messages import ObstacleTrajectoriesMessage
from pylot.perception.tracking.obstacle_trajectory import ObstacleTrajectory


class PerfectTrackerOperator(erdos.Operator):
    """Operator that gives past trajectories of other agents in the environment,
       i.e. their past (x,y,z) locations from an ego-vehicle perspective.
    """
    def __init__(self, ground_obstacles_stream, can_bus_stream,
                 ground_tracking_stream, flags):
        """Initializes the PerfectTracker Operator. """
        ground_obstacles_stream.add_callback(self.on_obstacles_update)
        can_bus_stream.add_callback(self.on_can_bus_update)
        erdos.add_watermark_callback([ground_obstacles_stream, can_bus_stream],
                                     [ground_tracking_stream],
                                     self.on_watermark)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags
        # Queues of incoming data.
        self._obstacles_raw_msgs = deque()
        self._can_bus_msgs = deque()

        # Processed data. Key is actor id, value is deque containing the past
        # trajectory of the corresponding actor. Trajectory is stored in world
        # coordinates, for ease of transformation.
        trajectory = lambda: deque(maxlen=self._flags.
                                   perfect_tracking_num_steps)
        self._obstacles = defaultdict(trajectory)

    @staticmethod
    def connect(ground_obstacles_stream, can_bus_stream):
        ground_tracking_stream = erdos.WriteStream()
        return [ground_tracking_stream]

    @erdos.profile_method()
    def on_watermark(self, timestamp, ground_tracking_stream):
        self._logger.debug('@{}: received watermark'.format(timestamp))
        obstacles_msg = self._obstacles_raw_msgs.popleft()
        can_bus_msg = self._can_bus_msgs.popleft()

        # Use the most recent can_bus message to convert the past frames
        # of vehicles and people to our current perspective.
        can_bus_transform = can_bus_msg.data.transform

        obstacle_trajectories = []
        # Only consider obstacles which still exist at the most recent
        # timestamp.
        for obstacle in obstacles_msg.obstacles:
            self._obstacles[obstacle.id].append(obstacle)
            cur_obstacle_trajectory = []
            # Iterate through past frames of this obstacle.
            for past_obstacle_loc in self._obstacles[obstacle.id]:
                # Get the location of the center of the obstacle's bounding
                # box, in relation to the CanBus measurement.
                v_transform = past_obstacle_loc.transform * \
                                past_obstacle_loc.bounding_box.transform
                new_location = can_bus_transform.inverse_transform_points(
                    [v_transform.location])[0]
                cur_obstacle_trajectory.append(
                    pylot.utils.Transform(location=new_location,
                                          rotation=pylot.utils.Rotation()))
            obstacle_trajectories.append(
                ObstacleTrajectory(obstacle.label, obstacle.id,
                                   obstacle.bounding_box,
                                   cur_obstacle_trajectory))

        output_msg = ObstacleTrajectoriesMessage(timestamp,
                                                 obstacle_trajectories)
        ground_tracking_stream.send(output_msg)

    def on_obstacles_update(self, msg):
        self._logger.debug('@{}: received obstacles message'.format(
            msg.timestamp))
        self._obstacles_raw_msgs.append(msg)

    def on_can_bus_update(self, msg):
        self._logger.debug('@{}: received can bus message'.format(
            msg.timestamp))
        self._can_bus_msgs.append(msg)
