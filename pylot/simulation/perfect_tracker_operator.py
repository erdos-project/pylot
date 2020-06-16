from collections import defaultdict, deque

import erdos

import pylot.utils
from pylot.perception.messages import ObstacleTrajectoriesMessage
from pylot.perception.tracking.obstacle_trajectory import ObstacleTrajectory


class PerfectTrackerOperator(erdos.Operator):
    """Operator that gives past trajectories of other agents in the environment,
       i.e. their past (x,y,z) locations from an ego-vehicle perspective.
    """
    def __init__(self, vehicle_id_stream, ground_obstacles_stream, pose_stream,
                 ground_tracking_stream, flags):
        """Initializes the PerfectTracker Operator. """
        self._vehicle_id_stream = vehicle_id_stream
        ground_obstacles_stream.add_callback(self.on_obstacles_update)
        pose_stream.add_callback(self.on_pose_update)
        erdos.add_watermark_callback([ground_obstacles_stream, pose_stream],
                                     [ground_tracking_stream],
                                     self.on_watermark)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags
        # Queues of incoming data.
        self._obstacles_raw_msgs = deque()
        self._pose_msgs = deque()

        # Processed data. Key is actor id, value is deque containing the past
        # trajectory of the corresponding actor. Trajectory is stored in world
        # coordinates, for ease of transformation.
        trajectory = lambda: deque(maxlen=self._flags.tracking_num_steps)
        self._obstacles = defaultdict(trajectory)

    @staticmethod
    def connect(vehicle_id_stream, ground_obstacles_stream, pose_stream):
        ground_tracking_stream = erdos.WriteStream()
        return [ground_tracking_stream]

    @erdos.profile_method()
    def on_watermark(self, timestamp, ground_tracking_stream):
        self._logger.debug('@{}: received watermark'.format(timestamp))
        obstacles_msg = self._obstacles_raw_msgs.popleft()
        pose_msg = self._pose_msgs.popleft()

        # Use the most recent pose message to convert the past frames
        # of vehicles and people to our current perspective.
        pose_transform = pose_msg.data.transform

        obstacle_trajectories = []
        # Only consider obstacles which still exist at the most recent
        # timestamp.
        for obstacle in obstacles_msg.obstacles:
            if obstacle.id == self._vehicle_id and not \
                (self._flags.prediction and self._flags.prediction_type == 'r2p2'):
                # Do not track the ego-vehicle.
                continue
            self._obstacles[obstacle.id].append(obstacle)
            cur_obstacle_trajectory = []
            # Iterate through past frames of this obstacle.
            for past_obstacle_loc in self._obstacles[obstacle.id]:
                # Get the location of the center of the obstacle's bounding
                # box, in relation to the Pose measurement.
                v_transform = past_obstacle_loc.transform * \
                                past_obstacle_loc.bounding_box.transform
                new_location = pose_transform.inverse_transform_locations(
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

    def on_pose_update(self, msg):
        self._logger.debug('@{}: received pose message'.format(msg.timestamp))
        self._pose_msgs.append(msg)

    def run(self):
        # Read the vehicle id from the vehicle id stream
        vehicle_id_msg = self._vehicle_id_stream.read()
        self._vehicle_id = vehicle_id_msg.data
