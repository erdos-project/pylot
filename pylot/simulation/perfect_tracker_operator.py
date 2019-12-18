from absl import flags
from collections import defaultdict, deque
import erdust

from pylot.perception.messages import ObjTrajectory, ObjTrajectoriesMessage

flags.DEFINE_integer(
    'perfect_tracking_num_steps',
    None,
    'Limit on number of past steps returned by the perfect object tracker.')


class PerfectTrackerOperator(erdust.Operator):
    """Operator that gives past trajectories of other agents in the environment,
       i.e. their past (x,y,z) locations from an ego-vehicle perspective.
    """

    def __init__(self,
                 ground_obstacles_stream,
                 can_bus_stream,
                 ground_tracking_stream,
                 name,
                 flags):
        """Initializes the PerfectTracker Operator. """
        ground_obstacles_stream.add_callback(self.on_obstacles_update)
        can_bus_stream.add_callback(self.on_can_bus_update)
        erdust.add_watermark_callback([ground_obstacles_stream,
                                       can_bus_stream],
                                      [ground_tracking_stream],
                                      self.on_watermark)
        self._name = name
        self._flags = flags

        # Queues of incoming data.
        self._obstacles_raw_msgs = deque()
        self._can_bus_msgs = deque()

        # Processed data. Key is actor id, value is deque containing the past
        # trajectory of the corresponding actor. Trajectory is stored in world
        # coordinates, for ease of transformation.
        trajectory = lambda: deque(maxlen=self._flags.perfect_tracking_num_steps)
        self._obstacles = defaultdict(trajectory)

    @staticmethod
    def connect(ground_obstacles_stream,
                can_bus_stream):
        ground_tracking_stream = erdust.WriteStream()
        return [ground_tracking_stream]

    def on_watermark(self, timestamp, ground_tracking_stream):
        obstacles_msg = self._obstacles_raw_msgs.popleft()
        can_bus_msg = self._can_bus_msgs.popleft()

        # Use the most recent can_bus message to convert the past frames
        # of vehicles and pedestrians to our current perspective.
        inv_can_bus_transform = can_bus_msg.data.transform.inverse_transform()

        obstacle_trajectories = []
        # Only consider obstacles which still exist at the most recent
        # timestamp.
        for obstacle in obstacles_msg.obstacles:
            self._obstacles[obstacle.id].append(obstacle)
            cur_obstacle_trajectory = []
            # Iterate through past frames of this obstacle.
            for past_obstacle_loc in self._obstacle[obstacle.id]:
                # Get the location of the center of the obstacle's bounding
                # box, in relation to the CanBus measurement.
                new_transform = inv_can_bus_transform * \
                                past_obstacle_loc.transform * \
                                past_obstacle_loc.bounding_box.transform
                cur_obstacle_trajectory.append(new_transform.location)
            obstacle_trajectories.append(
                ObjTrajectory(obstacle.label,
                              obstacle.id,
                              cur_obstacle_trajectory))

        output_msg = ObjTrajectoriesMessage(timestamp, obstacle_trajectories)
        ground_tracking_stream.send(output_msg)

    def on_obstacles_update(self, msg):
        self._obstacles_raw_msgs.append(msg)

    def on_can_bus_update(self, msg):
        self._can_bus_msgs.append(msg)
