import erdos


class ObstaclesMessage(erdos.Message):
    """ This class represents a detector output message to be sent by
    operators.

    Attributes:
        obstacles: List of detected obstacles.
        timestamp: Timestamp of the message.
        runtime: Operator runtime (in ms).
    """
    def __init__(self, obstacles, timestamp, runtime=0):
        """ Constructs the ObstaclesMessage with the given data and timestamp.

        Args:
            obstacles: List of perception.detection.utils.DetectedObstacle.
            runtime: Detector operator runtime.
            timestamp: The timestamp of the message.
        """
        super(ObstaclesMessage, self).__init__(timestamp, None)
        self.obstacles = obstacles
        self.runtime = runtime

    def __str__(self):
        return 'ObstaclesMessage(timestamp: {}, runtime: {}, '\
            'obstacles: {})'.format(
                self.timestamp, self.runtime, self.obstacles)


class SegmentedFrameMessage(erdos.Message):
    """ This class represents a message to be used to send segmented frames

    Attributes:
        frame: The segmented frame in Cityscapes palette.
        runtime: The runtime of the operator that produced the segmented frame.
    """
    def __init__(self, frame, timestamp, runtime=0):
        """ Constructs the SegmentedFrameMessage.

        Args:
            frame: The segmented frame.
            runtime: The runtime of the operator that produced the segmented
                frame.
            timestamp: The timestamp of the message.
        """
        super(SegmentedFrameMessage, self).__init__(timestamp, None)
        self.frame = frame
        self.runtime = runtime
        self.encoding = 'segmented'

    def __str__(self):
        return 'SegmentedFrameMessage(timestamp {}, width: {}, '\
            'height: {})'.format(self.timestamp, self.width, self.height)


class ObstaclePositionsSpeedsMessage(erdos.Message):
    """ This class represents a message to be used to send vehicle position and
        speed info.

    Attributes:
        obstacle_positions_speeds: 2D positions and speeds of the obstacles.
    """
    def __init__(self, obstacle_positions_speeds, timestamp):
        """ Constructs the ObstaclePositionsSpeedsMessage with the given data
        and timestamp.

        Args:
            obstacle_positions_speeds: A list of 2D obstacle positions.
            timestamp: The timestamp of the message.
        """
        super(ObstaclePositionsSpeedsMessage, self).__init__(timestamp, None)
        self.obstacle_positions_speeds = obstacle_positions_speeds

    def __str__(self):
        return ('ObstaclePositionsSpeedsMessage(timestamp {}, '
                'positions: {})'.format(self.timestamp,
                                        self.obstacle_positions_speeds))


class ObstacleTrajectory(object):
    def __init__(self, label, id, trajectory):
        """Constructs the obstacle trajectory using the given data.

        Args:
            label: String for the class of the obstacle.
            id: ID of the obstacle.
            trajectory: List of past pylot.util.simulation.Transforms.
        """

        self.label = label
        self.id = id
        self.trajectory = trajectory

    def __str__(self):
        return '{} {}, Trajectory {}'.format(self.label, self.id,
                                             self.trajectory)


class ObstacleTrajectoriesMessage(erdos.Message):
    """ This class represents a message to be used to send vehicle
        trajectory info.
    """
    def __init__(self, timestamp, obstacle_trajectories):
        """ Constructs the ObstacleTrajectoriesMessage with the given data and
        timestamp.

        Args:
            obstacle_trajectories: A list of ObstacleTrajectory instances.
        """
        super(ObstacleTrajectoriesMessage, self).__init__(timestamp, None)
        self.obstacle_trajectories = obstacle_trajectories

    def __str__(self):
        trajectories_str = '\n'.join(
            [str(traj) for traj in self.obstacle_trajectories])
        return ('ObstacleTrajectoriesMessage(timestamp {}, '
                'trajectories: {})'.format(self.timestamp, trajectories_str))
