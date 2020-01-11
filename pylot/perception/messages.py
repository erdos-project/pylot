import erdos

import pylot.perception.camera_frame
import pylot.perception.depth_frame
import pylot.perception.point_cloud
from pylot.perception.segmentation.segmented_frame import SegmentedFrame


class FrameMessage(erdos.Message):
    """ Message class to be used to send camera frames.

    Attributes:
        frame: A numpy array storing the frame.
    """
    def __init__(self, frame, timestamp):
        """ Initializes the frame messsage.

        Args:
            frame: The frame to be stored.
            timestamp: A erdos.timestamp.Timestamp of the message.
        """
        super(FrameMessage, self).__init__(timestamp, None)
        if not isinstance(frame, pylot.perception.camera_frame.CameraFrame):
            raise ValueError(
                'frame should be of type perception.camera_frame.CameraFrame')
        self.frame = frame

    def __str__(self):
        return 'FrameMessage(timestamp: {}, frame: {})'.format(
            self.timestamp, self.frame)


class DepthFrameMessage(erdos.Message):
    """ Message class to be used to send depth camera frames.

    Attributes:
        frame: A pylot.perception.depth_frame.DepthFrame.
    """
    def __init__(self, frame, timestamp):
        """ Initializes the depth frame messsage.

        Args:
            frame: A pylot.perception.depth_frame.DepthFrame.
            timestamp: A erdos.timestamp.Timestamp of the message.
        """
        super(DepthFrameMessage, self).__init__(timestamp, None)
        if not isinstance(frame, pylot.perception.depth_frame.DepthFrame):
            raise ValueError(
                'frame should be of type perception.depth_frame.DepthFrame')
        self.frame = frame

    def __str__(self):
        return 'DepthMessage(timestamp: {}, depth_frame: {})'.format(
            self.timestamp, self.frame)


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
        if not isinstance(frame, SegmentedFrame):
            raise ValueError('frame should be of type SegmentedFrame')
        self.frame = frame
        self.runtime = runtime

    def __str__(self):
        return 'SegmentedFrameMessage(timestamp {}, frame: {})'.format(
            self.timestamp, self.frame)


class PointCloudMessage(erdos.Message):
    """ Message class to be used to send point clouds.

    Attributes:
        point_cloud: A pylot.perception.point_cloud.PointCloud.
    """
    def __init__(self, pc, timestamp):
        """ Initializes the point cloud messsage.

        Args:
            pc: A pylot.perception.point_cloud.PointCloud.
            timestamp: A erdos.timestamp.Timestamp of the message.
        """
        super(PointCloudMessage, self).__init__(timestamp, None)
        if not isinstance(pc, pylot.perception.point_cloud.PointCloud):
            raise ValueError(
                'pc should be of type perception.point_cloud.PointCloud')
        self.point_cloud = pc

    def __str__(self):
        return 'PointCloudMessage(timestamp: {}, point cloud: {})'.format(
            self.timestamp, self.point_cloud)


class IMUMessage(erdos.Message):
    """ Message class to be used to send IMU measurements.

     Attributes:
        transform: simulation.utils.Transform of the IMU.
        acceleration: utils.Vector3D linear acceleration
            measurement in m/s^2
        gyro: utils.Vector3D angular velocity measurement in rad/sec
        compass: float orientation measurement w.r.t North direction
            ((0, -1, 0) in Unreal) in radians
    """
    def __init__(self, transform, acceleration, gyro, compass, timestamp):
        """ Initializes the IMU messsage.
         Args:
            transform: The simulation.utils.Transform of the IMU.
            acceleration: utils.Vector3D linear acceleration
                measurement in m/s^2
            gyro: utils.Vector3D angular velocity measurement in rad/sec
            compass: float orientation measurement w.r.t North direction
                ((0, -1, 0) in Unreal) in radians
        """
        super(IMUMessage, self).__init__(timestamp, None)
        self.transform = transform
        self.acceleration = acceleration
        self.gyro = gyro
        self.compass = compass

    def __str__(self):
        return 'IMUMessage(timestamp: {}, transform: {}, acceleration: {}, '\
            'gyro: {}, compass: {})'.format(
                self.timestamp, self.transform, self.acceleration, self.gyro,
                self.compass)


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


class ObstacleTrajectoriesMessage(erdos.Message):
    """ This class represents a message to be used to send vehicle
        trajectory info.
    """
    def __init__(self, timestamp, obstacle_trajectories):
        """ Constructs the ObstacleTrajectoriesMessage with the given data and
        timestamp.

        Args:
            obstacle_trajectories: A list of
                perception.tracking.obstacle_trajectory.ObstacleTrajectory.
        """
        super(ObstacleTrajectoriesMessage, self).__init__(timestamp, None)
        self.obstacle_trajectories = obstacle_trajectories

    def __str__(self):
        trajectories_str = '\n'.join(
            [str(traj) for traj in self.obstacle_trajectories])
        return ('ObstacleTrajectoriesMessage(timestamp {}, '
                'trajectories: {})'.format(self.timestamp, trajectories_str))


class DetectedLaneMessage(erdos.Message):
    """ Message class to be used to send group info about the lane.

    Attributes:
        detected_lanes: A list of detected lanes locations.
    """
    def __init__(self, timestamp, detected_lanes):
        super(DetectedLaneMessage, self).__init__(timestamp, None)
        self.detected_lanes = detected_lanes

    def __str__(self):
        return 'DetectedLaneMessage(timestamp: {}, '\
                'detected_lanes: {})'.format(
                    self.timestamp, self.detected_lanes)
