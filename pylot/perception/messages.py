"""This module implements perception related Message classes."""

import erdos

import pylot.perception.camera_frame
import pylot.perception.depth_frame
import pylot.perception.point_cloud
from pylot.perception.segmentation.segmented_frame import SegmentedFrame


class FrameMessage(erdos.Message):
    """Message class to be used to send camera frames.

    Args:
        frame (:py:class:`~pylot.perception.camera_frame.CameraFrame`): The
            camera frame.
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of the
            message.

    Attributes:
        frame (:py:class:`~pylot.perception.camera_frame.CameraFrame`): The
            camera frame.
    """
    def __init__(self, frame, timestamp):
        super(FrameMessage, self).__init__(timestamp, None)
        if not isinstance(frame, pylot.perception.camera_frame.CameraFrame):
            raise ValueError(
                'frame should be of type perception.camera_frame.CameraFrame')
        self.frame = frame

    def __str__(self):
        return 'FrameMessage(timestamp: {}, frame: {})'.format(
            self.timestamp, self.frame)


class DepthFrameMessage(erdos.Message):
    """Message class to be used to send depth camera frames.

    Args:
        frame (:py:class:`~pylot.perception.depth_frame.DepthFrame`): The
            depth frame.
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of the
            message.

    Attributes:
        frame (:py:class:`~pylot.perception.depth_frame.DepthFrame`): The
            depth frame.
    """
    def __init__(self, frame, timestamp):
        super(DepthFrameMessage, self).__init__(timestamp, None)
        if not isinstance(frame, pylot.perception.depth_frame.DepthFrame):
            raise ValueError(
                'frame should be of type perception.depth_frame.DepthFrame')
        self.frame = frame

    def __str__(self):
        return 'DepthMessage(timestamp: {}, depth_frame: {})'.format(
            self.timestamp, self.frame)


class SegmentedFrameMessage(erdos.Message):
    """This class represents a message to be used to send segmented frames

    Args:
        frame (:py:class:`~pylot.perception.segmentation.segmented_frame.SegmentedFrame`):
            The segmented frame.
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of the
            message.
        runtime (:obj:`float`, optional): The runtime of the operator that
            produced the segmented frame (in ms).

    Attributes:
        frame (:py:class:`~pylot.perception.segmentation.segmented_frame.SegmentedFrame`):
            The segmented frame.
        runtime (:obj:`float`): The runtime of the operator that produced the
            segmented frame (in ms).
    """
    def __init__(self, frame, timestamp, runtime=0):
        super(SegmentedFrameMessage, self).__init__(timestamp, None)
        if not isinstance(frame, SegmentedFrame):
            raise ValueError('frame should be of type SegmentedFrame')
        self.frame = frame
        self.runtime = runtime

    def __str__(self):
        return 'SegmentedFrameMessage(timestamp {}, frame: {})'.format(
            self.timestamp, self.frame)


class PointCloudMessage(erdos.Message):
    """Message class to be used to send point clouds.

    Args:
        pc (:py:class:`~pylot.perception.point_cloud.PointCloud`): A point cloud.
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of the
            message.

    Attributes:
        point_cloud (:py:class:`~pylot.perception.point_cloud.PointCloud`): The
            point cloud.
    """
    def __init__(self, pc, timestamp):
        super(PointCloudMessage, self).__init__(timestamp, None)
        if not isinstance(pc, pylot.perception.point_cloud.PointCloud):
            raise ValueError(
                'pc should be of type perception.point_cloud.PointCloud')
        self.point_cloud = pc

    def __str__(self):
        return 'PointCloudMessage(timestamp: {}, point cloud: {})'.format(
            self.timestamp, self.point_cloud)


class IMUMessage(erdos.Message):
    """Message class to be used to send IMU measurements.

    Args:
        transform (:py:class:`~pylot.utils.Transform`): Transform of the IMU.
        acceleration (:py:class:`~pylot.utils.Vector3D`): Linear acceleration
            measurement in m/s^2.
        gyro (:py:class:`~pylot.utils.Vector3D`): Angular velocity measurement
            in rad/sec.
        compass (:obj:`float`): Orientation measurement w.r.t North direction
            ((0, -1, 0) in Unreal) in radians.
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of the
            message.

    Attributes:
        transform (:py:class:`~pylot.utils.Transform`): Transform of the IMU.
        acceleration (:py:class:`~pylot.utils.Vector3D`): Linear acceleration
            measurement in m/s^2.
        gyro (:py:class:`~pylot.utils.Vector3D`): Angular velocity measurement
            in rad/sec.
        compass (:obj:`float`): Orientation measurement w.r.t North direction
            ((0, -1, 0) in Unreal) in radians.
    """
    def __init__(self, transform, acceleration, gyro, compass, timestamp):
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
    """Used to send detected obstacles.

    Args:
        obstacles (list(:py:class:`~pylot.perception.detection.utils.DetectedObstacle`)):
            Detected obstacles.
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of the
            message.
        runtime (:obj:`float`, optional): The runtime of the operator that
            produced the obstacles (in ms).


    Attributes:
        obstacles (list(:py:class:`~pylot.perception.detection.utils.DetectedObstacle`)):
            Detected obstacles.
        runtime (:obj:`float`, optional): The runtime of the operator that
            produced the obstacles (in ms).
    """
    def __init__(self, obstacles, timestamp, runtime=0):
        super(ObstaclesMessage, self).__init__(timestamp, None)
        self.obstacles = obstacles
        self.runtime = runtime

    def __str__(self):
        return 'ObstaclesMessage(timestamp: {}, runtime: {}, '\
            'obstacles: {})'.format(
                self.timestamp, self.runtime, self.obstacles)


class ObstaclePositionsSpeedsMessage(erdos.Message):
    """Represents a message to be used to send vehicle position and speed info.

    Args:
        obstacle_positions_speeds: A list of 2D obstacle positions.
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of the
            message.

    Attributes:
        obstacle_positions_speeds: A list of 2D obstacle positions.
    """
    def __init__(self, obstacle_positions_speeds, timestamp):
        super(ObstaclePositionsSpeedsMessage, self).__init__(timestamp, None)
        self.obstacle_positions_speeds = obstacle_positions_speeds

    def __str__(self):
        return ('ObstaclePositionsSpeedsMessage(timestamp {}, '
                'positions: {})'.format(self.timestamp,
                                        self.obstacle_positions_speeds))


class ObstacleTrajectoriesMessage(erdos.Message):
    """Message to be used to send obstacle trajectory info.

    Args:
        obstacle_trajectories (list(:py:class:`~pylot.perception.tracking.obstacle_trajectory.ObstacleTrajectory`)):
            Obstacle trajectories.
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of the
            message.

    Attributes:
        obstacle_trajectories (list(:py:class:`~pylot.perception.tracking.obstacle_trajectory.ObstacleTrajectory`)):
            Obstacle trajectories.
    """
    def __init__(self, timestamp, obstacle_trajectories):
        super(ObstacleTrajectoriesMessage, self).__init__(timestamp, None)
        self.obstacle_trajectories = obstacle_trajectories

    def __str__(self):
        trajectories_str = '\n'.join(
            [str(traj) for traj in self.obstacle_trajectories])
        return ('ObstacleTrajectoriesMessage(timestamp {}, '
                'trajectories: {})'.format(self.timestamp, trajectories_str))


class DetectedLaneMessage(erdos.Message):
    """Message to be used to send info about lanes.

    Args:
        detected_lanes (list(:py:class:`~pylot.utils.Location`)): Detected lane
            locations.

    Attributes:
        detected_lanes (list(:py:class:`~pylot.utils.Location`)): Detected lane
            locations.
    """
    def __init__(self, timestamp, detected_lanes):
        super(DetectedLaneMessage, self).__init__(timestamp, None)
        self.detected_lanes = detected_lanes

    def __str__(self):
        return 'DetectedLaneMessage(timestamp: {}, '\
                'detected_lanes: {})'.format(
                    self.timestamp, self.detected_lanes)
