"""This module implements perception related Message classes."""

import erdos

import pylot.perception.camera_frame
import pylot.perception.depth_frame
import pylot.perception.point_cloud
from pylot.perception.segmentation.segmented_frame import SegmentedFrame


class FrameMessage(erdos.Message):
    """Message class to be used to send camera frames.

    Args:
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of the
            message.
        frame (:py:class:`~pylot.perception.camera_frame.CameraFrame`): The
            camera frame.

    Attributes:
        frame (:py:class:`~pylot.perception.camera_frame.CameraFrame`): The
            camera frame.
    """
    def __init__(self, timestamp, frame):
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
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of the
            message.
        frame (:py:class:`~pylot.perception.depth_frame.DepthFrame`): The
            depth frame.

    Attributes:
        frame (:py:class:`~pylot.perception.depth_frame.DepthFrame`): The
            depth frame.
    """
    def __init__(self, timestamp, frame):
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
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of the
            message.
        frame (:py:class:`~pylot.perception.segmentation.segmented_frame.SegmentedFrame`):
            The segmented frame.
        runtime (:obj:`float`, optional): The runtime of the operator that
            produced the segmented frame (in ms).

    Attributes:
        frame (:py:class:`~pylot.perception.segmentation.segmented_frame.SegmentedFrame`):
            The segmented frame.
        runtime (:obj:`float`): The runtime of the operator that produced the
            segmented frame (in ms).
    """
    def __init__(self, timestamp, frame, runtime=0):
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
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of the
            message.
        pc (:py:class:`~pylot.perception.point_cloud.PointCloud`): The point
            cloud to send.

    Attributes:
        point_cloud (:py:class:`~pylot.perception.point_cloud.PointCloud`): The
            point cloud.
    """
    def __init__(self, timestamp, pc):
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
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of the
            message.
        transform (:py:class:`~pylot.utils.Transform`): Transform of the IMU.
        acceleration (:py:class:`~pylot.utils.Vector3D`): Linear acceleration
            measurement in m/s^2.
        gyro (:py:class:`~pylot.utils.Vector3D`): Angular velocity measurement
            in rad/sec.
        compass (:obj:`float`): Orientation measurement w.r.t North direction
            ((0, -1, 0) in Unreal) in radians.

    Attributes:
        transform (:py:class:`~pylot.utils.Transform`): Transform of the IMU.
        acceleration (:py:class:`~pylot.utils.Vector3D`): Linear acceleration
            measurement in m/s^2.
        gyro (:py:class:`~pylot.utils.Vector3D`): Angular velocity measurement
            in rad/sec.
        compass (:obj:`float`): Orientation measurement w.r.t North direction
            ((0, -1, 0) in Unreal) in radians.
    """
    def __init__(self, timestamp, transform, acceleration, gyro, compass):
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
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of the
            message.
        obstacles (list(:py:class:`~pylot.perception.detection.utils.DetectedObstacle`)):
            Detected obstacles.
        runtime (:obj:`float`, optional): The runtime of the operator that
            produced the obstacles (in ms).


    Attributes:
        obstacles (list(:py:class:`~pylot.perception.detection.utils.DetectedObstacle`)):
            Detected obstacles.
        runtime (:obj:`float`, optional): The runtime of the operator that
            produced the obstacles (in ms).
    """
    def __init__(self, timestamp, obstacles, runtime=0):
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
    def __init__(self, timestamp, obstacle_positions_speeds):
        super(ObstaclePositionsSpeedsMessage, self).__init__(timestamp, None)
        self.obstacle_positions_speeds = obstacle_positions_speeds

    def __str__(self):
        return ('ObstaclePositionsSpeedsMessage(timestamp {}, '
                'positions: {})'.format(self.timestamp,
                                        self.obstacle_positions_speeds))


class ObstacleTrajectoriesMessage(erdos.Message):
    """Message to be used to send obstacle trajectory info.

    Args:
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of the
            message.
        obstacle_trajectories (list(:py:class:`~pylot.perception.tracking.obstacle_trajectory.ObstacleTrajectory`)):
            Obstacle trajectories.

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
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of the
            message.
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


class TrafficLightsMessage(erdos.Message):
    """Message class to be used to send info about traffic lights.

    Args:
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of the
            message.
        traffic_lights (list(:py:class:`~pylot.perception.detection.traffic_light.TrafficLight`)):
            A list of traffic lights.

    Attributes:
        traffic_lights (list(:py:class:`~pylot.perception.detection.traffic_light.TrafficLight`)):
            A list of traffic lights.
    """
    def __init__(self, timestamp, traffic_lights):
        super(TrafficLightsMessage, self).__init__(timestamp, None)
        self.traffic_lights = traffic_lights

    def __str__(self):
        return 'TrafficLightsMessage(timestamp: {}, ' \
            'traffic lights: {})'.format(
                self.timestamp, self.traffic_lights)


class StopSignsMessage(erdos.Message):
    """Message class to be used to send info about stop signs.

    Args:
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of the
            message.
        stop_signs (list(:py:class:`~pylot.perception.detection.stop_sign.StopSign`)):
            A list of stop signs.

    Attributes:
        stop_signs (list(:py:class:`~pylot.perception.detection.stop_sign.StopSign`)):
            A list of stop signs.
    """
    def __init__(self, timestamp, stop_signs):
        super(StopSignsMessage, self).__init__(timestamp, None)
        self.stop_signs = stop_signs

    def __str__(self):
        return 'StopSignsMessage(timestamp: {}, stop signs: {})'.format(
            self.timestamp, self.stop_signs)


class SpeedSignsMessage(erdos.Message):
    """Message class to be used to send info about speed limit actors.

    Args:
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of the
            message.
        speed_signs (list(:py:class:`~pylot.perception.detection.speed_limit_sign.SpeedLimitSign`)):
            A list of speed limit signs.

    Attributes:
        speed_signs (list(:py:class:`~pylot.perception.detection.speed_limit_sign.SpeedLimitSign`)):
            A list of speed limit signs.
    """
    def __init__(self, timestamp, speed_signs):
        super(SpeedSignsMessage, self).__init__(timestamp, None)
        self.speed_signs = speed_signs

    def __str__(self):
        return 'SpeedSignsMessage(timestamp: {}, speed signs: {})'.format(
            self.timestamp, self.speed_signs)
