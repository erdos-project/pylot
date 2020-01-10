import erdos
import numpy as np

import pylot.utils


class FrameMessage(erdos.Message):
    """ Message class to be used to send camera frames.

    Attributes:
        frame: A numpy array storing the frame.
        width: The width of the frame.
        height: The height of the frame.
        encoding: The encoding of the frame.
    """
    def __init__(self, frame, timestamp, encoding='carla'):
        """ Initializes the frame messsage.

        Args:
            frame: The frame to be stored.
            timestamp: A erdos.timestamp.Timestamp of the message.
            encoding: The encoding of the message.
        """
        super(FrameMessage, self).__init__(timestamp, None)
        if encoding == 'carla':
            # If the message is from Carla, we convert it to a BGR image
            # and store it as a frame with the BGR encoding.
            _frame = np.frombuffer(frame.raw_data, dtype=np.dtype("uint8"))
            _frame = np.reshape(_frame, (frame.height, frame.width, 4))
            self.frame = _frame[:, :, :3]
            self.encoding = 'BGR'
        else:
            # If the image is sourced from somewhere else, we do not process
            # it and use the encoding provided.
            self.frame = frame
            self.encoding = encoding
        self.width = self.frame.shape[1]
        self.height = self.frame.shape[0]

    def __str__(self):
        return 'FrameMessage(timestamp: {}, width: {}, '\
            'height: {}, encoding: {})'.format(
                self.timestamp, self.width, self.height, self.encoding)


class DepthFrameMessage(erdos.Message):
    """ Message class to be used to send depth camera frames.

    Attributes:
        frame: A pylot.utils.DepthFrame.
    """
    def __init__(self, frame, timestamp):
        """ Initializes the depth frame messsage.

        Args:
            frame: A pylot.utils.DepthFrame.
            timestamp: A erdos.timestamp.Timestamp of the message.
        """
        super(DepthFrameMessage, self).__init__(timestamp, None)
        if not isinstance(frame, pylot.utils.DepthFrame):
            raise ValueError('frame should be of type pylot.utils.DepthFrame')
        self.frame = frame

    def __str__(self):
        return 'DepthMessage(timestamp: {}, depth_frame: {})'.format(
            self.timestamp, self.frame)


class PointCloudMessage(erdos.Message):
    """ Message class to be used to send point clouds.

    Attributes:
        point_cloud: A pylot.utils.PointCloud.
    """
    def __init__(self, point_cloud, transform, timestamp):
        """ Initializes the point cloud messsage.

        Args:
            point_cloud: A pylot.utils.PointCloud.
            timestamp: A erdos.timestamp.Timestamp of the message.
        """
        super(PointCloudMessage, self).__init__(timestamp, None)
        if not isinstance(point_cloud, pylot.utils.PointCloud):
            raise ValueError(
                'point_cloud should be of type pylot.utils.PointCloud')
        self.point_cloud = point_cloud

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


class GroundObstaclesMessage(erdos.Message):
    """ Message class used to send ground pedestrian and vehicle info.

    Attributes:
        obstacles: A list of simulation.utils.Obstacle tuples.
    """
    def __init__(self, timestamp, obstacles):
        super(GroundObstaclesMessage, self).__init__(timestamp, None)
        self.obstacles = obstacles

    def __str__(self):
        return 'GroundObstaclesMessage(timestamp: {}, obstacles: {})'.format(
            self.timestamp, self.obstacles)


class GroundTrafficLightsMessage(erdos.Message):
    """ Message class to be used to send ground info about traffic lights actors.

    Attributes:
        traffic_lights: A list of simulation.utils.TrafficLight tuples.
    """
    def __init__(self, timestamp, traffic_lights):
        super(GroundTrafficLightsMessage, self).__init__(timestamp, None)
        self.traffic_lights = traffic_lights

    def __str__(self):
        return 'GroundTrafficLightsMessage(timestamp: {}, '\
            'traffic lights: {})'.format(
                self.timestamp, self.traffic_lights)


class GroundSpeedSignsMessage(erdos.Message):
    """ Message class to be used to send ground info about speed limit actors.

    Attributes:
        speed_signs: A list of simulation.carla_utils.SpeedLimitSign tuples.
    """
    def __init__(self, timestamp, speed_signs):
        super(GroundSpeedSignsMessage, self).__init__(timestamp, None)
        self.speed_signs = speed_signs

    def __str__(self):
        return 'GroundSpeedSignsMessage(timestamp: {}, '\
            'speed signs: {})'.format(
                self.timestamp, self.speed_signs)


class GroundStopSignsMessage(erdos.Message):
    """ Message class to be used to send ground info about stop signs.

    Attributes:
        stop_signs: A list of stop marking transforms.
    """
    def __init__(self, timestamp, stop_signs):
        super(GroundStopSignsMessage, self).__init__(timestamp, None)
        self.stop_signs = stop_signs

    def __str__(self):
        return 'GroundStopSignsMessage(timestamp: {}, '\
            'stop signs: {})'.format(
                self.timestamp, self.stop_signs)


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
