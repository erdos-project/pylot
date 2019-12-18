import erdust


class FrameMessage(erdust.Message):
    """ Message class to be used to send camera frames.

    Attributes:
        frame: A numpy array storing the frame.
        width: The width of the frame.
        height: The height of the frame.
        encoding: The encoding of the frame.
    """
    def __init__(self,
                 frame_array,
                 timestamp,
                 encoding='BGR'):
        """ Initializes the frame messsage.

        Args:
            frame_array: A numpy array storing the frame.
            timestamp: A erdos.timestamp.Timestamp of the message.
            encoding: The encoding of the message.
        """
        super(FrameMessage, self).__init__(timestamp, None)
        self.frame = frame_array
        self.width = frame_array.shape[1]
        self.height = frame_array.shape[0]
        self.encoding = 'BGR'

    def __str__(self):
        return 'FrameMessage(timestamp: {}, width: {}, '\
            'height: {}, encoding: {})'.format(
                self.timestamp, self.width, self.height, self.encoding)


class DepthFrameMessage(erdust.Message):
    """ Message class to be used to send depth camera frames.

    Attributes:
        frame: A numpy array storing the frame.
        transform: The simulation.utils.Transform of the camera.
        width: The width of the frame.
        height: The height of the frame.
        fov: The field of view of the camera.
    """
    def __init__(self, frame, transform, fov, timestamp):
        """ Initializes the depth frame messsage.

        Args:
            frame_array: A numpy array storing the depth frame.
            transform: The simulation.utils.Transform of the camera.
            fov: The field of view of the camera.
            timestamp: A erdos.timestamp.Timestamp of the message.
        """
        super(DepthFrameMessage, self).__init__(timestamp, None)
        self.frame = frame
        self.transform = transform
        self.width = frame.shape[1]
        self.height = frame.shape[0]
        self.fov = fov

    def __str__(self):
        return 'DepthMessage(timestamp: {}, width: {}, '\
            'height: {}, fov: {})'.format(
                self.timestamp, self.width, self.height, self.fov)


class PointCloudMessage(erdust.Message):
    """ Message class to be used to send point clouds.

    Attributes:
        point_cloud: A numpy array storing the point cloud.
        transform: The simulation.utils.Transform of the Lidar.
    """
    def __init__(self, point_cloud, transform, timestamp):
        """ Initializes the point cloud messsage.

        Args:
            point_cloud: A numpy array storing the point cloud.
            transform: The simulation.utils.Transform of the Lidar.
            timestamp: A erdos.timestamp.Timestamp of the message.
        """
        super(PointCloudMessage, self).__init__(timestamp, None)
        self.point_cloud = point_cloud
        self.transform = transform

    def __str__(self):
        return 'PointCloudMessage(timestamp: {}, num point clouds: {})'.format(
            self.timestamp, len(self.point_cloud))


class IMUMessage(erdust.Message):
    """ Message class to be used to send IMU measurements.

     Attributes:
        transform: simulation.utils.Transform of the IMU.
        acceleration: simulation.utils.Vector3D linear acceleration
            measurement in m/s^2
        gyro: simulation.utils.Vector3D angular velocity measurement in rad/sec
        compass: float orientation measurement w.r.t North direction
            ((0, -1, 0) in Unreal) in radians
    """
    def __init__(self, transform, acceleration, gyro, compass, timestamp):
        """ Initializes the IMU messsage.
         Args:
            transform: The simulation.utils.Transform of the IMU.
            acceleration: simulation.utils.Vector3D linear acceleration
                measurement in m/s^2
            gyro: simulation.utils.Vector3D angular velocity measurement in
                rad/sec
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


class GroundObstaclesMessage(erdust.Message):
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


class GroundTrafficLightsMessage(erdust.Message):
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


class GroundSpeedSignsMessage(erdust.Message):
    """ Message class to be used to send ground info about speed limit actors.

    Attributes:
        speed_signs: A list of simulation.utils.SpeedLimitSign tuples.
    """
    def __init__(self, timestamp, speed_signs):
        super(GroundSpeedSignsMessage, self).__init__(timestamp, None)
        self.speed_signs = speed_signs

    def __str__(self):
        return 'GroundSpeedSignsMessage(timestamp: {}, '\
            'speed signs: {})'.format(
                self.timestamp, self.speed_signs)


class GroundStopSignsMessage(erdust.Message):
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


class DetectedLaneMessage(erdust.Message):
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
