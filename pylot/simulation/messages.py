from erdos.message import Message


class FrameMessage(Message):
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
        super(FrameMessage, self).__init__(None, timestamp, 'default')
        self.frame = frame_array
        self.width = frame_array.shape[1]
        self.height = frame_array.shape[0]
        self.encoding = 'BGR'

    def __str__(self):
        return 'FrameMessage(timestamp: {}, width: {}, height: {}, encoding: {})'.format(
            self.timestamp, self.width, self.height, self.encoding)


class DepthFrameMessage(Message):
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
        super(DepthFrameMessage, self).__init__(None, timestamp, 'default')
        self.frame = frame
        self.transform = transform
        self.width = frame.shape[1]
        self.height = frame.shape[0]
        self.fov = fov

    def __str__(self):
        return 'DepthMessage(timestamp: {}, width: {}, height: {}, fov: {})'.format(
            self.timestamp, self.width, self.height, self.fov)


class PointCloudMessage(Message):
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
        super(PointCloudMessage, self).__init__(None, timestamp, 'default')
        self.point_cloud = point_cloud
        self.transform = transform

    def __str__(self):
        return 'PointCloudMessage(timestamp: {}, num point clouds: {})'.format(
            self.timestamp, len(self.point_cloud))


class GroundVehiclesMessage(Message):
    """ Message class to be used to send ground info about vehicle actors.

    Attributes:
        vehicles: A list of simulation.utils.Vehicle tuples.
    """
    def __init__(self, vehicles, timestamp):
        super(GroundVehiclesMessage, self).__init__(None, timestamp, 'default')
        self.vehicles = vehicles

    def __str__(self):
        return 'GroundVehiclesMessage(timestamp: {}, vehicles: {})'.format(
            self.timestamp, self.vehicles)


class GroundPedestriansMessage(Message):
    """ Message class to be used to send ground info about pedestrian actors.

    Attributes:
        pedestrians: A list of simulation.utils.Pedestrian tuples.
    """
    def __init__(self, pedestrians, timestamp):
        super(GroundPedestriansMessage, self).__init__(
            None, timestamp, 'default')
        self.pedestrians = pedestrians

    def __str__(self):
        return 'GroundPedestriansMessage(timestmap: {}, '\
            'pedestrians: {})'.format(
                self.timestamp, self.pedestrians)


class GroundTrafficLightsMessage(Message):
    """ Message class to be used to send ground info about traffic lights actors.

    Attributes:
        traffic_lights: A list of simulation.utils.TrafficLight tuples.
    """
    def __init__(self, traffic_lights, timestamp):
        super(GroundTrafficLightsMessage, self).__init__(
            None, timestamp, 'default')
        self.traffic_lights = traffic_lights

    def __str__(self):
        return 'GroundTrafficLightsMessage(timestamp: {}, '\
            'traffic lights: {})'.format(
                self.timestamp, self.traffic_lights)


class GroundSpeedSignsMessage(Message):
    """ Message class to be used to send ground info about speed limit actors.

    Attributes:
        speed_signs: A list of simulation.utils.SpeedLimitSign tuples.
    """
    def __init__(self, speed_signs, timestamp):
        super(GroundSpeedSignsMessage, self).__init__(
            None, timestamp, 'default')
        self.speed_signs = speed_signs

    def __str__(self):
        return 'GroundSpeedSignsMessage(timestamp: {}, '\
            'speed signs: {})'.format(
                self.timestamp, self.speed_signs)


class GroundStopSignsMessage(Message):
    """ Message class to be used to send ground info about stop signs.

    Attributes:
        stop_signs: A list of stop marking transforms.
    """
    def __init__(self, stop_signs, timestamp):
        super(GroundStopSignsMessage, self).__init__(
            None, timestamp, 'default')
        self.stop_signs = stop_signs

    def __str__(self):
        return 'GroundStopSignsMessage(timestamp: {}, '\
            'stop signs: {})'.format(
                self.timestamp, self.stop_signs)


class DetectedLaneMessage(Message):
    """ Message class to be used to send group info about the lane.

    Attributes:
        detected_lanes: A list of detected lanes locations.
    """

    def __init__(self, detected_lanes, timestamp):
        super(DetectedLaneMessage, self).__init__(None, timestamp, 'default')
        self.detected_lanes = detected_lanes

    def __str__(self):
        return 'DetectedLaneMessage(timestamp: {}, '\
                'detected_lanes: {})'.format(self.timestamp, self.detected_lanes)
