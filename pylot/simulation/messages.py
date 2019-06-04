from erdos.message import Message


class FrameMessage(Message):
    def __init__(self,
                 frame_array,
                 timestamp,
                 encoding='BGR',
                 stream_name='default'):
        super(FrameMessage, self).__init__(None, timestamp, stream_name)
        self.frame = frame_array
        self.width = frame_array.shape[1]
        self.height = frame_array.shape[0]
        self.encoding = 'BGR'

    def __str__(self):
        return 'timestamp: {}, width: {}, height: {}, encoding: {}'.format(
            self.timestamp, self.width, self.height, self.encoding)


class DepthFrameMessage(Message):
    def __init__(self,
                 frame,
                 transform,
                 fov,
                 timestamp,
                 stream_name='default'):
        super(DepthFrameMessage, self).__init__(None, timestamp, stream_name)
        self.frame = frame
        self.transform = transform
        self.width = frame.shape[1]
        self.height = frame.shape[0]
        self.fov = fov

    def __str__(self):
        return 'timestamp: {}, width: {}, height: {}, fov: {}'.format(
            self.timestamp, self.width, self.height, self.fov)


class PointCloudMessage(Message):
    def __init__(self,
                 point_cloud,
                 transform,
                 timestamp,
                 stream_name='default'):
        super(PointCloudMessage, self).__init__(None, timestamp, stream_name)
        self.point_cloud = point_cloud
        self.transform = transform

    def __str__(self):
        return 'timestamp: {}, num point clouds: {}'.format(
            self.timestamp, len(self.point_cloud))


class GroundVehiclesMessage(Message):
    def __init__(self, vehicles, timestamp, stream_name='default'):
        super(GroundVehiclesMessage, self).__init__(None, timestamp, stream_name)
        self.vehicles = vehicles

    def __str__(self):
        return 'timestamp: {}, vehicles: {}'.format(
            self.timestamp, self.vehicles)


class GroundPedestriansMessage(Message):
    def __init__(self, pedestrians, timestamp, stream_name='default'):
        super(GroundPedestriansMessage, self).__init__(
            None, timestamp, stream_name)
        self.pedestrians = pedestrians

    def __str__(self):
        return 'timestmap: {}, pedestrians: {}'.format(
            self.timestamp, self.pedestrians)


class GroundTrafficLightsMessage(Message):
    def __init__(self, traffic_lights, timestamp, stream_name='default'):
        super(GroundTrafficLightsMessage, self).__init__(
            None, timestamp, stream_name)
        self.traffic_lights = traffic_lights

    def __str__(self):
        return 'timestamp: {}, traffic lights: {}'.format(
            self.timestamp, self.traffic_lights)


class GroundSpeedSignsMessage(Message):
    def __init__(self, speed_signs, timestamp, stream_name='default'):
        super(GroundSpeedSignsMessage, self).__init__(
            None, timestamp, stream_name)
        self.speed_signs = speed_signs

    def __str__(self):
        return 'timestamp: {}, speed signs: {}'.format(
            self.timestamp, self.speed_signs)
