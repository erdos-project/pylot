import erdos


class GroundObstaclesMessage(erdos.Message):
    """Message class used to send person and vehicle info.

    Args:
        obstacles (list(:py:class:`~pylot.perception.detection.obstacle.Obstacle`)):
            A list of obstacles.

    Attributes:
        obstacles (list(:py:class:`~pylot.perception.detection.obstacle.Obstacle`)):
            A list of obstacles.
    """
    def __init__(self, timestamp, obstacles):
        super(GroundObstaclesMessage, self).__init__(timestamp, None)
        self.obstacles = obstacles

    def __str__(self):
        return 'GroundObstaclesMessage(timestamp: {}, obstacles: {})'.format(
            self.timestamp, self.obstacles)


class GroundTrafficLightsMessage(erdos.Message):
    """Message class to be used to send info about traffic lights actors.

    Args:
        traffic_lights (list(:py:class:`~pylot.perception.detection.traffic_light.TrafficLight`)):
            A list of traffic lights.

    Attributes:
        traffic_lights (list(:py:class:`~pylot.perception.detection.traffic_light.TrafficLight`)):
            A list of traffic lights.
    """
    def __init__(self, timestamp, traffic_lights):
        super(GroundTrafficLightsMessage, self).__init__(timestamp, None)
        self.traffic_lights = traffic_lights

    def __str__(self):
        return 'GroundTrafficLightsMessage(timestamp: {}, '\
            'traffic lights: {})'.format(
                self.timestamp, self.traffic_lights)


class GroundSpeedSignsMessage(erdos.Message):
    """Message class to be used to send info about speed limit actors.

    Args:
        speed_signs (list(:py:class:`~pylot.perception.detection.speed_limit_sign.SpeedLimitSign`)):
            A list of speed limit signs.

    Attributes:
        speed_signs (list(:py:class:`~pylot.perception.detection.speed_limit_sign.SpeedLimitSign`)):
            A list of speed limit signs.
    """
    def __init__(self, timestamp, speed_signs):
        super(GroundSpeedSignsMessage, self).__init__(timestamp, None)
        self.speed_signs = speed_signs

    def __str__(self):
        return 'GroundSpeedSignsMessage(timestamp: {}, '\
            'speed signs: {})'.format(
                self.timestamp, self.speed_signs)


class GroundStopSignsMessage(erdos.Message):
    """Message class to be used to send info about stop signs.

    Args:
        stop_signs (list(:py:class:`~pylot.perception.detection.stop_sign.StopSign`)):
            A list of stop signs.

    Attributes:
        stop_signs (list(:py:class:`~pylot.perception.detection.stop_sign.StopSign`)):
            A list of stop signs.
    """
    def __init__(self, timestamp, stop_signs):
        super(GroundStopSignsMessage, self).__init__(timestamp, None)
        self.stop_signs = stop_signs

    def __str__(self):
        return 'GroundStopSignsMessage(timestamp: {}, '\
            'stop signs: {})'.format(
                self.timestamp, self.stop_signs)
