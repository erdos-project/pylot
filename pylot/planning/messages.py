import erdos


class WaypointsMessage(erdos.Message):
    """Message class to be used to send waypoints. Optionally can also send
    a target speed for each waypoint.

    Args:
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of
            the message.
        waypoints (list(:py:class:`~pylot.utils.Transform`), optional): List of
            waypoint transforms.
        target_speeds (list(float)), optional): List of target speeds.
    """
    def __init__(self, timestamp, waypoints, target_speeds=None):
        super(WaypointsMessage, self).__init__(timestamp, None)
        self.waypoints = waypoints
        if target_speeds is not None:
            assert len(target_speeds) == len(waypoints), \
                "Length of target speeds must match length of waypoints"
        self.target_speeds = target_speeds

    def __str__(self):
        return \
            'WaypointMessage(timestamp: {}, waypoints: {}, target speeds: {}'\
            .format(self.timestamp, self.waypoints, self.target_speeds)


class BehaviorMessage(erdos.Message):
    def __init__(self,
                 timestamp,
                 target_lane_id,
                 target_speed,
                 target_deadline,
                 target_leading_vehicle_id=None):
        super(BehaviorMessage, self).__init__(timestamp, None)
        self.target_lane_id = target_lane_id
        self.target_speed = target_speed
        self.target_deadline = target_deadline
        self.target_leading_vehicle_id = target_leading_vehicle_id

    def __str__(self):
        return 'BehaviorMessage(timestamp: {}, target_lane_id: {}, '\
            'target_speed: {}, target_deadline: {}, '\
            'target_leading_vehicle_id: {})'.format(
                self.timestamp, self.target_lane_id, self.target_speed,
                self.target_deadline, self.target_leading_vehicle_id)
