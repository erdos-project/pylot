import erdos


class WaypointsMessage(erdos.Message):
    """Message class to be used to send waypoints.

    Optionally can also send a target speed for each waypoint.

    Args:
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of
            the message.
        waypoints (:py:class:`~pylot.planning.Waypoints`): Waypoints.
    """
    def __init__(self,
                 timestamp: erdos.Timestamp,
                 waypoints,
                 agent_state=None):
        super(WaypointsMessage, self).__init__(timestamp, None)
        self.waypoints = waypoints
        self.agent_state = agent_state

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'WaypointMessage(timestamp: {}, waypoints: {}, '\
            'agent_state: {}'.format(self.timestamp, self.waypoints,
                                     self.agent_state)


class BehaviorMessage(erdos.Message):
    def __init__(self,
                 timestamp: erdos.Timestamp,
                 target_lane_id: int,
                 target_speed: float,
                 target_deadline: float,
                 target_leading_vehicle_id: int = None):
        super(BehaviorMessage, self).__init__(timestamp, None)
        self.target_lane_id = target_lane_id
        self.target_speed = target_speed
        self.target_deadline = target_deadline
        self.target_leading_vehicle_id = target_leading_vehicle_id

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'BehaviorMessage(timestamp: {}, target_lane_id: {}, '\
            'target_speed: {}, target_deadline: {}, '\
            'target_leading_vehicle_id: {})'.format(
                self.timestamp, self.target_lane_id, self.target_speed,
                self.target_deadline, self.target_leading_vehicle_id)
