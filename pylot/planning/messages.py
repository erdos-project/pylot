import erdust


class WaypointsMessage(erdust.Message):
    """ This class represents a message to be used to send waypoints."""

    def __init__(self, timestamp, waypoints=None, target_speed=0, wp_angle=0,
                 wp_vector=0, wp_angle_speed=0, wp_vector_speed=0):
        super(WaypointsMessage, self).__init__(timestamp, None)
        self.wp_angle = wp_angle
        self.wp_vector = wp_vector
        self.wp_angle_speed = wp_angle_speed
        self.wp_vector_speed = wp_vector_speed
        self.target_speed = target_speed
        self.waypoints = waypoints

    def __str__(self):
        return 'WaypointMessage(timestamp: {}, wp_angle: {}, wp_vector: {}, '\
            'wp_angle_speed: {}, wp_vector_speed: {}, waypoints: {}'.format(
                self.timestamp, self.wp_angle, self.wp_vector,
                self.wp_angle_speed, self.wp_vector_speed, self.waypoints)


class BehaviorMessage(erdust.Message):

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
