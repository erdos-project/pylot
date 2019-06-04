from erdos.message import Message


class WaypointsMessage(Message):
    """ This class represents a message to be used to send waypoints."""

    def __init__(self, timestamp, waypoints=None, target_speed=0, wp_angle=0,
                 wp_vector=0, wp_angle_speed=0, wp_vector_speed=0,
                 stream_name='default'):
        super(WaypointsMessage, self).__init__(None, timestamp, stream_name)
        # Values used in Carla 0.8.4.
        self.wp_angle = wp_angle
        self.wp_vector = wp_vector
        self.wp_angle_speed = wp_angle_speed
        self.wp_vector_speed = wp_vector_speed
        # Value used in Carla 0.9.x
        self.target_speed = target_speed
        self.waypoints = waypoints

    def __str__(self):
        return 'timestamp: {}, wp_angle: {}, wp_vector: {}, wp_angle_speed: {}, wp_vector_speed: {}, waypoints: {}'.format(
            self.timestamp, self.wp_angle, self.wp_vector, self.wp_angle_speed, self.wp_vector_speed, self.waypoints)
