from absl import flags

# PID controller parameters.
flags.DEFINE_float('pid_p', 1.0, 'PID p parameter')
flags.DEFINE_float('pid_d', 0.0, 'PID d parameter')
flags.DEFINE_float('pid_i', 0.05, 'PID i parameter')
flags.DEFINE_integer(
    'min_pid_speed_waypoint_distance', 5,
    'Waypoint used for target speed must be at least this many meters away')
flags.DEFINE_integer(
    'min_pid_steer_waypoint_distance', 5,
    'Waypoint used for steering must be at least this many meters away')
# Agent stopping configs.
flags.DEFINE_bool('stop_for_traffic_lights', True,
                  'True to enable traffic light stopping')
flags.DEFINE_bool('stop_for_people', True, 'True to enable person stopping')
flags.DEFINE_bool('stop_for_vehicles', True, 'True to enable vehicle stopping')
flags.DEFINE_bool('stop_at_uncontrolled_junctions', False,
                  'True to enable stopping at uncontroller junctions')
# Agent stopping parameters.
flags.DEFINE_integer(
    'traffic_light_min_distance', 5,
    'Min distance for a traffic light to be considered for stopping [m]')
flags.DEFINE_integer(
    'traffic_light_max_distance', 20,
    'Max distance for a traffic light to be considered for stopping [m]')
flags.DEFINE_float(
    'traffic_light_max_angle', 0.6,
    'Angle threshold for a traffic light to be considered for stopping [rad]')
flags.DEFINE_integer(
    'vehicle_max_distance', 18,
    'Max distance for a vehicle to be considered for stopping [m]')
flags.DEFINE_float(
    'vehicle_max_angle', 0.4,
    'Angle threshold for a vehicle to be considered for stopping [rad]')
flags.DEFINE_integer(
    'person_distance_hit_zone', 35,
    'Distance threhsold for a person to be in the hit zone [m]')
flags.DEFINE_float('person_angle_hit_zone', 0.15,
                   'Angle threshold for a person to be in the hit zone [rad]')
flags.DEFINE_integer(
    'person_distance_emergency_zone', 15,
    'Distance threshold for a person to be in the emergency zone [m]')
flags.DEFINE_float(
    'person_angle_emergency_zone', 0.5,
    'Angle threshold for a person to be in the emergency zone [rad]')
# Steering control parameters
flags.DEFINE_float('throttle_max', 1.0, 'Maximum throttle [0, 1]')
flags.DEFINE_float('steer_gain', 0.7, 'Gain on computed steering angle')
flags.DEFINE_float('brake_max', 1.0, 'Maximum brake [0, 1]')
flags.DEFINE_float('coast_factor', 1.75, 'Factor to control coasting')
