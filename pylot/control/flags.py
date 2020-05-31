from absl import flags

# PID controller parameters.
flags.DEFINE_float('pid_p', 0.25, 'PID p parameter')
flags.DEFINE_float('pid_i', 0.20, 'PID i parameter')
flags.DEFINE_float('pid_d', 0.0, 'PID d parameter')
flags.DEFINE_integer('pid_speed_wp', 10, 'PID speed waypoint number')
flags.DEFINE_integer('pid_steer_wp', 5, 'PID steer waypoint number')
# Agent stopping configs.
flags.DEFINE_bool('stop_for_traffic_lights', True,
                  'True to enable traffic light stopping')
flags.DEFINE_bool('stop_for_people', True, 'True to enable person stopping')
flags.DEFINE_bool('stop_for_vehicles', True, 'True to enable vehicle stopping')
# Agent stopping parameters.
flags.DEFINE_integer('traffic_light_min_dist_thres', 5,
                     'Min distance threshold traffic light')
flags.DEFINE_integer('traffic_light_max_dist_thres', 20,
                     'Max distance threshold traffic light')
flags.DEFINE_float('traffic_light_angle_thres', 0.5,
                   'Traffic light angle threshold')
flags.DEFINE_integer('vehicle_distance_thres', 15,
                     'Vehicle distance threshold')
flags.DEFINE_float('vehicle_angle_thres', 0.4, 'Vehicle angle threshold')
flags.DEFINE_float('person_angle_hit_thres', 0.15,
                   'Person hit zone angle threshold')
flags.DEFINE_integer('person_distance_emergency_thres', 12,
                     'Person emergency zone distance threshold')
flags.DEFINE_float('person_angle_emergency_thres', 0.5,
                   'Person emergency zone angle threshold')
flags.DEFINE_integer('person_distance_hit_thres', 35,
                     'Person hit zone distance threshold')
# Steering control parameters
flags.DEFINE_float('throttle_max', 0.75, 'Max throttle')
flags.DEFINE_float('steer_gain', 0.7, 'Gain on computed steering angle')
flags.DEFINE_float('brake_strength', 1,
                   'Strength for applying brake; between 0 and 1')
flags.DEFINE_integer('coast_factor', 2, 'Factor to control coasting')
