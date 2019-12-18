from absl import flags

# PID controller parameters.
flags.DEFINE_float('pid_p', 0.25, 'PID p parameter')
flags.DEFINE_float('pid_i', 0.20, 'PID i parameter')
flags.DEFINE_float('pid_d', 0.0, 'PID d parameter')
# Agent stopping configs.
flags.DEFINE_bool('stop_for_traffic_lights', True,
                  'True to enable traffic light stopping')
flags.DEFINE_bool('stop_for_pedestrians', True,
                  'True to enable pedestrian stopping')
flags.DEFINE_bool('stop_for_vehicles', True,
                  'True to enable vehicle stopping')
# Agent stopping parameters.
flags.DEFINE_integer('traffic_light_min_dist_thres', 5,
                     'Min distance threshold traffic light')
flags.DEFINE_integer('traffic_light_max_dist_thres', 20,
                     'Max distance threshold traffic light')
flags.DEFINE_float('traffic_light_angle_thres', 0.5,
                   'Traffic light angle threshold')
flags.DEFINE_integer('vehicle_distance_thres', 15,
                     'Vehicle distance threshold')
flags.DEFINE_float('vehicle_angle_thres', 0.4,
                   'Vehicle angle threshold')
flags.DEFINE_float('pedestrian_angle_hit_thres', 0.15,
                   'Pedestrian hit zone angle threshold')
flags.DEFINE_integer('pedestrian_distance_emergency_thres', 12,
                     'Pedestrian emergency zone distance threshold')
flags.DEFINE_float('pedestrian_angle_emergency_thres', 0.5,
                   'Pedestrian emergency zone angle threshold')
flags.DEFINE_integer('pedestrian_distance_hit_thres', 35,
                     'Pedestrian hit zone distance threshold')
# Steering control parameters
flags.DEFINE_float('default_throttle', 0.0, 'Default throttle')
flags.DEFINE_float('throttle_max', 0.75, 'Max throttle')
flags.DEFINE_integer(
    'target_speed', 10,
    'Target speed in m/s , could be controlled by the speed limit')
flags.DEFINE_float('steer_gain', 0.7, 'Gain on computed steering angle')
flags.DEFINE_float('brake_strength', 1,
                   'Strength for applying brake; between 0 and 1')
flags.DEFINE_integer('coast_factor', 2, 'Factor to control coasting')
