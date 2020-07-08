from absl import flags

flags.DEFINE_float('step_size_hybrid_astar', 3.0, 'Sampling distance [m]')
flags.DEFINE_integer('max_iterations_hybrid_astar', 2000,
                     'Maximum number of iterations')
flags.DEFINE_float('completion_threshold', 1.0,
                   'Threshold to end position [m]')
flags.DEFINE_float('angle_completion_threshold', 100.0,
                   ' Threshold to end yaw [rad]')
flags.DEFINE_float('rad_step', 0.5, 'Turning sampling discretization [rad]')
flags.DEFINE_float('rad_upper_range', 4.0,
                   'Maximum turning angle to the right [rad]')
flags.DEFINE_float('rad_lower_range', 4.0,
                   'Maximum turning angle to the left [rad]')
flags.DEFINE_float('obstacle_clearance_hybrid_astar', 1.0,
                   'Obstacle clearance threshold [m]')
flags.DEFINE_float('lane_width_hybrid_astar', 6, 'Road width')
flags.DEFINE_float('radius', 6.0, 'Minimum turning radius of the car [m]')
flags.DEFINE_float('car_length', 4.8, 'Length of car [m]')
flags.DEFINE_float('car_width', 1.8, 'Width of car [m]')
