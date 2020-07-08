from absl import flags

flags.DEFINE_float('step_size', 0.5, 'Sampling distance [m]')
flags.DEFINE_integer('max_iterations', 2000, 'Maximum number of iterations')
flags.DEFINE_float('end_dist_threshold', 2.0, 'Completion threshold [m]')
flags.DEFINE_float('obstacle_clearance_rrt', 0.5,
                   'Minimum clearance radius [m]')
flags.DEFINE_float('lane_width', 3.0, 'Width of the lane [m]')
