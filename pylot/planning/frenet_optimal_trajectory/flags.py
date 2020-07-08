from absl import flags

# Frenet Optimal Trajectory Hyperparameters
flags.DEFINE_float('max_speed', 35.0, 'Maximum vehicle speed [m/s]')
flags.DEFINE_float('max_accel', 6.0, 'Maximum vehicle acceleration [m/s^2]')
flags.DEFINE_float('max_curvature', 1.0, 'Maximum curvature speed [1/m]')
flags.DEFINE_float('max_road_width_l', 5.0, 'Maximum left road width [m]')
flags.DEFINE_float('max_road_width_r', 1.0, 'Maximum right road width [m]')
flags.DEFINE_float('d_road_w', 0.25, 'Road width sampling discretization [m]')
flags.DEFINE_float('dt', 0.25, 'Time sampling discretization [s]')
flags.DEFINE_float('maxt', 8.0, 'Max prediction horizon [s]')
flags.DEFINE_float('mint', 2.0, 'Min prediction horizon [s]')
flags.DEFINE_float('d_t_s', 0.25, 'Target speed sampling discretization [m/s]')
flags.DEFINE_float('n_s_sample', 2.0, 'Number speeds to sample')
flags.DEFINE_float('obstacle_clearance_fot', 0.5,
                   'Obstacle clearance threshold [m]')
flags.DEFINE_float('kd', 1.0, 'Deviation cost')
flags.DEFINE_float('kv', 0.1, 'Velocity cost')
flags.DEFINE_float('ka', 0.1, 'Acceleration cost')
flags.DEFINE_float('kj', 0.01, 'Jerk cost')
flags.DEFINE_float('kt', 0.01, 'Time cost')
flags.DEFINE_float('ko', 0.1, 'Obstacle cost')
flags.DEFINE_float('klat', 1.0, 'Lateral cost')
flags.DEFINE_float('klon', 1.0, 'Longitudinal cost')
