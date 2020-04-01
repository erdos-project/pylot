from absl import flags

import pylot.planning.frenet_optimal_trajectory.flags

flags.DEFINE_integer(
    'target_speed', 10,
    'Target speed in m/s , could be controlled by the speed limit')
