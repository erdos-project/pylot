from absl import flags

import pylot.planning.frenet_optimal_trajectory.flags

flags.DEFINE_float(
    'target_speed', 10.0,
    'Target speed in m/s , could be controlled by the speed limit')
flags.DEFINE_float(
    'distance_threshold', 30.0, 'Max distance to consider obstacles[m]')
flags.DEFINE_integer(
    'num_waypoints_ahead', 60, 'Number of ahead waypoints to plan for')
flags.DEFINE_integer(
    'num_waypoints_behind', 30, 'Number of waypoints behind to account for')

