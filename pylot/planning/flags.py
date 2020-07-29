from absl import flags

import pylot.planning.frenet_optimal_trajectory.flags
import pylot.planning.hybrid_astar.flags
import pylot.planning.rrt_star.flags  # noqa: F401

flags.DEFINE_float(
    'target_speed', 10.0,
    'Target speed in m/s , could be controlled by the speed limit')
flags.DEFINE_float('obstacle_radius', 1.0, 'Default obstacle radius [m]')
flags.DEFINE_integer('num_waypoints_ahead', 60,
                     'Number of ahead waypoints to plan for')
flags.DEFINE_integer('num_waypoints_behind', 30,
                     'Number of waypoints behind to account for')
flags.DEFINE_float('obstacle_filtering_distance', 1.0,
                   'distance at which to sample obstacles [m]')
