from fast_rrt_star import *
from rrg import *
from rrt import *
from datetime import datetime
from random_data_generator import *
import time
import numpy as np
from plotter import *

space_region = ((-7.5, 58), (7.5, 74))
# obstacle = get_random_obstacles(1, space_region)
# start = get_random_initial_state(space_region, obstacle)
# target = get_random_target_state(space_region, obstacle)

obstacle = {0: ((-1, 64.0), (3, 3)), 1: ((-8, 50), (5, 10))}
start = (0, 58)
target = ((-0.5, 73), (1, 1))
d_threshold = 1
n_samples = 500
granularity = 0.2
n_iter = 1

rrt_star_times = []
rrt_star_costs = []

rrt_times = []
rrt_costs = []

for i in range(n_iter):
    t = datetime.now()
    rrt_star, rrt_star_final_state, rrt_star_cost = apply_rrt_star(state_space=space_region,
                                                    starting_state=start,
                                                    target_space=target,
                                                    obstacle_map=obstacle,
                                                    d_threshold=d_threshold,
                                                    n_samples=n_samples,
                                                    granularity=granularity)
    rrt_star_costs.append(rrt_star_cost)
    rrt_star_times.append(datetime.now() - t)

    t = datetime.now()
    rrt, rrt_final_state, rrt_cost = apply_rrt(state_space=space_region,
                                     starting_state=start,
                                     target_space=target,
                                     obstacle_map=obstacle,
                                     d_threshold=d_threshold,
                                     n_samples=n_samples,
                                     granularity=granularity)
    rrt_costs.append(rrt_cost)
    rrt_times.append(datetime.now() - t)

    # plot the trees
    # Plotting takes some time as it has to iterate and create each edge. Could not find a better solution
    id = int(time.time())
    plot(rrt_star, start, rrt_star_final_state, target, obstacle, 'RRT*', id)
    plot(rrt, start, rrt_final_state, target, obstacle, 'RRT', id)

print("==============================")
print("RRTStar Mean Time: ", np.mean(rrt_star_times))
print("RRTStar Max Time: ", np.max(rrt_star_times))
print("RRTStar Min Time: ", np.min(rrt_star_times))

print("RRTStar Mean Cost: ", np.mean([x for x in rrt_star_costs if x is not None]))
print("RRTStar Max Cost: ", np.max([x for x in rrt_star_costs if x is not None]))
print("RRTStar Min Cost: ", np.min([x for x in rrt_star_costs if x is not None]))

print("RRTStar Failre Rate: ", len([x for x in rrt_star_costs if x is None])/len(rrt_star_costs))

print("==============================")
print("RRT Mean Time: ", np.mean(rrt_times))
print("RRT Max Time: ", np.max(rrt_times))
print("RRT Min Time: ", np.min(rrt_times))

print("RRT Mean Cost: ", np.mean([x for x in rrt_costs if x is not None]))
print("RRT Max Cost: ", np.max([x for x in rrt_costs if x is not None]))
print("RRT Min Cost: ", np.min([x for x in rrt_costs if x is not None]))

print("RRT Failre Rate: ", len([x for x in rrt_costs if x is None])/len(rrt_costs))
