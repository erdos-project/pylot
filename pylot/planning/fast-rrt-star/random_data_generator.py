import numpy as np
from commons import *

obstacle_range_factor = (10, 10)
target_range_factor = (2, 2)


def get_random_obstacle(space_region):
    global obstacle_range_factor
    space_origin, space_range = space_region
    space_range = np.asarray(space_range)
    obstacle_range_factor = np.asarray(obstacle_range_factor)
    obstacle_range = np.random.rand(2) * obstacle_range_factor + obstacle_range_factor
    obstacle_origin = np.random.rand(2) * (space_range - obstacle_range)

    return (tuple(obstacle_origin), tuple(obstacle_range))


def get_random_obstacles(n, space_region):
    obstacles = {}
    for i in range(n):
        obstacles[i] = get_random_obstacle(space_region)
    return obstacles


def get_random_space_region(max_space_range):
    space_range = np.random.rand(2) * np.asarray(max_space_range)
    return ((0, 0), tuple(space_range))


def get_random_target_state(space_region, obstacle_map):
    _, space_range = space_region
    space_range = np.asarray(space_range)
    target_range = np.random.rand(2) * np.asarray(target_range_factor) + target_range_factor
    target_origin = np.random.rand(2) * (space_range - target_range)
    for obstacle in obstacle_map.values():

        # checking only 2 corners.
        if lies_in_area(target_origin, obstacle):
            return get_random_target_state(space_region, obstacle_map)
        if lies_in_area(target_origin + target_range, obstacle):
            return get_random_target_state(space_region, obstacle_map)

    return (tuple(target_origin), tuple(target_range))


def get_random_initial_state(space_region, obstacle_map):
    _, space_range = space_region
    space_range = np.asarray(space_range)
    initial_state = np.random.rand(2) * space_range
    for obstacle in obstacle_map.values():
        if lies_in_area(initial_state, obstacle):
            return get_random_initial_state(space_region, obstacle_map)

    return tuple(initial_state)
