import numpy as np

volume_of_unit_ball = {
    1: 2,
    2: 3.142,
    3: 4.189,
    4: 4.935,
    5: 5.264,
    6: 5.168,
    7: 4.725,
    8: 4.059,
    9: 3.299,
    10: 2.550
}

collision_cache = {}

free_space_cache = {}


def select_node_to_expand(tree, space_range):
    space_range = np.asarray(space_range)
    random_point = np.random.rand(space_range.shape[1]) * (space_range[1] - space_range[0]) + space_range[0]
    nodes = list(tree.nodes())
    d = cartesian_distance(nodes, random_point)
    return nodes[np.asscalar(np.argmin(d))], random_point


def sample_new_point(m_g, random_point, d_threshold):
    m_g = np.asarray(m_g)
    random_point = np.asarray(random_point)

    # get distance to random point
    d = cartesian_distance(m_g, random_point)
    if d <= d_threshold:
        return tuple(random_point)

    # rescale the point
    m_new = m_g + d_threshold * (random_point - m_g) / d
    return tuple(m_new)


def get_free_area(space_region, obstacle_map):
    _, space_range = space_region
    l, b = space_range
    space_area = l * b

    obstacle_area = 0
    for obstacle in obstacle_map.values():
        _, obstacle_range = obstacle
        l, b = obstacle_range
        obstacle_area += l * b

    return space_area - obstacle_area


def lies_in_area(point, area):
    frame, _range = area
    frame = np.array(frame)
    point = np.array(point)

    diff = point - frame

    return np.all(diff <= _range) and np.all(diff >= 0)


def dist_to_target(point, area):
    frame, _range = area
    return np.linalg.norm(np.array(point) - np.array(frame))


def nearest_neighbours(nodes, center, radius):
    nodes = np.asarray(nodes)
    d = cartesian_distance(nodes, center)
    nearest_nodes = nodes[d < radius]
    return tuple(map(tuple, nearest_nodes))


def cartesian_distance(x, y):
    x = np.array(x)
    y = np.array(y)

    if x.ndim == 1:
        x = x.reshape(1, -1)

    if y.ndim == 1:
        y = y.reshape(1, -1)

    dist = np.sqrt(np.sum((y - x) ** 2, axis=1))
    return dist


def is_obstacle_space(point, obstacle_map):
    if obstacle_map is None:
        return False

    for key in obstacle_map.keys():
        if lies_in_area(point, obstacle_map[key]):
            return True
    return False


def is_collision_free(x, y, obstacle_map, granularity):
    if collision_cache.get(y, False):
        return False

    if is_obstacle_space(y, obstacle_map):
        collision_cache[y] = True
        return False

    x = np.array(x)
    y = np.array(y)
    d = np.asscalar(cartesian_distance(x, y))
    unit_vector = (y - x) / d
    floor = int(np.floor(d / granularity))

    for i in range(floor):
        _m = x + i * granularity * unit_vector

        if collision_cache.get(tuple(_m), False):
            return False

        # can be skipped as the hit ratio is not that much, so time for cache checking adds up
        if free_space_cache.get(tuple(_m), False):
            continue

        if is_obstacle_space(_m, obstacle_map):
            collision_cache[tuple(_m)] = True
            return False

        free_space_cache[tuple(_m)] = True

    return True
