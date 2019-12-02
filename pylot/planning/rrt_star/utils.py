"""
Author: Edward Fang
Email: edward.fang@berkeley.edu
This code is adapted from: https://github.com/dixantmittal/fast-rrt-star
"""
import numpy as np

volume_of_unit_ball = {
    1: 2,
    2: 3.142,
    3: 4.189,
}

collision_cache = {}

free_space_cache = {}


def start_target_to_space(start, target, length, width):
    """
    Create a state space for RRT* search given a start, target and length / width buffer.

    :param start: tuple
        (x, y)
    :param target: tuple
        (x, y), (range_x, range_y)
    :param length: float
    :param width: float
    :return: tuple
        (origin_x, origin_y), (range_x, range_y)
    """
    origin = (min(start[0], target[0][0] + length / 2) - length, min(start[1], target[0][1] + width / 2) - width)
    bounds = (max(start[0], target[0][0] + length / 2) - origin[0] + width,
              max(start[1], target[0][1] + width / 2) - origin[1] + width)
    return origin, bounds


def select_node_to_expand(tree, space_range):
    """
    Return a random node from tree to expand for RRT* given space_range.
    :param tree: nx graph
        the RRT* tree
    :param space_range: tuple
        (origin_x, origin_y), (range_x, range_y)
    :return: tuple
        (closest node to random point, random point)
    """
    space_range = np.asarray(space_range)
    random_point = np.random.rand(space_range.shape[1]) * (space_range[1]) + space_range[0]
    nodes = list(tree.nodes())
    d = cartesian_distance(nodes, random_point)
    return nodes[np.asscalar(np.argmin(d))], random_point


def sample_new_point(m_g, random_point, d_threshold):
    """
    Return a randomly sampled point d_threshold away from a node m_g.

    :param m_g: nx node
    :param random_point: tuple
        (x, y)
    :param d_threshold: float
    :return:
        if random_point is greater than d_threshold away from node m_g, rescale it
        otherwise return the random_point
    """
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
    """
    Return the total free area in space region, accounting for obstacles.

    :param space_region: tuple
        (origin_x, origin_y), (range_x, range_y)
    :param obstacle_map: dict
        id: (x, y), (range_x, range_y)
    :return: float
    """
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
    """
    Return whether a point lies in given area.

    :param point: tuple
        (x, y)
    :param area: tuple
        (origin_x, origin_y), (range_x, range_y)
    :return: bool
    """
    frame, _range = area
    frame = np.array(frame)
    point = np.array(point)

    diff = point - frame

    return np.all(diff <= _range) and np.all(diff >= 0)


def dist_to_target(point, area):
    """
    Return the distance from point to an area.

    :param point: tuple
        (x, y)
    :param area: tuple
        (origin_x, origin_y), (range_x, range_y)
    :return: float
    """
    frame, _range = area
    return np.linalg.norm(np.array(point) - np.array(frame))


def nearest_neighbours(nodes, center, radius):
    """
    Return the nearest neighbors of center given nodes and a search radius.

    :param nodes: list
        list of nx nodes
    :param center: tuple
        (x, y)
    :param radius: float
    :return: tuple
        a tuple of nearest node tuples
    """
    nodes = np.asarray(nodes)
    d = cartesian_distance(nodes, center)
    nearest_nodes = nodes[d < radius]
    return tuple(map(tuple, nearest_nodes))


def cartesian_distance(x, y):
    """
    Return the cartesian distance between two points x, y.

    :param x: tuple
        (x0, x1)
    :param y: tuple
        (y0, y1)
    :return: float
    """
    x = np.array(x)
    y = np.array(y)

    if x.ndim == 1:
        x = x.reshape(1, -1)

    if y.ndim == 1:
        y = y.reshape(1, -1)

    dist = np.sqrt(np.sum((y - x) ** 2, axis=1))
    return dist


def is_obstacle_space(point, obstacle_map):
    """
    Return if given point intersects an obstacle defined by obstacle_map.

    :param point: tuple
        (x, y)
    :param obstacle_map: dict
        id: (x, y), (range_x, range_y)
    :return: bool
        whether point intersects an obstacle
    """
    if obstacle_map is None:
        return False

    for key in obstacle_map.keys():
        if lies_in_area(point, obstacle_map[key]):
            return True
    return False


def is_collision_free(x, y, obstacle_map, granularity):
    """
    Determine if a path from x to y is collision free given an obstacle map and granularity.

    :param x: tuple
        (x0, x1)
    :param y:
        (y0, y1)
    :param obstacle_map: dict
        id: (x, y), (range_x, range_y)
    :param granularity: float
    :return: bool
        whether path from x to y is collision free
    """
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
