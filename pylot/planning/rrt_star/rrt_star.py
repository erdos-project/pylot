import networkx as nx
from pylot.planning.rrt_star.utils import *


def apply_rrt_star(state_space, starting_state, target_space, obstacle_map, n_samples=5000, granularity=0.1,
                   d_threshold=0.5):
    tree = nx.DiGraph()
    tree.add_node(starting_state)

    space_dim = len(starting_state)

    final_state = None
    closest_state = None
    min_dist = np.inf

    # cost for each vertex
    cost = {starting_state: 0}

    gamma = 1 + np.power(2, space_dim) * (1 + 1.0 / space_dim) * get_free_area(state_space, obstacle_map)

    for i in range(n_samples):

        # update cost cache
        if i % 500 == 0:
            cost = nx.single_source_dijkstra_path_length(tree, starting_state)

        # select node to expand
        m_g, random_point = select_node_to_expand(tree, state_space)

        # sample a new point
        m_new = sample_new_point(m_g, random_point, d_threshold)

        # check if m_new lies in space_region
        if not lies_in_area(m_new, state_space):
            continue

        # if m_new is not collision free, sample any other point
        if not is_collision_free(m_g, m_new, obstacle_map, granularity):
            continue

        # find k nearest neighbours
        radius = np.minimum(np.power(gamma / volume_of_unit_ball[space_dim] * np.log(i + 1) / (i + 1),
                                     1 / space_dim), d_threshold)
        m_near = nearest_neighbours(list(tree.nodes), m_new, radius=radius)

        min_cost = m_g
        d_min_cost = cartesian_distance(m_g, m_new)

        # look for shortest cost path to m_new
        for m_g in m_near:

            # find the possible cost for m_new through m_g
            d = cartesian_distance(m_g, m_new)
            c = cost[m_g] + d

            # if cost is less than current lowest cost, that means m_new to m_g could be a potential link
            if c < cost[min_cost] + d_min_cost:

                # check if path between(m_g,m_new) defined by motion-model is collision free
                if not is_collision_free(m_g, m_new, obstacle_map, granularity):
                    continue

                # if path is free, update the minimum distance
                min_cost = m_g
                d_min_cost = d

        tree.add_weighted_edges_from([(min_cost, m_new, d_min_cost)])
        cost[m_new] = cost[min_cost] + d_min_cost

        # update m_new's neighbours for paths through m_new
        for m_g in m_near:

            # find the cost for m_g through m_new
            d = cartesian_distance(m_new, m_g)
            c = cost[m_new] + d

            # if cost is less than current cost, that means m_new to m_g could be a potential link
            if c < cost[m_g]:
                # check if path between(m_g,m_new) is collision free
                is_free = is_collision_free(m_g, m_new, obstacle_map, granularity)

                # if path is free, update the links
                if is_free:
                    tree.remove_edge(list(tree.predecessors(m_g))[0], m_g)
                    tree.add_weighted_edges_from([(m_new, m_g, d)])
                    cost[m_g] = c

        # if target is reached, return the tree and final state
        if lies_in_area(m_new, target_space):
            cost = nx.single_source_dijkstra_path_length(tree, starting_state)
            if final_state is None:
                final_state = m_new
            elif cost[m_new] < cost[final_state]:
                final_state = m_new

        # keep track of best in case of failure
        cur_dist = dist_to_target(m_new, target_space)
        if cur_dist < min_dist:
            closest_state = m_new
            min_dist = cur_dist


    if final_state is None:
        final_state = closest_state
        final_cost = None
    else:
        final_cost = cost[final_state]
    return tree, final_state, final_cost
