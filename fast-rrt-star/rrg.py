import numpy as np
import networkx as nx
from commons import *
from tqdm import tqdm


def apply_rrg(state_space, starting_state, target_region, obstacle_map, n_samples=1000, granularity=0.1,
              d_threshold=0.5):
    tree = nx.DiGraph()
    tree.add_node(starting_state)

    space_dim = len(starting_state)

    final_state = None

    min_cost = None

    gamma = 1 + np.power(2, space_dim) * (1 + 1.0 / space_dim) * get_free_area(state_space, obstacle_map)

    for i in tqdm(range(n_samples)):
        # select node to expand
        m_g, random_point = select_node_to_expand(tree, state_space)

        # sample a new point
        m_new = sample_new_point(m_g, random_point, d_threshold)

        # check if m_new lies in space_region
        if not lies_in_area(m_new, state_space):
            continue

        # find k nearest neighbours
        radius = np.minimum(np.power(gamma / volume_of_unit_ball[space_dim] * np.log(i + 1) / (i + 1),
                                     1 / space_dim), d_threshold)
        m_near = nearest_neighbours(list(tree.nodes), m_new, radius)

        for m_g in m_near:

            # check if path between(m_g,m_new) defined by motion-model is collision free
            if not is_collision_free(m_g, m_new, obstacle_map, granularity):
                continue
            # if path is free, add new node to tree
            tree.add_weighted_edges_from([(m_g, m_new, cartesian_distance(m_g, m_new))])

        # if target is reached, update the final state
        if lies_in_area(m_new, target_region):
            print('Target reached at i:', i)
            if final_state is None:
                final_state = m_new
                min_cost = nx.dijkstra_path_length(tree, starting_state, m_new)
            else:
                # if new final state has shorter cost, set it as final state
                cost = nx.dijkstra_path_length(tree, starting_state, m_new)
                if cost < min_cost:
                    final_state = m_new
                    min_cost = cost

    if final_state is None:
        print("Target not reached.")
    return tree, final_state
