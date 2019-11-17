import numpy as np
import networkx as nx
from commons import *
from tqdm import tqdm


def apply_rrt(state_space, starting_state, target_space, obstacle_map, granularity=0.1, d_threshold=0.5,
              n_samples=1000, find_optimal=True):
    tree = nx.DiGraph()
    tree.add_node(starting_state)

    final_state = None

    min_cost = None

    for i in range(n_samples):
        # select node to expand
        m_g, random_point = select_node_to_expand(tree, state_space)

        # sample a new point
        m_new = sample_new_point(m_g, random_point, d_threshold)

        # check if m_new lies in space_region
        if not lies_in_area(m_new, state_space):
            continue

        # check if path between(m_g,m_new) defined by motion-model is collision free
        if not is_collision_free(m_g, m_new, obstacle_map, granularity):
            continue

        # if path is free, add new node to tree
        tree.add_weighted_edges_from([(m_g, m_new, cartesian_distance(m_g, m_new))])
        if lies_in_area(m_new, target_space):
            if final_state is None:
                final_state = m_new
                min_cost = nx.dijkstra_path_length(tree, starting_state, m_new)
                if not find_optimal:
                    break
            else:
                # if new final state has shorter cost, set it as final state
                cost = nx.dijkstra_path_length(tree, starting_state, m_new)
                if cost < min_cost:
                    final_state = m_new
                    min_cost = cost

    if final_state is None:
        min_cost = None

    return tree, final_state, min_cost
