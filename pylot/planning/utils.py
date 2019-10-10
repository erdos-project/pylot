import enum
import math
import numpy as np

from pylot.control.utils import get_angle


class BehaviorPlannerState(enum.Enum):
    """ States in which the FSM behavior planner can be in."""
    READY = 1
    KEEP_LANE = 2
    PREPARE_LANE_CHANGE_LEFT = 3
    LANGE_CHANGE_LEFT = 4
    PREPARE_LANE_CHANGE_RIGHT = 5
    LANE_CHANGE_RIGHT = 6


def get_xy_vector_dist(loc1, loc2):
    vec = np.array([loc1.x, loc1.y] - np.array([loc2.x, loc2.y]))
    dist = math.sqrt(vec[0]**2 + vec[1]**2)
    if abs(dist) < 0.00001:
        return vec, dist
    else:
        return vec / dist, dist


def get_waypoint_vector_and_angle(wp_transform, ego_transform):
    wp_vector, wp_mag = get_xy_vector_dist(
        wp_transform.location,
        ego_transform.location)
    if wp_mag > 0:
        wp_angle = get_angle(
            wp_vector,
            [ego_transform.orientation.x, ego_transform.orientation.y])
    else:
        wp_angle = 0

    return wp_vector, wp_angle


def get_distance(loc1, loc2):
    """ Computes the Euclidian distance between two 2D points."""
    x_diff = loc1.x - loc2.x
    y_diff = loc1.y - loc2.y
    return math.sqrt(x_diff**2 + y_diff**2)
