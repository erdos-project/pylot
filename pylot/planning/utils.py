import enum

from pylot.control.utils import get_angle


class BehaviorPlannerState(enum.Enum):
    """States in which the FSM behavior planner can be in."""
    READY = 1
    KEEP_LANE = 2
    PREPARE_LANE_CHANGE_LEFT = 3
    LANGE_CHANGE_LEFT = 4
    PREPARE_LANE_CHANGE_RIGHT = 5
    LANE_CHANGE_RIGHT = 6


def get_waypoint_vector_and_angle(wp_transform, ego_transform):
    """Computes the 2D vector between the two transforms and the angle."""
    wp_vector, wp_mag = wp_transform.location.get_vector_and_magnitude(
        ego_transform.location)
    if wp_mag > 0:
        wp_angle = get_angle(
            wp_vector,
            [ego_transform.forward_vector.x, ego_transform.forward_vector.y])
    else:
        wp_angle = 0

    return wp_vector, wp_angle
