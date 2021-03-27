import enum
import math


class BehaviorPlannerState(enum.Enum):
    """States in which the FSM behavior planner can be in."""
    FOLLOW_WAYPOINTS = 0
    READY = 1
    KEEP_LANE = 2
    PREPARE_LANE_CHANGE_LEFT = 3
    LANGE_CHANGE_LEFT = 4
    PREPARE_LANE_CHANGE_RIGHT = 5
    LANE_CHANGE_RIGHT = 6
    OVERTAKE = 7


def compute_person_speed_factor(ego_location_2d, person_location_2d, wp_vector,
                                flags, logger) -> float:
    speed_factor_p = 1
    p_vector = person_location_2d - ego_location_2d
    p_dist = person_location_2d.l2_distance(ego_location_2d)
    p_angle = p_vector.get_angle(wp_vector)
    logger.debug('Person vector {}; dist {}; angle {}'.format(
        p_vector, p_dist, p_angle))
    # Maximum braking is applied if the person is in the emergency
    # hit zone. Otherwise, gradual braking is applied if the person
    # is in the hit zone.
    if (math.fabs(p_angle) < flags.person_angle_hit_zone
            and p_dist < flags.person_distance_hit_zone):
        # Person is in the hit zone.
        speed_factor_p = min(
            speed_factor_p,
            p_dist / (flags.coast_factor * flags.person_distance_hit_zone))
    if (math.fabs(p_angle) < flags.person_angle_emergency_zone
            and p_dist < flags.person_distance_emergency_zone):
        # Person is in emergency hit zone.
        speed_factor_p = 0
    return speed_factor_p


def compute_vehicle_speed_factor(ego_location_2d, vehicle_location_2d,
                                 wp_vector, flags, logger) -> float:
    speed_factor_v = 1
    v_vector = vehicle_location_2d - ego_location_2d
    v_dist = vehicle_location_2d.l2_distance(ego_location_2d)
    v_angle = v_vector.get_angle(wp_vector)
    logger.debug('Vehicle vector {}; dist {}; angle {}'.format(
        v_vector, v_dist, v_angle))
    min_angle = -0.5 * flags.vehicle_max_angle / flags.coast_factor
    if (min_angle < v_angle < flags.vehicle_max_angle
            and v_dist < flags.vehicle_max_distance):
        # The vehicle is within the angle limit, and nearby.
        speed_factor_v = min(
            speed_factor_v,
            v_dist / (flags.coast_factor * flags.vehicle_max_distance))

    if (min_angle < v_angle < flags.vehicle_max_angle / flags.coast_factor
            and v_dist < flags.vehicle_max_distance * flags.coast_factor):
        # The vehicle is a bit far away, but it's on ego vehicle's path.
        speed_factor_v = min(
            speed_factor_v,
            v_dist / (flags.coast_factor * flags.vehicle_max_distance))

    min_nearby_angle = -0.5 * flags.vehicle_max_angle * flags.coast_factor
    if (min_nearby_angle < v_angle <
            flags.vehicle_max_angle * flags.coast_factor
            and v_dist < flags.vehicle_max_distance / flags.coast_factor):
        # The vehicle is very close; the angle can be higher.
        speed_factor_v = 0
    return speed_factor_v
