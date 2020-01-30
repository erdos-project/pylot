import enum
import math
from pylot.perception.detection.traffic_light import TrafficLightColor

MAX_VEL = 20  # m/s
VEHICLE_LABELS = {'car', 'bicycle', 'motorcycle', 'bus', 'truck', 'vehicle'}


class BehaviorPlannerState(enum.Enum):
    """States in which the FSM behavior planner can be in."""
    READY = 1
    KEEP_LANE = 2
    PREPARE_LANE_CHANGE_LEFT = 3
    LANGE_CHANGE_LEFT = 4
    PREPARE_LANE_CHANGE_RIGHT = 5
    LANE_CHANGE_RIGHT = 6


def compute_waypoint_vector_and_angle(vehicle_transform, waypoints, wp_num):
    # Get vectors and angles to corresponding speed and steer waypoints.
    next_waypoint = waypoints[min(len(waypoints) - 1, wp_num)]
    wp_vector, _, wp_angle = \
        vehicle_transform.get_vector_magnitude_angle(
            next_waypoint.location)
    return wp_vector, wp_angle


def stop_person(ego_vehicle_location, person_location, wp_vector,
                speed_factor_p, flags):
    """Computes a stopping factor for ego vehicle given a person location.

    Args:
        ego_vehicle_location (:py:class:`~pylot.utils.Location`): Location of
            the ego vehicle in world coordinates.
        person_location (:py:class:`~pylot.utils.Location`): Location of the
            person in world coordinates.
        flags (absl.flags): The flags object.

    Returns:
        :obj:`float`: A stopping factor between 0 and 1 (i.e., no braking).
    """
    speed_factor_p_temp = 1
    p_vector, p_dist = person_location.get_vector_and_magnitude(
        ego_vehicle_location)
    p_angle = p_vector.get_angle(wp_vector)
    # Check if person is on hit zone.
    if (math.fabs(p_angle) < flags.person_angle_hit_thres
            and p_dist < flags.person_distance_hit_thres):
        speed_factor_p_temp = p_dist / (flags.coast_factor *
                                        flags.person_distance_hit_thres)
    # Check if person is near hit zone.
    if (math.fabs(p_angle) < flags.person_angle_emergency_thres
            and p_dist < flags.person_distance_emergency_thres):
        speed_factor_p_temp = 0
    return min(speed_factor_p, speed_factor_p_temp)


def stop_vehicle(ego_vehicle_location, obs_vehicle_location, wp_vector,
                 speed_factor_v, flags):
    """Computes a stopping factor for ego vehicle given a vehicle pos.

    Args:
        ego_vehicle_location (:py:class:`~pylot.utils.Location`): Location of
            the ego vehicle in world coordinates.
        obs_vehicle_location (:py:class:`~pylot.utils.Location`): Location of
            the obstacle vehicle in world coordinates.
        flags (absl.flags): The flags object.

    Returns:
        :obj:`float`: A stopping factor between 0 and 1 (i.e., no braking).
    """
    speed_factor_v_temp = 1
    v_vector, v_dist = obs_vehicle_location.get_vector_and_magnitude(
        ego_vehicle_location)
    v_angle = v_vector.get_angle(wp_vector)

    min_angle = -0.5 * flags.vehicle_angle_thres / flags.coast_factor
    max_angle = flags.vehicle_angle_thres / flags.coast_factor
    max_dist = flags.vehicle_distance_thres * flags.coast_factor
    medium_max_angle = flags.vehicle_angle_thres
    medium_dist = flags.vehicle_distance_thres
    if ((min_angle < v_angle < max_angle and v_dist < max_dist) or
            (min_angle < v_angle < medium_max_angle and v_dist < medium_dist)):
        speed_factor_v_temp = v_dist / (flags.coast_factor *
                                        flags.vehicle_distance_thres)

    min_nearby_angle = -0.5 * flags.vehicle_angle_thres * flags.coast_factor
    max_nearby_angle = flags.vehicle_angle_thres * flags.coast_factor
    nearby_dist = flags.vehicle_distance_thres / flags.coast_factor
    if (min_nearby_angle < v_angle < max_nearby_angle
            and v_dist < nearby_dist):
        speed_factor_v_temp = 0

    return min(speed_factor_v, speed_factor_v_temp)


def stop_traffic_light(ego_vehicle_location, tl_location, tl_state, wp_vector,
                       wp_angle, speed_factor_tl, flags):
    """ Computes a stopping factor for ego vehicle given a traffic light.

    Args:
        ego_vehicle_location (:py:class:`~pylot.utils.Location`): Location of
            the ego vehicle in world coordinates.
        tl_location (:py:class:`~pylot.utils.Location`): Location of the
            traffic light in world coordinates.
        flags (absl.flags): The flags object.

    Returns:
        :obj:`float`: A stopping factor between 0 and 1 (i.e., no braking).
    """
    speed_factor_tl_temp = 1
    if (tl_state == TrafficLightColor.YELLOW
            or tl_state == TrafficLightColor.RED):
        tl_vector, tl_dist = tl_location.get_vector_and_magnitude(
            ego_vehicle_location)
        tl_angle = tl_vector.get_angle(wp_vector)

        if ((0 < tl_angle <
             flags.traffic_light_angle_thres / flags.coast_factor and
             tl_dist < flags.traffic_light_max_dist_thres * flags.coast_factor)
                or (0 < tl_angle < flags.traffic_light_angle_thres
                    and tl_dist < flags.traffic_light_max_dist_thres)
                and math.fabs(wp_angle) < 0.2):

            speed_factor_tl_temp = tl_dist / (
                flags.coast_factor * flags.traffic_light_max_dist_thres)

        if ((0 < tl_angle <
             flags.traffic_light_angle_thres * flags.coast_factor and
             tl_dist < flags.traffic_light_max_dist_thres / flags.coast_factor)
                and math.fabs(wp_angle) < 0.2):
            speed_factor_tl_temp = 0

    return min(speed_factor_tl, speed_factor_tl_temp)


def stop_for_agents(ego_vehicle_location,
                    wp_angle,
                    wp_vector,
                    obstacles,
                    traffic_lights,
                    flags,
                    logger,
                    hd_map=None,
                    timestamp=None):
    speed_factor = 1
    speed_factor_tl = 1
    speed_factor_p = 1
    speed_factor_v = 1

    for obstacle in obstacles:
        if obstacle.label == 'person' and flags.stop_for_people:
            # Only brake for people that are on the road.
            if (not hd_map or hd_map.is_on_lane(obstacle.transform.location)):
                new_speed_factor_p = stop_person(ego_vehicle_location,
                                                 obstacle.transform.location,
                                                 wp_vector, speed_factor_p,
                                                 flags)
                if new_speed_factor_p < speed_factor_p:
                    speed_factor_p = new_speed_factor_p
                    logger.debug(
                        '@{}: person {} reduced speed factor to {}'.format(
                            timestamp, obstacle, speed_factor_p))
        elif obstacle.label in VEHICLE_LABELS and flags.stop_for_vehicles:
            # Only brake for vehicles that are in ego vehicle's lane.
            if (not hd_map or hd_map.are_on_same_lane(
                    ego_vehicle_location, obstacle.transform.location)):
                new_speed_factor_v = stop_vehicle(ego_vehicle_location,
                                                  obstacle.transform.location,
                                                  wp_vector, speed_factor_v,
                                                  flags)
                if new_speed_factor_v < speed_factor_v:
                    speed_factor_v = new_speed_factor_v
                    logger.debug(
                        '@{}: vehicle {} reduced speed factor to {}'.format(
                            timestamp, obstacle, speed_factor_v))
        else:
            logger.debug('@{}: filtering obstacle {}'.format(
                timestamp, obstacle))

    if flags.stop_for_traffic_lights:
        for tl in traffic_lights:
            if ((not hd_map or hd_map.must_obbey_traffic_light(
                    ego_vehicle_location, tl.transform.location))
                    and _is_traffic_light_visible(
                        ego_vehicle_location, tl.transform.location, flags)):
                logger.debug(
                    '@{}: ego vehicle is obbeying traffic light {}'.format(
                        timestamp, ego_vehicle_location, tl))
                new_speed_factor_tl = stop_traffic_light(
                    ego_vehicle_location, tl.transform.location, tl.state,
                    wp_vector, wp_angle, speed_factor_tl, flags)
                if new_speed_factor_tl < speed_factor_tl:
                    speed_factor_tl = new_speed_factor_tl
                    logger.debug(
                        '@{}: traffic light {} reduced speed factor to {}'.
                        format(timestamp, tl, speed_factor_tl))

    speed_factor = min(speed_factor_tl, speed_factor_p, speed_factor_v)

    state = {
        'stop_person': speed_factor_p,
        'stop_vehicle': speed_factor_v,
        'stop_traffic_lights': speed_factor_tl
    }

    logger.debug('@{}: agent speed factors {}'.format(timestamp, state))

    return speed_factor, state


def _is_traffic_light_visible(ego_vehicle_location, tl_location, flags):
    _, tl_dist = ego_vehicle_location.get_vector_and_magnitude(tl_location)
    return tl_dist > flags.traffic_light_min_dist_thres
