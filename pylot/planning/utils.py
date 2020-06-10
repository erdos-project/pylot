import enum
import math

import pylot.perception.detection.utils
from pylot.perception.detection.traffic_light import TrafficLightColor


class BehaviorPlannerState(enum.Enum):
    """States in which the FSM behavior planner can be in."""
    READY = 1
    KEEP_LANE = 2
    PREPARE_LANE_CHANGE_LEFT = 3
    LANGE_CHANGE_LEFT = 4
    PREPARE_LANE_CHANGE_RIGHT = 5
    LANE_CHANGE_RIGHT = 6


def stop_person(ego_transform, obstacle, wp_vector, flags, logger, hd_map):
    """Computes a stopping factor for ego vehicle given a person obstacle.

    Args:
        ego_transform (:py:class:`~pylot.utils.Transform`): Transform of
            the ego vehicle in world coordinates.
        obstacle: A DetectedObstacle or a ObstaclePrediction of a person.
        wp_vector (:py:class:`~pylot.utils.Vector2D`): vector from the ego
            vehicle to the target waypoint.
        flags (absl.flags): The flags object.

    Returns:
        :obj:`float`: A stopping factor between 0 and 1 (i.e., no braking).
    """
    if hd_map is not None:
        if not hd_map.is_on_lane(obstacle.transform.location):
            # Person is not on the road.
            if (not hasattr(obstacle, 'trajectory') or not any(
                    map(hd_map.is_transform_on_lane, obstacle.trajectory))):
                # The person is not going to be on the road.
                logger.debug(
                    'Ignoring ({},{}); it is not going to be on the road'.
                    format(obstacle.label, obstacle.id))
                return 1
    else:
        logger.warning('No HDMap. All people are considered for stopping.')

    speed_factor_p = 1
    ego_location_2d = ego_transform.location.as_vector_2D()
    person_location_2d = obstacle.transform.location.as_vector_2D()
    p_vector = person_location_2d - ego_location_2d
    p_dist = person_location_2d.l2_distance(ego_location_2d)
    p_angle = p_vector.get_angle(wp_vector)
    # Maximum braking is applied if the person is in the emergency
    # hit zone. Otherwise, gradual braking is applied if the person
    # is in the hit zone.
    if (math.fabs(p_angle) < flags.person_angle_hit_zone
            and p_dist < flags.person_distance_hit_zone):
        # Person is in the hit zone.
        speed_factor_p = p_dist / (flags.coast_factor *
                                   flags.person_distance_hit_zone)
    if (math.fabs(p_angle) < flags.person_angle_emergency_zone
            and p_dist < flags.person_distance_emergency_zone):
        # Person is in emergency hit zone.
        speed_factor_p = 0
    return speed_factor_p


def stop_vehicle(ego_transform, obstacle, wp_vector, flags, logger, hd_map):
    """Computes a stopping factor for ego vehicle given a vehicle pos.

    Args:
        ego_transform (:py:class:`~pylot.utils.Transform`): Transform of
            the ego vehicle in world coordinates.
        obstacle: A DetectedObstacle or a ObstaclePrediction of a vehicle.
        flags (absl.flags): The flags object.

    Returns:
        :obj:`float`: A stopping factor between 0 and 1 (i.e., no braking).
    """
    if hd_map is not None:
        if not hd_map.are_on_same_lane(ego_transform.location,
                                       obstacle.transform.location):
            # TODO(ionel): Should only consider same lane.
            # Vehicle is not on the same lane as the ego.
            if (not hasattr(obstacle, 'trajectory') or not any(
                    map(hd_map.is_transform_on_lane, obstacle.trajectory))):
                # The vehicle is not going to be on the road.
                logger.debug(
                    'Ignoring ({},{}); it is not going to be on the road'.
                    format(obstacle.label, obstacle.id))
                return 1
    else:
        logger.warning('No HDMap. All vehicles are considered for stopping.')

    speed_factor_v = 1
    ego_location_2d = ego_transform.location.as_vector_2D()
    vehicle_location_2d = obstacle.transform.location.as_vector_2D()
    v_vector = vehicle_location_2d - ego_location_2d
    v_dist = vehicle_location_2d.l2_distance(ego_location_2d)
    v_angle = v_vector.get_angle(wp_vector)

    min_angle = -0.5 * flags.vehicle_max_angle / flags.coast_factor
    if (min_angle < v_angle < flags.vehicle_max_angle
            and v_dist < flags.vehicle_max_distance):
        # The vehicle is within the angle limit, and nearby.
        speed_factor_v = v_dist / (flags.coast_factor *
                                   flags.vehicle_max_distance)

    if (min_angle < v_angle < flags.vehicle_max_angle / flags.coast_factor
            and v_dist < flags.vehicle_max_distance * flags.coast_factor):
        # The vehicle is a bit far away, but it's on ego vehicle's path.
        speed_factor_v = v_dist / (flags.coast_factor *
                                   flags.vehicle_max_distance)

    min_nearby_angle = -0.5 * flags.vehicle_max_angle * flags.coast_factor
    if (min_nearby_angle < v_angle <
            flags.vehicle_max_angle * flags.coast_factor
            and v_dist < flags.vehicle_max_distance / flags.coast_factor):
        # The vehicle is very close; the angle can be higher.
        speed_factor_v = 0
    return speed_factor_v


def stop_traffic_light(ego_transform, tl, wp_vector, wp_angle, flags, logger,
                       hd_map):
    """Computes a stopping factor for ego vehicle given a traffic light.

    Args:
        ego_location_2d (:py:class:`~pylot.utils.Vector2D`): 2D location of
            the ego vehicle in world coordinates.
        tl (:py:class:`~pylot.perception.detection.traffic_light.TrafficLight`):
            the traffic light.
        wp_vector (:py:class:`~pylot.utils.Vector2D`): vector from the ego
            vehicle to the target waypoint.
        flags (absl.flags): The flags object.

    Returns:
        :obj:`float`: A stopping factor between 0 and 1 (i.e., no braking).
    """
    if hd_map is not None:
        # The traffic light is not relevant to the ego vehicle.
        if not hd_map.must_obey_traffic_light(ego_transform.location,
                                              tl.transform.location):
            logger.debug(
                'Ignoring traffic light {} that must not be obeyed'.format(tl))
            return 1
    else:
        logger.warning(
            'No HDMap. All traffic lights are considered for stopping.')
    # The ego vehicle is too close to the traffic light.
    if (ego_transform.location.distance(tl.transform.location) <
            flags.traffic_light_min_distance):
        logger.debug(
            'Ignoring traffic light {}; vehicle is too close'.format(tl))
        return 1
    # The ego vehicle can carry on driving.
    if (tl.state == TrafficLightColor.GREEN
            or tl.state == TrafficLightColor.OFF):
        return 1

    speed_factor_tl = 1
    ego_location_2d = ego_transform.location.as_vector_2D()
    tl_location_2d = tl.transform.location.as_vector_2D()
    tl_vector = tl_location_2d - ego_location_2d
    tl_dist = tl_location_2d.l2_distance(ego_location_2d)
    tl_angle = tl_vector.get_angle(wp_vector)

    if (0 <= tl_angle < flags.traffic_light_max_angle
            and tl_dist < flags.traffic_light_max_distance):
        # The traffic light is at most x radians to the right of the
        # vehicle path, and is not too far away.
        speed_factor_tl = tl_dist / (flags.coast_factor *
                                     flags.traffic_light_max_distance)

    if (0 <= tl_angle < flags.traffic_light_max_angle / flags.coast_factor
            and tl_dist < flags.traffic_light_max_distance * flags.coast_factor
            and math.fabs(wp_angle) < 0.2):
        # The ego is pretty far away, so the angle to the traffic light has
        # to be smaller, and the vehicle must be driving straight.
        speed_factor_tl = tl_dist / (flags.coast_factor *
                                     flags.traffic_light_max_distance)

    if (0 < tl_angle < flags.traffic_light_max_angle * flags.coast_factor
            and tl_dist < flags.traffic_light_max_distance / flags.coast_factor
            and math.fabs(wp_angle) < 0.2):
        # The traffic light is nearby and the vehicle is driving straight;
        # the angle to the traffic light can be higher.
        speed_factor_tl = 0

    return speed_factor_tl


def stop_for_agents(ego_transform,
                    wp_angle,
                    wp_vector,
                    obstacles,
                    traffic_lights,
                    flags,
                    logger,
                    hd_map=None,
                    timestamp=None):
    speed_factor_tl = 1
    speed_factor_p = 1
    speed_factor_v = 1

    for obstacle in obstacles:
        if obstacle.label == 'person' and flags.stop_for_people:
            new_speed_factor_p = stop_person(ego_transform, obstacle,
                                             wp_vector, flags, logger, hd_map)
            if new_speed_factor_p < speed_factor_p:
                speed_factor_p = new_speed_factor_p
                logger.debug(
                    '@{}: person {} reduced speed factor to {}'.format(
                        timestamp, obstacle, speed_factor_p))
        elif (obstacle.label in pylot.perception.detection.utils.VEHICLE_LABELS
              and flags.stop_for_vehicles):
            new_speed_factor_v = stop_vehicle(ego_transform, obstacle,
                                              wp_vector, flags, logger, hd_map)
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
            new_speed_factor_tl = stop_traffic_light(ego_transform, tl,
                                                     wp_vector, wp_angle,
                                                     flags, logger, hd_map)
            if new_speed_factor_tl < speed_factor_tl:
                speed_factor_tl = new_speed_factor_tl
                logger.debug(
                    '@{}: traffic light {} reduced speed factor to {}'.format(
                        timestamp, tl, speed_factor_tl))

    speed_factor = min(speed_factor_tl, speed_factor_p, speed_factor_v)
    logger.debug(
        '@{}: speed factors: person {}, vehicle {}, traffic light {}'.format(
            timestamp, speed_factor_p, speed_factor_v, speed_factor_tl))
    return speed_factor
