import enum
import itertools
import math

from pylot.perception.detection.traffic_light import TrafficLightColor

# Number of predicted locations to consider when computing speed factors.
NUM_FUTURE_TRANSFORMS = 10


class BehaviorPlannerState(enum.Enum):
    """States in which the FSM behavior planner can be in."""
    READY = 1
    KEEP_LANE = 2
    PREPARE_LANE_CHANGE_LEFT = 3
    LANGE_CHANGE_LEFT = 4
    PREPARE_LANE_CHANGE_RIGHT = 5
    LANE_CHANGE_RIGHT = 6


def _compute_person_speed_factor(ego_location_2d, person_location_2d,
                                 wp_vector, flags, logger):
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


def stop_person(ego_transform, obstacle, wp_vector, flags, logger, hd_map):
    """Computes a stopping factor for ego vehicle given a person obstacle.

    Args:
        ego_transform (:py:class:`~pylot.utils.Transform`): Transform of
            the ego vehicle in world coordinates.
        obstacle: An Obstacle or an ObstaclePrediction of a person.
        wp_vector (:py:class:`~pylot.utils.Vector2D`): vector from the ego
            vehicle to the target waypoint.
        flags (absl.flags): The flags object.

    Returns:
        :obj:`float`: A stopping factor between 0 and 1 (i.e., no braking).
    """
    if hd_map is not None:
        if hasattr(obstacle, 'predicted_trajectory'):
            if not any(
                    map(
                        lambda transform: hd_map.are_on_same_lane(
                            transform.location, ego_transform.location),
                        obstacle.predicted_trajectory)):
                # The person is not going to be on the road.
                logger.debug(
                    'Ignoring ({},{}); it is not going to be on the same lane'.
                    format(obstacle.label, obstacle.id))
                return 1
        else:
            # Person is not on the road.
            if not hd_map.is_on_lane(obstacle.transform.location):
                logger.debug('Ignoring ({},{}); it is not on the road'.format(
                    obstacle.label, obstacle.id))
                return 1
    else:
        logger.warning('No HDMap. All people are considered for stopping.')
    ego_location_2d = ego_transform.location.as_vector_2D()
    min_speed_factor_p = _compute_person_speed_factor(
        ego_location_2d, obstacle.transform.location.as_vector_2D(), wp_vector,
        flags, logger)
    if hasattr(obstacle, 'predicted_trajectory'):
        transforms = itertools.islice(
            obstacle.predicted_trajectory, 0,
            min(NUM_FUTURE_TRANSFORMS, len(obstacle.predicted_trajectory)))
        for person_transform in transforms:
            speed_factor_p = _compute_person_speed_factor(
                ego_location_2d, person_transform.location.as_vector_2D(),
                wp_vector, flags, logger)
            min_speed_factor_p = min(min_speed_factor_p, speed_factor_p)
    return min_speed_factor_p


def _compute_vehicle_speed_factor(ego_location_2d, vehicle_location_2d,
                                  wp_vector, flags, logger):
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


def stop_vehicle(ego_transform, obstacle, wp_vector, flags, logger, hd_map):
    """Computes a stopping factor for ego vehicle given a vehicle pos.

    Args:
        ego_transform (:py:class:`~pylot.utils.Transform`): Transform of
            the ego vehicle in world coordinates.
        obstacle: An Obstacle or an ObstaclePrediction of a vehicle.
        flags (absl.flags): The flags object.

    Returns:
        :obj:`float`: A stopping factor between 0 and 1 (i.e., no braking).
    """
    if ego_transform.location.x == obstacle.transform.location.x and \
       ego_transform.location.y == obstacle.transform.location.y and \
       ego_transform.location.z == obstacle.transform.location.z:
        # Don't stop for ourselves.
        return 1
    if hd_map is not None:
        if not hd_map.are_on_same_lane(ego_transform.location,
                                       obstacle.transform.location):
            # Vehicle is not on the same lane as the ego.
            if (not hasattr(obstacle, 'predicted_trajectory') or not any(
                    map(
                        lambda transform: hd_map.are_on_same_lane(
                            transform.location, ego_transform.location),
                        obstacle.predicted_trajectory))):
                # The vehicle is not going to be on the road.
                logger.debug(
                    'Ignoring ({},{}); it is not going to be on the road'.
                    format(obstacle.label, obstacle.id))
                return 1
    else:
        logger.warning('No HDMap. All vehicles are considered for stopping.')

    ego_location_2d = ego_transform.location.as_vector_2D()
    min_speed_factor_v = _compute_vehicle_speed_factor(
        ego_location_2d, obstacle.transform.location.as_vector_2D(), wp_vector,
        flags, logger)
    if hasattr(obstacle, 'predicted_trajectory'):
        transforms = itertools.islice(
            obstacle.predicted_trajectory, 0,
            min(NUM_FUTURE_TRANSFORMS, len(obstacle.predicted_trajectory)))
        for vehicle_transform in transforms:
            speed_factor_v = _compute_vehicle_speed_factor(
                ego_location_2d, vehicle_transform.location.as_vector_2D(),
                wp_vector, flags, logger)
            min_speed_factor_v = min(min_speed_factor_v, speed_factor_v)
    return min_speed_factor_v


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
            return False, 1
    else:
        logger.warning(
            'No HDMap. All traffic lights are considered for stopping.')
    # The ego vehicle is too close to the traffic light.
    if (ego_transform.location.distance(tl.transform.location) <
            flags.traffic_light_min_distance):
        logger.debug(
            'Ignoring traffic light {}; vehicle is too close'.format(tl))
        return True, 1
    # The ego vehicle can carry on driving.
    if (tl.state == TrafficLightColor.GREEN
            or tl.state == TrafficLightColor.OFF):
        return True, 1

    height_delta = tl.transform.location.z - ego_transform.location.z
    if height_delta > 4:
        logger.debug('Traffic light is American style')
        # The traffic ligh is across the road. Increase the max distance.
        traffic_light_max_distance = flags.traffic_light_max_distance * 2.5
        traffic_light_max_angle = flags.traffic_light_max_angle / 4
        american_tl = True
    else:
        logger.debug('Traffic light is European style')
        traffic_light_max_distance = flags.traffic_light_max_distance
        traffic_light_max_angle = flags.traffic_light_max_angle
        american_tl = False
    speed_factor_tl = 1
    ego_location_2d = ego_transform.location.as_vector_2D()
    tl_location_2d = tl.transform.location.as_vector_2D()
    tl_vector = tl_location_2d - ego_location_2d
    tl_dist = tl_location_2d.l2_distance(ego_location_2d)
    tl_angle = tl_vector.get_angle(wp_vector)
    logger.debug(
        'Traffic light vector {}; dist {}; angle {}; wp_angle {}'.format(
            tl_vector, tl_dist, tl_angle, wp_angle))
    if (-0.2 <= tl_angle < traffic_light_max_angle
            and tl_dist < traffic_light_max_distance):
        # The traffic light is at most x radians to the right of the
        # vehicle path, and is not too far away.
        speed_factor_tl = min(
            speed_factor_tl,
            tl_dist / (flags.coast_factor * traffic_light_max_distance))

    if (-0.2 <= tl_angle < traffic_light_max_angle / flags.coast_factor
            and tl_dist < traffic_light_max_distance * flags.coast_factor
            and math.fabs(wp_angle) < 0.2):
        # The ego is pretty far away, so the angle to the traffic light has
        # to be smaller, and the vehicle must be driving straight.
        speed_factor_tl = min(
            speed_factor_tl,
            tl_dist / (flags.coast_factor * traffic_light_max_distance))

    if (-0.2 <= tl_angle < traffic_light_max_angle * flags.coast_factor
            and math.fabs(wp_angle) < 0.2):
        if american_tl:
            if (-0.1 <= tl_angle < traffic_light_max_angle and tl_dist < 60):
                dist_to_intersection = hd_map.distance_to_intersection(
                    ego_transform.location, max_distance_to_check=20)
                if (dist_to_intersection is not None
                        and dist_to_intersection < 12
                        and tl.bounding_box.get_width() *
                        tl.bounding_box.get_height() > 400):
                    speed_factor_tl = 0
                if (dist_to_intersection is not None and tl_dist < 27
                        and 12 <= dist_to_intersection <= 20):
                    speed_factor_tl = 0
        else:
            if tl_dist < traffic_light_max_distance / flags.coast_factor:
                # The traffic light is nearby and the vehicle is driving
                # straight; the angle to the traffic light can be higher.
                speed_factor_tl = 0
    if speed_factor_tl < 1:
        dist_to_intersection = hd_map.distance_to_intersection(
            ego_transform.location, max_distance_to_check=15)
        if dist_to_intersection is None:
            # Our lidar-based depth estimation does not work when
            # we're on a hill.
            # XXX(ionel): Hack to avoid getting stuck when we're far
            # from intersections (see scenario 28 in the challenge training
            # routes).
            logger.warning(
                'Ignored traffic light speed factor because junction '
                'is not nearby')
            return True, 1
        else:
            return True, speed_factor_tl
    else:
        # The traffic light doesn't affect the vehicle.
        return False, speed_factor_tl


def stop_for_agents(ego_transform,
                    wp_angle,
                    wp_vector,
                    obstacles,
                    traffic_lights,
                    flags,
                    logger,
                    hd_map=None,
                    timestamp=None,
                    distance_since_last_full_stop=0):
    speed_factor_tl = 1
    speed_factor_p = 1
    speed_factor_v = 1
    speed_factor_stop = 1

    for obstacle in obstacles:
        if obstacle.is_person() and flags.stop_for_people:
            new_speed_factor_p = stop_person(ego_transform, obstacle,
                                             wp_vector, flags, logger, hd_map)
            if new_speed_factor_p < speed_factor_p:
                speed_factor_p = new_speed_factor_p
                logger.debug(
                    '@{}: person {} reduced speed factor to {}'.format(
                        timestamp, obstacle.id, speed_factor_p))
        elif obstacle.is_vehicle() and flags.stop_for_vehicles:
            new_speed_factor_v = stop_vehicle(ego_transform, obstacle,
                                              wp_vector, flags, logger, hd_map)
            if new_speed_factor_v < speed_factor_v:
                speed_factor_v = new_speed_factor_v
                logger.debug(
                    '@{}: vehicle {} reduced speed factor to {}'.format(
                        timestamp, obstacle.id, speed_factor_v))
        else:
            logger.debug('@{}: filtering obstacle {}'.format(
                timestamp, obstacle.label))

    semaphorized_junction = False
    if flags.stop_for_traffic_lights:
        for tl in traffic_lights:
            valid_tl, new_speed_factor_tl = stop_traffic_light(
                ego_transform, tl, wp_vector, wp_angle, flags, logger, hd_map)
            semaphorized_junction = semaphorized_junction or valid_tl
            if new_speed_factor_tl < speed_factor_tl:
                speed_factor_tl = new_speed_factor_tl
                logger.debug(
                    '@{}: traffic light {} reduced speed factor to {}'.format(
                        timestamp, tl, speed_factor_tl))
    if flags.stop_at_uncontrolled_junctions:
        if (hd_map is not None and not semaphorized_junction
                and not hd_map.is_intersection(ego_transform.location)):
            dist_to_junction = hd_map.distance_to_intersection(
                ego_transform.location, max_distance_to_check=13)
            logger.debug('@{}: dist to junc {}, last stop {}'.format(
                timestamp, dist_to_junction, distance_since_last_full_stop))
            if (dist_to_junction is not None
                    and distance_since_last_full_stop > 13):
                speed_factor_stop = 0

    speed_factor = min(speed_factor_tl, speed_factor_p, speed_factor_v,
                       speed_factor_stop)
    logger.debug(
        '@{}: speed factors: person {}, vehicle {}, traffic light {}, stop {}'.
        format(timestamp, speed_factor_p, speed_factor_v, speed_factor_tl,
               speed_factor_stop))
    return speed_factor
