import math
import numpy as np


def get_world_vec_dist(x_dst, y_dst, x_src, y_src):
    vec = np.array([x_dst, y_dst] - np.array([x_src, y_src]))
    dist = math.sqrt(vec[0]**2 + vec[1]**2)
    if abs(dist) < 0.00001:
        return vec, dist
    else:
        return vec / dist, dist


def get_angle(vec_dst, vec_src):
    angle = (math.atan2(vec_dst[1], vec_dst[0]) -
             math.atan2(vec_src[1], vec_src[0]))
    if angle > math.pi:
        angle -= 2 * math.pi
    elif angle < -math.pi:
        angle += 2 * math.pi
    return angle


def get_speed(velocity_vector):
    """ Compute the speed of the vehicle in km/h.
    Args:
        vehicle: A velocity vector.
    Returns:
        The speed of the given vehicle as a float in km/h.
    """
    speed = 3.6 * math.sqrt(velocity_vector.x**2 + velocity_vector.y**2 +
                            velocity_vector.z**2)
    return speed


def is_pedestrian_hitable(vehicle_transform, ped_pos):
    # TODO(ionel): Implement.
    return True


def is_traffic_light_active(vehicle_transform, tl_pos):
    # TODO(ionel): Implement.
    return True


def is_vehicle_on_same_lane(vehicle_transform, obs_vehicle_pos):
    # TODO(ionel): Implement.
    # vec_loc = carla.Location(vehicle_transform.location.x,
    #                          vehicle_transform.location.y,
    #                          vehicle_transform.location.z)
    # vec_waypoint = hd_map.get_waypoint(vec_loc,
    #                                    project_to_road=True,
    #                                    lane_type=carla.LaneType.Driving)
    # obs_loc = carla.Location(obstacle_pos.x,
    #                          obstacle_pos.y,
    #                          obstacle_pos.z)
    # obs_waypoint = hd_map.get_waypoint(obs_loc)
    return True


def is_pedestrian_on_hit_zone(p_dist, p_angle, flags):
    return (math.fabs(p_angle) < flags.pedestrian_angle_hit_thres and
            p_dist < flags.pedestrian_distance_hit_thres)


def is_pedestrian_on_near_hit_zone(p_dist, p_angle, flags):
    return (math.fabs(p_angle) < flags.pedestrian_angle_emergency_thres and
            p_dist < flags.pedestrian_distance_emergency_thres)


def is_traffic_light_visible(vehicle_transform, tl_pos, flags):
    _, tl_dist = get_world_vec_dist(
        vehicle_transform.location.x,
        vehicle_transform.location.y,
        tl_pos.x,
        tl_pos.y)
    return tl_dist > flags.traffic_light_min_dist_thres


def stop_pedestrian(vehicle_transform,
                    pedestrian_pos,
                    wp_vector,
                    speed_factor_p,
                    flags):
    speed_factor_p_temp = 1
    p_vector, p_dist = get_world_vec_dist(
        pedestrian_pos.x,
        pedestrian_pos.y,
        vehicle_transform.location.x,
        vehicle_transform.location.y)
    p_angle = get_angle(p_vector, wp_vector)
    if is_pedestrian_on_hit_zone(p_dist, p_angle, flags):
        speed_factor_p_temp = p_dist / (flags.coast_factor * flags.pedestrian_distance_hit_thres)
    if is_pedestrian_on_near_hit_zone(p_dist, p_angle, flags):
        speed_factor_p_temp = 0
    if (speed_factor_p_temp < speed_factor_p):
        speed_factor_p = speed_factor_p_temp
    return speed_factor_p



def stop_vehicle(vehicle_transform, obs_vehicle_pos, wp_vector, speed_factor_v, flags):
    speed_factor_v_temp = 1
    v_vector, v_dist = get_world_vec_dist(
        obs_vehicle_pos.x, obs_vehicle_pos.y, vehicle_transform.location.x, vehicle_transform.location.y)
    v_angle = get_angle(v_vector, wp_vector)

    if ((-0.5 * flags.vehicle_angle_thres / flags.coast_factor <
         v_angle < flags.vehicle_angle_thres / flags.coast_factor and
         v_dist < flags.vehicle_distance_thres * flags.coast_factor) or
        (-0.5 * flags.vehicle_angle_thres / flags.coast_factor <
         v_angle < flags.vehicle_angle_thres and
         v_dist < flags.vehicle_distance_thres)):
        speed_factor_v_temp = v_dist / (
            flags.coast_factor * flags.vehicle_distance_thres)

    if (-0.5 * flags.vehicle_angle_thres * flags.coast_factor <
        v_angle < flags.vehicle_angle_thres * flags.coast_factor and
        v_dist < flags.vehicle_distance_thres / flags.coast_factor):
        speed_factor_v_temp = 0

    if speed_factor_v_temp < speed_factor_v:
        speed_factor_v = speed_factor_v_temp

    return speed_factor_v


def stop_traffic_light(vehicle_transform,
                       tl_pos,
                       tl_state,
                       wp_vector,
                       wp_angle,
                       speed_factor_tl,
                       flags):
    speed_factor_tl_temp = 1
    if tl_state != 0:  # Not green
        tl_vector, tl_dist = get_world_vec_dist(
            tl_pos.x, tl_pos.y, vehicle_transform.location.x, vehicle_transform.location.y)
        tl_angle = get_angle(tl_vector, wp_vector)

        if ((0 < tl_angle < flags.traffic_light_angle_thres / flags.coast_factor and
             tl_dist < flags.traffic_light_max_dist_thres * flags.coast_factor) or
            (0 < tl_angle < flags.traffic_light_angle_thres and
             tl_dist < flags.traffic_light_max_dist_thres) and
            math.fabs(wp_angle) < 0.2):

            speed_factor_tl_temp = tl_dist / (
                flags.coast_factor *
                flags.traffic_light_max_dist_thres)

        if ((0 < tl_angle < flags.traffic_light_angle_thres * flags.coast_factor and
             tl_dist < flags.traffic_light_max_dist_thres / flags.coast_factor) and
            math.fabs(wp_angle) < 0.2):
            speed_factor_tl_temp = 0

        if (speed_factor_tl_temp < speed_factor_tl):
            speed_factor_tl = speed_factor_tl_temp

    return speed_factor_tl
