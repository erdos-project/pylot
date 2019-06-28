import math
import numpy as np

from pylot.control.utils import get_world_vec_dist


def is_pedestrian_hitable(pos, city_map):
    return city_map.is_point_on_lane([pos.x, pos.y, 38])


def is_traffic_light_active(vehicle_transform, tl_pos, city_map):

    def search_closest_lane_point(x_agent, y_agent, depth):
        step_size = 4
        if depth > 1:
            return None
        try:
            degrees = city_map.get_lane_orientation_degrees(
                [x_agent, y_agent, 38])
        except:
            return None
        if not city_map.is_point_on_lane([x_agent, y_agent, 38]):
            result = search_closest_lane_point(x_agent + step_size,
                                               y_agent, depth + 1)
            if result is not None:
                return result
            result = search_closest_lane_point(
                x_agent, y_agent + step_size, depth + 1)
            if result is not None:
                return result
            result = search_closest_lane_point(
                x_agent + step_size, y_agent + step_size, depth + 1)
            if result is not None:
                return result
            result = search_closest_lane_point(
                x_agent + step_size, y_agent - step_size, depth + 1)
            if result is not None:
                return result
            result = search_closest_lane_point(
                x_agent - step_size, y_agent + step_size, depth + 1)
            if result is not None:
                return result
            result = search_closest_lane_point(x_agent - step_size,
                                               y_agent, depth + 1)
            if result is not None:
                return result
            result = search_closest_lane_point(
                x_agent, y_agent - step_size, depth + 1)
            if result is not None:
                return result
            result = search_closest_lane_point(
                x_agent - step_size, y_agent - step_size, depth + 1)
            if result is not None:
                return result
        else:
            if degrees < 6:
                return [x_agent, y_agent]
            else:
                return None

    closest_lane_point = search_closest_lane_point(tl_pos.x, tl_pos.y, 0)

    if closest_lane_point is not None:
        return (math.fabs(
            city_map.get_lane_orientation_degrees(
                [vehicle_transform.location.x, vehicle_transform.location.y, 38])
            - city_map.get_lane_orientation_degrees(
                [closest_lane_point[0], closest_lane_point[1], 38])) < 1)
    else:
        return None


def is_traffic_light_visible(vehicle_transform, tl_pos, flags):
    _, tl_dist = get_world_vec_dist(
        vehicle_transform.location.x,
        vehicle_transform.location.y,
        tl_pos.x,
        tl_pos.y)
    return tl_dist > flags.traffic_light_min_dist_thres


def is_vehicle_on_same_lane(vehicle_transform, obs_vehicle_pos, city_map):
    if city_map.is_point_on_intersection([obs_vehicle_pos.x, obs_vehicle_pos.y, 38]):
        return True
    return (math.fabs(
        city_map.get_lane_orientation_degrees(
            [vehicle_transform.location.x, vehicle_transform.location.y, 38]) -
        city_map.get_lane_orientation_degrees([obs_vehicle_pos.x, obs_vehicle_pos.y, 38])) < 1)
