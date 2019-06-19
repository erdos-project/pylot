import enum
import math


class BehaviorPlannerState(enum.Enum):
    READY = 1
    KEEP_LANE = 2
    PREPARE_LANE_CHANGE_LEFT = 3
    LANGE_CHANGE_LEFT = 4
    PREPARE_LANE_CHANGE_RIGHT = 5
    LANE_CHANGE_RIGHT = 6


def get_distance(loc1, loc2):
    x_diff = loc1.x - loc2.x
    y_diff = loc1.y - loc2.y
    return math.sqrt(x_diff**2 + y_diff**2)


def get_target_speed(vehicle_loc, waypoint):
    if get_distance(waypoint.location, vehicle_loc) > 0.08:
        target_speed = 10
    else:
        # We are reaching a waypoint reduce the speed to 0.
        target_speed = 0
    return target_speed
