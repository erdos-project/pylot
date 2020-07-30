import math
from collections import namedtuple

from pylot.planning.utils import BehaviorPlannerState

VehicleInfo = namedtuple(
    "VehicleInfo",
    [
        'next_speed',  # The next vehicle speed.
        'target_speed',  # The target speed of the vehicle.
        'goal_lane',  # The id of the goal lane.
        'delta_s',  # Delta s distance to the goal location.
        'speed_limit',  # Speed limit in the area.
    ])

Trajectory = namedtuple(
    "Trajectory",
    [
        'intended_lane',  # The id of the lane change to (i.e., next lane).
        'final_lane',  # Resulting lane for the current behavior.
    ])


def get_trajectory_data(state, vehicle_info, trajectory):
    final_lane = trajectory[:-1]
    if state == BehaviorPlannerState.PREPARE_LANE_CHANGE_LEFT:
        intended_lane = final_lane + 1
    elif state == BehaviorPlannerState.PREPARE_LANE_CHANGE_RIGHT:
        intended_lane = final_lane - 1
    else:
        intended_lane = final_lane
    return Trajectory(intended_lane, final_lane)


def cost_speed(vehicle_info, predictions, trajectory):
    """ Computes cost of driving at a given speed.

    Args:
        vehicle_info: A VehicleInfo tuple.
        predictions: A dict of predictions for vehicles.
        trajectory: A Trajectory tuple.
     Returns:
         A cost in [0, 1].
     """
    # Cost of the car stopping.
    STOP_COST = 0.7
    # How many km/h to drive at bellow speed limit.
    BUFFER_SPEED = 5.0
    vehicle_info.target_speed = vehicle_info.speed_limit - BUFFER_SPEED
    if vehicle_info.next_speed < vehicle_info.target_speed:
        # Cost linearly decreases the closer we drive to target speed.
        return (STOP_COST *
                (vehicle_info.target_speed - vehicle_info.next_speed) /
                vehicle_info.target_speed)
    elif (vehicle_info.next_speed >= vehicle_info.target_speed
          and vehicle_info.next_speed < vehicle_info.speed_limit):
        # Cost linearly increases if we drive above target speed.
        return (vehicle_info.next_speed -
                vehicle_info.target_speed) / BUFFER_SPEED
    else:
        # Cost is always 1 if we drive above speed limit.
        return 1


def cost_lane_change(vehicle_info, predictions, trajectory):
    """ Computes cost of changing lanes.

    Args:
        vehicle_info: A VehicleInfo tuple.
        predictions: A dict of predictions for vehicles.
        trajectory: A Trajectory tuple.
    Returns:
         A cost in [0, 1].
    """
    # We want to penalize lane changes (i.e., high delta_d).
    delta_d = (2.0 * vehicle_info.goal_lane - trajectory.intended_lane -
               trajectory.final_lane)
    # Ensure that max cost is in [0, 1].
    if abs(vehicle_info.delta_s) < 0.0001:
        # We're very close to the goal.
        return 1
    else:
        return 1 - math.exp(-abs(delta_d) / vehicle_info.delta_s)


def cost_inefficiency(vehicle_info, predictions, trajectory):
    """ Computes cost of driving in the fastest lane.

    Args:
        vehicle_info: A VehicleInfo tuple.
        predictions: A dict of predictions for vehicles.
        trajectory: A Trajectory tuple.
    Returns:
        A cost in [0, 1].
    """
    # Cost becomes higher for trajectories with intended and final_lane
    # lane that have traffic slower than target_speed.
    proposed_speed_intended = get_lane_speed(predictions,
                                             trajectory.intended_lane)
    if not proposed_speed_intended:
        proposed_speed_intended = vehicle_info.target_speed
    proposed_speed_final = get_lane_speed(predictions, trajectory.final_lane)
    if not proposed_speed_final:
        proposed_speed_final = vehicle_info.target_speed
    cost = (2.0 * vehicle_info.target_speed - proposed_speed_intended -
            proposed_speed_final) / vehicle_info.target_speed
    return cost


def cost_overtake(current_state, future_state, ego_info):
    if ego_info.current_time - ego_info.last_time_moving > 50000:
        # Switch to OVERTAKE if ego hasn't moved for a while.
        if future_state == BehaviorPlannerState.OVERTAKE:
            return 0
        return 1
    else:
        if current_state == BehaviorPlannerState.OVERTAKE:
            # Do not speed too long in OVERTAKE state.
            if ego_info.current_time - ego_info.last_time_stopped > 3000:
                if future_state == BehaviorPlannerState.OVERTAKE:
                    return 1
                else:
                    return 0
            else:
                if future_state == BehaviorPlannerState.OVERTAKE:
                    return 0
                else:
                    return 1
        else:
            # Do not switch to overtake because the ego is not blocked.
            if future_state == BehaviorPlannerState.OVERTAKE:
                return 1
            return 0
    raise NotImplementedError


def get_lane_speed(predictions, lane_id):
    """ Returns the speed vehicles are driving in a lane.
    Assumes that all vehicles are driving at the same speed.

    Returns: The speed of an vehicle in lane_id, or None if no vehicle
        exists in lane lane_id.
    """
    for vehicle_id, trajectory in predictions.items():
        if trajectory[0].lane_id == lane_id and vehicle_id != -1:
            return trajectory[0].speed
    return None
