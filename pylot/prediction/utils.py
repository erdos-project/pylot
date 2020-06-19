import math
import numpy as np

from pylot.utils import Location, Rotation, Transform, Vector2D

def pad_trajectory(trajectory, num_steps):
    """Take the appropriate number of past steps as specified by num_steps.
    If we have not seen enough past locations of the obstacle, pad the
    trajectory with the appropriate number of copies of the earliest
    location.
    """
    num_past_locations = trajectory.shape[0]
    if num_past_locations < num_steps:
        initial_copies = np.repeat([np.array(trajectory[0])],
                                   num_steps - num_past_locations,
                                   axis=0)
        trajectory = np.vstack((initial_copies, trajectory))
    elif num_past_locations > num_steps:
        trajectory = trajectory[-num_steps:]
    assert trajectory.shape[0] == num_steps
    return trajectory

def get_nearby_obstacles(all_obstacles, radius):
    """Given a list of obstacles with past trajectories in the ego-vehicle's
       frame of reference, return a list of obstacles that are within a
       specified radius of the ego-vehicle, sorted by increasing distance
       from the ego-vehicle.
    """
    distances = [
        v.trajectory[-1].get_angle_and_magnitude(Location())[1]
        for v in all_obstacles
    ]
    sorted_vehicles = [
        v for v, d in sorted(zip(all_obstacles, distances),
                             key=lambda pair: pair[1])
        if d <= radius
    ]

    return sorted_vehicles

def get_nearby_obstacles_ego_transforms(nearby_obstacles):
    """ Gets the transform relative to the ego-vehicle for each obstacle
        in a list of nearby obstacles.
    """
    if len(nearby_obstacles) == 0:
        return []
    nearby_obstacles_ego_locations = np.stack(
        [v.trajectory[-1] for v in nearby_obstacles])
    nearby_obstacles_ego_transforms = []

    # Add appropriate rotations to closest_vehicles_ego_transforms, which
    # we estimate using the direction determined by the last two distinct
    # locations
    for i in range(len(nearby_obstacles)):
        cur_obstacle_angle = _estimate_obstacle_orientation(nearby_obstacles[i])
        nearby_obstacles_ego_transforms.append(
            Transform(location=nearby_obstacles_ego_locations[i].location,
                      rotation=Rotation(yaw=cur_obstacle_angle)))
    return nearby_obstacles_ego_transforms

def _estimate_obstacle_orientation(obstacle):
    """ Uses the obstacle's past trajectory to estimate its angle from the
        positive x-axis (trajectory points are in the ego-vehicle's coordinate
        frame).
    """
    other_idx = len(obstacle.trajectory) - 2
    yaw = 0.0  # Default orientation.
    current_loc = obstacle.trajectory[-1].location.as_vector_2D()
    while other_idx >= 0:
        past_ref_loc = obstacle.trajectory[other_idx].location.as_vector_2D(
        )
        vec = current_loc - past_ref_loc
        displacement = current_loc.l2_distance(past_ref_loc)
        if displacement > 0.001:
            yaw = vec.get_angle(Vector2D(0, 0))
            break
        else:
            other_idx -= 1
    # Force angle to be between -180 and 180 degrees.
    if yaw > 180:
        yaw -= 360
    elif yaw < -180:
        yaw += 360
    return math.degrees(yaw)

def get_occupancy_grid(point_cloud, lidar_z, lidar_meters_max):
    """Get occupancy grids for two different ranges of heights."""

    point_cloud = _point_cloud_to_precog_coordinates(point_cloud)

    # Threshold that was used when generating the PRECOG
    # (https://arxiv.org/pdf/1905.01296.pdf) dataset
    z_threshold = -lidar_z - 2.0
    above_mask = point_cloud[:, 2] > z_threshold

    feats = ()
    # Above z_threshold.
    feats += (_get_occupancy_from_masked_lidar(
        point_cloud[above_mask], lidar_meters_max), )
    # Below z_threshold.
    feats += (_get_occupancy_from_masked_lidar(
        point_cloud[(1 - above_mask).astype(np.bool)], lidar_meters_max), )

    stacked_feats = np.stack(feats, axis=-1)
    return np.expand_dims(stacked_feats, axis=0)

def _get_occupancy_from_masked_lidar(masked_lidar, meters_max):
    """Given an array of lidar points, return the corresponding occupancy grid
       with bins every 0.5 meters."""
    pixels_per_meter = 2
    xbins = np.linspace(-meters_max, meters_max,
                        meters_max * 2 * pixels_per_meter + 1)
    ybins = np.linspace(-meters_max, meters_max,
                        meters_max * 2 * pixels_per_meter + 1)
    grid = np.histogramdd(masked_lidar[..., :2], bins=(xbins, ybins))[0]
    grid[grid > 0.] = 1
    return grid

def _point_cloud_to_precog_coordinates(point_cloud):
    """Converts a LIDAR PointCloud, which is in camera coordinates,
       to the coordinates used in the PRECOG dataset, which is LIDAR
       coordinates but with the y- and z-coordinates negated (for
       a reference describing PRECOG coordinates, see e.g. https://github.com/nrhine1/deep_imitative_models/blob/0d52edfa54cb79da28bd7cf965ebccbe8514fc10/dim/env/preprocess/carla_preprocess.py#L584)
    """
    to_precog_transform = Transform(matrix=np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]))
    transformed_point_cloud = to_precog_transform.transform_points(
        point_cloud)
    return transformed_point_cloud

