import numpy as np

from pylot.utils import Transform


def get_occupancy_grid(point_cloud, lidar_z, lidar_meters_max):
    """Get occupancy grids for two different ranges of heights."""

    point_cloud = _point_cloud_to_precog_coordinates(point_cloud)

    # Threshold that was used when generating the PRECOG
    # (https://arxiv.org/pdf/1905.01296.pdf) dataset
    z_threshold = -lidar_z - 2.0
    above_mask = point_cloud[:, 2] > z_threshold

    feats = ()
    # Above z_threshold.
    feats += (_get_occupancy_from_masked_lidar(point_cloud[above_mask],
                                               lidar_meters_max), )
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
       a reference describing PRECOG coordinates, see e.g. https://github.com/nrhine1/deep_imitative_models/blob/0d52edfa54cb79da28bd7cf965ebccbe8514fc10/dim/env/preprocess/carla_preprocess.py#L584)  # noqa: E501
    """
    to_precog_transform = Transform(matrix=np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]))
    transformed_point_cloud = to_precog_transform.transform_points(point_cloud)
    return transformed_point_cloud
