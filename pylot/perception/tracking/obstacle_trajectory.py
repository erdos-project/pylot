import math

from pylot.perception.detection.obstacle import Obstacle
from pylot.utils import Transform, Vector2D


class ObstacleTrajectory(object):
    """Used to store the trajectory of an obstacle.

    Args:
        obstacle (:py:class:`~pylot.perception.detection.obstacle.Obstacle`):
            The obstacle for which the trajectory is computed.
        trajectory (list(:py:class:`~pylot.utils.Transform`)): List of past
            transforms.
    """
    def __init__(self, obstacle: Obstacle, trajectory):
        self.obstacle = obstacle
        self.trajectory = trajectory

    def draw_on_frame(self,
                      frame,
                      bbox_color_map,
                      ego_transform: Transform = None):
        """Draws the tracked obstacle as a 2D bounding box."""
        self.obstacle.draw_on_frame(frame, bbox_color_map, ego_transform)

    def draw_trajectory_on_frame(self, frame, draw_label: bool = False):
        """Draws the trajectory on a bird's eye view frame."""
        if self.obstacle.is_person():
            color = [255, 0, 0]
        elif self.obstacle.is_vehicle():
            color = [128, 128, 0]
        else:
            color = [255, 255, 0]
        self.obstacle.draw_trajectory_on_frame(self.trajectory, frame, color,
                                               draw_label)

    def estimate_obstacle_orientation(self):
        """Uses the obstacle's past trajectory to estimate its angle from the
           positive x-axis (assumes trajectory points are in the ego-vehicle's
           coordinate frame)."""
        other_idx = len(self.trajectory) - 2
        # TODO: Setting a default yaw is dangerous. Find some way to estimate
        # the orientation of a stationary object (e.g. 3D object detection).
        yaw = 0.0  # Default orientation for stationary objects.
        current_loc = self.trajectory[-1].location.as_vector_2D()
        while other_idx >= 0:
            past_ref_loc = self.trajectory[other_idx].location.as_vector_2D()
            vec = current_loc - past_ref_loc
            displacement = current_loc.l2_distance(past_ref_loc)
            if displacement > 0.001:
                # Angle between displacement vector and the x-axis, i.e.
                # the (1,0) vector.
                yaw = vec.get_angle(Vector2D(1, 0))
                break
            else:
                other_idx -= 1
        return math.degrees(yaw)

    def get_last_n_transforms(self, n: int):
        """Returns the last n steps of the trajectory. If we have not seen
        enough past locations of the obstacle, pad the trajectory with the
        appropriate number of copies of the earliest location."""
        num_past_locations = len(self.trajectory)
        if num_past_locations < n:
            initial_copies = [self.trajectory[0]] * (n - num_past_locations)
            last_n_steps = initial_copies + self.trajectory
        else:
            last_n_steps = self.trajectory[-n:]
        return last_n_steps

    def to_world_coordinates(self, ego_transform: Transform):
        """Transforms the trajectory into world coordinates."""
        cur_trajectory = []
        for past_transform in self.trajectory:
            cur_trajectory.append(ego_transform * past_transform)
        self.trajectory = cur_trajectory

    @property
    def id(self):
        return self.obstacle.id

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Obstacle {}, trajectory {}'.format(self.obstacle,
                                                   self.trajectory)
