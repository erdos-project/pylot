class ObstacleTrajectory(object):
    """Used to store the trajectory of an obstacle.

    Args:
        obstacle (:py:class:`~pylot.perception.detection.obstacle.Obstacle):
            The obstacle for which the trajectory is computed.
        trajectory (list(:py:class:`~pylot.utils.Transform`)): List of past
            transforms.
    """
    def __init__(self, obstacle, trajectory):
        self.obstacle = obstacle
        self.trajectory = trajectory

    def draw_on_frame(self, frame, bbox_color_map, ego_transform=None):
        """Draws the tracked obstacle as a 2D bounding box."""
        self.obstacle.draw_on_frame(frame, bbox_color_map, ego_transform)

    def draw_trajectory_on_frame(self, frame, draw_label=False):
        """Draws the trajectory on a bird's eye view frame."""
        if self.obstacle.is_person():
            color = [255, 0, 0]
        elif self.obstacle.is_vehicle():
            color = [128, 128, 0]
        else:
            color = [255, 255, 0]
        self.obstacle.draw_trajectory_on_frame(self.trajectory, frame, color,
                                               draw_label)

    def to_world_coordinates(self, ego_transform):
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
