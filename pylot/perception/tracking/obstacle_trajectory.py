class ObstacleTrajectory(object):
    def __init__(self, label, id, trajectory):
        """Constructs the obstacle trajectory using the given data.

        Args:
            label: String for the class of the obstacle.
            id: ID of the obstacle.
            trajectory: List of past pylot.util.simulation.Transforms.
        """

        self.label = label
        self.id = id
        self.trajectory = trajectory

    def __str__(self):
        return '{} {}, Trajectory {}'.format(self.label, self.id,
                                             self.trajectory)
