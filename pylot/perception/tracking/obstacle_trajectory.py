from pylot.perception.detection.utils import BoundingBox3D


class ObstacleTrajectory(object):
    """Used to store the trajectory of an obstacle.

    Args:
        label (:obj:`str`): The label of the obstacle.
        id (:obj:`int`): The identifier of the obstacle.
        bounding_box (:py:class:`~pylot.perception.detection.utils.BoundingBox3D`):
            Bounding box of the obstacle.
        trajectory (list(:py:class:`~pylot.utils.Transform`)): List of past
            transforms.
    """
    def __init__(self, label, id, bounding_box, trajectory):
        self.label = label
        self.id = id
        if not isinstance(bounding_box, BoundingBox3D):
            raise ValueError('bounding box should be of type BoundingBox3D')
        self.bounding_box = bounding_box
        self.trajectory = trajectory

    def __str__(self):
        return 'Obstacle {}, label: {}, bbox: {}, trajectory {}'.format(
            self.id, self.label, self.bounding_box, self.trajectory)
