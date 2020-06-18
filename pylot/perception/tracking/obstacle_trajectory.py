from pylot.perception.detection.utils import BoundingBox2D, BoundingBox3D

VEHICLE_LABELS = {'car', 'bicycle', 'motorcycle', 'bus', 'truck', 'vehicle'}


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
        if not (isinstance(bounding_box, BoundingBox3D)
                or isinstance(bounding_box, BoundingBox2D)):
            raise ValueError(
                'bounding box should be of type BoundingBox2D or BoundingBox3D'
            )
        self.bounding_box = bounding_box
        self.trajectory = trajectory

    def draw_on_image(self, image_np, bbox_color_map, ego_transform=None):
        import cv2
        txt_font = cv2.FONT_HERSHEY_SIMPLEX
        text = '{}, id: {}'.format(self.label, self.id)
        if ego_transform is not None:
            text += ', {:.1f}m'.format(
                ego_transform.location.distance(self.trajectory[-1].location))
        txt_size = cv2.getTextSize(text, txt_font, 0.5, 2)[0]
        if self.label in bbox_color_map:
            color = bbox_color_map[self.label]
        else:
            color = [255, 255, 255]
        if isinstance(self.bounding_box, BoundingBox2D):
            # Show bounding box.
            cv2.rectangle(image_np, self.bounding_box.get_min_point(),
                          self.bounding_box.get_max_point(), color, 2)
            # Show text.
            cv2.rectangle(image_np,
                          (self.bounding_box.x_min,
                           self.bounding_box.y_min - txt_size[1] - 2),
                          (self.bounding_box.x_min + txt_size[0],
                           self.bounding_box.y_min - 2), color, -1)
            cv2.putText(image_np,
                        text,
                        (self.bounding_box.x_min, self.bounding_box.y_min - 2),
                        txt_font,
                        0.5, (0, 0, 0),
                        thickness=1,
                        lineType=cv2.LINE_AA)

    def is_person(self):
        return self.label == 'person'

    def is_speed_limit(self):
        return self.label in [
            'speed limit 30', 'speed limit 60', 'speed limit 90'
        ]

    def is_stop_sign(self):
        return self.label == 'stop sign' or self.label == 'stop marking'

    def is_traffic_light(self):
        return self.label in [
            'red traffic light', 'yellow traffic light', 'green traffic light',
            'off traffic light'
        ]

    def is_vehicle(self):
        # Might want to include train.
        return self.label in VEHICLE_LABELS

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Obstacle {}, label: {}, bbox: {}, trajectory {}'.format(
            self.id, self.label, self.bounding_box, self.trajectory)
