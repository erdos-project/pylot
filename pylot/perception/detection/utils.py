import copy

import numpy as np

import pylot.utils

OBSTACLE_LABELS = {
    'car', 'bicycle', 'motorcycle', 'bus', 'truck', 'vehicle', 'person',
    'stop sign', 'parking meter', 'cat', 'dog', 'speed limit 30',
    'speed limit 60', 'speed limit 90'
}

PYLOT_BBOX_COLOR_MAP = {
    'person': [0, 128, 0],
    'vehicle': [128, 0, 0],
    'car': [128, 0, 0],
    'bicycle': [128, 0, 0],
    'motorcycle': [128, 0, 0],
    'bus': [128, 0, 0],
    'truck': [128, 0, 0],
    'stop marking': [128, 128, 0],
    'speed limit': [255, 255, 0],
    'red traffic light': [0, 0, 255],
    'yellow traffic light': [0, 255, 255],
    'green traffic light': [0, 255, 0],
    'off traffic light': [0, 0, 0],
    '': [255, 255, 255],
}

coco_bbox_color_list = np.array([
    1.000, 1.000, 1.000, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494,
    0.184, 0.556, 0.466, 0.674, 0.188, 0.301, 0.745, 0.933, 0.635, 0.078,
    0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000, 0.000,
    1.000, 0.500, 0.000, 0.749, 0.749, 0.000, 0.000, 1.000, 0.000, 0.000,
    0.000, 1.000, 0.667, 0.000, 1.000, 0.333, 0.333, 0.000, 0.333, 0.667,
    0.000, 0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000,
    0.667, 1.000, 0.000, 1.000, 0.333, 0.000, 1.000, 0.667, 0.000, 1.000,
    1.000, 0.000, 0.000, 0.333, 0.500, 0.000, 0.667, 0.500, 0.000, 1.000,
    0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500, 0.333, 0.667, 0.500,
    0.333, 1.000, 0.500, 0.667, 0.000, 0.500, 0.667, 0.333, 0.500, 0.667,
    0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500, 1.000, 0.333,
    0.500, 1.000, 0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000,
    0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000, 1.000, 0.333,
    0.333, 1.000, 0.333, 0.667, 1.000, 0.333, 1.000, 1.000, 0.667, 0.000,
    1.000, 0.667, 0.333, 1.000, 0.667, 0.667, 1.000, 0.667, 1.000, 1.000,
    1.000, 0.000, 1.000, 1.000, 0.333, 1.000, 1.000, 0.667, 1.000, 0.167,
    0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000,
    0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000,
    0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000,
    0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000,
    0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833,
    0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.143, 0.143, 0.143, 0.286,
    0.286, 0.286, 0.429, 0.429, 0.429, 0.571, 0.571, 0.571, 0.714, 0.714,
    0.714, 0.857, 0.857, 0.857, 0.000, 0.447, 0.741, 0.50, 0.5, 0
]).astype(np.float32)


class BoundingBox2D(object):
    """Class that stores a 2D bounding box."""
    def __init__(self, x_min, x_max, y_min, y_max):
        assert x_min < x_max and y_min < y_max
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def get_min_point(self) -> pylot.utils.Vector2D:
        return pylot.utils.Vector2D(self.x_min, self.y_min)

    def get_max_point(self) -> pylot.utils.Vector2D:
        return pylot.utils.Vector2D(self.x_max, self.y_max)

    def get_height(self):
        return self.y_max - self.y_min

    def get_width(self):
        return self.x_max - self.x_min

    def get_center_point(self) -> pylot.utils.Vector2D:
        return pylot.utils.Vector2D((self.x_min + self.x_max) // 2,
                                    (self.y_min + self.y_max) // 2)

    def as_width_height_bbox(self):
        return [self.x_min, self.y_min, self.get_width(), self.get_height()]

    def is_within(self, point) -> bool:
        """Checks if a point is within the bounding box."""
        return (point.x >= self.x_min and point.x <= self.x_max
                and point.y >= self.y_min and point.y <= self.y_max)

    def calculate_iou(self, other_bbox) -> float:
        """Calculate the IoU of a single bounding box.

        Args:
            other_bbox (:py:class:`.BoundingBox2D`): The other bounding box.

        Returns:
            :obj:`float`: The IoU of the two bounding boxes.
        """
        if (other_bbox.x_min > other_bbox.x_max
                or other_bbox.y_min > other_bbox.y_max):
            raise AssertionError(
                "Other bbox is malformed {}".format(other_bbox))

        if self.x_min > self.x_max or self.y_min > self.y_max:
            raise AssertionError("Bounding box is malformed {}".format(self))

        if (self.x_max < other_bbox.x_min or other_bbox.x_max < self.x_min
                or self.y_max < other_bbox.y_min
                or other_bbox.y_max < self.y_min):
            return 0.0

        inter_x1 = max([self.x_min, other_bbox.x_min])
        inter_x2 = min([self.x_max, other_bbox.x_max])

        inter_y1 = max([self.y_min, other_bbox.y_min])
        inter_y2 = min([self.y_max, other_bbox.y_max])

        inter_area = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
        gt_area = (self.x_max - self.x_min + 1) * (self.y_max - self.y_min + 1)
        pred_area = (other_bbox.x_max - other_bbox.x_min +
                     1) * (other_bbox.y_max - other_bbox.y_min + 1)
        return float(inter_area) / (gt_area + pred_area - inter_area)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'BoundingBox2D(xmin: {}, xmax: {}, ymin: {}, ymax: {})'.format(
            self.x_min, self.x_max, self.y_min, self.y_max)


class BoundingBox3D(object):
    """Class used to store a 3D bounding box.

    Args:
        transform (:py:class:`~pylot.utils.Transform`): Transform of the
            bounding box (rotation is (0, 0, 0)).
        extent (:py:class:`~pylot.utils.Vector3D`): The extent of the bounding
            box.

    Attributes:
        transform (:py:class:`~pylot.utils.Transform`): Transform of the
            bounding box (rotation is (0, 0, 0)).
        extent (:py:class:`~pylot.utils.Vector3D`): The extent of the bounding
            box.
    """
    def __init__(self,
                 transform: pylot.utils.Transform = None,
                 extent: pylot.utils.Vector3D = None,
                 corners=None):
        self.transform = transform
        self.extent = extent
        self.corners = corners

    @classmethod
    def from_dimensions(cls, bbox_dimensions, location, rotation_y):
        """Creates a 3D bounding box.

        Args:
            bbox_dimensions: The height, width and length of the bbox.
            location: The location of the box in the camera frame.
            rotation: The rotation of the bbox.

        Returns:
            :py:class:`.BoundingBox3D`: A bounding box instance.
        """
        c, s = np.cos(rotation_y), np.sin(rotation_y)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        l, w, h = bbox_dimensions[2], bbox_dimensions[1], bbox_dimensions[0]
        x_corners = [
            l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2
        ]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [
            w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2
        ]
        corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
        corners_3d = np.dot(R, corners).transpose(1, 0)
        corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(
            1, 3)
        return cls(corners=corners_3d)

    @classmethod
    def from_simulator_bounding_box(cls, bbox):
        """Creates a pylot bounding box from a simulator bounding box.

        Args:
            bbox: The bounding box to transform.

        Returns:
            :py:class:`.BoundingBox3D`: A bounding box instance.
        """
        transform = pylot.utils.Transform(
            pylot.utils.Location.from_simulator_location(bbox.location),
            pylot.utils.Rotation())
        extent = pylot.utils.Vector3D.from_simulator_vector(bbox.extent)
        return cls(transform, extent)

    def as_simulator_bounding_box(self):
        """Retrieves the bounding box as instance of a simulator bounding box.

        Returns:
            A instance of a simulator class that represents the bounding box.
        """
        from carla import BoundingBox
        bb_loc = self.transform.location.as_simulator_location()
        bb_extent = self.extent.as_simulator_vector()
        return BoundingBox(bb_loc, bb_extent)

    def visualize(self, world, actor_transform, time_between_frames=100):
        """Visualizes the bounding box on the world.

        Args:
            world: The simulator world instance to visualize the bounding
                box on.
            actor_transform (:py:class:`~pylot.utils.Transform`): The current
                transform of the actor that the bounding box is of.
            time_between_frames (:obj:`float`): Time in ms to show the bounding
                box for.
        """
        bb = self.as_simulator_bounding_box()
        bb.location += actor_transform.location()
        world.debug.draw_box(bb,
                             actor_transform.rotation.as_simulator_rotation(),
                             life_time=time_between_frames / 1000.0)

    def to_camera_view(self, obstacle_transform: pylot.utils.Transform,
                       extrinsic_matrix, intrinsic_matrix):
        """Converts the coordinates of the bounding box for the given obstacle
        to the coordinates in the view of the camera.

        This method retrieves the extent of the bounding box, transforms them
        to coordinates relative to the bounding box origin, then converts those
        to coordinates relative to the obstacle.

        These coordinates are then considered to be in the world coordinate
        system, which is mapped into the camera view. A negative z-value
        signifies that the bounding box is behind the camera plane.

        Note that this function does not cap the coordinates to be within the
        size of the camera image.

        Args:
            obstacle_transform (:py:class:`~pylot.utils.Transform`): The
                transform of the obstacle that the bounding box is associated
                with.
            extrinsic_matrix: The extrinsic matrix of the camera.
            intrinsic_matrix: The intrinsic matrix of the camera.

        Returns:
            A list of 8 Location instances specifying the 8 corners of the
            bounding box.
        """
        # Retrieve the eight coordinates of the bounding box with respect to
        # the origin of the bounding box.
        import numpy as np
        if self.corners is not None:
            pts_2d = np.dot(intrinsic_matrix,
                            self.corners.transpose(1, 0)).transpose(1, 0)
            pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
            camera_coordinates = [
                pylot.utils.Vector2D(pt[0], pt[1]) for pt in pts_2d
            ]
            return camera_coordinates

        extent = self.extent
        bbox = np.array([
            pylot.utils.Location(x=+extent.x, y=+extent.y, z=-extent.z),
            pylot.utils.Location(x=-extent.x, y=+extent.y, z=-extent.z),
            pylot.utils.Location(x=-extent.x, y=-extent.y, z=-extent.z),
            pylot.utils.Location(x=+extent.x, y=-extent.y, z=-extent.z),
            pylot.utils.Location(x=+extent.x, y=+extent.y, z=+extent.z),
            pylot.utils.Location(x=-extent.x, y=+extent.y, z=+extent.z),
            pylot.utils.Location(x=-extent.x, y=-extent.y, z=+extent.z),
            pylot.utils.Location(x=+extent.x, y=-extent.y, z=+extent.z)
        ])

        # Transform the vertices with respect to the bounding box transform.
        bbox = self.transform.transform_locations(bbox)

        # Convert the bounding box relative to the world.
        bbox = obstacle_transform.transform_locations(bbox)

        # Obstacle's transform is relative to the world. Thus, the bbox
        # contains the 3D bounding box vertices relative to the world.
        camera_coordinates = []
        for vertex in bbox:
            location_2D = vertex.to_camera_view(extrinsic_matrix,
                                                intrinsic_matrix)

            # Add the points to the image.
            camera_coordinates.append(location_2D)

        return camera_coordinates

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "BoundingBox3D(transform: {}, extent: {})".format(
            self.transform, self.extent)


def get_bounding_box_in_camera_view(bb_coordinates, image_width, image_height):
    """Creates the bounding box in the view of the camera image using the
    coordinates generated with respect to the camera transform.

    Args:
        bb_coordinates: 8 :py:class:`~pylot.utils.Location` coordinates of
            the bounding box relative to the camera transform.
        image_width (:obj:`int`): The width of the image being published by the
            camera.
        image_height (:obj:`int`): The height of the image being published by
            the camera.

    Returns:
        :py:class:`.BoundingBox2D`: a bounding box, or None if the bounding box
            does not fall into the view of the camera.
    """
    # Make sure that atleast 2 of the bounding box coordinates are in front.
    z_vals = [loc.z for loc in bb_coordinates if loc.z >= 0]
    if len(z_vals) < 2:
        return None

    # Create the thresholding line segments of the camera view.
    from shapely.geometry import LineString
    left = LineString(((0, 0), (0, image_height)))
    bottom = LineString(((0, image_height), (image_width, image_height)))
    right = LineString(((image_width, image_height), (image_width, 0)))
    top = LineString(((image_width, 0), (0, 0)))
    camera_thresholds = [left, bottom, right, top]

    def threshold(p1, p2):
        points = []
        # If the points are themselves within the image, add them to the
        # set of thresholded points.
        if (p1[0] >= 0 and p1[0] < image_width and p1[1] >= 0
                and p1[1] < image_height):
            points.append(p1)

        if (p2[0] >= 0 and p2[0] < image_width and p2[1] >= 0
                and p2[1] < image_height):
            points.append(p2)

        # Compute the intersection of the line segment formed by p1 -- p2
        # with all the thresholds of the camera image.
        p12 = LineString((p1, p2))
        for camera_threshold in camera_thresholds:
            p = p12.intersection(camera_threshold)
            if not p.is_empty:
                if p.geom_type == 'Point':
                    points.append((p.x, p.y))
                elif p.geom_type == 'LineString':
                    for coord in p.coords:
                        points.append((coord[0], coord[1]))
        return points

    # Go over each of the segments of the bounding box and threshold it to
    # be inside the image.
    thresholded_points = []
    points = [(int(loc.x), int(loc.y)) for loc in bb_coordinates]
    # Bottom plane thresholded.
    thresholded_points.extend(threshold(points[0], points[1]))
    thresholded_points.extend(threshold(points[1], points[2]))
    thresholded_points.extend(threshold(points[2], points[3]))
    thresholded_points.extend(threshold(points[3], points[0]))

    # Top plane thresholded.
    thresholded_points.extend(threshold(points[4], points[5]))
    thresholded_points.extend(threshold(points[5], points[6]))
    thresholded_points.extend(threshold(points[6], points[7]))
    thresholded_points.extend(threshold(points[7], points[4]))

    # Remaining segments thresholded.
    thresholded_points.extend(threshold(points[0], points[4]))
    thresholded_points.extend(threshold(points[1], points[5]))
    thresholded_points.extend(threshold(points[2], points[6]))
    thresholded_points.extend(threshold(points[3], points[7]))

    if len(thresholded_points) == 0:
        return None
    else:
        x = [int(x) for x, _ in thresholded_points]
        y = [int(y) for _, y in thresholded_points]
        if min(x) < max(x) and min(y) < max(y):
            return BoundingBox2D(min(x), max(x), min(y), max(y))
        else:
            return None


def load_coco_labels(labels_path):
    """Returns a map from index to label.

    Args:
        labels_path (:obj:`str`): Path to a file storing a label on each line.
    """
    labels_map = {}
    with open(labels_path) as labels_file:
        labels = labels_file.read().splitlines()
        index = 1
        for label in labels:
            labels_map[index] = label
            index += 1
    return labels_map


def load_coco_bbox_colors(coco_labels):
    """Returns a map from label to color."""
    # Transform to RGB values.
    bbox_color_list = coco_bbox_color_list.reshape((-1, 3)) * 255
    # Transform to ints
    bbox_colors = [(bbox_color_list[_]).astype(np.uint8)
                   for _ in range(len(bbox_color_list))]
    bbox_colors = np.array(bbox_colors,
                           dtype=np.uint8).reshape(len(bbox_colors), 1, 1, 3)

    colors = {}
    for category, label in coco_labels.items():
        colors[label] = bbox_colors[category - 1][0][0].tolist()
    return colors


def get_prediction_results(ground_truths, predictions, iou_threshold):
    """Calculate the number of true positives, false positives and false
    negatives from the given ground truth and predictions."""
    true_pos, false_pos, false_neg = None, None, None

    # If there are no predictions, then everything is a false negative.
    if len(predictions) == 0:
        true_pos, false_pos = 0, 0
        false_neg = len(ground_truths)
        return true_pos, false_pos, false_neg

    # If there is no ground truth, everything is a false positive.
    if len(ground_truths) == 0:
        true_pos, false_neg = 0, 0
        false_pos = len(predictions)
        return true_pos, false_pos, false_neg

    # Iterate over the predictions and calculate the IOU of each prediction
    # with each ground truth.
    ious = []
    for i, prediction in enumerate(predictions):
        for j, ground_truth in enumerate(ground_truths):
            iou = prediction.calculate_iou(ground_truth)
            if iou > iou_threshold:
                ious.append((i, j, iou))

    # If no IOUs were over the threshold, return all predictions as false
    # positives and all ground truths as false negatives.
    if len(ious) == 0:
        true_pos = 0
        false_pos, false_neg = len(predictions), len(ground_truths)
    else:
        # Sort the IOUs and match each box only once.
        ground_truths_matched, predictions_matched = set(), set()
        matched = []
        for prediction, ground_truth, iou in sorted(ious,
                                                    key=lambda x: x[-1],
                                                    reverse=True):
            if (ground_truth not in ground_truths_matched
                    and prediction not in predictions_matched):
                ground_truths_matched.add(ground_truth)
                predictions_matched.add(prediction)
                matched.append((prediction, ground_truth, iou))

        # The matches are the true positives.
        true_pos = len(matched)
        # The unmatched predictions are the false positives.
        false_pos = len(predictions) - len(predictions_matched)
        # The umatched ground truths are the false negatives.
        false_neg = len(ground_truths) - len(ground_truths_matched)

    return true_pos, false_pos, false_neg


def get_precision_recall(true_positives, false_positives, false_negatives):
    precision, recall = None, None
    if true_positives + false_positives == 0:
        precision = 0.0
    else:
        precision = float(true_positives) / float(true_positives +
                                                  false_positives)

    if true_positives + false_negatives == 0:
        recall = 0.0
    else:
        recall = float(true_positives) / float(true_positives +
                                               false_negatives)

    return (precision, recall)


def get_precision_recall_at_iou(ground_truths, predictions, iou_threshold):
    true_pos, false_pos, false_neg = get_prediction_results(
        ground_truths, predictions, iou_threshold)
    return get_precision_recall(true_pos, false_pos, false_neg)


def get_mAP(ground_obstacles, obstacles):
    """Return mAP with IoU threshold of 0.5"""
    # Sort bboxes descending by score.
    sorted_obstacles = \
        sorted(obstacles, key=lambda o: o.confidence, reverse=True)
    detected_bboxes = [o.bounding_box_2D for o in sorted_obstacles]
    ground_bboxes = [obstacle.bounding_box_2D for obstacle in ground_obstacles]
    # Compute recall precision. The results are sorted in descending
    # order by recall.
    prec_rec = []
    while (len(detected_bboxes) > 0):
        # Get precision recall with 0.5 IoU threshold .
        precision, recall = get_precision_recall_at_iou(
            ground_bboxes, detected_bboxes, 0.5)
        prec_rec.append((precision, recall))
        detected_bboxes.pop()
    # Append (0, 0) to also cover the area from first recall point to 0 recall.
    prec_rec.append((0, 0))
    avg_precision = 0.0
    max_precision = 0.0
    max_precision = None
    last_recall = None
    for (precision, recall) in prec_rec:
        if max_precision is None:
            max_precision = precision
            last_recall = recall
        else:
            avg_precision += (last_recall - recall) * max_precision
            max_precision = max(max_precision, precision)
            last_recall = recall
    return avg_precision


def get_obstacle_locations(obstacles, depth_msg, ego_transform, camera_setup,
                           logger):
    from pylot.perception.messages import DepthFrameMessage, PointCloudMessage
    if isinstance(depth_msg, PointCloudMessage):
        point_cloud = depth_msg.point_cloud
        # Get the position of the camera in world frame of reference.
        transformed_camera_setup = copy.deepcopy(camera_setup)
        transformed_camera_setup.set_transform(
            ego_transform * transformed_camera_setup.transform)

        obstacles_with_location = []
        for obstacle in obstacles:
            location = point_cloud.get_pixel_location(
                obstacle.bounding_box_2D.get_center_point(),
                transformed_camera_setup)
            if location is not None:
                obstacle.transform = pylot.utils.Transform(
                    location, pylot.utils.Rotation())
                obstacles_with_location.append(obstacle)
            else:
                logger.error(
                    'Could not find world location for obstacle {}'.format(
                        obstacle))
        return obstacles_with_location
    elif isinstance(depth_msg, DepthFrameMessage):
        depth_frame = depth_msg.frame
        depth_frame.camera_setup.set_transform(
            ego_transform * depth_frame.camera_setup.transform)

        for obstacle in obstacles:
            center_point = obstacle.bounding_box_2D.get_center_point()
            # Sample several points around the center of the bounding box
            # in case the bounding box is not well centered on the obstacle.
            # In such situations the center point might be in between legs,
            # and thus we might overestimate the distance.
            sample_points = []
            for delta_x in range(-30, 30, 5):
                for delta_y in range(-30, 30, 5):
                    sample_point = center_point + pylot.utils.Vector2D(
                        delta_x, delta_y)
                    if obstacle.bounding_box.is_within(sample_point):
                        sample_points.append(sample_point)
            locations = depth_frame.get_pixel_locations(sample_points)
            # Choose the closest from the locations of the sampled points.
            min_distance = np.infty
            closest_location = None
            for location in locations:
                dist = location.distance(ego_transform.location)
                if dist < min_distance:
                    min_distance = dist
                    closest_location = location
            obstacle.transform = pylot.utils.Transform(closest_location,
                                                       pylot.utils.Rotation())
        return obstacles
    else:
        raise ValueError('Unexpected depth message type')
