import numpy as np

import pylot.utils
from pylot.perception.detection.utils import BoundingBox2D, BoundingBox3D, \
    get_bounding_box_in_camera_view

VEHICLE_LABELS = {'car', 'bicycle', 'motorcycle', 'bus', 'truck', 'vehicle'}


class Obstacle(object):
    """Class used to store info about obstacles.

    This class provides helper functions to detect obstacles and provide
    bounding boxes for them.

    Args:
        bounding_box (:py:class:`.BoundingBox2D`): The bounding box of the
            obstacle (can be 2D or 3D).
        confidence (:obj:`float`): The confidence of the detection.
        label (:obj:`str`): The label of the obstacle.
        id (:obj:`int`): The identifier of the obstacle.
        transform (:py:class:`~pylot.utils.Transform`, optional): Transform of
            the obstacle in the world.

    Attributes:
        bounding_box (:py:class:`~pylot.utisl.BoundingBox2D`): Bounding box of
            the obstacle (can be 2D or 3D).
        confidence (:obj:`float`): The confidence of the detection.
        label (:obj:`str`): The label of the obstacle.
        id (:obj:`int`): The identifier of the obstacle.
        transform (:py:class:`~pylot.utils.Transform`): Transform of the
            obstacle.
    """
    def __init__(self,
                 bounding_box,
                 confidence: float,
                 label: str,
                 id: int = -1,
                 transform: pylot.utils.Transform = None,
                 detailed_label: str = '',
                 bounding_box_2D: BoundingBox2D = None):
        self.bounding_box = bounding_box
        if isinstance(bounding_box, BoundingBox2D):
            self.bounding_box_2D = bounding_box
        else:
            self.bounding_box_2D = bounding_box_2D
        self.confidence = confidence
        self.label = label
        self.id = id
        self.transform = transform
        self.detailed_label = detailed_label
        if label == 'vehicle':
            self.segmentation_class = 10
        elif label == 'person':
            self.segmentation_class = 4
        else:
            self.segmentation_class = None
        # Thresholds to be used for detection of the obstacle.
        self.__segmentation_threshold = 0.20
        self.__depth_threshold = 5

    @classmethod
    def from_simulator_actor(cls, actor):
        """Creates an Obstacle from a simulator actor.

        Args:
            actor: The actor to initialize the obstacle with.

        Returns:
            :py:class:`.Obstacle`: An obstacle instance.
        """
        from carla import Vehicle, Walker
        if not isinstance(actor, (Vehicle, Walker)):
            raise ValueError("The actor should be of type Vehicle or "
                             "Walker to initialize the Obstacle class.")
        # We do not use everywhere from_simulator* methods in order to reduce
        # runtime.
        # Convert the transform provided by the simulation to the Pylot class.
        transform = pylot.utils.Transform.from_simulator_transform(
            actor.get_transform())
        # Convert the bounding box from the simulation to the Pylot one.
        bounding_box = BoundingBox3D.from_simulator_bounding_box(
            actor.bounding_box)
        if isinstance(actor, Vehicle):
            label = 'vehicle'
        else:
            label = 'person'
        # Get the simulator actor from type_id (e.g. vehicle.ford.mustang).
        detailed_label = actor.type_id
        # TODO (Sukrit): Move from vehicles and people to separate classes
        # for bicycles, motorcycles, cars and persons.
        return cls(bounding_box, 1.0, label, actor.id, transform,
                   detailed_label)

    def as_mot16_str(self, timestamp):
        if not self.bounding_box_2D:
            raise ValueError(
                'Obstacle {} does not have 2D bounding box'.format(self.id))
        log_line = "{},{},{},{},{},{},{},{},{},{}\n".format(
            timestamp, self.id, self.bounding_box_2D.x_min,
            self.bounding_box_2D.y_min, self.bounding_box_2D.get_width(),
            self.bounding_box_2D.get_height(), 1.0, -1, -1, -1)
        return log_line

    def _distance(self, other_transform: pylot.utils.Transform):
        """Computes the distance from the obstacle to the other transform.

        The distance provides an estimate of the depth returned by the depth
        camera sensor in the simulator. As a result, the distance is defined
        as the displacement of the obstacle along either the X or the Y axis.

        Args:
            other_transform (:py:class:`~pylot.utils.Transform`): The other
                transform.

        Returns:
            :obj:`float`: The distance (in metres) of the obstacle from the
            transform.
        """
        import numpy as np
        if self.transform is None:
            raise ValueError('Obstacle {} does not have a transform'.format(
                self.id))
        # Get the location of the vehicle and the obstacle as numpy arrays.
        other_location = other_transform.location.as_numpy_array()
        obstacle_location = self.transform.location.as_numpy_array()

        # Calculate the vector from the vehicle to the obstacle.
        # Scale it by the forward vector, and calculate the norm.
        relative_vector = other_location - obstacle_location
        distance = np.linalg.norm(
            relative_vector * other_transform.forward_vector.as_numpy_array())
        return distance

    def draw_on_frame(self,
                      frame,
                      bbox_color_map,
                      ego_transform: pylot.utils.Transform = None,
                      text: str = None):
        """Annotate the image with the bounding box of the obstacle."""
        if text is None:
            text = '{}, {:.1f}'.format(self.label, self.confidence)
            if self.id != -1:
                text += ', id:{}'.format(self.id)
            if ego_transform is not None and self.transform is not None:
                text += ', {:.1f}m'.format(
                    ego_transform.location.distance(self.transform.location))
        if self.label in bbox_color_map:
            color = bbox_color_map[self.label]
        else:
            color = [255, 255, 255]
        # Show bounding box.
        if self.bounding_box_2D:
            # Draw the 2D bounding box if available.
            frame.draw_box(self.bounding_box_2D.get_min_point(),
                           self.bounding_box_2D.get_max_point(), color)
            frame.draw_text(self.bounding_box_2D.get_min_point(), text, color)
        elif isinstance(self.bounding_box, BoundingBox3D):
            if self.bounding_box.corners is None:
                raise ValueError(
                    'Obstacle {} does not have bbox corners'.format(self.id))
            corners = self.bounding_box.to_camera_view(
                None, frame.camera_setup.get_extrinsic_matrix(),
                frame.camera_setup.get_intrinsic_matrix())
            frame.draw_3d_box(corners, color)
        else:
            raise ValueError('Obstacle {} does not have bounding box'.format(
                self.id))

    def draw_trajectory_on_frame(self,
                                 trajectory,
                                 frame,
                                 point_color,
                                 draw_label: bool = False):
        # Intrinsic and extrinsic matrix of the top down camera.
        extrinsic_matrix = frame.camera_setup.get_extrinsic_matrix()
        intrinsic_matrix = frame.camera_setup.get_intrinsic_matrix()
        if isinstance(self.bounding_box, BoundingBox3D):
            # Draw bounding boxes.
            start_location = self.bounding_box.transform.location - \
                self.bounding_box.extent
            end_location = self.bounding_box.transform.location + \
                self.bounding_box.extent
            for transform in trajectory:
                [start_transform,
                 end_transform] = transform.transform_locations(
                     [start_location, end_location])
                start_point = start_transform.to_camera_view(
                    extrinsic_matrix, intrinsic_matrix)
                end_point = end_transform.to_camera_view(
                    extrinsic_matrix, intrinsic_matrix)
                if frame.in_frame(start_point) or frame.in_frame(end_point):
                    frame.draw_box(start_point, end_point, point_color)
        else:
            # Draw points.
            for transform in trajectory:
                screen_point = transform.location.to_camera_view(
                    extrinsic_matrix, intrinsic_matrix)
                if frame.in_frame(screen_point):
                    # Draw trajectory on frame.
                    frame.draw_point(screen_point, point_color)
        if draw_label and len(trajectory) > 0:
            text = '{}, {}'.format(self.label, self.id)
            screen_point = trajectory[-1].location.to_camera_view(
                extrinsic_matrix, intrinsic_matrix)
            frame.draw_text(screen_point, text, point_color)

    def get_bounding_box_corners(self,
                                 obstacle_transform,
                                 obstacle_radius=None):
        """Gets the corners of the obstacle's bounding box.
        Note:
            The bounding box is applied on the given obstacle transfom, and not
            on the default obstacle transform.
        """
        # Use 3d bounding boxes if available, otherwise use default
        if isinstance(self.bounding_box, BoundingBox3D):
            start_location = (self.bounding_box.transform.location -
                              self.bounding_box.extent)
            end_location = (self.bounding_box.transform.location +
                            self.bounding_box.extent)
            [start_location,
             end_location] = obstacle_transform.transform_locations(
                 [start_location, end_location])
        else:
            obstacle_radius_loc = pylot.utils.Location(obstacle_radius,
                                                       obstacle_radius)
            start_location = obstacle_transform.location - obstacle_radius_loc
            end_location = obstacle_transform.location + obstacle_radius_loc
        return [
            min(start_location.x, end_location.x),
            min(start_location.y, end_location.y),
            max(start_location.x, end_location.x),
            max(start_location.y, end_location.y)
        ]

    def get_in_log_format(self):
        if not self.bounding_box_2D:
            raise ValueError(
                'Obstacle {} does not have 2D bounding box'.format(self.id))
        min_point = self.bounding_box_2D.get_min_point()
        max_point = self.bounding_box_2D.get_max_point()
        return (self.label, self.detailed_label, self.id,
                ((min_point.x, min_point.y), (max_point.x, max_point.y)))

    def is_animal(self):
        return self.label in [
            'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe'
        ]

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

    def populate_bounding_box_2D(self, depth_frame, segmented_frame):
        """Populates the 2D bounding box for the obstacle.

        Heuristically uses the depth frame and segmentation frame to figure out
        if the obstacle is in view of the camera or not.

        Args:
            depth_frame (:py:class:`~pylot.perception.depth_frame.DepthFrame`):
                Depth frame used to compare the depth to the distance of the
                obstacle from the sensor.
            segmented_frame (:py:class:`~pylot.perception.segmentation.segmented_frame.SegmentedFrame`):  # noqa: E501
                Segmented frame used to refine the conversions.

        Returns:
            :py:class:`~pylot.utisl.BoundingBox2D`: An instance representing a
            rectangle over the obstacle if the obstacle is deemed to be
            visible, None otherwise.
        """
        if self.bounding_box_2D:
            return self.bounding_box_2D
        # Convert the bounding box of the obstacle to the camera coordinates.
        bb_coordinates = self.bounding_box.to_camera_view(
            self.transform, depth_frame.camera_setup.get_extrinsic_matrix(),
            depth_frame.camera_setup.get_intrinsic_matrix())

        # Threshold the bounding box to be within the camera view.
        bbox_2d = get_bounding_box_in_camera_view(
            bb_coordinates, depth_frame.camera_setup.width,
            depth_frame.camera_setup.height)
        if not bbox_2d:
            return None
        # Crop the segmented and depth image to the given bounding box.
        cropped_image = segmented_frame.as_numpy_array()[
            bbox_2d.y_min:bbox_2d.y_max, bbox_2d.x_min:bbox_2d.x_max]
        cropped_depth = depth_frame.as_numpy_array()[
            bbox_2d.y_min:bbox_2d.y_max, bbox_2d.x_min:bbox_2d.x_max]

        # If the size of the bounding box is greater than 0, ensure that the
        # bounding box contains more than a threshold of pixels corresponding
        # to the required segmentation class.
        if cropped_image.size > 0:
            masked_image = np.zeros_like(cropped_image)
            masked_image[np.where(
                cropped_image == self.segmentation_class)] = 1
            seg_threshold = self.__segmentation_threshold * masked_image.size
            if np.sum(masked_image) >= seg_threshold:
                # The bounding box contains the required number of pixels that
                # belong to the required class. Ensure that the depth of the
                # obstacle is the depth in the image.
                masked_depth = cropped_depth[np.where(masked_image == 1)]
                mean_depth = np.mean(masked_depth) * 1000
                depth = self._distance(
                    depth_frame.camera_setup.get_transform())
                if abs(depth - mean_depth) <= self.__depth_threshold:
                    self.bounding_box_2D = bbox_2d
                    return bbox_2d
        return None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        obstacle = 'Obstacle(id: {}, label: {}, confidence: {}, '\
            'bbox: {})'.format(self.id, self.label, self.confidence,
                               self.bounding_box)
        if self.transform:
            return obstacle + ' at ' + str(self.transform)
        else:
            return obstacle
