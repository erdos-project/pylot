from enum import Enum

import numpy as np

import pylot.utils
from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.detection.utils import BoundingBox2D, \
    get_bounding_box_in_camera_view


class TrafficLightColor(Enum):
    """Enum to represent the states of a traffic light."""
    RED = 1
    YELLOW = 2
    GREEN = 3
    OFF = 4

    def get_label(self):
        """Gets the label of a traffic light color.

        Returns:
            :obj:`str`: The label string.
        """
        if self.value == 1:
            return 'red traffic light'
        elif self.value == 2:
            return 'yellow traffic light'
        elif self.value == 3:
            return 'green traffic light'
        else:
            return 'off traffic light'

    def get_color(self):
        if self.value == 1:
            return [255, 0, 0]
        elif self.value == 2:
            return [255, 165, 0]
        elif self.value == 3:
            return [0, 255, 0]
        else:
            return [0, 0, 0]


class TrafficLight(Obstacle):
    """Class used to store info about traffic lights.

    Args:
        confidence (:obj:`float`): The confidence of the detection.
        state (:py:class:`.TrafficLightColor`): The state of the traffic light.
        id (:obj:`int`, optional): The identifier of the traffic light.
        transform (:py:class:`~pylot.utils.Transform`, optional): Transform of
            the traffic light.
        trigger_volume_extent (:py:class:`pylot.utils.Vector3D`, optional): The
            extent of the trigger volume of the light.
        bounding_box (:py:class:`.BoundingBox2D`, optional): The bounding box
            of the traffic light in camera view.

    Attributes:
        confidence (:obj:`float`): The confidence of the detection.
        state (:py:class:`.TrafficLightColor`): The state of the traffic light.
        id (:obj:`int`): The identifier of the traffic light.
        transform (:py:class:`~pylot.utils.Transform`): Transform of the
            traffic light.
        trigger_volume_extent (:py:class:`pylot.utils.Vector3D`): The extent
            of the trigger volume of the light.
        bounding_box (:py:class:`.BoundingBox2D`, optional): The bounding box
            of the traffic light in camera view.
    """
    def __init__(self,
                 confidence: float,
                 state: TrafficLightColor,
                 id: int = -1,
                 transform: pylot.utils.Transform = None,
                 trigger_volume_extent: pylot.utils.Vector3D = None,
                 bounding_box: BoundingBox2D = None):
        super(TrafficLight, self).__init__(bounding_box, confidence,
                                           state.get_label(), id, transform)
        self.state = state
        self.trigger_volume_extent = trigger_volume_extent

    @classmethod
    def from_simulator_actor(cls, traffic_light):
        """ Creates a TrafficLight from a simulator traffic light actor.

        Args:
            traffic_light: A simulator traffic light actor.

        Returns:
            :py:class:`.TrafficLight`: A traffic light.
        """
        from carla import TrafficLight, TrafficLightState
        if not isinstance(traffic_light, TrafficLight):
            raise ValueError('The traffic light must be a TrafficLight')
        # Retrieve the Transform of the TrafficLight.
        transform = pylot.utils.Transform.from_simulator_transform(
            traffic_light.get_transform())
        # Retrieve the Trigger Volume of the TrafficLight.
        trigger_volume_extent = pylot.utils.Vector3D(
            traffic_light.trigger_volume.extent.x,
            traffic_light.trigger_volume.extent.y,
            traffic_light.trigger_volume.extent.z)
        traffic_light_state = traffic_light.get_state()
        state = TrafficLightColor.OFF
        if traffic_light_state == TrafficLightState.Red:
            state = TrafficLightColor.RED
        elif traffic_light_state == TrafficLightState.Yellow:
            state = TrafficLightColor.YELLOW
        elif traffic_light_state == TrafficLightState.Green:
            state = TrafficLightColor.GREEN
        return cls(1.0, state, traffic_light.id, transform,
                   trigger_volume_extent)

    def draw_on_bird_eye_frame(self, frame):
        # Intrinsic and extrinsic matrix of the top down camera.
        extrinsic_matrix = frame.camera_setup.get_extrinsic_matrix()
        intrinsic_matrix = frame.camera_setup.get_intrinsic_matrix()
        point = self.transform.location.to_camera_view(extrinsic_matrix,
                                                       intrinsic_matrix)
        frame.draw_point(point, self.state.get_color(), r=10)
        frame.draw_text(point, self.state.get_label(), self.state.get_color())

    def is_traffic_light_visible(self,
                                 camera_transform: pylot.utils.Transform,
                                 town_name: str = None,
                                 distance_threshold: int = 70):
        """Checks if the traffic light is visible from the camera transform.

        Args:
            transform (:py:class:`~pylot.utils.Transform`): Transform of the
                camera in the world frame of reference.
            distance_threshold (:obj:`int`): Maximum distance to the camera
                (in m).

        Returns:
            bool: True if the traffic light is visible from the camera
            transform.
        """
        # We dot product the forward vectors (i.e., orientation).
        # Note: we have to rotate the traffic light forward vector
        # so that it's pointing out from the traffic light in the
        # opposite direction in which the ligth is beamed.
        prod = np.dot([
            self.transform.forward_vector.y, -self.transform.forward_vector.x,
            self.transform.forward_vector.z
        ], [
            camera_transform.forward_vector.x,
            camera_transform.forward_vector.y,
            camera_transform.forward_vector.z
        ])
        if self.transform.location.distance(
                camera_transform.location) > distance_threshold:
            return prod > 0.4

        if town_name is None:
            return prod > -0.80
        else:
            if town_name == 'Town01' or town_name == 'Town02':
                return prod > 0.3
        return prod > -0.80

    def get_all_detected_traffic_light_boxes(self, town_name: str, depth_frame,
                                             segmented_image):
        """ Returns traffic lights for all boxes of a simulator traffic light.

        Note:
            All the traffic lights returned will have the same id and
            transform.

        Args:
            town_name (:obj:`str`): Name of the town in which the traffic light
                is.
            depth_frame (:py:class:`~pylot.perception.depth_frame.DepthFrame`):
                 Depth frame.
            segmented_image: A segmented image np array used to refine the
                 bounding boxes.

        Returns:
            list(:py:class:`~pylot.perception.detection.traffic_light.TrafficLight`):
            Detected traffic lights, one for each traffic light box.
        """
        traffic_lights = []
        bboxes = self._get_bboxes(town_name)
        # Convert the returned bounding boxes to 2D and check if the
        # light is occluded. If not, add it to the traffic lights list.
        for bbox in bboxes:
            bounding_box = [
                loc.to_camera_view(
                    depth_frame.camera_setup.get_extrinsic_matrix(),
                    depth_frame.camera_setup.get_intrinsic_matrix())
                for loc in bbox
            ]
            bbox_2d = get_bounding_box_in_camera_view(
                bounding_box, depth_frame.camera_setup.width,
                depth_frame.camera_setup.height)
            if not bbox_2d:
                continue

            # Crop the segmented and depth image to the given bounding box.
            cropped_image = segmented_image[bbox_2d.y_min:bbox_2d.y_max,
                                            bbox_2d.x_min:bbox_2d.x_max]
            cropped_depth = depth_frame.frame[bbox_2d.y_min:bbox_2d.y_max,
                                              bbox_2d.x_min:bbox_2d.x_max]

            if cropped_image.size > 0:
                masked_image = np.zeros_like(cropped_image)
                masked_image[np.where(
                    np.logical_or(cropped_image == 12,
                                  cropped_image == 18))] = 1
                if np.sum(masked_image) >= 0.20 * masked_image.size:
                    masked_depth = cropped_depth[np.where(masked_image == 1)]
                    mean_depth = np.mean(masked_depth) * 1000
                    if abs(mean_depth -
                           bounding_box[0].z) <= 2 and mean_depth < 150:
                        traffic_lights.append(
                            TrafficLight(1.0, self.state, self.id,
                                         self.transform,
                                         self.trigger_volume_extent, bbox_2d))
        return traffic_lights

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'TrafficLight(confidence: {}, state: {}, id: {}, ' \
            'transform: {}, trigger_volume_extent: {}, bbox: {})'.format(
                self.confidence, self.state, self.id, self.transform,
                self.trigger_volume_extent, self.bounding_box)

    def _relative_to_traffic_light(self, points):
        """Transforms the bounding box specified in the points relative to the
        light.

        Args:
            points: An array of length 4 representing the 4 points of the
                rectangle.
        """
        def rotate(yaw, location):
            """ Rotate a given 3D vector around the Z-axis. """
            rotation_matrix = np.identity(3)
            rotation_matrix[0, 0] = np.cos(yaw)
            rotation_matrix[0, 1] = -np.sin(yaw)
            rotation_matrix[1, 0] = np.sin(yaw)
            rotation_matrix[1, 1] = np.cos(yaw)
            location_vector = np.array([[location.x], [location.y],
                                        [location.z]])
            transformed = np.dot(rotation_matrix, location_vector)
            return pylot.utils.Location(x=transformed[0, 0],
                                        y=transformed[1, 0],
                                        z=transformed[2, 0])

        transformed_points = [
            rotate(np.radians(self.transform.rotation.yaw), point)
            for point in points
        ]
        base_relative_points = [
            self.transform.location + point for point in transformed_points
        ]
        return base_relative_points

    def _get_bboxes(self, town_name: str):
        if town_name == 'Town01' or town_name == 'Town02':
            return self._get_bboxes_for_town1_or_2()
        elif town_name == 'Town03':
            return self._get_bboxes_for_town3()
        elif town_name == 'Town04':
            return self._get_bboxes_for_town4()
        elif town_name == 'Town05':
            return self._get_bboxes_for_town5()
        else:
            raise ValueError(
                'Could not find a town named {}'.format(town_name))

    def _get_bboxes_for_town1_or_2(self):
        points = [
            # Back Plane
            pylot.utils.Location(x=-0.5, y=-0.1, z=2),
            pylot.utils.Location(x=+0.1, y=-0.1, z=2),
            pylot.utils.Location(x=+0.1, y=-0.1, z=3),
            pylot.utils.Location(x=-0.5, y=-0.1, z=3),
            # Front Plane
            pylot.utils.Location(x=-0.5, y=0.5, z=2),
            pylot.utils.Location(x=+0.1, y=0.5, z=2),
            pylot.utils.Location(x=+0.1, y=0.5, z=3),
            pylot.utils.Location(x=-0.5, y=0.5, z=3),
        ]
        return [self._relative_to_traffic_light(points)]

    def _get_bboxes_for_town3(self):
        bboxes = []
        if (self.trigger_volume_extent.x > 2 or self.id in [
                66,
                67,
                68,
                71,
                72,
                73,
                75,
                81,
        ]):
            points = [
                # Back Plane
                pylot.utils.Location(x=-5.2, y=-0.2, z=5.5),
                pylot.utils.Location(x=-4.8, y=-0.2, z=5.5),
                pylot.utils.Location(x=-4.8, y=-0.2, z=6.5),
                pylot.utils.Location(x=-5.2, y=-0.2, z=6.5),
                # Front Plane
                pylot.utils.Location(x=-5.2, y=0.4, z=5.5),
                pylot.utils.Location(x=-4.8, y=0.4, z=5.5),
                pylot.utils.Location(x=-4.8, y=0.4, z=6.5),
                pylot.utils.Location(x=-5.2, y=0.4, z=6.5),
            ]
            bboxes.append(self._relative_to_traffic_light(points))
            right_points = [
                point + pylot.utils.Location(x=-3.0) for point in points
            ]
            bboxes.append(self._relative_to_traffic_light(right_points))
            if self.id not in [51, 52, 53]:
                left_points = [
                    point + pylot.utils.Location(x=-6.5) for point in points
                ]
                bboxes.append(self._relative_to_traffic_light(left_points))
        else:
            points = [
                # Back Plane
                pylot.utils.Location(x=-0.5, y=-0.1, z=2),
                pylot.utils.Location(x=+0.1, y=-0.1, z=2),
                pylot.utils.Location(x=+0.1, y=-0.1, z=3),
                pylot.utils.Location(x=-0.5, y=-0.1, z=3),
                # Front Plane
                pylot.utils.Location(x=-0.5, y=0.5, z=2),
                pylot.utils.Location(x=+0.1, y=0.5, z=2),
                pylot.utils.Location(x=+0.1, y=0.5, z=3),
                pylot.utils.Location(x=-0.5, y=0.5, z=3),
            ]
            bboxes.append(self._relative_to_traffic_light(points))

        return bboxes

    def _get_bboxes_for_town4(self):
        bboxes = []
        points = [
            # Back Plane
            pylot.utils.Location(x=-5.2, y=-0.2, z=5.5),
            pylot.utils.Location(x=-4.8, y=-0.2, z=5.5),
            pylot.utils.Location(x=-4.8, y=-0.2, z=6.5),
            pylot.utils.Location(x=-5.2, y=-0.2, z=6.5),
            # Front Plane
            pylot.utils.Location(x=-5.2, y=0.4, z=5.5),
            pylot.utils.Location(x=-4.8, y=0.4, z=5.5),
            pylot.utils.Location(x=-4.8, y=0.4, z=6.5),
            pylot.utils.Location(x=-5.2, y=0.4, z=6.5),
        ]
        bboxes.append(self._relative_to_traffic_light(points))
        if self.trigger_volume_extent.x > 5:
            # This is a traffic light with 4 signs, we need to come up with
            # more bounding boxes.
            middle_points = [  # Light in the middle of the pole.
                # Back Plane
                pylot.utils.Location(x=-0.5, y=-0.1, z=2.5),
                pylot.utils.Location(x=+0.1, y=-0.1, z=2.5),
                pylot.utils.Location(x=+0.1, y=-0.1, z=3.5),
                pylot.utils.Location(x=-0.5, y=-0.1, z=3.5),
                # Front Plane
                pylot.utils.Location(x=-0.5, y=0.5, z=2.5),
                pylot.utils.Location(x=+0.1, y=0.5, z=2.5),
                pylot.utils.Location(x=+0.1, y=0.5, z=3.5),
                pylot.utils.Location(x=-0.5, y=0.5, z=3.5),
            ]
            right_points = [
                point + pylot.utils.Location(x=-3.0) for point in points
            ]
            left_points = [
                point + pylot.utils.Location(x=-5.5) for point in points
            ]
            bboxes.append(self._relative_to_traffic_light(middle_points))
            bboxes.append(self._relative_to_traffic_light(right_points))
            bboxes.append(self._relative_to_traffic_light(left_points))
        return bboxes

    def _get_bboxes_for_town5(self):
        bboxes = []
        points = [
            # Back Plane
            pylot.utils.Location(x=-5.2, y=-0.2, z=5.5),
            pylot.utils.Location(x=-4.8, y=-0.2, z=5.5),
            pylot.utils.Location(x=-4.8, y=-0.2, z=6.5),
            pylot.utils.Location(x=-5.2, y=-0.2, z=6.5),
            # Front Plane
            pylot.utils.Location(x=-5.2, y=0.4, z=5.5),
            pylot.utils.Location(x=-4.8, y=0.4, z=5.5),
            pylot.utils.Location(x=-4.8, y=0.4, z=6.5),
            pylot.utils.Location(x=-5.2, y=0.4, z=6.5),
        ]
        # Town05 randomizes the identifiers for the traffic light at each
        # reload of the world. We cannot depend on static identifiers for
        # figuring out which lights only have a single traffic light.
        bboxes.append(self._relative_to_traffic_light(points))
        # There's a traffic light with extent.x < 2, which only has one box.
        if self.trigger_volume_extent.x >= 2:
            # This is a traffids light with 4 signs, we need to come up
            # with more bounding boxes.
            middle_points = [  # Light in the middle of the pole.
                # Back Plane
                pylot.utils.Location(x=-0.4, y=-0.1, z=2.55),
                pylot.utils.Location(x=+0.2, y=-0.1, z=2.55),
                pylot.utils.Location(x=+0.2, y=-0.1, z=3.55),
                pylot.utils.Location(x=-0.4, y=-0.1, z=3.55),
                # Front Plane
                pylot.utils.Location(x=-0.4, y=0.5, z=2.55),
                pylot.utils.Location(x=+0.2, y=0.5, z=2.55),
                pylot.utils.Location(x=+0.2, y=0.5, z=3.55),
                pylot.utils.Location(x=-0.5, y=0.5, z=3.55),
            ]
            right_points = [
                point + pylot.utils.Location(x=-3.0) for point in points
            ]
            left_points = [
                point + pylot.utils.Location(x=-5.5) for point in points
            ]
            bboxes.append(self._relative_to_traffic_light(middle_points))
            bboxes.append(self._relative_to_traffic_light(right_points))
            bboxes.append(self._relative_to_traffic_light(left_points))
        return bboxes
