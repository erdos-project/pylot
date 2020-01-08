import numpy as np

from pylot.perception.detection.utils import TrafficLightColor
import pylot.utils


class TrafficLight(object):
    """ The Pylot version of a carla TrafficLight that defines helper
    functions needed in Pylot, and makes the class serializable.

    Args:
        id: The identifier of the TrafficLight.
        transform: The transform of the TrafficLight.
        trigger_volume_extent: The extent of the trigger volume of the light.
        state: The state of the light. (Green/Yellow/Red/Off)
    """
    def __init__(self, id, transform, trigger_volume_extent, state):
        self.id = id
        self.transform = transform
        self.trigger_volume_extent = trigger_volume_extent
        self.state = state

    @classmethod
    def from_carla_actor(cls, traffic_light):
        """ Creates a TrafficLight from a carla traffic light actor.

        Args:
            traffic_light: The carla.TrafficLight instance to initialize this
                instance with.
        """
        import carla
        if not isinstance(traffic_light, carla.TrafficLight):
            raise ValueError('The traffic light must be a carla.TrafficLight')
        # Retrieve the Transform of the TrafficLight.
        transform = pylot.utils.Transform.from_carla_transform(
            traffic_light.get_transform())
        # Retrieve the Trigger Volume of the TrafficLight.
        trigger_volume_extent = pylot.utils.Vector3D(
            traffic_light.trigger_volume.extent.x,
            traffic_light.trigger_volume.extent.y,
            traffic_light.trigger_volume.extent.z)
        traffic_light_state = traffic_light.get_state()
        state = TrafficLightColor.OFF
        if traffic_light_state == carla.TrafficLightState.Red:
            state = TrafficLightColor.RED
        elif traffic_light_state == carla.TrafficLightState.Yellow:
            state = TrafficLightColor.YELLOW
        elif traffic_light_state == carla.TrafficLightState.Green:
            state = TrafficLightColor.GREEN
        return cls(traffic_light.id, transform, trigger_volume_extent, state)

    def is_traffic_light_visible(self,
                                 camera_transform,
                                 town_name=None,
                                 distance_threshold=70):
        """ Returns True if the traffic light is visible from the
        camera_transform.

        Args:
            camera_transform: The transform of the camera.
            distance_threshold: Maximum distance to the camera (in m).
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

    def relative_to_traffic_light(self, points):
        """ Transforms the bounding box specified in the points relative to the
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

    def get_bbox_state(self, town_name):
        if town_name == 'Town01' or town_name == 'Town02':
            return self._get_bbox_state_town1_or_2()
        elif town_name == 'Town03':
            return self._get_bbox_state_town3()
        elif town_name == 'Town04':
            return self._get_bbox_state_town4()
        elif town_name == 'Town05':
            return self._get_bbox_state_town5()
        else:
            raise ValueError(
                'Could not find a town named {}'.format(town_name))

    def _get_bbox_state_town1_or_2(self):
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
        return [(self.relative_to_traffic_light(points), self.state)]

    def _get_bbox_state_town3(self):
        bbox_state = []
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
            bbox_state.append(
                (self.relative_to_traffic_light(points), self.state))
            right_points = [
                point + pylot.utils.Location(x=-3.0) for point in points
            ]
            bbox_state.append(
                (self.relative_to_traffic_light(right_points), self.state))
            if self.id not in [51, 52, 53]:
                left_points = [
                    point + pylot.utils.Location(x=-6.5) for point in points
                ]
                bbox_state.append(
                    (self.relative_to_traffic_light(left_points), self.state))
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
            bbox_state.append(
                (self.relative_to_traffic_light(points), self.state))

        return bbox_state

    def _get_bbox_state_town4(self):
        bbox_state = []
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
        bbox_state.append((self.relative_to_traffic_light(points), self.state))
        if self.trigger_volume_extent.x > 5:
            # This is a traffic light with 4 signs, we need to come up with
            # more bounding boxes.
            bbox_state.append(
                (self.relative_to_traffic_light(middle_points), self.state))
            bbox_state.append(
                (self.relative_to_traffic_light(right_points), self.state))
            bbox_state.append(
                (self.relative_to_traffic_light(left_points), self.state))
        return bbox_state

    def _get_bbox_state_town5(self):
        bbox_state = []
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

        # Town05 randomizes the identifiers for the traffic light at each
        # reload of the world. We cannot depend on static identifiers for
        # figuring out which lights only have a single traffic light.
        bbox_state.append((self.relative_to_traffic_light(points), self.state))
        # There's a traffic light with extent.x < 2, which only has one box.
        if self.trigger_volume_extent.x >= 2:
            # This is a traffids light with 4 signs, we need to come up
            # with more bounding boxes.
            bbox_state.append(
                (self.relative_to_traffic_light(middle_points), self.state))
            bbox_state.append(
                (self.relative_to_traffic_light(right_points), self.state))
            bbox_state.append(
                (self.relative_to_traffic_light(left_points), self.state))
        return bbox_state
