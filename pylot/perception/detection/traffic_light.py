import carla
import numpy as np

from pylot.perception.detection.utils import TrafficLightColor
import pylot.simulation.utils


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

    def from_carla_actor(traffic_light):
        """ Creates a TrafficLight from a carla traffic light actor.

        Args:
            traffic_light: The carla.TrafficLight instance to initialize this
                instance with.
        """
        # Retrieve the Transform of the TrafficLight.
        transform = pylot.simulation.utils.Transform(
            carla_transform=traffic_light.get_transform())
        # Retrieve the Trigger Volume of the TrafficLight.
        trigger_volume_extent = pylot.simulation.utils.Vector3D(
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
        return TrafficLight(traffic_light.id, transform, trigger_volume_extent,
                            state)

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
            return pylot.simulation.utils.Location(x=transformed[0, 0],
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
