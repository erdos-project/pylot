import carla
import numpy as np
from numpy.linalg import inv

import pylot.utils
from pylot.utils import Location, Transform
from pylot.perception.detection.utils import BoundingBox2D, BoundingBox3D, \
    DetectedObstacle, DetectedSpeedLimit


class Obstacle(object):
    """ An Obstacle represents a dynamic obstacle that we could encounter on the
    road. This class provides helper functions to detect obstacles and provide
    bounding boxes for them.
    """
    def __init__(self, actor):
        """ Initializes the Obstacle class with the given arguments.

        Args:
            actor: The actor to initialize the obstacle with. (should be of
            type carla.Vehicle or carla.Walker)
        """
        if not isinstance(actor, (carla.Vehicle, carla.Walker)):
            raise ValueError("The actor should be of type carla.Vehicle or "
                             "carla.Walker to initialize the Obstacle class.")

        # Retrieve the unique identifier of this actor in the simulation.
        self.id = actor.id

        # Convert the transform provided by the simulation to the Pylot class.
        self.transform = Transform.from_carla_transform(actor.get_transform())

        # Convert the bounding box from the simulation to the Pylot one.
        self.bounding_box = BoundingBox3D.from_carla_bounding_box(
            actor.bounding_box)

        # Get the speed of the obstacle.
        velocity_vector = pylot.utils.Vector3D.from_carla_vector(
            actor.get_velocity())
        self.forward_speed = velocity_vector.magnitude()

        # Get the blueprint of the actor.
        # TODO (Sukrit): Move from vehicles and pedestrians to separate classes
        # for bicycles, motorcycles, cars and persons.
        if isinstance(actor, carla.Vehicle):
            self.label = 'vehicle'
            self.segmentation_class = 10
        else:
            self.label = 'pedestrian'
            self.segmentation_class = 4

        # Thresholds to be used for detection of the obstacle.
        self.__segmentation_threshold = 0.20
        self.__depth_threshold = 5

    def distance(self, vehicle_transform):
        """ Returns the distance of the obstacle from the vehicle represented
        by the vehicle transform.

        The distance provides an estimate of the depth returned by the depth
        camera sensor in Carla. As a result, the distance is defined as the
        displacement of the obstacle along either the X or the Y axis.

        Args:
            vehicle_transform: The Pylot Transform of the vehicle to find the
            distance of the obstacle from.

        Returns:
            The distance (in metres) of the obstacle from the vehicle.
        """
        import numpy as np

        # Get the location of the vehicle and the obstacle as numpy arrays.
        vehicle_location = vehicle_transform.location.as_numpy_array()
        obstacle_location = self.transform.location.as_numpy_array()

        # Calculate the vector from the vehicle to the obstacle.
        # Scale it by the forward vector, and calculate the norm.
        relative_vector = vehicle_location - obstacle_location
        distance = np.linalg.norm(
            relative_vector *
            vehicle_transform.forward_vector.as_numpy_array())
        return distance

    def to_camera_view(self, camera_setup, depth_array, segmented_frame):
        """ Retrieves the 2D bounding box for the obstacle with respect to the
        given camera setup.

        Heuristically uses the depth frame and segmentation frame to figure out
        if the obstacle is in view of the camera or not.

        Args:
            camera_setup: The CameraSetup instance, with respect to which the
                obstacle needs to be viewed.
            depth_array: The depth frame array to be used to compare the depth
                to the distance of the obstacle from the sensor.
            segmented_frame: The SegmentedFrame to be used to compare
                the segmentation class.

        Returns:
            A BoundingBox2D instance representing a rectangle over the obstacle
            if the obstacle is deemed to be visible, None otherwise.
        """
        # Convert the bounding box of the obstacle to the camera coordinates.
        bb_coordinates = self.bounding_box.to_camera_view(
            self.transform, camera_setup.get_extrinsic_matrix(),
            camera_setup.get_intrinsic_matrix())

        # Threshold the bounding box to be within the camera view.
        bb_coordinates = [(bb.x, bb.y, bb.z) for bb in bb_coordinates]
        bbox_2d = get_bounding_box_in_camera_view(bb_coordinates,
                                                  camera_setup.width,
                                                  camera_setup.height)
        if not bbox_2d:
            return None
        # Crop the segmented and depth image to the given bounding box.
        cropped_image = segmented_frame.as_numpy_array()[
            bbox_2d.y_min:bbox_2d.y_max, bbox_2d.x_min:bbox_2d.x_max]
        cropped_depth = depth_array[bbox_2d.y_min:bbox_2d.y_max,
                                    bbox_2d.x_min:bbox_2d.x_max]

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
                depth = self.distance(camera_setup.get_transform())
                if abs(depth - mean_depth) <= self.__depth_threshold:
                    return bbox_2d
        return None


def get_top_down_transform(transform, top_down_lateral_view):
    # Height calculation relies on the fact that the camera's FOV is 90.
    top_down_location = (transform.location +
                         Location(0, 0, top_down_lateral_view))
    return Transform(top_down_location, pylot.utils.Rotation(-90, 0, 0))


def lidar_point_cloud_to_camera_coordinates(point_cloud):
    """ Transforms a point cloud from lidar to camera coordinates."""
    point_cloud = [Location(x, y, z) for x, y, z in np.asarray(point_cloud)]
    identity_transform = Transform(matrix=np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    # Converts a transform that occurs in camera coordinates into a transform
    # that goes from lidar coordinates to camera coordinates.
    to_camera_transform = Transform(matrix=np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]))
    transform = identity_transform * to_camera_transform
    transformed_points = transform.transform_points(point_cloud)
    return [[loc.x, loc.y, loc.z] for loc in transformed_points]


def get_3d_world_position_with_point_cloud(x, y, pc, camera_setup):
    """ Gets the 3D world position from pixel coordinates using a Lidar
        point cloud.

        Args:
            x: Pixel x coordinate.
            y: Pixel y coordinate.
            pc: Point cloud in camera coordinates.
       Returns:
            3D world location or None if all point clouds are behind.
    """
    intrinsic_mat = camera_setup.get_intrinsic_matrix()
    # Project our 2D pixel location into 3D space, onto the z=1 plane.
    p3d = np.dot(inv(intrinsic_mat), np.array([[x], [y], [1.0]]))
    pixel = pylot.utils.Vector2D(p3d[0], p3d[1])
    location = pixel.get_closest_point_in_point_cloud(np.array(pc))
    if location is not None:
        # Normalize our point to have the same depth as our closest point.
        p3d *= np.array([location[2]])
        p3d_locations = [
            Location(px, py, pz) for px, py, pz in np.asarray(p3d.transpose())
        ]
        # Convert from camera to unreal coordinates.
        to_world_transform = camera_setup.get_unreal_transform()
        point_cloud = to_world_transform.transform_points(p3d_locations)
        return point_cloud[0]
    else:
        return None


def get_bounding_box_in_camera_view(bb_coordinates, image_width, image_height):
    """ Creates the bounding box in the view of the camera image using the
    coordinates generated with respect to the camera transform.

    Args:
        bb_coordinates: The coordinates of the bounding box relative to the
            camera transform.
        image_width: The width of the image being published by the camera.
        image_height: The height of the image being published by the camera.

    Returns:
        None, if the bounding box does not fall into the view of the camera,
        otherwise it returns a BoundingBox2D.
    """
    # Make sure that atleast 2 of the bounding box coordinates are in front.
    z_vals = [z for _, _, z in bb_coordinates if z >= 0]
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
    points = [(int(x), int(y)) for x, y, _ in bb_coordinates]
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
        return BoundingBox2D(min(x), max(x), min(y), max(y))


def get_traffic_lights_obstacles(traffic_lights, depth_frame, segmented_image,
                                 town_name):
    """ Get the traffic lights that are within the camera frame.
    Note: This method should be used with Carla 0.9.*
    """
    # Create the extrinsic and intrinsic matrices for the given camera.
    extrinsic_matrix = depth_frame.camera_setup.get_extrinsic_matrix()
    intrinsic_matrix = depth_frame.camera_setup.get_intrinsic_matrix()
    camera_transform = depth_frame.camera_setup.get_transform()

    # Iterate over all the traffic lights, and figure out which ones are
    # facing us and are visible in the camera view.
    detected = []
    for light in traffic_lights:
        if not light.is_traffic_light_visible(camera_transform, town_name):
            continue

        bboxes = light.get_bbox_state(town_name)

        # Convert the returned bounding boxes to 2D and check if the
        # light is occluded. If not, add it to the detected obstacle list.
        for box, color in bboxes:
            bounding_box = [
                loc.to_camera_view(extrinsic_matrix, intrinsic_matrix)
                for loc in box
            ]
            bounding_box = [(bb.x, bb.y, bb.z) for bb in bounding_box]
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
                masked_image[np.where(cropped_image == 12)] = 1
                if np.sum(masked_image) >= 0.20 * masked_image.size:
                    masked_depth = cropped_depth[np.where(masked_image == 1)]
                    mean_depth = np.mean(masked_depth) * 1000
                    if abs(mean_depth -
                           bounding_box[0][-1]) <= 2 and mean_depth < 150:
                        detected.append(
                            DetectedObstacle(bbox_2d, 1.0, color.get_label()))
    return detected


def get_detected_speed_limits(speed_signs, depth_frame, segmented_frame):
    """ Get the speed limit signs that are withing the camera frame.

    Args:
        speed_signs: List of speed limit signs in the world.
        depth_frame: A pylot.utils.DepthFrame, with a camera_setup relative to
            the world.
        segmented_frame: pylot.perception.segmentation.SegmentedFrame

    Returns:
        A list of pylot.perception.detection.DetectedSpeedLimit
    """
    def match_bboxes_with_speed_signs(camera_transform, loc_bboxes,
                                      speed_signs):
        result = []
        for location, bbox in loc_bboxes:
            best_ts = None
            best_dist = 1000000
            for ts in speed_signs:
                dist = location.distance(ts.transform.location)
                if dist < best_dist:
                    best_dist = dist
                    best_ts = ts
            if not best_ts:
                continue
            # Check that the sign is facing the ego vehicle.
            yaw_diff = (best_ts.transform.rotation.yaw -
                        camera_transform.rotation.yaw)
            if yaw_diff < 0:
                yaw_diff += 360
            elif yaw_diff >= 360:
                yaw_diff -= 360
            if best_dist < 5**2 and yaw_diff > 30 and yaw_diff < 150:
                result.append(
                    DetectedSpeedLimit(bbox, best_ts.limit, 1.0,
                                       'speed limit'))
        return result

    if not isinstance(depth_frame, pylot.utils.DepthFrame):
        raise ValueError(
            'depth_frame should be of type pylot.utils.DepthFrame')
    # Compute the 2D bounding boxes.
    bboxes_2d = segmented_frame.get_traffic_sign_bounding_boxes(min_width=8,
                                                                min_height=9)
    # Transform the centers of 2D bounding boxes to 3D locations.
    coordinates = [bbox.get_center_point() for bbox in bboxes_2d]
    locations = depth_frame.get_pixel_locations(coordinates)
    loc_and_bboxes = zip(locations, bboxes_2d)
    det_speed_limits = match_bboxes_with_speed_signs(
        depth_frame.camera_setup.transform, loc_and_bboxes, speed_signs)
    return det_speed_limits


def get_detected_traffic_stops(traffic_stops, depth_frame):
    """ Get traffic stop lane markings that are withing the camera frame.

    Args:
        traffic_stops: List of traffic stop actors in the world.
        depth_frame: A pylot.utils.DepthFrame, with a camera_setup relative to
            the world.

    Returns:
        List of DetectedObstacles.
    """
    def get_stop_markings_bbox(bbox3d, depth_frame):
        """ Gets a 2D stop marking bounding box from a 3D bounding box."""
        # Move trigger_volume by -0.85 so that the top plane is on the ground.
        ext_z_value = bbox3d.extent.z - 0.85
        ext = [
            Location(x=+bbox3d.extent.x, y=+bbox3d.extent.y, z=ext_z_value),
            Location(x=+bbox3d.extent.x, y=-bbox3d.extent.y, z=ext_z_value),
            Location(x=-bbox3d.extent.x, y=+bbox3d.extent.y, z=ext_z_value),
            Location(x=-bbox3d.extent.x, y=-bbox3d.extent.y, z=ext_z_value),
        ]
        bbox = bbox3d.transform.transform_points(ext)
        camera_transform = depth_frame.camera_setup.get_transform()
        coords = []
        for loc in bbox:
            loc_view = loc.to_camera_view(
                camera_transform.matrix,
                depth_frame.camera_setup.get_intrinsic_matrix())
            if (loc_view.z >= 0 and loc_view.x >= 0 and loc_view.y >= 0
                    and loc_view.x < depth_frame.camera_setup.width
                    and loc_view.y < depth_frame.camera_setup.height):
                coords.append(loc_view)
        if len(coords) == 4:
            xmin = min(coords[0].x, coords[1].x, coords[2].x, coords[3].x)
            xmax = max(coords[0].x, coords[1].x, coords[2].x, coords[3].x)
            ymin = min(coords[0].y, coords[1].y, coords[2].y, coords[3].y)
            ymax = max(coords[0].y, coords[1].y, coords[2].y, coords[3].y)
            # Check if the bbox is not obstructed and if it's sufficiently
            # big for the text to be readable.
            if (ymax - ymin > 15 and depth_frame.pixel_has_same_depth(
                    int(coords[0].x), int(coords[0].y), coords[0].z, 0.4)):
                return BoundingBox2D(int(xmin), int(xmax), int(ymin),
                                     int(ymax))
        return None

    if not isinstance(depth_frame, pylot.utils.DepthFrame):
        raise ValueError(
            'depth_frame should be of type pylot.utils.DepthFrame')
    det_obstacles = []
    for transform, bbox in traffic_stops:
        bbox_2d = get_stop_markings_bbox(bbox, depth_frame)
        if bbox_2d is not None:
            det_obstacles.append(DetectedObstacle(bbox_2d, 1.0,
                                                  'stop marking'))
    return det_obstacles
