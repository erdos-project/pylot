import carla
from collections import namedtuple
import numpy as np
from numpy.linalg import inv
from numpy.matlib import repmat

import pylot.utils
from pylot.utils import Location, Transform
from pylot.perception.detection.utils import BoundingBox2D, BoundingBox3D, \
    DetectedObstacle, DetectedSpeedLimit

SpeedLimitSign = namedtuple('SpeedLimitSign', 'transform, limit')
StopSign = namedtuple('StopSign', 'transform, bounding_box')
LocationGeo = namedtuple('LocationGeo', 'latitude, longitude, altitude')


class CanBus(object):
    def __init__(self, transform, forward_speed):
        if not isinstance(transform, Transform):
            raise ValueError(
                'transform should be of type pylot.utils.Transform')
        self.transform = transform
        # Forward speed in m/s.
        self.forward_speed = forward_speed

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "CanBus(transform: {}, forward speed: {})".format(
            self.transform, self.forward_speed)


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


def depth_to_local_point_cloud(depth_frame, camera_setup, max_depth=0.9):
    """
    Convert a CARLA-encoded depth-map to a 2D array containing
    the 3D position (relative to the camera) of each pixel.
    "max_depth" is used to omit the points that are far enough.

    Args:
        depth_frame: the normalized depth frame
        width: frame width
        height: frame height
        fov: camera field of view
    """
    far = 1000.0  # max depth in meters.
    normalized_depth = depth_frame
    intrinsic_mat = camera_setup.get_intrinsic_matrix()
    width, height = camera_setup.width, camera_setup.height
    # 2d pixel coordinates
    pixel_length = width * height
    u_coord = repmat(np.r_[0:width:1], height, 1).reshape(pixel_length)
    v_coord = repmat(np.c_[0:height:1], 1, width).reshape(pixel_length)
    normalized_depth = np.reshape(normalized_depth, pixel_length)

    # Search for pixels where the depth is greater than max_depth to
    # delete them
    max_depth_indexes = np.where(normalized_depth > max_depth)
    normalized_depth = np.delete(normalized_depth, max_depth_indexes)
    u_coord = np.delete(u_coord, max_depth_indexes)
    v_coord = np.delete(v_coord, max_depth_indexes)

    # p2d = [u,v,1]
    p2d = np.array([u_coord, v_coord, np.ones_like(u_coord)])

    # P = [X,Y,Z]
    p3d = np.dot(inv(intrinsic_mat), p2d)
    p3d *= normalized_depth * far

    # [[X1,Y1,Z1],[X2,Y2,Z2], ... [Xn,Yn,Zn]]
    # Return the points as location,
    locations = [
        Location(x, y, z) for x, y, z in np.asarray(np.transpose(p3d))
    ]
    return locations


def lidar_to_camera_transform(transform):
    """
    Takes in a Transform that occurs in camera coordinates,
    and converts it into a Transform that goes from lidar
    coordinates to camera coordinates.
    """
    to_camera_transform = Transform(matrix=np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]))
    return transform * to_camera_transform


def get_3d_world_position_with_depth_map(x, y, depth_frame, camera_setup):
    """ Gets the 3D world position from pixel coordinates using a depth frame.

        Args:
            x: Pixel x coordinate.
            y: Pixel y coordinate.
            depth_frame: Normalized depth frame.
            width: frame width
            height: frame height
            fov: camera field of view
            camera_transform: Camera transform relative to the world.

       Returns:
            3D world location.
    """
    far = 1.0
    point_cloud = depth_to_local_point_cloud(depth_frame,
                                             camera_setup,
                                             max_depth=far)
    # Transform the points in 3D world coordinates.
    to_world_transform = camera_setup.get_unreal_transform()
    point_cloud = to_world_transform.transform_points(point_cloud)
    return point_cloud[y * camera_setup.width + x]


def batch_get_3d_world_position_with_depth_map(xs, ys, depth_frame,
                                               camera_setup):
    """ Gets the 3D world positions from pixel coordinates using a depth frame.

        Args:
            xs: List of pixel x coordinate.
            ys: List of pixel y coordinate.
            depth_frame: Normalized depth frame.
            width: frame width
            height: frame height
            fov: camera field of view
            camera_transform: Camera transform relative to the world.

       Returns:
            List of 3D world locations.
    """
    assert len(xs) == len(ys)
    far = 1.0
    point_cloud = depth_to_local_point_cloud(depth_frame,
                                             camera_setup,
                                             max_depth=far)
    # Transform the points in 3D world coordinates.
    to_world_transform = camera_setup.get_unreal_transform()
    point_cloud = to_world_transform.transform_points(point_cloud)
    return [
        point_cloud[ys[i] * camera_setup.width + xs[i]] for i in range(len(xs))
    ]


def find_point_depth(x, y, point_cloud):
    """ Finds the closest depth normalized point cloud point to x, y."""
    if len(point_cloud) == 0:
        return None
    # Select only points that are in front.
    point_cloud = point_cloud[np.where(point_cloud[:, 2] > 0.0)]
    # Select x and y.
    pc_xy = point_cloud[:, 0:2]
    # Select z
    pc_z = point_cloud[:, 2]
    # Divize x, y by z
    normalized_pc = pc_xy / pc_z[:, None]
    xy = np.array([x, y]).transpose()
    # Compute distance
    dist = np.sum((normalized_pc - xy)**2, axis=1)
    # Select index of the closest point.
    closest_index = np.argmin(dist)
    # Return the closest point.
    return tuple(point_cloud[closest_index])


def lidar_point_cloud_to_camera_coordinates(point_cloud):
    """ Transforms a point cloud from lidar to camera coordinates."""
    point_cloud = [Location(x, y, z) for x, y, z in np.asarray(point_cloud)]
    identity_transform = Transform(matrix=np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    transform = lidar_to_camera_transform(identity_transform)
    transformed_points = transform.transform_points(point_cloud)
    return [[loc.x, loc.y, loc.z] for loc in transformed_points]


def get_3d_world_position_with_point_cloud(u, v, pc, camera_setup):
    """ Gets the 3D world position from pixel coordinates using a Lidar
        point cloud.

        Args:
            u: Pixel x coordinate.
            v: Pixel y coordinate.
            pc: Point cloud in camera coordinates.
            camera_transform: Camera transform relative to the world.
            width: frame width
            height: frame height
            fov: camera field of view
       Returns:
            3D world location or None if it could not be computed.
    """
    intrinsic_mat = camera_setup.get_intrinsic_matrix()
    # Project our 2D pixel location into 3D space, onto the z=1 plane.
    p3d = np.dot(inv(intrinsic_mat), np.array([[u], [v], [1.0]]))
    depth = find_point_depth(p3d[0], p3d[1], np.array(pc))
    if depth:
        # Normalize our point to have the same depth as our closest point.
        p3d *= np.array([depth[2]])
        p3d_locations = [
            Location(x, y, z) for x, y, z in np.asarray(p3d.transpose())
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
        if p1[0] >= 0 and p1[0] < image_width and p1[1] >= 0 and p1[
                1] < image_height:
            points.append(p1)

        if p2[0] >= 0 and p2[0] < image_width and p2[1] >= 0 and p2[
                1] < image_height:
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

    points_2D = [(int(x), int(y)) for x, y, _ in bb_coordinates]

    # Bottom plane thresholded.
    thresholded_points.extend(threshold(points_2D[0], points_2D[1]))
    thresholded_points.extend(threshold(points_2D[1], points_2D[2]))
    thresholded_points.extend(threshold(points_2D[2], points_2D[3]))
    thresholded_points.extend(threshold(points_2D[3], points_2D[0]))

    # Top plane thresholded.
    thresholded_points.extend(threshold(points_2D[4], points_2D[5]))
    thresholded_points.extend(threshold(points_2D[5], points_2D[6]))
    thresholded_points.extend(threshold(points_2D[6], points_2D[7]))
    thresholded_points.extend(threshold(points_2D[7], points_2D[4]))

    # Remaining segments thresholded.
    thresholded_points.extend(threshold(points_2D[0], points_2D[4]))
    thresholded_points.extend(threshold(points_2D[1], points_2D[5]))
    thresholded_points.extend(threshold(points_2D[2], points_2D[6]))
    thresholded_points.extend(threshold(points_2D[3], points_2D[7]))

    if len(thresholded_points) == 0:
        return None
    else:
        x = [int(x) for x, _ in thresholded_points]
        y = [int(y) for _, y in thresholded_points]
        return BoundingBox2D(min(x), max(x), min(y), max(y))


def get_traffic_lights_obstacles(traffic_lights, depth_array, segmented_image,
                                 town_name, camera_setup):
    """ Get the traffic lights that are within the camera frame.
    Note: This method should be used with Carla 0.9.*
    """
    # Create the extrinsic and intrinsic matrices for the given camera.
    extrinsic_matrix = camera_setup.get_extrinsic_matrix()
    intrinsic_matrix = camera_setup.get_intrinsic_matrix()
    camera_transform = camera_setup.get_transform()

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
            bbox_2d = get_bounding_box_in_camera_view(bounding_box,
                                                      camera_setup.width,
                                                      camera_setup.height)
            if not bbox_2d:
                continue

            # Crop the segmented and depth image to the given bounding box.
            cropped_image = segmented_image[bbox_2d.y_min:bbox_2d.y_max,
                                            bbox_2d.x_min:bbox_2d.x_max]
            cropped_depth = depth_array[bbox_2d.y_min:bbox_2d.y_max,
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


def get_detected_speed_limits(speed_signs, vehicle_transform, depth_frame,
                              segmented_frame, camera_setup):
    """ Get the speed limit signs that are withing the camera frame.

    Args:
        speed_signs: List of speed limit signs in the world.
        vehicle_transform: Ego-vehicle transform in world coordinates.
        camera_transform: Camera transform in world coordinates.
        fov: Camera field of view.
        segmented_frame: Segmented frame.
    """
    # Compute the bounding boxes.
    bboxes = segmented_frame.get_traffic_sign_bounding_boxes(min_width=8,
                                                             min_height=9)

    # Get the positions of the bounding box centers.
    x_mids = [(bbox[0] + bbox[1]) // 2 for bbox in bboxes]
    y_mids = [(bbox[2] + bbox[3]) // 2 for bbox in bboxes]
    pos_3d = batch_get_3d_world_position_with_depth_map(
        x_mids, y_mids, depth_frame, camera_setup)
    pos_and_bboxes = zip(pos_3d, bboxes)
    ts_bboxes = _match_bboxes_with_speed_signs(vehicle_transform,
                                               pos_and_bboxes, speed_signs)

    det_obstacles = [
        DetectedSpeedLimit(bbox, limit, 1.0, 'speed limit')
        for (bbox, limit) in ts_bboxes
    ]
    return det_obstacles


def _match_bboxes_with_speed_signs(vehicle_transform, pos_bboxes, speed_signs):
    result = []
    for pos, bbox in pos_bboxes:
        best_ts = None
        best_dist = 1000000
        for ts in speed_signs:
            dist = ((pos.x - ts.transform.location.x)**2 +
                    (pos.y - ts.transform.location.y)**2)
            if (dist < best_dist):
                best_dist = dist
                best_ts = ts
        if not best_ts:
            continue
        # Check that the sign is facing the ego vehicle.
        yaw_diff = (best_ts.transform.rotation.yaw -
                    vehicle_transform.rotation.yaw)
        if yaw_diff < 0:
            yaw_diff += 360
        elif yaw_diff >= 360:
            yaw_diff -= 360
        if best_dist < 5**2 and yaw_diff > 30 and yaw_diff < 150:
            result.append((bbox, best_ts.limit))
    return result


def _get_stop_markings_bbox(bbox3d, depth_frame, camera_transform,
                            camera_intrinsic, frame_width, frame_height):
    """ Gets a 2D stop marking bouding box from a 3D bounding box."""
    # Offset trigger_volume by -0.85 so that the top plane is on the ground.
    ext_z_value = bbox3d.extent.z - 0.85
    ext = [
        Location(x=+bbox3d.extent.x, y=+bbox3d.extent.y, z=ext_z_value),
        Location(x=+bbox3d.extent.x, y=-bbox3d.extent.y, z=ext_z_value),
        Location(x=-bbox3d.extent.x, y=+bbox3d.extent.y, z=ext_z_value),
        Location(x=-bbox3d.extent.x, y=-bbox3d.extent.y, z=ext_z_value),
    ]
    bbox = bbox3d.transform.transform_points(ext)
    coords = []
    for loc in bbox:
        loc_view = loc.to_camera_view(camera_transform.matrix,
                                      camera_intrinsic)
        if (loc_view.z >= 0 and loc_view.x >= 0 and loc_view.y >= 0
                and loc_view.x < frame_width and loc_view.y < frame_height):
            coords.append(loc_view)
    if len(coords) == 4:
        xmin = min(coords[0].x, coords[1].x, coords[2].x, coords[3].x)
        xmax = max(coords[0].x, coords[1].x, coords[2].x, coords[3].x)
        ymin = min(coords[0].y, coords[1].y, coords[2].y, coords[3].y)
        ymax = max(coords[0].y, coords[1].y, coords[2].y, coords[3].y)
        # Check if the bbox is not obstructed and if it's sufficiently
        # big for the text to be readable.
        if (ymax - ymin > 15
                and have_same_depth(int(coords[0].x), int(coords[0].y),
                                    coords[0].z, depth_frame, 0.4)):
            return (int(xmin), int(xmax), int(ymin), int(ymax))
    return None


def have_same_depth(x, y, z, depth_array, threshold):
    x, y = int(x), int(y)
    return abs(depth_array[y][x] * 1000 - z) < threshold


def get_detected_traffic_stops(traffic_stops, depth_frame, camera_setup):
    """ Get traffic stop lane markings that are withing the camera frame.

    Args:
        traffic_stops: List of traffic stop actors in the world.
        camera_transform: Camera transform in world coordinates.
        fov: Camera field of view.

    Returns:
        List of DetectedObstacles.
    """
    det_obstacles = []
    bgr_intrinsic = camera_setup.get_intrinsic_matrix()
    for transform, bbox in traffic_stops:
        bbox_2d = _get_stop_markings_bbox(bbox, depth_frame,
                                          camera_setup.get_transform(),
                                          bgr_intrinsic, camera_setup.width,
                                          camera_setup.height)
        if bbox_2d:
            det_obstacles.append(DetectedObstacle(bbox_2d, 1.0,
                                                  'stop marking'))
    return det_obstacles
