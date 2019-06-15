from collections import namedtuple
from itertools import combinations
import math
import numpy as np
from numpy.linalg import inv
from numpy.matlib import repmat

Acceleration = namedtuple('Acceleration', 'x, y, z')
Orientation = namedtuple('Orientation', 'x, y, z')
Rotation = namedtuple('Rotation', 'pitch, yaw, roll')
Position = namedtuple('Position', 'location, orientation')
Vehicle = namedtuple('Vehicle',
                     'location, transform, bounding_box, forward_speed')
Pedestrian = namedtuple('Pedestrian',
                        'id, location, transform, bounding_box, forward_speed')
TrafficLight = namedtuple('TrafficLight', 'location, transform, state')
SpeedLimitSign = namedtuple('SpeedLimitSign', 'location, transform, limit')
LocationGeo = namedtuple('LocationGeo', 'latitude, longitude, altitude')
Extent = namedtuple('Extent', 'x, y, z')
Scale = namedtuple('Scale', 'x y z')
Scale.__new__.__defaults__ = (1.0, 1.0, 1.0)


class CameraSetup(object):
    def __init__(self,
                 name,
                 camera_type,
                 width,
                 height,
                 transform,
                 fov=90):
        self.name = name
        self.camera_type = camera_type
        self.width = width
        self.height = height
        self.transform = transform
        self.fov = fov
        self.intrinsic_mat = create_intrinsic_matrix(
            self.width, self.height, self.fov)

    def get_intrinsic(self):
        return self.intrinsic_mat

    def get_transform(self):
        return self.transform

    def get_unreal_transform(self):
        return camera_to_unreal_transform(self.transform)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'CameraSetup(name: {}, type: {}, width: {}, height: {}, transform: {}, fov: {}'.format(
            self.name, self.camera_type, self.width, self.height,
            self.transform, self.fov)


class LidarSetup(object):
    def __init__(self,
                 name,
                 lidar_type,
                 transform,
                 range,
                 rotation_frequency,
                 channels,
                 upper_fov,
                 lower_fov,
                 points_per_second):
        self.name = name
        self.lidar_type = lidar_type
        self.transform = transform
        self.range = range
        self.rotation_frequency = rotation_frequency
        self.channels = channels
        self.upper_fov = upper_fov
        self.lower_fov = lower_fov
        self.points_per_second = points_per_second

    def get_transform(self):
        return self.transform

    def get_unreal_transform(self):
        return lidar_to_unreal_transform(self.transform)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'LidarSetup(name: {}, type: {}, transform: {}, range: {}, rotation freq: {}, channels: {}, upper_fov: {}, lower_fov: {}, points_per_second: {}'.format(
            self.name, self.lidar_type, self.transform, self.range,
            self.rotation_frequency, self.channels, self.upper_fov,
            self.lower_fov, self.points_per_second)


class CanBus(object):
    def __init__(self, transform, forward_speed):
        self.transform = transform
        self.forward_speed = forward_speed

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "CanBus(transform: {}, forward speed: {})".format(
            self.transform, self.forward_speed)


class BoundingBox(object):
    def __init__(self, bb):
        if hasattr(bb, 'location'):
            # Path for Carla 0.9.x.
            loc = Location(bb.location.x, bb.location.y, bb.location.z)
            rot = Rotation(0, 0, 0)
        else:
            # Path for Carla 0.8.4.
            loc = Location(bb.transform.location.x,
                           bb.transform.location.y,
                           bb.transform.location.z)
            rot = Rotation(bb.transform.rotation.pitch,
                           bb.transform.rotation.yaw,
                           bb.transform.rotation.roll)
        self.transform = Transform(loc, rot)
        self.extent = Extent(bb.extent.x, bb.extent.y, bb.extent.z)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "BoundingBox(transform: {}, extent: {})".format(
            self.transform, self.extent)


class Location(object):
    def __init__(self, x=None, y=None, z=None, carla_loc=None):
        if carla_loc:
            self.x = carla_loc.x
            self.y = carla_loc.y
            self.z = carla_loc.z
        else:
            assert x is not None and y is not None and z is not None
            self.x = x
            self.y = y
            self.z = z

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Location({}, {}, {})'.format(self.x, self.y, self.z)


class Transform(object):

    def __init__(self, pos=None, rotation=None, orientation=None, scale=None,
                 matrix=None):
        self.orientation = orientation
        self.rotation = rotation
        self.location = pos
        self.scale = scale
        if scale is None:
            scale = Scale()
        if matrix is None:
            self.matrix = np.matrix(np.identity(4))
            cy = math.cos(np.radians(rotation.yaw))
            sy = math.sin(np.radians(rotation.yaw))
            cr = math.cos(np.radians(rotation.roll))
            sr = math.sin(np.radians(rotation.roll))
            cp = math.cos(np.radians(rotation.pitch))
            sp = math.sin(np.radians(rotation.pitch))
            self.matrix[0, 3] = pos.x
            self.matrix[1, 3] = pos.y
            self.matrix[2, 3] = pos.z
            self.matrix[0, 0] = scale.x * (cp * cy)
            self.matrix[0, 1] = scale.y * (cy * sp * sr - sy * cr)
            self.matrix[0, 2] = -scale.z * (cy * sp * cr + sy * sr)
            self.matrix[1, 0] = scale.x * (sy * cp)
            self.matrix[1, 1] = scale.y * (sy * sp * sr + cy * cr)
            self.matrix[1, 2] = scale.z * (cy * sr - sy * sp * cr)
            self.matrix[2, 0] = scale.x * (sp)
            self.matrix[2, 1] = -scale.y * (cp * sr)
            self.matrix[2, 2] = scale.z * (cp * cr)
        else:
            self.matrix = matrix

    def transform_points(self, points):
        """
        Given a 4x4 transformation matrix, transform an array of 3D points.
        Expected point foramt: [[X0,Y0,Z0],..[Xn,Yn,Zn]]
        """
        # Needed format: [[X0,..Xn],[Z0,..Zn],[Z0,..Zn]]. So let's transpose
        # the point matrix.
        points = points.transpose()
        # Add 0s row: [[X0..,Xn],[Y0..,Yn],[Z0..,Zn],[0,..0]]
        points = np.append(points, np.ones((1, points.shape[1])), axis=0)
        # Point transformation
        points = self.matrix * points
        # Return all but last row
        return points[0:3].transpose()

    def inverse_transform(self):
        return Transform(matrix=inv(self.matrix))

    def __mul__(self, other):
        return Transform(matrix=np.dot(self.matrix, other.matrix))

    def __str__(self):
        if self.location:
            return "Transform(location: {}, rotation: {})".format(
                self.location, self.rotation)
        else:
            return "Trastorm({})".format(str(self.matrix))


def to_erdos_transform(transform):
    orientation = None
    # get_forward_vector() is only available in carla 0.9.5.
    get_fwd_vector = getattr(transform, "get_forward_vector", None)
    if callable(get_fwd_vector):
        fwd_vector = transform.get_forward_vector()
        orientation = Orientation(fwd_vector.x, fwd_vector.y, fwd_vector.z)

    return Transform(
        Location(carla_loc=transform.location),
        Rotation(transform.rotation.pitch,
                 transform.rotation.yaw,
                 transform.rotation.roll),
        orientation)


def depth_to_array(image):
    """
    Convert an image containing CARLA encoded depth-map to a 2D array containing
    the depth value of each pixel normalized between [0.0, 1.0].
    """
    array = to_bgra_array(image)
    array = array.astype(np.float32)
    # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
    normalized_depth = np.dot(array[:, :, :3], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
    return normalized_depth


def to_bgra_array(image):
    """Convert a CARLA raw image to a BGRA np array."""
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    return array


def labels_to_array(image):
    """
    Convert an image containing CARLA semantic segmentation labels to a 2D array
    containing the label of each pixel.
    """
    return to_bgra_array(image)[:, :, 2]


def get_speed(velocity_vector):
    """ Compute the speed of the vehicle in km/h.
    Args:
        vehicle: A velocity vector.
    Returns:
        The speed of the given vehicle as a float in km/h.
    """
    speed = 3.6 * math.sqrt(velocity_vector.x**2 + velocity_vector.y**2 +
                            velocity_vector.z**2)
    return speed


def create_intrinsic_matrix(width, height, fov=90.0):
    # (Intrinsic) K Matrix
    k = np.identity(3)
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0
    k[0, 0] = k[1, 1] = width / (2.0 * math.tan(fov * math.pi / 360.0))
    return k


def depth_to_local_point_cloud(depth_msg, max_depth=0.9):
    far = 1000.0  # max depth in meters.
    normalized_depth = depth_msg.frame
    intrinsic_mat = create_intrinsic_matrix(
        depth_msg.width, depth_msg.height, depth_msg.fov)
    # 2d pixel coordinates
    pixel_length = depth_msg.width * depth_msg.height
    u_coord = repmat(np.r_[depth_msg.width-1:-1:-1],
                     depth_msg.height, 1).reshape(pixel_length)
    v_coord = repmat(np.c_[depth_msg.height-1:-1:-1],
                     1, depth_msg.width).reshape(pixel_length)
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
    return np.transpose(p3d)


def camera_to_unreal_transform(camera_transform):
    to_unreal_transform = Transform(
        Location(0, 0, 0),
        Rotation(pitch=0, yaw=90, roll=-90),
        scale=Scale(x=-1))
    return camera_transform * to_unreal_transform


def lidar_to_unreal_transform(lidar_transform):
    to_unreal_transform = Transform(
        Location(0, 0, 0),
        Rotation(pitch=0, yaw=90, roll=0),
        scale=Scale(z=-1))
    return lidar_transform * to_unreal_transform


def get_3d_world_position_with_depth_map(x, y, depth_msg, vehicle_transform):
    far = 1.0
    point_cloud = depth_to_local_point_cloud(depth_msg, max_depth=far)
    # Transform the points in 3D world coordinates.
    to_world_transform = depth_msg.transform * vehicle_transform
    point_cloud = to_world_transform.transform_points(point_cloud)
    (x, y, z) = point_cloud.tolist()[y * depth_msg.width + x]
    return Location(x, y, z)


def batch_get_3d_world_position_with_depth_map(
        xs, ys, depth_msg, vehicle_transform):
    assert len(xs) == len(ys)
    far = 1.0
    point_cloud = depth_to_local_point_cloud(depth_msg, max_depth=far)
    # Transform the points in 3D world coordinates.
    to_world_transform = vehicle_transform * depth_msg.transform
    point_cloud = to_world_transform.transform_points(point_cloud)
    point_cloud = point_cloud.tolist()
    locs = [point_cloud[ys[i] * depth_msg.width + xs[i]]
            for i in range(len(xs))]
    return [Location(loc[0], loc[1], loc[2]) for loc in locs]


def find_depth(x, y, point_cloud, max_x_dist=2, max_y_dist=2):
    closest_point = None
    dist = 1000000
    # Find the closest lidar point to the point we're trying to get depth for.
    for (px, py, pz) in point_cloud:
        # Ignore if the point is behind.
        if pz <= 0:
            continue
        x_dist = abs(x - px / pz)
        y_dist = abs(y - py / pz)
        if x_dist < max_x_dist and y_dist < max_y_dist:
            if y_dist + x_dist < dist:
                closest_point = (px, py, pz)
                dist = y_dist + x_dist
    if closest_point:
        return closest_point
    else:
        return None


def get_3d_world_position_with_point_cloud(
        x, y, pc, camera_transform, width, height, fov, vehicle_transform):
    intrinsic_mat = create_intrinsic_matrix(width, height, fov)
    u = width - 1 - x
    v = height - 1 - y
    p3d = np.dot(inv(intrinsic_mat), np.array([[u], [v], [1.0]]))
    depth = find_depth(p3d[0], p3d[1], pc)
    if depth:
        scale = depth[2] / p3d[2]
        p3d *= np.array([scale])
        to_world_transform = camera_transform * vehicle_transform
        point_cloud = to_world_transform.transform_points(p3d.transpose())
        (x, y, z) = point_cloud.tolist()[0]
        return Location(x, y, z)
    else:
        None


def get_camera_intrinsic_and_transform(image_size=(800, 600),
                                       position=(2.0, 0.0, 1.4),
                                       rotation_pitch=0,
                                       rotation_roll=0,
                                       rotation_yaw=0,
                                       fov=90.0):
    intrinsic_mat = create_intrinsic_matrix(image_size[0], image_size[1], fov)
    loc = Location(position[0], position[1], position[2])
    rot = Rotation(rotation_pitch, rotation_yaw, rotation_roll)
    camera_transform = Transform(loc, rot)
    camera_unreal_transform = camera_to_unreal_transform(camera_transform)
    return (intrinsic_mat, camera_unreal_transform, image_size)


def get_bounding_box_from_corners(corners):
    """
    Gets the bounding box of the pedestrian given the corners of the plane.
    """
    # Figure out the opposite ends of the rectangle. Our 2D mapping doesn't
    # return perfect rectangular coordinates and also doesn't return them
    # in clockwise order.
    max_distance = 0
    opp_ends = None
    for (a, b) in combinations(corners, r=2):
        if abs(a[0] - b[0]) <= 0.8 or abs(a[1] - b[1]) <= 0.8:
            # The points are too close. They may be lying on the same axis.
            # Move forward.
            pass
        else:
            # The points possibly lie on different axis. Choose the two
            # points which are the farthest.
            distance = (b[0] - a[0])**2 + (b[1] - a[1])**2
            if distance > max_distance:
                max_distance = distance
                if a[0] < b[0] and a[1] < b[1]:
                    opp_ends = (a, b)
                else:
                    opp_ends = (b, a)

    # If we were able to find two points far enough to be considered as
    # possible bounding boxes, return the results.
    return opp_ends


def get_bounding_box_sampling_points(ends):
    """
    Get the sampling points given the ends of the rectangle.
    """
    a, b = ends

    # Find the middle point of the rectangle, and see if the points
    # around it are visible from the camera.
    middle_point = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2,
                    b[2].flatten().item(0))
    sampling_points = [
        middle_point,
        (middle_point[0] + 2, middle_point[1], middle_point[2]),
        (middle_point[0] + 1, middle_point[1] + 1, middle_point[2]),
        (middle_point[0] + 1, middle_point[1] - 1, middle_point[2]),
        (middle_point[0] - 2, middle_point[1], middle_point[2]),
        (middle_point[0] - 1, middle_point[1] + 1, middle_point[2]),
        (middle_point[0] - 1, middle_point[1] - 1, middle_point[2])
    ]
    return (middle_point, sampling_points)


def get_2d_bbox_from_3d_box(
        depth_array, vehicle_transform, obj_transform,
        bounding_box, rgb_transform, rgb_intrinsic, rgb_img_size,
        middle_depth_threshold, neighbor_threshold):
    corners = map_ground_bounding_box_to_2D(
        vehicle_transform, obj_transform,
        bounding_box, rgb_transform, rgb_intrinsic,
        rgb_img_size)
    if len(corners) == 8:
        ends = get_bounding_box_from_corners(corners)
        if ends:
            (middle_point, points) = get_bounding_box_sampling_points(ends)
            # Select bounding box if the middle point in inside the frame
            # and has the same depth
            if (inside_image(middle_point[0], middle_point[1],
                             rgb_img_size[0], rgb_img_size[1]) and
                have_same_depth(middle_point[0],
                                middle_point[1],
                                middle_point[2],
                                depth_array,
                                middle_depth_threshold)):
                (xmin, xmax, ymin, ymax) = select_max_bbox(ends)
                width = xmax - xmin
                height = ymax - ymin
                # Filter out the small bounding boxes (they're far away).
                # We use thresholds that are proportional to the image size.
                # XXX(ionel): Reduce thresholds to 0.01, 0.01, and 0.0002 if
                # you want to include objects that are far away.
                if (width > rgb_img_size[0] * 0.01 and
                    height > rgb_img_size[1] * 0.02 and
                    width * height > rgb_img_size[0] * rgb_img_size[1] * 0.0004):
                    return (xmin, xmax, ymin, ymax)
            else:
                # The mid point doesn't have the same depth. It can happen
                # for valid boxes when the mid point is between the legs.
                # In this case, we check that a fraction of the neighbouring
                # points have the same depth.
                # Filter the points inside the image.
                points_inside_image = [
                    (x, y, z)
                    for (x, y, z) in points if inside_image(
                            x, y, rgb_img_size[0], rgb_img_size[1])
                ]
                same_depth_points = [
                    have_same_depth(x, y, z, depth_array, neighbor_threshold)
                            for (x, y, z) in points_inside_image
                ]
                if len(same_depth_points) > 0 and \
                        same_depth_points.count(True) >= 0.4 * len(same_depth_points):
                    (xmin, xmax, ymin, ymax) = select_max_bbox(ends)
                    width = xmax - xmin
                    height = ymax - ymin
                    width = xmax - xmin
                    height = ymax - ymin
                    # Filter out the small bounding boxes (they're far away).
                    # We use thresholds that are proportional to the image size.
                    # XXX(ionel): Reduce thresholds to 0.01, 0.01, and 0.0002 if
                    # you want to include objects that are far away.
                    if (width > rgb_img_size[0] * 0.01 and
                        height > rgb_img_size[1] * 0.02 and
                        width * height > rgb_img_size[0] * rgb_img_size[1] * 0.0004):
                        return (xmin, xmax, ymin, ymax)


def have_same_depth(x, y, z, depth_array, threshold):
    x, y = int(x), int(y)
    return abs(depth_array[y][x] * 1000 - z) < threshold


def inside_image(x, y, img_width, img_height):
    return x >= 0 and y >= 0 and x < img_width and y < img_height


def select_max_bbox(ends):
    (xmin, ymin) = tuple(map(int, ends[0][:2]))
    (xmax, ymax) = tuple(map(int, ends[0][:2]))
    corner = tuple(map(int, ends[1][:2]))
    # XXX(ionel): This is not quite correct. We get the
    # minimum and maximum x and y values, but these may
    # not be valid points. However, it works because the
    # bboxes are parallel to x and y axis.
    xmin = min(xmin, corner[0])
    ymin = min(ymin, corner[1])
    xmax = max(xmax, corner[0])
    ymax = max(ymax, corner[1])
    return (xmin, xmax, ymin, ymax)


def map_ground_bounding_box_to_2D(vehicle_transform,
                                  obj_transform,
                                  bounding_box,
                                  rgb_transform,
                                  rgb_intrinsic,
                                  rgb_img_size):
    (image_width, image_height) = rgb_img_size
    extrinsic_mat = vehicle_transform * rgb_transform

    # 8 bounding box vertices relative to (0,0,0)
    bbox = np.array([
        [  bounding_box.extent.x,   bounding_box.extent.y,   bounding_box.extent.z],
        [  bounding_box.extent.x, - bounding_box.extent.y,   bounding_box.extent.z],
        [  bounding_box.extent.x,   bounding_box.extent.y, - bounding_box.extent.z],
        [  bounding_box.extent.x, - bounding_box.extent.y, - bounding_box.extent.z],
        [- bounding_box.extent.x,   bounding_box.extent.y,   bounding_box.extent.z],
        [- bounding_box.extent.x, - bounding_box.extent.y,   bounding_box.extent.z],
        [- bounding_box.extent.x,   bounding_box.extent.y, - bounding_box.extent.z],
        [- bounding_box.extent.x, - bounding_box.extent.y, - bounding_box.extent.z]
    ])

    # Transform the vertices with respect to the bounding box transform.
    bbox = bounding_box.transform.transform_points(bbox)

    # The bounding box transform is with respect to the object transform.
    # Transform the points relative to its transform.
    bbox = obj_transform.transform_points(bbox)

    # Object's transform is relative to the world. Thus, the bbox contains
    # the 3D bounding box vertices relative to the world.

    coords = []
    for vertex in bbox:
        pos_vector = np.array([
            [vertex[0,0]],  # [[X,
            [vertex[0,1]],  #   Y,
            [vertex[0,2]],  #   Z,
            [1.0]           #   1.0]]
        ])
        # Transform the points to camera.
        transformed_3d_pos = np.dot(inv(extrinsic_mat.matrix), pos_vector)
        # Transform the points to 2D.
        pos2d = np.dot(rgb_intrinsic, transformed_3d_pos[:3])

        # Normalize the 2D points.
        loc_2d = Location(float(image_width - pos2d[0] / pos2d[2]),
                          float(image_height - pos2d[1] / pos2d[2]),
                          pos2d[2])
        # Add the points to the image.
        if loc_2d.z > 0: # If the point is in front of the camera.
            if (loc_2d.x >= 0 or loc_2d.y >= 0) and (loc_2d.x < image_width or loc_2d.y < image_height):
                coords.append((loc_2d.x, loc_2d.y, loc_2d.z))

    return coords


def map_ground_3D_transform_to_2D(location,
                                  vehicle_transform,
                                  rgb_transform,
                                  rgb_intrinsic,
                                  rgb_img_size):
    transform = vehicle_transform * rgb_transform
    extrinsic_mat = transform.matrix
    # The position in world 3D coordiantes.
    pos_vector = np.array([[location.x], [location.y], [location.z], [1.0]])
    transformed_3d_pos = np.dot(inv(extrinsic_mat), pos_vector)
    pos2d = np.dot(rgb_intrinsic, transformed_3d_pos[:3])
    (img_width, img_height) = rgb_img_size
    loc_2d = Location(img_width - pos2d[0] / pos2d[2],
                      img_height - pos2d[1] / pos2d[2],
                      pos2d[2])
    if (loc_2d.z > 0 and loc_2d.x >= 0 and loc_2d.x < img_width and
        loc_2d.y >= 0 and loc_2d.y < img_height):
        return (loc_2d.x, loc_2d.y, loc_2d.z)
    return None

def match_bboxes_with_traffic_lights(bboxes, traffic_lights):
    # Match bounding boxes with traffic lights. In order to match,
    # the bounding box must be within 20 m of the base of the traffic light
    # in the (x,y) plane, and must be between 5 and 7 meters above the base
    # of the traffic light. If there are multiple possibilities, take the closest.
    result = []
    for bbox in bboxes:
        best_tl_idx = 0
        best_dist = 1000000
        for idx in range(len(traffic_lights)):
            tl = traffic_lights[idx]
            dist = (bbox[0].x - tl[0].x)**2 + (bbox[0].y - tl[0].y)**2
            # Check whether the traffic light is the closest so far to the
            # bounding box, and that the traffic light is between 5 and 7
            # meters above the base of the traffic light.
            if dist < best_dist and bbox[0].z - tl[0].z > 5 and bbox[0].z - tl[0].z < 7:
                best_tl_idx = idx
                best_dist = dist
        if best_dist < 20 ** 2:
            result.append((bbox[1], traffic_lights[best_tl_idx][1]))
    return result
