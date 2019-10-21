import numpy as np


SPEED_LIMIT = 35  # m/s
WP_PRECISION = 1.0  # meters


def lateral_shift(transform, shift):
    """
    Get the location corresponding to orthogonal vector.

    :param transform: carla.Transform
    :param shift: float in meters
    :return: carla.Location
    """
    transform.rotation.yaw += 90
    return transform.location + shift * transform.get_forward_vector()


def walk_lane(waypoint, wp_prec):
    """
    Get the next waypoints at distance wp_prec until no more points exist in the lane.

    :param waypoint: carla.Waypoint
    :param wp_prec: float in meters
    :return: list of carla.Waypoint
    """
    next_wp = [waypoint]
    waypoints = []
    while len(next_wp) == 1:
        waypoints.append(next_wp[0])
        next_wp = next_wp[0].next(wp_prec)
    return waypoints


def bounds(waypoints, kind="right"):
    """
    Return the left or right boundary markings for the given waypoints.
    Calculated by adding the vector orthogonal to the direction of the waypoints scaled by lane width.

    :param waypoints: List of carla.Waypoint
    :param kind: str ("left" or "right")
    :return: list of carla.Location
    """
    mul = 0.5
    if kind == "left":
        mul = -0.5

    return [lateral_shift(w.transform, mul * w.lane_width) for w in waypoints]


def bbox_2_global(bbox, location):
    """
    Transform the relative bbox to global coordinates using location.

    :param bbox: carla.BoundingBox
    :param location: carla.Location
    :return: [vertex_1, vertex_2, vertex_3, vertex_4]
    """
    bbox_location, bbox_extent = bbox.location, bbox.extent
    global_x, global_y = location.x + bbox_location.x, location.y + bbox_location.y

    global_bbox = [
        [global_x + bbox_extent.x, global_y + bbox_extent.y],
        [global_x - bbox_extent.y, global_y + bbox_extent.y],
        [global_x - bbox_extent.y, global_y - bbox_extent.y],
        [global_x + bbox.extent.x, global_y - bbox_extent.y]
    ]
    return global_bbox


class Planner(object):
    """
    Temporary planner class for scenarios involving one ego_vehicle and one pedestrian.
    """
    def __init__(self, ego_vehicle, pedestrian, waypoint_precision=WP_PRECISION, speed_limit=SPEED_LIMIT):
        # Set world and map
        self.world = ego_vehicle.get_world()
        self.map = self.world.get_map()

        # Set ego and pedestrian agent
        self.ego_vehicle = ego_vehicle
        self.pedestrian = pedestrian

        # Set misc. params
        self.waypoint_precision = waypoint_precision
        self.speed_limit = speed_limit

    def get_time(self):
        """
        Return simulated seconds elapsed since the beginning of the current episode.


        :return: float
        """
        return self.world.get_snapshot().timestamp.elapsed_seconds

    def get_road_bounds_and_lane_marks(self):
        """
        Return the bounds of the current road the vehicle is driving on. Bounds on both the current lane and
        entire current road are returned.


        :return: dict of {
            current_lane_center_markings: [center_markings]
            current_lane_bounds: [left_bounds, right_bounds],
            maximal_road_bounds: [maximal_left_bounds, maximal_right_bounds]
        }
        """
        location = self.ego_vehicle.get_location()
        waypoint = self.map.get_waypoint(location)

        # get the current lane bounds
        lane_waypoints = walk_lane(waypoint, self.waypoint_precision)
        left_bounds = bounds(lane_waypoints, kind="left")
        right_bounds = bounds(lane_waypoints)
        return [[[b.x, b.y] for b in left_bounds], [[b.x, b.y] for b in right_bounds]]

        # TODO: this is taking a long time, check for infinite loops
        # # get current road bounds
        # left_most_lane = waypoint
        # right_most_lane = waypoint
        #
        # while left_most_lane.get_left_lane():
        #     left_most_lane = left_most_lane.get_left_lane()
        #
        # while right_most_lane.get_right_lane():
        #     right_most_lane = right_most_lane.get_right_lane()
        #
        # left_most_waypoints = walk_lane(left_most_lane, self.waypoint_precision)
        # right_most_waypoints = walk_lane(right_most_lane, self.waypoint_precision)
        # left_most_bounds = bounds(left_most_waypoints, kind="left")
        # right_most_bounds = bounds(right_most_waypoints)
        #
        # return {
        #     "current_lane_center_markings": lane_waypoints,
        #     "current_lane_bounds": [left_bounds, right_bounds],
        #     "maximal_road_bounds": [left_most_bounds, right_most_bounds]
        # }

    def get_static_bboxes(self):
        """
        Return the bounding boxes of all static objects in view.

        :return: list of bboxes
        """
        static_bboxes = []

        # get ped bboxes (minus dynamic pedestrian)
        for pedestrian in self.world.get_actors().filter('walker.*'):
            if pedestrian.attributes["role_name"] == "pedestrian":
                pass
            static_bboxes.append(bbox_2_global(pedestrian.bounding_box, pedestrian.get_location()))

        # get vehicle bboxes (minus dynamic hero)
        for vehicle in self.world.get_actors().filter('vehicle.*'):
            if vehicle.attributes["role_name"] == "hero":
                pass
            static_bboxes.append(bbox_2_global(vehicle.bounding_box, vehicle.get_location()))

        return static_bboxes

    def get_dynamic_bboxes(self):
        """
        Return the bounding boxes of all the dynamic objects in view.

        :return: list of bboxes
        """
        # get dynamic pedestrian bbox
        for pedestrian in self.world.get_actors().filter('walker.*'):
            if pedestrian.attributes["role_name"] == "pedestrian":
                return [bbox_2_global(pedestrian.bounding_box, pedestrian.get_location())]
        return []

    def get_speed_limit(self):
        """
        Return the speed limit in m/s of the current lane the vehicle is driving on.
        TODO: get the true speed limit

        :return: float in m/s
        """
        return self.speed_limit

    def get_ego_bbox(self):
        """
        Return the bbox of the ego vehicle.

        :return: dict of {"hero": {lower_left, upper_right}}
        """
        return bbox_2_global(self.ego_vehicle.bounding_box, self.ego_vehicle.get_location())

    def get_ego_location(self):
        """
        Return the center location of the ego vehicle.

        :return: np.ndarray of [x, y]
        """
        location = self.ego_vehicle.get_location()
        return [location.x, location.y]

    def get_ego_speed(self):
        """
        Return the speed in m/s of the ego vehicle.

        :return: float in m/s
        """
        vel_vec = self.ego_vehicle.get_velocity()
        speed = np.sqrt(vel_vec.x**2 + vel_vec.y**2)
        return speed

    def get_ego_accel(self):
        """
        Return the acceleration in m/s^2 of the ego vehicle.

        :return: float in m/s^2
        """
        acc_vec = self.ego_vehicle.get_acceleration()
        acceleration = np.sqrt(acc_vec.x**2 + acc_vec.y**2)
        return acceleration

    def get_ego_path(self):
        """
        Return the desired path of the ego vehicle.

        :return: np.ndarray of [waypoint_1, waypoint_2, ... waypoint_n]

        TODO: this is hacked to for our straight road, need to incorporate idea of global intent
        """
        location = self.ego_vehicle.get_location()
        waypoint = self.map.get_waypoint(location)
        lane_waypoints = walk_lane(waypoint, self.waypoint_precision)
        return [[wp.transform.location.x, wp.transform.location.y] for wp in lane_waypoints]

