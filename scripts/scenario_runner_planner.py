import numpy as np


SPEED_LIMIT = 35  # m/s
WP_PRECISION = 1.0  # meters


def lateral_shift(transform, shift):
    transform.rotation.yaw += 90
    return transform.location + shift * transform.get_forward_vector()


def walk_lane(waypoint, wp_prec):
    next_wp = [waypoint]
    waypoints = []
    while len(next_wp) == 1:
        waypoints.append(next_wp[0])
        next_wp = next_wp[0].next(wp_prec)
    return waypoints


def bounds(waypoints, kind="right"):
    mul = 0.5
    if kind == "left":
        mul = -0.5

    return [lateral_shift(w.transform, mul * w.lane_width) for w in waypoints]


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

        # get current road bounds
        left_most_lane = waypoint
        right_most_lane = waypoint

        while left_most_lane.get_left_lane():
            left_most_lane = left_most_lane.get_left_lane()

        while right_most_lane.get_right_lane():
            right_most_lane = right_most_lane.get_right_lane()

        left_most_waypoints = walk_lane(left_most_lane, self.waypoint_precision)
        right_most_waypoints = walk_lane(right_most_lane, self.waypoint_precision)
        left_most_bounds = bounds(left_most_waypoints, kind="left")
        right_most_bounds = bounds(right_most_waypoints)

        return {
            "current_lane_center_markings": lane_waypoints,
            "current_lane_bounds": [left_bounds, right_bounds],
            "maximal_road_bounds": [left_most_bounds, right_most_bounds]
        }

    def get_static_objects(self):
        """
        Return the bounding boxes of all static objects in view.

        :return: list of bboxes
        """
        static_bboxes = []

        # get ped bboxes (minus dynamic pedestrian)
        for pedestrian in self.world.get_actors().filter('walker.*'):
            if pedestrian.attributes["role_name"] == "pedestrian":
                pass
            static_bboxes.append(pedestrian.bounding_box)

        # get vehicle bboxes (minus dynamic hero)
        for vehicle in self.world.get_actors().filter('vehicle.*'):
            if vehicle.attributes["role_name"] == "hero":
                pass
            static_bboxes.append(vehicle.bounding_box)

        return static_bboxes

    def get_dynamic_objects(self):
        """
        Return the bounding boxes of all the dynamic objects in view.

        :return: list of bboxes
        """
        # get dynamic pedestrian bbox
        for pedestrian in self.world.get_actors().filter('walker.*'):
            if pedestrian.attributes["role_name"] == "pedestrian":
                return [pedestrian.bounding_box]
        return []

    def get_speed_limit(self):
        """
        Return the speed limit in m/s of the current lane the vehicle is driving on.
        TODO: get the true speed limit

        :return: float in m/s
        """
        return self.speed_limit

    def get_ego_size(self):
        """
        Return the bbox of the ego vehicle.

        :return: dict of {"hero": {lower_left, upper_right}}
        """
        return self.ego_bbox

    def get_ego_location(self):
        """
        Return the center location of the ego vehicle.

        :return: np.ndarray of [x, y]
        """
        location = self.ego_vehicle.get_location()
        return np.array([location.x, location.y])

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
        return lane_waypoints

