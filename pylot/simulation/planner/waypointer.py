# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
# Modified by ERDOS team.

import math
import os
import random
import numpy as np
from numpy import linalg as LA

from converter import Converter
from city_track import CityTrack
import bezier


def angle_between(v1, v2):
    return np.arccos(np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2))


def sldist(t1, t2):
    return math.sqrt((t1[0] - t2[0]) * (t1[0] - t2[0]) +
                     (t1[1] - t2[1]) * (t1[1] - t2[1]))


class Waypointer(object):
    def __init__(self, city_name):
        # Open the necessary files
        dir_path = os.path.dirname(__file__)
        self.city_file = os.path.join(dir_path, city_name + '.txt')
        self.city_name = city_name

        # Define the specif parameter for the waypointer. Where is the middle of the road,
        # how open are the curves being made, etc.

        self.lane_shift_distance = 13  # The amount of shifting from the center the car should go
        self.extra_spacing_rights = -3
        self.extra_spacing_lefts = 7  # This is wrong, since it is expressed in world units
        self.way_key_points_predicted = 7
        self.number_of_waypoints = 30

        self._converter = Converter(self.city_file, 0.1643, 50.0)
        self._city_track = CityTrack(self.city_name)
        self._map = self._city_track.get_map()

        # Define here some specific configuration to produce waypoints
        self.last_trajectory = []
        self.lane_shift_distance = self.lane_shift_distance  # The amount of shifting from the center the car should go
        self.extra_spacing_rights = self.extra_spacing_rights
        self.extra_spacing_lefts = self.extra_spacing_lefts
        self.way_key_points_predicted = self.way_key_points_predicted
        self.number_of_waypoints = self.number_of_waypoints
        self.previous_map = [0, 0]

        # The internal state variable
        self.last_trajectory = []
        self._route = []
        self.previous_map = [0, 0]
        self._previous_source = None
        self.last_map_points = None
        self.points = None

    def reset(self):
        self.last_trajectory = []
        self._route = []
        self.previous_map = [0, 0]
        self._previous_source = None
        self.last_map_points = None
        self.points = None

    def _search_around_square(self, map_point, map_central_2d):
        """
            Function to search the map point in the central line.
        Args:
            map_point: the used map point
            map_central_2d: the 2d map containing the central lines in red

        Returns:
            projected point in the central line

        """

        x = int(map_point[0])
        y = int(map_point[1])

        square_crop = map_central_2d[(y - 30):(y + 30), (x - 30):(x + 30)]
        small_distance = 10000
        closest_point = [
            15 - square_crop.shape[1] / 2, 15 - square_crop.shape[0] / 2
        ]

        for t in np.transpose(np.nonzero(square_crop)):
            distance = sldist(
                t, [square_crop.shape[1] / 2, square_crop.shape[0] / 2])
            if distance < small_distance:
                small_distance = distance
                closest_point = [
                    t[0] - square_crop.shape[1] / 2,
                    t[1] - square_crop.shape[0] / 2
                ]

        return np.array([x + closest_point[0], y + closest_point[1]])

    def _shift_points(self, distance_to_center, lane_points,
                      inflection_position):
        """
            Function to take the route points in the middle of the road and shift then to the
            center of the lane
        Args:
            distance_to_center: The distance you want to shift
            lane_points: the lane points used
            inflection_position: A corner case, when there is a turn.

        Returns:

        """
        shifted_lane_vec = []
        for i in range(len(lane_points[:-1])):
            lane_point = lane_points[i]
            unit_vec = self._get_unit(lane_points[i + 1], lane_points[i])
            shifted_lane = [
                lane_point[0] + unit_vec[0] * distance_to_center[i],
                lane_point[1] + unit_vec[1] * distance_to_center[i]
            ]
            if i == inflection_position:
                unit_vec = self._get_unit(lane_points[i], lane_points[i - 1])
                shifted_lane_vec.append([
                    lane_point[0] + unit_vec[0] * distance_to_center[i],
                    lane_point[1] + unit_vec[1] * distance_to_center[i]
                ])
            shifted_lane_vec.append(shifted_lane)
        last_lane_point = lane_points[-1]
        shifted_lane = [
            last_lane_point[0] + unit_vec[0] * distance_to_center[-1],
            last_lane_point[1] + unit_vec[1] * distance_to_center[-1]
        ]
        shifted_lane_vec.append(shifted_lane)
        return shifted_lane_vec

    # Given a list, find the 3 curve points that this list correspond
    def _find_curve_points(self, points):
        """
            Function to find points when there is a curve.
        Args:
            points: the search space

        Returns:
            the points when there is a curve.
        """
        curve_points = None
        first_time = True
        prev_unit_vec = None
        for i in range(len(points) - 1):
            unit_vec = self._get_unit(points[i + 1], points[i])
            unit_vec = [round(unit_vec[0]), round(unit_vec[1])]
            if not first_time:
                if unit_vec != prev_unit_vec:
                    curve_points = [points[i + 1], points[i], points[i - 1]]
                    return curve_points, [i + 1, i, i - 1], np.cross(
                        unit_vec, prev_unit_vec)
            first_time = False
            prev_unit_vec = unit_vec
        return curve_points, None, None

    def _get_unit(self, last_pos, first_pos):
        """
            Get a unity vector from two points point
        """
        vector_dir = ((last_pos - first_pos) / LA.norm(last_pos - first_pos))
        vector_s_dir = [0, 0]
        vector_s_dir[0] = -vector_dir[1]
        vector_s_dir[1] = vector_dir[0]
        return vector_s_dir

    def generate_final_trajectory(self, coarse_trajectory):
        """
            Smooth the waypoints trajectory using a bezier curve.
        Args:
            coarse_trajectory:

        Returns:
        """
        total_course_trajectory_distance = 0
        previous_point = coarse_trajectory[0]
        for i in range(1, len(coarse_trajectory)):
            total_course_trajectory_distance += sldist(coarse_trajectory[i],
                                                       previous_point)

        points = bezier.bezier_curve(
            coarse_trajectory,
            max(1, int(total_course_trajectory_distance / 10.0)))
        world_points = []
        points = np.transpose(points)
        points_list = []
        for point in points:
            world_points.append(self._converter.convert_to_world(point))
            points_list.append(point.tolist())
        return world_points, points_list

    def get_free_node_direction_target(self, pos, pos_ori, source):
        """
            Get free positions to drive in the direction of the target point
        """
        free_nodes = self._map.get_adjacent_free_nodes(pos)
        added_walls = set()
        heading_start = np.array([pos_ori[0], pos_ori[1]])

        for adj in free_nodes:
            start_to_goal = np.array([adj[0] - pos[0], adj[1] - pos[1]])
            angle = angle_between(heading_start, start_to_goal)
            if angle < 2 and adj != source:
                added_walls.add((adj[0], adj[1]))

        return added_walls

    def graph_to_waypoints(self, next_route):
        """
            Convert the graph to raw waypoints, with the same size as as the route.
            Basically just project the route to the map and shift to the center of the lane.
        Args:
            next_route: the graph points (nodes) that are going to be converted.

        Returns:
            the list of waypoints

        """
        # Take the map with the central lines
        lane_points = []
        for point in next_route:
            map_point = self._converter.convert_to_pixel(
                [int(point[0]), int(point[1])])
            lane_points.append(
                self._search_around_square(map_point,
                                           self._map.map_image_center))

        # THE CURVE POINTS
        _, points_indexes, curve_direction = self._find_curve_points(
            lane_points)
        # If it is a intersection we divide this in two parts

        lan_shift_distance_vec = [self.lane_shift_distance] * len(lane_points)
        if points_indexes is not None:
            for i in points_indexes:
                if curve_direction > 0:
                    lan_shift_distance_vec[i] += (self.extra_spacing_lefts * 1)
                else:
                    lan_shift_distance_vec[i] += (
                        self.extra_spacing_rights * -1)

            shifted_lane_vec = self._shift_points(
                lan_shift_distance_vec, lane_points, points_indexes[1])
        else:
            shifted_lane_vec = self._shift_points(lan_shift_distance_vec,
                                                  lane_points, None)
        return shifted_lane_vec

    def add_extra_points(self, node_target, target_ori, node_source):
        """
            Hacky: Add extra points after the target. The route needs to
        """
        direction = node_target
        direction_ori = target_ori

        while len(self._route) < 10:  # ADD EXTRA POINTS AFTER
            try:
                free_nodes = list(
                    self.get_free_node_direction_target(
                        direction, direction_ori, node_source))

                direction_ori = self._get_unit(
                    np.array(direction), np.array(free_nodes[0]))
                aux = -direction_ori[1]
                direction_ori[1] = direction_ori[0]
                direction_ori[0] = aux

                direction = free_nodes[0]
            except IndexError:

                # Repeate some route point, there is no problem.
                direction = [
                    round(self._route[-1][0] + direction_ori[0]),
                    round(self._route[-1][1] + direction_ori[1])
                ]

            self._route.append(direction)

    def convert_list_of_nodes_to_pixel(self, route):
        map_points = []
        for point in route:
            map_point = self._converter.convert_to_pixel(
                [int(point[0]), int(point[1])])
            map_points.append(map_point)
        return map_points

    def get_next_waypoints(self, source, source_ori, target, target_ori):
        """
            Get the next waypoints, from a list of generated waypoints.
        Args:
            source: source position
            source_ori: source orientation
            target: the desired end position
            target_ori: the desired target orientation

        Returns:
        """
        # Project the source and target on the node space.
        track_source = self._city_track.project_node(source)
        track_target = self._city_track.project_node(target)

        # Test if it is already at the goal
        if track_source == track_target:
            self.reset()
            return self.last_trajectory, self.last_map_points, self.convert_list_of_nodes_to_pixel(
                self._route)

        # This is to avoid computing a new route when inside the route
        # The the distance to the closest intersection.
        distance_node = self._city_track.closest_curve_position(track_source)

        # Potential problem, if the car goest too fast, there can be problems for the turns.
        # I will keep this for a while.
        if distance_node > 2 and self._previous_source != track_source:
            self._route = self._city_track.compute_route(
                track_source, source_ori, track_target, target_ori)

            # IF needed we add points after the objective, that is very hacky.
            self.add_extra_points(track_target, target_ori, track_source)

            self.points = self.graph_to_waypoints(
                self._route[1:(1 + self.way_key_points_predicted)])

            self.last_trajectory, self.last_map_points = self.generate_final_trajectory(
                [np.array(self._converter.convert_to_pixel(source))] +
                self.points)

            # Store the previous position, to avoid recomputation
            self._previous_source = track_source
            return self.last_trajectory, self.last_map_points, self.points
        else:
            if sldist(self.previous_map,
                      self._converter.convert_to_pixel(source)) > 1.0:
                # That is because no route was ever computed. This is a problem we should solve.
                if not self._route:
                    self._route = self._city_track.compute_route(
                        track_source, source_ori, track_target, target_ori)
                    self.add_extra_points(track_target, target_ori,
                                          track_source)

                    self.points = self.graph_to_waypoints(
                        self._route[1:(1 + self.way_key_points_predicted)])

                    self.last_trajectory, self.last_map_points = self.generate_final_trajectory(
                        [np.array(self._converter.convert_to_pixel(source))] +
                        self.points)

                # We have to find the current node position
                self.previous_map = self._converter.convert_to_pixel(source)
                # Make a source not replaced

                for point in self.last_map_points:
                    point_vec = self._get_unit(
                        np.array(self._converter.convert_to_pixel(source)),
                        point)
                    cross_product = np.cross(source_ori[0:2], point_vec)

                    if (cross_product > 0.0 and sldist(
                            point, self._converter.convert_to_pixel(source)) <
                            50) or sldist(
                                point, self._converter.convert_to_pixel(
                                    source)) < 15.0:
                        self.last_trajectory.remove(
                            self._converter.convert_to_world(point)
                        )  # = [self.make_world_map(point)] + self.last_trajc
                        self.last_map_points.remove(point)

            # Store the previous position, to avoid recomputation
            #self._previous_source = track_source

            return self.last_trajectory, self.last_map_points, self.points

    def route_test(self, node_source, source_ori, node_target, target_ori):
        route = self._city_track.compute_route(node_source, source_ori,
                                               node_target, target_ori)
        return not route == None
