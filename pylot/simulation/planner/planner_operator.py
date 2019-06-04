import numpy as np

from erdos.data_stream import DataStream
from erdos.message import Message
from erdos.op import Op
from erdos.utils import setup_logging

import city_track

# Constants Used for the high level commands
REACH_GOAL = 0.0
GO_STRAIGHT = 5.0
TURN_RIGHT = 4.0
TURN_LEFT = 3.0
LANE_FOLLOW = 2.0


def signal(v1, v2):
    return np.cross(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)


class PlannerOperator(Op):
    def __init__(self,
                 city_name,
                 goal_location,
                 goal_orientation,
                 log_file_name=None,
                 name='planner'):
        super(PlannerOperator, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self._city_name = city_name
        self._goal_location = goal_location
        self._goal_orientation = goal_orientation
        self._commands = []

    def setup_streams(self, input_streams):
        input_streams.add_callback(self.on_goal)
        return [DataStream(name='planner_output')]

    def on_goal(self, msg):
        (location, orientation) = msg.data
        direction = self.get_next_command(
            location, orientation, self._goal_location, self._goal_orientation)
        print('Computed directions {}'.format(direction))
        self.get_output_stream('planner_output').send(
            Message(direction, msg.timestamp))

    def execute(self):
        self._city_track = city_track.CityTrack(self._city_name)
        self.spin()

    def get_next_command(self, source, source_ori, target, target_ori):
        """
        Computes the full plan and returns the next command,
        Args
            source: source position
            source_ori: source orientation
            target: target position
            target_ori: target orientation
        Returns
            a command ( Straight,Lane Follow, Left or Right)
        """

        track_source = self._city_track.project_node(source)
        track_target = self._city_track.project_node(target)

        # reach the goal

        if self._city_track.is_at_goal(track_source, track_target):
            return REACH_GOAL

        if (self._city_track.is_at_new_node(track_source)
                and self._city_track.is_away_from_intersection(track_source)):

            route = self._city_track.compute_route(track_source, source_ori,
                                                   track_target, target_ori)
            if route is None:
                raise RuntimeError('Impossible to find route')

            self._commands = self._route_to_commands(route)

            if self._city_track.is_far_away_from_route_intersection(
                    track_source):
                return LANE_FOLLOW
            else:
                if self._commands:
                    return self._commands[0]
                else:
                    return LANE_FOLLOW
        else:

            if self._city_track.is_far_away_from_route_intersection(
                    track_source):
                return LANE_FOLLOW

            # If there are computed commands
            if self._commands:
                return self._commands[0]
            else:
                return LANE_FOLLOW

    def get_shortest_path_distance(self, source, source_ori, target,
                                   target_ori):

        distance = 0
        track_source = self._city_track.project_node(source)
        track_target = self._city_track.project_node(target)

        current_pos = track_source

        route = self._city_track.compute_route(track_source, source_ori,
                                               track_target, target_ori)
        # No Route, distance is zero
        if route is None:
            return 0.0

        for node_iter in route:
            distance += sldist(node_iter, current_pos)
            current_pos = node_iter

        # We multiply by these values to convert distance to world coordinates
        return distance * float(self._city_track.get_pixel_density()) \
               * float(self._city_track.get_node_density())

    def is_there_posible_route(self, source, source_ori, target, target_ori):

        track_source = self._city_track.project_node(source)
        track_target = self._city_track.project_node(target)

        return not self._city_track.compute_route(
            track_source, source_ori, track_target, target_ori) is None

    def test_position(self, source):

        node_source = self._city_track.project_node(source)

        return self._city_track.is_away_from_intersection(node_source)

    def _route_to_commands(self, route):
        """
        from the shortest path graph, transform it into a list of commands

        :param route: the sub graph containing the shortest path
        :return: list of commands encoded from 0-5
        """
        commands_list = []
        for i in range(0, len(route)):
            if route[i] not in self._city_track.get_intersection_nodes():
                continue
            current = route[i]
            past = route[i - 1]
            future = route[i + 1]
            past_to_current = np.array(
                [current[0] - past[0], current[1] - past[1]])
            current_to_future = np.array(
                [future[0] - current[0], future[1] - current[1]])
            angle = signal(current_to_future, past_to_current)
            if angle < -0.1:
                command = TURN_RIGHT
            elif angle > 0.1:
                command = TURN_LEFT
            else:
                command = GO_STRAIGHT
            commands_list.append(command)
        return commands_list
