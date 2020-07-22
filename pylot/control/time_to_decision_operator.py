import sys
from collections import deque

import erdos


class TimeToDecisionOperator(erdos.Operator):
    def __init__(self, pose_stream, obstacles_stream, time_to_decision_stream,
                 flags):
        pose_stream.add_callback(self.on_pose_update,
                                 [time_to_decision_stream])
        obstacles_stream.add_callback(self.on_obstacles_update)
        erdos.add_watermark_callback([pose_stream, obstacles_stream],
                                     [time_to_decision_stream],
                                     self.on_watermark)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags
        self._initialized = False
        self._simulator_in_sync = False
        self._pose_msgs = deque()
        self._obstacles_msgs = deque()

    @staticmethod
    def connect(pose_stream, obstacles_stream):
        return [erdos.WriteStream()]

    def on_pose_update(self, msg, time_to_decision_stream):
        """Called for each sensor reading."""
        self._logger.debug('@{}: received pose message'.format(msg.timestamp))
        if not self._initialized:
            self._initialized = True
            self._logger.debug('@{}: Sending ttd for {}'.format(
                msg.timestamp, msg.timestamp))
            # Send a first TTD message so that the operators can start running.
            ttd = self.time_to_decision(msg.data.transform,
                                        msg.data.forward_speed, [])
            time_to_decision_stream.send(erdos.Message(msg.timestamp, ttd))
            time_to_decision_stream.send(erdos.WatermarkMessage(msg.timestamp))
        self._pose_msgs.append(msg)

    def on_obstacles_update(self, msg):
        self._logger.debug('@{}: received obstacles message'.format(
            msg.timestamp))
        self._obstacles_msgs.append(msg)

    def on_watermark(self, timestamp, time_to_decision_stream):
        self._logger.debug('@{}: received watermark'.format(timestamp))
        obstacles_msg = self._obstacles_msgs.popleft()
        pose_msg = self._pose_msgs.popleft()
        ttd = self.time_to_decision(pose_msg.data.transform,
                                    pose_msg.data.forward_speed,
                                    obstacles_msg.obstacle_trajectories)
        # Send a TTD for the next round.
        if (self._flags.carla_localization_frequency == -1
                or (not self._simulator_in_sync
                    and self._flags.carla_localization_frequency != 8
                    and self._flags.carla_localization_frequency != 10)):
            self._simulator_in_sync = True
            next_sensor_time = (
                timestamp.coordinates[0] + int(1000 / self._flags.carla_fps) +
                int(1000 / self._flags.carla_localization_frequency))
        else:
            next_sensor_time = (
                timestamp.coordinates[0] +
                int(1000 / self._flags.carla_localization_frequency))
        next_sensor_timestamp = erdos.Timestamp(coordinates=[next_sensor_time])
        # next_sensor_timestamp = erdos.Timestamp(coordinates=[
        #     timestamp.coordinates[0] + int(1000 / self._flags.carla_fps)
        # ])
        time_to_decision_stream.send(erdos.Message(next_sensor_timestamp, ttd))
        time_to_decision_stream.send(
            erdos.WatermarkMessage(next_sensor_timestamp))
        self._logger.debug('@{}: Sending ttd for {}'.format(
            timestamp, next_sensor_timestamp))

    def time_to_decision(self, ego_transform, forward_speed, obstacles):
        """Computes time to decision (in ms)."""
        # planning_runtimes = [309, 208, 148, 67, 40]
        # List of detection distances and runtimes.
        detection_distances = [(10, 24), (15, 42), (25, 74), (30, 141)]
        # We should be working with predictions, ignore obstacles that are
        # headed in the same direction as the ego-vehicle, and the ones that
        # are not going to invade ego vehicle's lane.
        min_dist = sys.maxsize
        for obstacle in obstacles:
            obs_loc = (ego_transform * obstacle.trajectory[-1]).location
            if self._map.are_on_same_lane(ego_transform.location, obs_loc):
                dist_to_obstacle = ego_transform.location.distance(obs_loc)
                if dist_to_obstacle > 1:
                    min_dist = min(dist_to_obstacle, min_dist)
        if min_dist == sys.maxsize or forward_speed < 0.1:
            return 600, 190

        braking_dist = forward_speed * forward_speed / (2.0 * 0.8 * 9.81)
        time_to_decision = min(
            600, max(100,
                     (min_dist - braking_dist) * 1000 / forward_speed / 3))

        # Min runtime of other components.
        time_for_detection = time_to_decision - 80
        det_deadline = None
        for detection_dist, det_runtime in detection_distances:
            if (detection_dist >= dist_to_obstacle
                    and det_runtime <= time_for_detection):
                det_deadline = det_runtime
        if det_deadline is None:
            if dist_to_obstacle > 30:
                det_deadline = 190
            else:
                det_deadline = 24
        self._logger.debug('TTD computed: braking distance {}, ttd {}, '
                           'detection deadline {}, min dist {}'.format(
                               braking_dist, time_to_decision, det_deadline,
                               min_dist))
        return time_to_decision, det_deadline

    def run(self):
        from pylot.map.hd_map import HDMap
        from pylot.simulation.utils import get_map
        self._map = HDMap(
            get_map(self._flags.carla_host, self._flags.carla_port,
                    self._flags.carla_timeout))
