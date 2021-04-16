from collections import deque

import erdos

from pylot.utils import time_epoch_ms


class TimeToDecisionOperator(erdos.Operator):
    def __init__(self, pose_stream: erdos.ReadStream,
                 obstacles_prediction_stream: erdos.ReadStream,
                 time_to_decision_stream: erdos.WriteStream, flags):
        pose_stream.add_callback(self.on_pose_update,
                                 [time_to_decision_stream])
        obstacles_prediction_stream.add_callback(
            self.on_obstacles_prediction_update, [time_to_decision_stream])
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            self.config.name + '-csv', self.config.csv_log_file_name)
        self._last_pose = None
        self._last_obstacles = {}
        self._new_obstacles = deque([], maxlen=8)
        self._e2e_runtime = 433

    @staticmethod
    def connect(pose_stream: erdos.ReadStream,
                obstacles_prediction_stream: erdos.ReadStream):
        return [erdos.WriteStream()]

    def destroy(self):
        self._logger.warn('destroying {}'.format(self.config.name))

    def on_pose_update(self, msg: erdos.Message,
                       time_to_decision_stream: erdos.WriteStream):
        self._logger.debug('@{}: {} received pose message'.format(
            msg.timestamp, self.config.name))
        self._last_pose = msg.data
        # ttd = TimeToDecisionOperator.time_to_decision(msg.data.transform,
        #                                               msg.data.forward_speed,
        #                                               None)
        # time_to_decision_stream.send(erdos.Message(msg.timestamp, ttd))

    def on_obstacles_prediction_update(
            self, msg: erdos.Message,
            time_to_decision_stream: erdos.WriteStream):
        self._logger.debug(
            '@{}: {} received obstacle predictions message'.format(
                msg.timestamp, self.config.name))
        ttd = self.time_to_decision(self._last_pose, msg.predictions)
        # TODO: What timestamp to use?
        time_to_decision_stream.send(erdos.Message(msg.timestamp, ttd))
        self._csv_logger.info("{},{},ttd,{}".format(
            time_epoch_ms(), msg.timestamp.coordinates[0], ttd))

    # @staticmethod
    # def time_to_decision(pose, forward_speed, obstacles):
    #     """Computes time to decision (in ms)."""
    #     # Time to deadline is 400 ms when driving at 10 m/s
    #     # Deadline decreases by 10ms for every 1m/s of extra speed.
    #     time_to_deadline = 400 - (forward_speed - 10) * 10
    #     # TODO: Include other environment information in the calculation.
    #     return time_to_deadline

    def time_to_decision(self, pose, obstacles_predictions):
        obstacles_transform = {}
        cur_new_obstacles = []
        for obstacle_prediction in obstacles_predictions:
            obs_id = obstacle_prediction.obstacle_trajectory.obstacle.id
            obstacles_transform[obs_id] = obstacle_prediction.transform
            if obs_id not in self._last_obstacles:
                # New obstacle
                cur_new_obstacles.append(
                    (obs_id, obstacle_prediction.transform))
        self._last_obstacles = obstacles_transform

        self._new_obstacles.append(cur_new_obstacles)
        speed_km = pose.forward_speed * 3.6
        # Friction factor 0.7, which means that the asphalt is pretty dry.
        braking_distance = speed_km * speed_km / (250 * 0.7)

        min_ttd = 274
        for index, obstacles in enumerate(self._new_obstacles):
            for (obs_id, obs_transform) in obstacles:
                # Need to build a history of 8 readings so we can predict
                # reliably. So we multiply 50ms with the number of readings
                # that are still necessary.
                time_to_build_history = 50 * (index + 1)
                reaction_distance = (time_to_build_history +
                                     self._e2e_runtime) * pose.forward_speed
                stopping_distance = reaction_distance + braking_distance
                if obs_id in obstacles_transform:
                    # Get the latest location of the obstacle.
                    dist_to_obstacle = pose.transform.location.distance(
                        obstacles_transform[obs_id].location)
                else:
                    # Lost the obstacle; use the old location.
                    dist_to_obstacle = pose.transform.location.distance(
                        obs_transform.location)
                safety_distance = dist_to_obstacle - stopping_distance
                if safety_distance < 3:
                    min_ttd = min(min_ttd, 60)
                elif safety_distance < 9:
                    min_ttd = min(min_ttd, 110)
        # Detection time (ttd) + P99.9 runtime of the other components.
        self._e2e_runtime = min_ttd + 159
        return min_ttd
