import time
from collections import deque

import erdos

from pylot.planning.messages import WaypointsMessage
from pylot.planning.waypoints import Waypoints
from pylot.utils import time_epoch_ms


class PlanningPoseSynchronizerOperator(erdos.Operator):
    """ Synchronizes and relays waypoints and pose messages to the control
    module to enable pseudo-synchronization of the simulation with a
    high-frequency control.

    This operator receives waypoints (and the corresponding localization
    messages) along with a high-frequency control stream. It applies the
    waypoints after the given runtime (deduced from the time of the receipt of
    the localization and the waypoints message), and sends both control and
    waypoint messages applicable at the given time to the downstream control
    operator.

    Args:
        waypoints_read_stream (:py:class:`erdos.ReadStream`): Stream on which
            the waypoint updates are received.
        pose_read_stream (:py:class:`erdos.ReadStream`): Stream on which the
            high-frequency pose messages are received. These messages trigger
            a control update from the pipeline.
        localization_read_stream (:py:class:`erdos.ReadStream`): Stream on
            which the localization messages that would trigger a future
            waypoint update are received.
        notify_stream1 (:py:class:`erdos.ReadStream`): Stream on which the
            notifications from the first sensor are received.
        notify_stream2 (:py:class:`erdos.ReadStream`): Stream on which the
            notifications from the second sensor are received.
        waypoints_write_stream (:py:class:`erdos.WriteStream`): Stream on which
            the waypoints matched to the given pose message are sent to the
            downstream control operator.
        pose_write_stream (:py:class:`erdos.WriteStream`): Stream that relays
            the pose messages from the simulator bridge operator to the control
            module.
        release_sensor_stream (:py:class:`erdos.WriteStream`): Stream that
            synchronizes all the sensors and waits for the slowest sensor in
            order to release data from all the sensor simultaneously.
        pipeline_finish_notify_stream (:py:class:`erdos.WriteStream`): Stream
            that notifies the simulation that it has finished and that a new
            sensor input should be released into the system.
    """
    def __init__(self, waypoints_read_stream, pose_read_stream,
                 localization_pose_stream, notify_stream1, notify_stream2,
                 waypoints_write_stream, pose_write_stream,
                 release_sensor_stream, pipeline_finish_notify_stream, flags):
        # Register callbacks on both the waypoints and the pose stream.
        waypoints_read_stream.add_callback(self.on_waypoints_update)
        pose_read_stream.add_callback(self.on_pose_update)
        localization_pose_stream.add_callback(self.on_localization_update)
        erdos.add_watermark_callback([notify_stream1, notify_stream2],
                                     [release_sensor_stream],
                                     self.on_sensor_ready)

        # Register watermark callback on pose and the joined stream.
        erdos.add_watermark_callback(
            [pose_read_stream], [waypoints_write_stream, pose_write_stream],
            self.on_pose_watermark)

        # Save the write streams.
        self._waypoints_write_stream = waypoints_write_stream
        self._pipeline_finish_notify_stream = pipeline_finish_notify_stream

        # Initialize a logger.
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            self.config.name + '-csv', self.config.csv_log_file_name)

        # Data used by the operator.
        self._pose_map = dict()
        self._waypoints = deque()
        self._first_waypoint = True
        self._waypoint_num = 0
        self._last_highest_applicable_time = None
        self._last_localization_update = None

    @staticmethod
    def connect(waypoints_read_stream, pose_read_stream,
                localization_pose_stream, notify_stream1, notify_stream2):
        waypoints_write_stream = erdos.WriteStream()
        pose_write_stream = erdos.WriteStream()
        release_sensor_stream = erdos.WriteStream()
        pipeline_finish_notify_stream = erdos.WriteStream()
        return [
            waypoints_write_stream,
            pose_write_stream,
            release_sensor_stream,
            pipeline_finish_notify_stream,
        ]

    @erdos.profile_method()
    def on_waypoints_update(self, msg):
        """ Invoked upon receipt of a waypoints message from the pipeline.

        This method retrieves the pose message for the timestamp, calculates
        the runtime of the pipeline, logs it and saves the waypoints for the
        future.

        Args:
            msg (:py:class:`pylot.planning.messages.WaypointsMessage`): The
                waypoints message received for the given timestamp.
        """
        waypoint_recv_time = time.time()
        self._logger.debug("@{}: received waypoints update.".format(
            msg.timestamp))

        # Retrieve the game time.
        game_time = msg.timestamp.coordinates[0]

        # Ensure that a single invocation of the pipeline is happening.
        assert self._last_localization_update == game_time, \
            "Concurrent Execution of the pipeline."

        watermark = erdos.WatermarkMessage(msg.timestamp)
        if self._waypoint_num < 10:
            self._logger.debug(
                "@{}: received waypoint num {}. "
                "Skipping because the simulator might not be in sync.".format(
                    msg.timestamp, self._waypoint_num))
            self._first_waypoint = False
            self._waypoint_num += 1
            # Send a message on the notify stream to ask the simulator to send
            # a new sensor stream.
            self._pipeline_finish_notify_stream.send(watermark)
            return

        # Retrieve the pose message for this timestamp.
        (pose_msg, pose_recv_time) = self._pose_map[game_time]

        # Calculate and log the processing time for this waypoints message.
        processing_time = int((waypoint_recv_time - pose_recv_time) * 1000)
        self._csv_logger.info('{},{},{},{:.4f}'.format(time_epoch_ms(),
                                                       game_time,
                                                       'end-to-end-runtime',
                                                       processing_time))

        # Apply the waypoints at the timestamp + processing time.
        applicable_time = game_time + processing_time
        if (self._last_highest_applicable_time is None
                or self._last_highest_applicable_time < applicable_time):
            self._last_highest_applicable_time = applicable_time
            self._waypoints.append((applicable_time, msg))
            self._logger.debug(
                "@{}: waypoints will be applicable at {}".format(
                    msg.timestamp, applicable_time))
        else:
            # The last waypoint applicable time was higher, we should purge
            # the ones higher than this one and add this entry.
            self._logger.debug(
                "@{}: Popping the last applicable time: {}".format(
                    msg.timestamp, self._waypoints[-1][0]))

            assert (
                self._waypoints.pop()[0] == self._last_highest_applicable_time)
            while self._waypoints[-1][0] >= applicable_time:
                self._logger.debug(
                    "@{}: Popping the last applicable time: {}".format(
                        msg.timestamp, self._waypoints[-1][0]))
                self._waypoints.pop()
            self._last_highest_applicable_time = applicable_time
            self._waypoints.append((applicable_time, msg))
            self._logger.debug("@{}: the waypoints were adjusted "
                               "and will be applicable at {}".format(
                                   msg.timestamp, applicable_time))

        # Delete the pose from the map.
        self._pose_map.pop(game_time, None)

        # Send a message on the notify stream to ask the simulator to send a
        # new sensor stream.
        self._pipeline_finish_notify_stream.send(watermark)

    def on_pose_update(self, msg):
        """ Invoked when we receive a pose message from the simulation.

        Save the pose message to be used on the receipt of the watermark.

        Args:
            msg (:py:class:`erdos.Message`): A message containing the pose
                of the ego-vehicle.
        """
        self._logger.debug("@{}: received pose message.".format(msg.timestamp))

        # Retrieve the game time.
        game_time = msg.timestamp.coordinates[0]

        # Save the pose message along with the time at which it was received.
        if game_time in self._pose_map:
            self._pose_map[game_time][0] = msg
        else:
            self._pose_map[game_time] = [msg, None]

    def on_localization_update(self, msg):
        """ Invoked upon receipt of a localization message that will lead
        to a future waypoint update.

        This callback logs the arrival time of the message in order to compute
        the runtime of the pipeline.

        Args:
            msg (:py:class:`erdos.Message`): A message containing the pose
                of the ego-vehicle.
        """
        self._logger.debug("@{}: received localization message.".format(
            msg.timestamp))

        # Retrieve the game time.
        game_time = msg.timestamp.coordinates[0]

        # Save the pose message along with the time at which it was received.
        self._pose_map[game_time] = [msg, time.time()]

        # Save the last localization message received to ensure that only a
        # single invocation of the pipeline happens at a time.
        self._last_localization_update = game_time

    def on_pose_watermark(self, timestamp, waypoint_stream, pose_stream):
        """ Invoked upon receipt of the watermark on the pose stream.

        This callback matches the waypoints to the given timestamp and releases
        both the waypoints and the pose message to the control operator.

        Args:
            timestamp (:py:class:`erdos.Timestamp`): The timestamp of the
                watermark.
            waypoint_stream (:py:class:`erdos.WriteStream`): The stream to send
                the waypoints out on.
            pose_stream (:py:class:`erdos.WriteStream`): The stream to send
                the pose out on.
        """
        self._logger.info("@{}: received pose watermark.".format(timestamp))

        # Retrieve the game time.
        game_time = timestamp.coordinates[0]

        # Retrieve the pose message for the given timestamp.
        pose_msg, pose_ingress_time = self._pose_map[game_time]

        # Match the waypoints to the given timestamp.
        waypoint_index, waypoints = -1, None
        for i, (sim_time, _waypoints) in enumerate(self._waypoints):
            if sim_time <= game_time:
                waypoint_index, waypoints = i, _waypoints
            else:
                break
        self._logger.debug("@{} waypoint index is {}".format(
            timestamp, waypoint_index))

        if waypoints is None:
            # If we haven't received a single waypoint, send an empty message.
            self._waypoints_write_stream.send(
                WaypointsMessage(timestamp, Waypoints(deque([]), deque([]))))
        else:
            # Send the trimmed waypoints on the write stream.
            waypoints.remove_completed(pose_msg.data.transform.location,
                                       pose_msg.data.transform)
            self._waypoints_write_stream.send(
                WaypointsMessage(timestamp, waypoints))

        # Send the pose and the watermark messages.
        watermark = erdos.WatermarkMessage(timestamp)
        pose_stream.send(pose_msg)
        pose_stream.send(watermark)
        waypoint_stream.send(watermark)

        # Clean up the pose from the dict.
        self._pose_map.pop(game_time, None)

    @erdos.profile_method()
    def on_sensor_ready(self, timestamp, release_sensor_stream):
        """ Invoked upon receipt of a notification of the sensors being
        ready for the given timestamp.

        Releases a watermark on the release_sensor_stream to notify all the
        sensors to release their data for the given timestamp.

        Args:
            timestamp (:py:class:`erdos.Timestamp`): The timestamp of the
                watermark.
            sensor_ready_stream (:py:class:`erdos.WriteStream`): The stream
                on which to write the notification.
        """
        self._logger.debug("@{}: the sensors are all ready.".format(timestamp))
        release_sensor_stream.send(erdos.WatermarkMessage(timestamp))

        # Retrieve the game time.
        game_time = timestamp.coordinates[0]

        # Also rewrite the receive time for the pose update because the sensor
        # callbacks might take too long.
        if game_time in self._pose_map:
            self._pose_map[game_time][1] = time.time()
        else:
            self._pose_map[game_time] = [None, time.time()]
