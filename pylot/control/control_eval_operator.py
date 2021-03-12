from collections import deque

import erdos
from erdos import Message, ReadStream, Timestamp

import numpy as np

from pylot.utils import Transform, time_epoch_ms


class ControlEvalOperator(erdos.Operator):
    """Operator that computes the accuracy metrics using reference waypoints
    and the achieved waypoints.

    Args:
        pose_stream (:py:class:`erdos.ReadStream`): The stream on which the
            vehicle transform is received.
        waypoints_stream (:py:class:`erdos.ReadStream`): The stream on which
            the waypoints are received from the planner.
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, pose_stream: ReadStream, waypoints_stream: ReadStream,
                 flags):
        # Register callbacks to retrieve pose and waypoints messages.
        pose_stream.add_callback(self.on_pose_update)
        waypoints_stream.add_callback(self.on_waypoint_update)

        # Add a watermark callback on pose stream and waypoints stream.
        erdos.add_watermark_callback([pose_stream, waypoints_stream], [],
                                     self.on_watermark)

        # Initialize state.
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            self.config.name + '-csv', self.config.csv_log_file_name)
        self._flags = flags
        self._waypoints_msgs = deque()
        self._pose_messages = deque()

        # Keep track of the last 2 waypoints that the ego vehicle was
        # supposed to achieve.
        self.last_waypoints = deque(maxlen=2)

    @staticmethod
    def connect(pose_stream: ReadStream, waypoints_stream: ReadStream):
        # This operator does not have any write streams.
        return []

    def destroy(self):
        self._logger.warn('destroying {}'.format(self.config.name))

    def on_pose_update(self, msg: Message):
        """Callback function for the pose update messages.

        This function appends the received message to the operator state.

        Args:
            msg (:py:class:`erdos.Message`): The message contains an instance
                of :py:class:`~pylot.utils.Pose`.
        """
        self._logger.debug('@{}: pose update.'.format(msg.timestamp))
        self._pose_messages.append(msg)

    def on_waypoint_update(self, msg: Message):
        """Callback function for the waypoint update messages.

        This function appends the received message to the operator state.

        Args:
            msg (:py:class:`~pylot.planning.messages.WaypointsMessage`): The
                message contains the waypoints for the future trajectory, as
                computed by the planner.
        """
        self._logger.debug('@{}: waypoints update.'.format(msg.timestamp))
        self._waypoints_msgs.append(msg)

    @erdos.profile_method()
    def on_watermark(self, timestamp: Timestamp):
        """Computes and logs the metrics of accuracy for the control module.

        This operator uses two different metrics of accuracy, as follows:
            1. Crosstrack error: The distance between the waypoint that the
                planner intended the vehicle to achieve, and the location
                achieved by the control module.
            2. Heading error: The angle between the heading of the vehicle,
                and the reference trajectory that the planner intended the
                vehicle to follow.

        The log format is (timestamp, crosstrack_error, heading_error).

        Args:
            timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp
                of the watermark.
        """
        self._logger.debug('@{}: received watermark.'.format(timestamp))
        if timestamp.is_top:
            return

        # Get the transform of the ego vehicle.
        pose_msg = self._pose_messages.popleft()
        vehicle_transform = pose_msg.data.transform

        if len(self.last_waypoints) == 2:
            # Compute the metrics of accuracy using the last two waypoints.
            self._logger.debug(
                "@{}: vehicle location: {}, waypoints: {}".format(
                    timestamp, vehicle_transform.location,
                    self.last_waypoints))
            crosstrack_err, heading_err = ControlEvalOperator.\
                compute_control_metrics(vehicle_transform,
                                        self.last_waypoints)

            self._csv_logger.info("{},{},control,{:.4f},{:.4f}".format(
                time_epoch_ms(), timestamp.coordinates[0], crosstrack_err,
                heading_err))

        # Add the first waypoint from the last waypoints received
        # by the operator.
        waypoints = self._waypoints_msgs.popleft().waypoints.waypoints
        if len(waypoints) > 0:
            next_waypoint = waypoints.popleft()
            while len(self.last_waypoints) >= 1 and len(
                    waypoints) > 0 and np.isclose(
                        next_waypoint.location.distance(
                            self.last_waypoints[-1].location), 0.0):
                next_waypoint = waypoints.popleft()
            self.last_waypoints.append(next_waypoint)

    @staticmethod
    def compute_control_metrics(vehicle_transform: Transform,
                                reference_waypoints):
        """Compute the metrics of accuracy for the control module.

        Args:
            vehicle_transform (:py:class:`~pylot.utils.Transform`): The
                transform of the vehicle.
            reference_waypoints (:py:class:`tuple`): A tuple of two reference
                waypoints.

        Returns:
            A tuple of (crosstrack_error, heading_error), as defined in the
            on_watermark function.
        """
        if len(reference_waypoints) != 2:
            raise ValueError("Expected 2 reference waypoints, got {}".format(
                len(reference_waypoints)))

        old_point, new_point = reference_waypoints

        # Calculate the crosstrack error.
        crosstrack_error = vehicle_transform.location.distance(
            new_point.location)

        # Calculate the heading error.
        vehicle_forward_vector = vehicle_transform.forward_vector.\
            as_numpy_array()

        # Create a unit vector between the two reference waypoints.
        waypoint_vector = new_point.location.as_numpy_array() - \
            old_point.location.as_numpy_array()
        waypoint_vector = waypoint_vector / np.linalg.norm(waypoint_vector)

        # Get the angle between the two vectors.
        heading_error = np.arccos(
            np.clip(np.dot(waypoint_vector, vehicle_forward_vector), -1.0,
                    1.0))

        return (crosstrack_error, heading_error)
