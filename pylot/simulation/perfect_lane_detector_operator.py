from collections import deque
import numpy as np
import erdos
from erdos import Message, ReadStream, Timestamp, WriteStream

from pylot.perception.messages import LanesMessage
from pylot.perception.camera_frame import CameraFrame


class PerfectLaneDetectionOperator(erdos.Operator):
    """Operator that uses the simulator to perfectly detect lanes.

    Args:
        pose_stream (:py:class:`erdos.ReadStream`): Stream on which pose
            info is received.
        open_drive_stream (:py:class:`erdos.ReadStream`): Stream on which open
            drive string representations are received. The operator can
            construct HDMaps out of the open drive strings.
        detected_lane_stream (:py:class:`erdos.WriteStream`): Stream on which
            the operator writes
            :py:class:`~pylot.perception.messages.LanesMessage` messages.
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, pose_stream: ReadStream, open_drive_stream: ReadStream,
                 center_camera_stream: ReadStream,
                 detected_lane_stream: WriteStream, flags):
        pose_stream.add_callback(self.on_pose_update)
        center_camera_stream.add_callback(self.on_bgr_camera_update)
        erdos.add_watermark_callback([pose_stream, center_camera_stream],
                                     [detected_lane_stream],
                                     self.on_position_update)
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._bgr_msgs = deque()
        self._pose_msgs = deque()
        self._frame_cnt = 0

    @staticmethod
    def connect(pose_stream: ReadStream, open_drive_stream: ReadStream,
                center_camera_stream: ReadStream):
        detected_lane_stream = erdos.WriteStream()
        return [detected_lane_stream]

    def run(self):
        # Run method is invoked after all operators finished initializing.
        # Thus, we're sure the world is up-to-date here.
        if self._flags.execution_mode == 'simulation':
            from pylot.map.hd_map import HDMap
            from pylot.simulation.utils import get_map
            self._map = HDMap(
                get_map(self._flags.simulator_host, self._flags.simulator_port,
                        self._flags.simulator_timeout),
                self.config.log_file_name)
            from pylot.simulation.utils import get_world
            _, self._world = get_world(self._flags.simulator_host,
                                       self._flags.simulator_port,
                                       self._flags.simulator_timeout)

    def on_opendrive_map(self, msg: Message):
        """Invoked whenever a message is received on the open drive stream.

        Args:
            msg (:py:class:`~erdos.message.Message`): Message that contains
                the open drive string.
        """
        self._logger.debug('@{}: received open drive message'.format(
            msg.timestamp))
        from pylot.simulation.utils import map_from_opendrive
        self._map = map_from_opendrive(msg.data)

    def on_bgr_camera_update(self, msg: Message):
        self._logger.debug('@{}: received BGR frame'.format(msg.timestamp))
        self._bgr_msgs.append(msg)

    def on_pose_update(self, msg: Message):
        self._logger.debug('@{}: received pose message'.format(msg.timestamp))
        self._pose_msgs.append(msg)

    @erdos.profile_method()
    def on_position_update(self, timestamp: Timestamp,
                           detected_lane_stream: WriteStream):
        """Invoked on the receipt of an update to the position of the vehicle.

        Uses the position of the vehicle to get future waypoints and draw
        lane markings using those waypoints.

        Args:
            pose_msg: Contains the current location of the ego vehicle.
        """
        self._logger.debug('@{}: received watermark'.format(timestamp))
        bgr_msg = self._bgr_msgs.popleft()
        pose_msg = self._pose_msgs.popleft()
        vehicle_location = pose_msg.data.transform.location
        if self._map:
            lanes = self._map.get_all_lanes(vehicle_location)
            if self._flags.log_lane_detection_camera:
                camera_setup = bgr_msg.frame.camera_setup
                black_img = np.zeros(
                    (camera_setup.height, camera_setup.width, 3),
                    dtype=np.dtype("uint8"))
                frame = CameraFrame(black_img, 'BGR', camera_setup)
                binary_frame = CameraFrame(black_img.copy(), 'BGR',
                                           camera_setup)
                for lane in lanes:
                    lane.draw_on_frame(frame,
                                       inverse_transform=pose_msg.data.
                                       transform.inverse_transform(),
                                       binary_frame=binary_frame)
                self._logger.debug('@{}: detected {} lanes'.format(
                    bgr_msg.timestamp, len(lanes)))
                self._frame_cnt += 1
                if self._frame_cnt % self._flags.log_every_nth_message == 0:
                    frame.save(bgr_msg.timestamp.coordinates[0],
                               self._flags.data_path, "lane")
                    binary_frame.save(bgr_msg.timestamp.coordinates[0],
                                      self._flags.data_path, "binary-lane")
            else:
                for lane in lanes:
                    lane.draw_on_world(self._world)
        else:
            self._logger.debug('@{}: map is not ready yet'.format(
                pose_msg.timestamp))
            lanes = []
        output_msg = LanesMessage(pose_msg.timestamp, lanes)
        detected_lane_stream.send(output_msg)
        detected_lane_stream.send(erdos.WatermarkMessage(pose_msg.timestamp))
