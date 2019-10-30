import threading
import collections

from erdos.op import Op
from erdos.utils import setup_logging
from erdos.message import WatermarkMessage

import pylot.utils
from pylot.perception.messages import DetectorMessage
from pylot.simulation.utils import get_2d_bbox_from_3d_box
from pylot.simulation.carla_utils import get_world
from pylot.perception.detection.utils import DetectedObject


class PerfectPedestrianDetectorOperator(Op):
    """ PerfectPedestrianDetectorOperator returns a bounding box from the
    view of the camera for all the pedestrians visible from the camera.

    This operator depends on input from the depth camera, the location of the
    pedestrians in the world coordinates, and the transform of the vehicle
    for a given timestamp.
    """

    def __init__(self,
                 name,
                 output_stream_name,
                 camera_setup,
                 flags,
                 log_file_name=None):
        """ Initializes the operator with the given information.

        Args:
            name: The name to be used for the operator in the dataflow graph.
            output_stream_name: The name of the stream to output the result to.
            camera_setup: The location and the rotation of the camera to
                return the bounding boxes in respect to.
            flags: The command line flags passed to the driver.
            log_file_name: The file name to log the intermediate messages to.
        """
        super(PerfectPedestrianDetectorOperator, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self._flags = flags
        self._output_stream_name = output_stream_name
        _, world = get_world(self._flags.carla_host, self._flags.carla_port,
                             self._flags.carla_timeout)

        if world is None:
            raise ValueError("There was an issue connecting to the simulator.")

        # Camera information.
        self._camera_intrinsic = camera_setup.get_intrinsic()
        self._camera_transform = camera_setup.get_unreal_transform()
        self._camera_img_size = (camera_setup.width, camera_setup.height)

        # Input retrieved from the various input streams.
        self._lock = threading.Lock()
        self._depth_imgs = collections.deque()
        self._can_bus_msgs = collections.deque()
        self._pedestrians = collections.deque()

    @staticmethod
    def setup_streams(input_streams, output_stream_name):
        """ Registers the appropriate callbacks on the input stream, and
        returns the output stream with the given name.

        Args:
            input_streams: The streams to receive data on.
            output_stream_name: The name of the stream to output the results
                to.

        Returns:
            The stream instance to which the output results are sent.
        """
        # Register a callback on the depth frames data stream.
        input_streams.filter(pylot.utils.is_depth_camera_stream).add_callback(
            PerfectPedestrianDetectorOperator.on_depth_camera_update)

        # Register a callback on can bus messages data stream.
        input_streams.filter(pylot.utils.is_can_bus_stream).add_callback(
            PerfectPedestrianDetectorOperator.on_can_bus_update)

        # Register a callback to retrieve the pedestrian updates from Carla.
        input_streams.filter(
            pylot.utils.is_ground_pedestrians_stream).add_callback(
                PerfectPedestrianDetectorOperator.on_pedestrians_update)

        # Register a callback function to be called when a time is closed.
        input_streams.add_completion_callback(
            PerfectPedestrianDetectorOperator.on_notification)

        # Return the stream on which the output will be sent.
        return [pylot.utils.create_obstacles_stream(output_stream_name)]

    def on_depth_camera_update(self, msg):
        """ Receives the depth camera update and adds it to the queue of
        messages.

        Args:
            msg: The message received for the given timestamp.
        """
        self._logger.info(
            "Received a depth camera update for the timestamp {}".format(
                msg.timestamp))
        with self._lock:
            self._depth_imgs.append(msg)

    def on_can_bus_update(self, msg):
        """ Receives the CAN Bus update and adds it to the queue of messages.

        Args:
            msg: The message received for the given timestamp.
        """
        self._logger.info(
            "Received a CAN Bus update for the timestamp {}".format(
                msg.timestamp))
        with self._lock:
            self._can_bus_msgs.append(msg)

    def on_pedestrians_update(self, msg):
        """ Receives the pedestrian update and adds it to the queue of
        messages.

        Args:
            msg: The message received for the given timestamp.
        """
        self._logger.info(
            "Received a pedestrian update for the timestamp {}".format(
                msg.timestamp))
        with self._lock:
            self._pedestrians.append(msg)

    def synchronize_msg_buffers(self, timestamp, buffers):
        """ Synchronizes the given buffers for the given timestamp.

       Args:
           timestamp: The timestamp to push all the top of the buffers to.
           buffers: The buffers to synchronize.

       Returns:
           True, if the buffers were successfully synchronized. False,
           otherwise.
       """
        for buffer in buffers:
            while (len(buffer) > 0 and buffer[0].timestamp < timestamp):
                buffer.popleft()
            if len(buffer) == 0:
                return False
            assert buffer[0].timestamp == timestamp
        return True

    def on_notification(self, msg):
        """ Initiates the computation at the arrival of a watermark.
        Uses the vehicle transform, the list of pedestrians retrieved from the
        simulation and the depth array retrieved from the depth camera to
        return the bounding boxes for the pedestrian on the output stream.

        Args:
            msg: The WatermarkMessage for the given timestamp.
        """
        self._logger.info("Received a watermark for the timestamp {}".format(
            msg.timestamp))
        with self._lock:
            if not self.synchronize_msg_buffers(
                    msg.timestamp,
                [self._depth_imgs, self._can_bus_msgs, self._pedestrians]):
                self._logger.info("Could not synchronize the message buffers "
                                  "for the timestamp {}".format(msg.timestamp))

            depth_msg = self._depth_imgs.popleft()
            can_bus_msg = self._can_bus_msgs.popleft()
            pedestrian_msg = self._pedestrians.popleft()

        # Assert that the timestamp of all the messages are the same.
        assert (depth_msg.timestamp == can_bus_msg.timestamp ==
                pedestrian_msg.timestamp)

        self._logger.info(
            "Depth Message: {}, Can Bus Message: {}, Pedestrian Message: {}".
            format(depth_msg.timestamp, can_bus_msg.timestamp,
                   pedestrian_msg.timestamp))

        detected_pedestrians = self.__get_pedestrians(
            pedestrian_msg.pedestrians, can_bus_msg.data.transform,
            depth_msg.frame)
        self._logger.info("Detected a total of {} pedestrians: {}".format(
            len(detected_pedestrians), detected_pedestrians))
        output_message = DetectorMessage(detected_pedestrians, 0,
                                         msg.timestamp)
        self.get_output_stream(self._output_stream_name).send(output_message)
        self.get_output_stream(self._output_stream_name).send(
            WatermarkMessage(msg.timestamp))
        self._logger.info(
            "Sent the ground truth of pedestrians for the timestamp {}".format(
                msg.timestamp))

    def __get_pedestrians(self, pedestrians, vehicle_transform, depth_array):
        """ Transforms pedestrians into detected objects.
        Args:
            pedestrians: List of Pedestrian objects.
            vehicle_transform: Ego-vehicle transform.
            depth_array: Depth frame taken at the time when pedestrians were
                         collected.
        """
        det_objs = []
        for pedestrian in pedestrians:
            bbox = get_2d_bbox_from_3d_box(depth_array, vehicle_transform,
                                           pedestrian.transform,
                                           pedestrian.bounding_box,
                                           self._camera_transform,
                                           self._camera_intrinsic,
                                           self._camera_img_size, 1.5, 3.0)
            if bbox is not None:
                det_objs.append(DetectedObject(bbox, 1.0, 'pedestrian'))
        return det_objs

    def execute(self):
        self.spin()
